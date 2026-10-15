"""Online METRA with OGBench environment interaction and SAC skill learning."""

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, MLP


def normalize(x, eps=1e-6):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def sample_unit_sphere(key, batch_size, latent_dim):
    return normalize(jax.random.normal(key, (batch_size, latent_dim)))


class OnlineMETRAAgent(flax.struct.PyTreeNode):
    """METRA's online representation objective and skill-conditioned SAC."""

    rng: Any
    representation_network: Any
    actor_network: Any
    critic_network: Any
    log_alpha: Any
    alpha_opt_state: Any
    log_lambda: Any
    lambda_opt_state: Any
    config: Any = nonpytree_field()
    alpha_tx: Any = nonpytree_field()
    lambda_tx: Any = nonpytree_field()

    def representation_loss(self, batch, grad_params):
        phi_s = self.representation_network.select('phi')(batch['observations'], params=grad_params)
        phi_next = self.representation_network.select('phi')(batch['next_observations'], params=grad_params)
        delta = phi_next - phi_s
        progress = jnp.sum(delta * batch['skills'], axis=-1)

        # METRA's dual_dist='one': cst_dist is one, and the squared latent
        # displacement is averaged across latent coordinates in the reference code.
        constraint_penalty = 1.0 - jnp.mean(delta**2, axis=-1)
        constraint_penalty = jnp.minimum(constraint_penalty, self.config['dual_slack'])
        objective = progress + jax.lax.stop_gradient(jnp.exp(self.log_lambda)) * constraint_penalty
        loss = -jnp.mean(objective)
        return loss, {
            'representation/objective': jnp.mean(objective),
            'representation/progress': jnp.mean(progress),
            'representation/constraint': jnp.mean(constraint_penalty),
            'representation/lambda': jnp.exp(self.log_lambda),
            'representation/delta_norm': jnp.mean(jnp.linalg.norm(delta, axis=-1)),
            'representation/latent_variance': jnp.mean(jnp.var(phi_s, axis=0)),
        }

    def critic_loss(self, batch, grad_params, key):
        next_dist = self.actor_network.select('actor')(batch['next_observations'], batch['skills'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=key)
        target_q1 = self.critic_network.select('target_q1')(
            batch['next_observations'], batch['skills'], actions=next_actions
        )
        target_q2 = self.critic_network.select('target_q2')(
            batch['next_observations'], batch['skills'], actions=next_actions
        )
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = target_q - jax.lax.stop_gradient(jnp.exp(self.log_alpha)) * next_log_probs
        q_target = jax.lax.stop_gradient(
            batch['rewards'] + self.config['discount'] * (1.0 - batch['dones']) * target_q
        )

        q1 = self.critic_network.select('q1')(
            batch['observations'], batch['skills'], actions=batch['actions'], params=grad_params
        )
        q2 = self.critic_network.select('q2')(
            batch['observations'], batch['skills'], actions=batch['actions'], params=grad_params
        )
        q1_loss = 0.5 * jnp.mean((q1 - q_target) ** 2)
        q2_loss = 0.5 * jnp.mean((q2 - q_target) ** 2)
        return q1_loss + q2_loss, {
            'sac/q1_loss': q1_loss,
            'sac/q2_loss': q2_loss,
            'sac/q_mean': jnp.mean(jnp.minimum(q1, q2)),
            'sac/q_target_mean': jnp.mean(q_target),
        }

    def actor_loss(self, batch, grad_params, key, critic_network):
        dist = self.actor_network.select('actor')(
            batch['observations'], batch['skills'], params=grad_params
        )
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q1 = critic_network.select('q1')(batch['observations'], batch['skills'], actions=actions)
        q2 = critic_network.select('q2')(batch['observations'], batch['skills'], actions=actions)
        q = jnp.minimum(q1, q2)
        alpha = jax.lax.stop_gradient(jnp.exp(self.log_alpha))
        loss = jnp.mean(alpha * log_probs - q)
        return loss, {
            'sac/actor_loss': loss,
            'sac/policy_log_prob': jnp.mean(log_probs),
            'sac/policy_q': jnp.mean(q),
        }

    @jax.jit
    def update(self, batch):
        rng, critic_key, actor_key = jax.random.split(self.rng, 3)

        new_representation_network, rep_info = self.representation_network.apply_loss_fn(
            lambda params: self.representation_loss(batch, params)
        )
        # METRA updates log(lambda) by minimizing log(lambda) * mean(constraint).
        constraint = rep_info['representation/constraint']
        lambda_grad = jax.grad(lambda value: value * jax.lax.stop_gradient(constraint))(self.log_lambda)
        lambda_updates, lambda_opt_state = self.lambda_tx.update(
            lambda_grad, self.lambda_opt_state, self.log_lambda
        )
        log_lambda = optax.apply_updates(self.log_lambda, lambda_updates)

        # The original implementation updates phi first, then computes rewards
        # from that updated encoder before the SAC updates.
        phi_s = new_representation_network.select('phi')(batch['observations'])
        phi_next = new_representation_network.select('phi')(batch['next_observations'])
        rewards = jax.lax.stop_gradient(jnp.sum((phi_next - phi_s) * batch['skills'], axis=-1))
        sac_batch = dict(batch, rewards=rewards)

        new_critic_network, critic_info = self.critic_network.apply_loss_fn(
            lambda params: self.critic_loss(sac_batch, params, critic_key)
        )
        new_actor_network, actor_info = self.actor_network.apply_loss_fn(
            lambda params: self.actor_loss(sac_batch, params, actor_key, new_critic_network)
        )

        # METRA's target entropy is -action_dim / 2 (target_coef=1).
        alpha_grad = jax.grad(
            lambda value: -value * jnp.mean(
                jax.lax.stop_gradient(actor_info['sac/policy_log_prob'] + self.config['target_entropy'])
            )
        )(self.log_alpha)
        alpha_updates, alpha_opt_state = self.alpha_tx.update(
            alpha_grad, self.alpha_opt_state, self.log_alpha
        )
        log_alpha = optax.apply_updates(self.log_alpha, alpha_updates)

        # Polyak update of both target critics after the SAC updates.
        critic_params = flax.core.unfreeze(new_critic_network.params)
        for name in ('q1', 'q2'):
            source = critic_params[f'modules_{name}']
            target_name = f'modules_target_{name}'
            target = critic_params[target_name]
            critic_params[target_name] = jax.tree_util.tree_map(
                lambda online, old: self.config['tau'] * online + (1.0 - self.config['tau']) * old,
                source,
                target,
            )
        new_critic_network = new_critic_network.replace(params=flax.core.freeze(critic_params))

        info = dict(rep_info)
        info.update(critic_info)
        info.update(actor_info)
        info.update({
            'representation/lambda': jnp.exp(log_lambda),
            'sac/alpha': jnp.exp(log_alpha),
            'sac/reward_mean': jnp.mean(rewards),
            'sac/reward_std': jnp.std(rewards),
            'sac/terminal_fraction': jnp.mean(batch['dones']),
            'sac/skill_norm': jnp.mean(jnp.linalg.norm(batch['skills'], axis=-1)),
        })
        return self.replace(
            rng=rng,
            representation_network=new_representation_network,
            actor_network=new_actor_network,
            critic_network=new_critic_network,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            log_lambda=log_lambda,
            lambda_opt_state=lambda_opt_state,
        ), info

    def compute_skill(self, observations, goals):
        phi_s = self.representation_network.select('phi')(observations)
        phi_goal = self.representation_network.select('phi')(goals)
        return normalize(phi_goal - phi_s)

    @jax.jit
    def sample_actions(self, observations, goals=None, skill=None, seed=None, temperature=1.0):
        if skill is None:
            if goals is None:
                raise ValueError('Provide a goal or a skill direction.')
            skill = self.compute_skill(observations, goals)
        dist = self.actor_network.select('actor')(observations, skill, temperature=temperature)
        actions = jax.lax.cond(
            temperature <= 0,
            lambda _: dist.mode(),
            lambda _: dist.sample(seed=seed),
            operand=None,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng, rep_rng, actor_rng, critic_rng = jax.random.split(jax.random.PRNGKey(seed), 4)
        latent_dim = config['latent_dim']
        example_skills = jnp.ones((ex_observations.shape[0], latent_dim), dtype=jnp.float32) / jnp.sqrt(latent_dim)

        rep_def = ModuleDict({
            'phi': MLP(
                (*config['hidden_dims'], latent_dim),
                activations=nn.relu,
                activate_final=False,
                layer_norm=False,
            )
        })
        rep_params = rep_def.init(rep_rng, phi=(ex_observations,))['params']
        representation_network = TrainState.create(
            rep_def, rep_params, tx=optax.adam(config['learning_rate'])
        )

        actor_def = ModuleDict({
            'actor': GCActor(
                hidden_dims=config['hidden_dims'],
                action_dim=ex_actions.shape[-1],
                log_std_min=config['actor_log_std_min'],
                log_std_max=config['actor_log_std_max'],
                tanh_squash=True,
                state_dependent_std=True,
                const_std=False,
                activations=nn.tanh,
                final_fc_init_scale=1.0,
                output_kernel_init=nn.initializers.xavier_normal(),
                log_std_bias_init=0.0,
            )
        })
        actor_params = actor_def.init(actor_rng, actor=(ex_observations, example_skills))['params']
        actor_network = TrainState.create(actor_def, actor_params, tx=optax.adam(config['learning_rate']))

        def make_q():
            return GCValue(
                hidden_dims=config['hidden_dims'], layer_norm=False, ensemble=False, activations=nn.relu
            )

        critic_def = ModuleDict({
            'q1': make_q(),
            'q2': make_q(),
            'target_q1': make_q(),
            'target_q2': make_q(),
        })
        critic_args = {
            name: (ex_observations, example_skills, ex_actions)
            for name in ('q1', 'q2', 'target_q1', 'target_q2')
        }
        critic_params = flax.core.unfreeze(critic_def.init(critic_rng, **critic_args)['params'])
        for name in ('q1', 'q2'):
            critic_params[f'modules_target_{name}'] = jax.tree_util.tree_map(
                lambda value: value.copy(), critic_params[f'modules_{name}']
            )
        critic_network = TrainState.create(
            critic_def, flax.core.freeze(critic_params), tx=optax.adam(config['learning_rate'])
        )

        config = dict(config)
        config['target_entropy'] = -ex_actions.shape[-1] / 2.0 * config['target_entropy_coef']
        log_alpha = jnp.asarray(jnp.log(config['alpha_init']), dtype=jnp.float32)
        log_lambda = jnp.asarray(jnp.log(config['lambda_init']), dtype=jnp.float32)
        alpha_tx = optax.adam(config['learning_rate'])
        lambda_tx = optax.adam(config['learning_rate'])
        return cls(
            rng=rng,
            representation_network=representation_network,
            actor_network=actor_network,
            critic_network=critic_network,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_tx.init(log_alpha),
            log_lambda=log_lambda,
            lambda_opt_state=lambda_tx.init(log_lambda),
            config=flax.core.FrozenDict(**config),
            alpha_tx=alpha_tx,
            lambda_tx=lambda_tx,
        )


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='online_metra',
        # METRA reference settings used by run_metra_ant.sh / run_metra_maze.sh.
        latent_dim=2,
        hidden_dims=(1024, 1024),
        learning_rate=1e-4,
        batch_size=256,
        traj_batch_size=8,
        trans_optimization_epochs=50,
        min_buffer_size=10000,
        max_buffer_size=1000000,
        max_episode_steps=200,
        eval_skill_horizon=1,
        alpha_init=0.01,
        target_entropy_coef=1.0,
        actor_log_std_min=-13.815510557964274,
        actor_log_std_max=2.0,
        discount=0.99,
        tau=0.005,
        lambda_init=30.0,
        dual_slack=1e-3,
    ))
