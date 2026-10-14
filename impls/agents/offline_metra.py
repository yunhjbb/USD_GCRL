"""Offline METRA-style representation learning with hindsight goal directions."""

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


def representation_objective(
    phi_s,
    phi_next,
    phi_goal,
    lagrange_multiplier,
    constraint_epsilon,
):
    """Return the negative linear METRA objective for a fixed hindsight skill."""
    delta = phi_next - phi_s
    # Compute a hindsight skill from phi, then hold it fixed for this update.
    direction = jax.lax.stop_gradient(normalize(phi_goal - phi_s))
    progress = jnp.sum(delta * direction, axis=-1)
    # Match the neighboring METRA implementation's per-coordinate squared-distance scaling.
    mean_delta_sq = jnp.mean(delta**2, axis=-1)
    constraint = jnp.minimum(constraint_epsilon, 1.0 - mean_delta_sq)
    objective = progress + lagrange_multiplier * constraint
    loss = -jnp.mean(objective)
    info = {
        'objective': jnp.mean(objective),
        'progress': jnp.mean(progress),
        'constraint': jnp.mean(constraint),
        'delta_norm': jnp.mean(jnp.linalg.norm(delta, axis=-1)),
        'delta_norm_sq': jnp.mean(jnp.sum(delta**2, axis=-1)),
        'delta_mean_sq': jnp.mean(mean_delta_sq),
        'lipschitz_violation': jnp.mean((mean_delta_sq > 1.0).astype(jnp.float32)),
        'latent_variance': jnp.mean(jnp.var(phi_s, axis=0)),
        'direction_norm': jnp.mean(jnp.linalg.norm(direction, axis=-1)),
    }
    return loss, info


def intrinsic_reward(phi_s, phi_next, skills):
    """Compute a detached directional reward for the offline skill learner."""
    reward = jnp.sum((phi_next - phi_s) * skills, axis=-1)
    return jax.lax.stop_gradient(reward)


def lagrange_dual_loss(log_lambda, mean_constraint):
    """METRA's log-parameterized dual loss, minimized by gradient descent."""
    return log_lambda * jax.lax.stop_gradient(mean_constraint)


def sample_unit_sphere(key, batch_size, latent_dim):
    skills = jax.random.normal(key, (batch_size, latent_dim))
    return normalize(skills)


class OfflineMETRAAgent(flax.struct.PyTreeNode):
    rng: Any
    representation_network: Any
    iql_network: Any
    log_lambda: Any
    lambda_opt_state: Any
    config: Any = nonpytree_field()
    lambda_tx: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(advantage, diff, expectile):
        weight = jnp.where(advantage >= 0, expectile, 1.0 - expectile)
        return weight * diff**2

    def representation_loss(self, batch, grad_params):
        phi_s = self.representation_network.select('phi')(batch['observations'], params=grad_params)
        phi_next = self.representation_network.select('phi')(batch['next_observations'], params=grad_params)
        phi_goal = self.representation_network.select('phi')(batch['future_goals'], params=grad_params)
        loss, info = representation_objective(
            phi_s,
            phi_next,
            phi_goal,
            jax.lax.stop_gradient(jnp.exp(self.log_lambda)),
            self.config['constraint_epsilon'],
        )
        return loss, {f'representation/{key}': value for key, value in info.items()}

    def iql_loss(self, batch, skills, rewards, grad_params):
        observations = batch['observations']
        next_observations = batch['next_observations']
        actions = batch['actions']
        dones = batch['terminals']

        target_q1 = self.iql_network.select('target_q1')(observations, skills, actions)
        target_q2 = self.iql_network.select('target_q2')(observations, skills, actions)
        target_q = jax.lax.stop_gradient(jnp.minimum(target_q1, target_q2))
        value = self.iql_network.select('value')(observations, skills, params=grad_params)
        value_advantage = target_q - value
        value_loss = self.expectile_loss(value_advantage, value_advantage, self.config['expectile']).mean()

        next_value = self.iql_network.select('value')(next_observations, skills)
        q_target = jax.lax.stop_gradient(
            rewards + self.config['discount'] * (1.0 - dones) * next_value
        )
        q1 = self.iql_network.select('q1')(observations, skills, actions, params=grad_params)
        q2 = self.iql_network.select('q2')(observations, skills, actions, params=grad_params)
        q1_loss = jnp.mean((q1 - q_target) ** 2)
        q2_loss = jnp.mean((q2 - q_target) ** 2)

        actor_dist = self.iql_network.select('actor')(observations, skills, params=grad_params)
        log_prob = actor_dist.log_prob(actions)
        advantage = jax.lax.stop_gradient(target_q - self.iql_network.select('value')(observations, skills))
        log_weight_cap = jnp.log(self.config['max_weight'])
        weights = jnp.exp(jnp.minimum(self.config['awr_beta'] * advantage, log_weight_cap))
        weights = jax.lax.stop_gradient(weights)
        actor_loss = -jnp.mean(weights * log_prob)

        return value_loss + q1_loss + q2_loss + actor_loss, {
            'iql/value_loss': value_loss,
            'iql/q1_loss': q1_loss,
            'iql/q2_loss': q2_loss,
            'iql/actor_loss': actor_loss,
            'iql/q_mean': jnp.mean(target_q),
            'iql/value_mean': jnp.mean(value),
            'iql/advantage_mean': jnp.mean(advantage),
            'iql/awr_weight_mean': jnp.mean(weights),
            'iql/reward_mean': jnp.mean(rewards),
            'iql/reward_std': jnp.std(rewards),
            'iql/terminal_fraction': jnp.mean(dones),
        }

    @jax.jit
    def update(self, batch):
        rep_batch = batch['representation']
        skill_batch = batch['skill']
        new_rng, skill_rng = jax.random.split(self.rng, 2)
        # Update the goal-directed representation from hindsight goals.
        new_representation_network, rep_info = self.representation_network.apply_loss_fn(
            lambda params: self.representation_loss(rep_batch, params)
        )

        # Match METRA's log-lambda dual update.
        constraint = rep_info['representation/constraint']
        lambda_grad = jax.grad(lagrange_dual_loss)(self.log_lambda, constraint)
        lambda_updates, new_lambda_opt_state = self.lambda_tx.update(
            lambda_grad, self.lambda_opt_state, self.log_lambda
        )
        new_log_lambda = optax.apply_updates(self.log_lambda, lambda_updates)

        # Draw one independent unit-sphere skill per fixed-dataset transition.
        skills = sample_unit_sphere(skill_rng, len(skill_batch['observations']), self.config['latent_dim'])
        phi_s = new_representation_network.select('phi')(skill_batch['observations'])
        phi_next = new_representation_network.select('phi')(skill_batch['next_observations'])
        rewards = intrinsic_reward(phi_s, phi_next, skills)

        new_iql_network, iql_info = self.iql_network.apply_loss_fn(
            lambda params: self.iql_loss(skill_batch, skills, rewards, params)
        )
        new_iql_network = self._update_target_qs(new_iql_network)

        info = dict(rep_info)
        info.update(iql_info)
        info['representation/lambda'] = jnp.exp(new_log_lambda)
        info['representation/lambda_loss'] = new_log_lambda * constraint
        info['iql/skill_norm'] = jnp.mean(jnp.linalg.norm(skills, axis=-1))
        return self.replace(
            rng=new_rng,
            representation_network=new_representation_network,
            iql_network=new_iql_network,
            log_lambda=new_log_lambda,
            lambda_opt_state=new_lambda_opt_state,
        ), info

    def _update_target_qs(self, network):
        params = flax.core.unfreeze(network.params)
        for name in ('q1', 'q2'):
            target_name = f'modules_target_{name}'
            source_name = f'modules_{name}'
            params[target_name] = jax.tree_util.tree_map(
                lambda online, target: self.config['tau'] * online + (1.0 - self.config['tau']) * target,
                params[source_name],
                params[target_name],
            )
        return network.replace(params=flax.core.freeze(params))

    @jax.jit
    def compute_skill(self, observations, goals):
        phi_s = self.representation_network.select('phi')(observations)
        phi_g = self.representation_network.select('phi')(goals)
        return normalize(phi_g - phi_s)

    @jax.jit
    def compute_intrinsic_reward(self, observations, next_observations, skills):
        phi_s = self.representation_network.select('phi')(observations)
        phi_next = self.representation_network.select('phi')(next_observations)
        return intrinsic_reward(phi_s, phi_next, skills)

    @jax.jit
    def sample_actions(self, observations, goals=None, skill=None, seed=None, temperature=1.0):
        if seed is None:
            seed = self.rng
        if skill is None:
            if goals is None:
                raise ValueError('Provide a goal or an explicit skill direction.')
            skill = self.compute_skill(observations, goals)
        dist = self.iql_network.select('actor')(observations, skill, temperature=temperature)
        actions = jax.lax.cond(
            temperature <= 0,
            lambda _: dist.mode(),
            lambda _: dist.sample(seed=seed),
            operand=None,
        )
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rep_rng, iql_rng = jax.random.split(rng, 3)
        latent_dim = config['latent_dim']
        ex_skills = jnp.ones((ex_observations.shape[0], latent_dim), dtype=jnp.float32) / jnp.sqrt(latent_dim)

        rep_def = ModuleDict({
            'phi': MLP(
                (*config['representation_hidden_dims'], latent_dim),
                activations=nn.relu,
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        })
        rep_args = {'phi': (ex_observations,)}
        rep_params = rep_def.init(rep_rng, **rep_args)['params']
        rep_network = TrainState.create(
            rep_def,
            rep_params,
            tx=optax.adam(config['representation_lr']),
        )

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            activations=nn.relu,
        )
        def make_q():
            return GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                activations=nn.relu,
            )

        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ex_actions.shape[-1],
            log_std_min=config['actor_log_std_min'],
            log_std_max=config['actor_log_std_max'],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            activations=nn.relu,
            final_fc_init_scale=1.0,
            output_kernel_init=nn.initializers.xavier_normal(),
            log_std_bias_init=1.0,
        )
        iql_modules = {
            'value': value_def,
            'q1': make_q(),
            'q2': make_q(),
            'target_q1': make_q(),
            'target_q2': make_q(),
            'actor': actor_def,
        }
        iql_args = {
            'value': (ex_observations, ex_skills),
            'q1': (ex_observations, ex_skills, ex_actions),
            'q2': (ex_observations, ex_skills, ex_actions),
            'target_q1': (ex_observations, ex_skills, ex_actions),
            'target_q2': (ex_observations, ex_skills, ex_actions),
            'actor': (ex_observations, ex_skills),
        }
        iql_def = ModuleDict(iql_modules)
        iql_params = flax.core.unfreeze(iql_def.init(iql_rng, **iql_args)['params'])
        for name in ('q1', 'q2'):
            iql_params[f'modules_target_{name}'] = jax.tree_util.tree_map(
                lambda x: x.copy(), iql_params[f'modules_{name}']
            )
        iql_network = TrainState.create(
            iql_def,
            flax.core.freeze(iql_params),
            tx=optax.adam(config['iql_lr']),
        )

        lambda_tx = optax.adam(config['lambda_lr'])
        log_lambda = jnp.asarray(jnp.log(config['lambda_init']), dtype=jnp.float32)
        return cls(
            rng=rng,
            representation_network=rep_network,
            iql_network=iql_network,
            log_lambda=log_lambda,
            lambda_opt_state=lambda_tx.init(log_lambda),
            config=flax.core.FrozenDict(**config),
            lambda_tx=lambda_tx,
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='offline_metra',
            dataset_class='OfflineMETRADataset',
            batch_size=256,
            updates_per_step=50,
            latent_dim=2,
            representation_hidden_dims=(1024, 1024),
            actor_hidden_dims=(1024, 1024),
            value_hidden_dims=(1024, 1024),
            actor_log_std_min=-13.815510557964274,
            actor_log_std_max=2.0,
            representation_lr=1e-4,
            iql_lr=3e-4,
            lambda_lr=1e-4,
            lambda_init=30.0,
            constraint_epsilon=1e-3,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            awr_beta=10.0,
            max_weight=100.0,
            min_future_gap=1,
            layer_norm=False,
            discrete=False,
            encoder=None,
            frame_stack=None,
        )
    )
