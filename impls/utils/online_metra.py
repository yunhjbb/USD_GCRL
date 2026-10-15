"""Online collection and replay for the METRA agent."""

import jax
import numpy as np

from utils.datasets import ReplayBuffer


def create_metra_replay_buffer(observation, action, config):
    transition = {
        'observations': np.asarray(observation, dtype=np.float32),
        'next_observations': np.asarray(observation, dtype=np.float32),
        'actions': np.asarray(action, dtype=np.float32),
        'skills': np.zeros((config['latent_dim'],), dtype=np.float32),
        'dones': np.asarray(0.0, dtype=np.float32),
    }
    return ReplayBuffer.create(transition, config['max_buffer_size'])


def _unit_skill(rng, latent_dim):
    skill = rng.normal(size=(latent_dim,)).astype(np.float32)
    return skill / max(np.linalg.norm(skill), 1e-6)


def collect_and_update(agent, env, replay_buffer, config, *, seed):
    """Collect METRA skill rollouts, then run its configured replay updates."""
    rng = np.random.default_rng(seed)
    action_rng = jax.random.PRNGKey(seed)
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    rollout_lengths = []
    external_returns = []
    transitions_collected = 0

    for _ in range(config['traj_batch_size']):
        task_id = int(rng.integers(1, len(task_infos) + 1))
        observation, _ = env.reset(options={'task_id': task_id})
        observation = np.asarray(observation, dtype=np.float32)
        skill = _unit_skill(rng, config['latent_dim'])
        episode_return = 0.0
        episode_length = 0

        while True:
            action_rng, sample_key = jax.random.split(action_rng)
            action = np.asarray(
                agent.sample_actions(
                    observation,
                    skill=skill,
                    seed=sample_key,
                    temperature=1.0,
                ),
                dtype=np.float32,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_length += 1
            # The reference sampler marks the final transition of every path as done.
            done = bool(terminated or truncated or episode_length >= config['max_episode_steps'])
            next_observation = np.asarray(next_observation, dtype=np.float32)
            # On an environment terminal, the reference worker keeps its previous
            # observation as the path's last observation. A sampler cutoff keeps
            # the actual next observation instead.
            stored_next_observation = observation if terminated or truncated else next_observation
            replay_buffer.add_transition({
                'observations': observation,
                'next_observations': stored_next_observation,
                'actions': action,
                'skills': skill,
                'dones': np.asarray(float(done), dtype=np.float32),
            })
            observation = next_observation
            episode_return += float(reward)
            if done:
                break

        rollout_lengths.append(episode_length)
        external_returns.append(episode_return)
        transitions_collected += episode_length

    update_info = {}
    if replay_buffer.size >= config['min_buffer_size']:
        for _ in range(config['trans_optimization_epochs']):
            sampled = replay_buffer.sample(config['batch_size'])
            batch = {
                'observations': sampled['observations'],
                'next_observations': sampled['next_observations'],
                'actions': sampled['actions'],
                'skills': sampled['skills'],
                'dones': sampled['dones'],
            }
            agent, update_info = agent.update(batch)

    update_info = dict(update_info)
    update_info.update({
        'rollout/episodes': float(config['traj_batch_size']),
        'rollout/length_mean': float(np.mean(rollout_lengths)),
        'rollout/length_max': float(np.max(rollout_lengths)),
        'rollout/transitions_collected': float(transitions_collected),
        'rollout/external_return_mean': float(np.mean(external_returns)),
        'rollout/replay_size': float(replay_buffer.size),
        'rollout/updates': float(config['trans_optimization_epochs'] if replay_buffer.size >= config['min_buffer_size'] else 0),
    })
    return agent, update_info
