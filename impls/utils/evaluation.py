from collections import defaultdict
import os

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    skill_horizon=1,
    max_episode_steps=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    if skill_horizon < 1:
        raise ValueError('skill_horizon must be at least 1.')
    trajs = []
    stats = defaultdict(list)
    episode_lengths = []

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        skill = None
        render = []
        while not done and (max_episode_steps is None or step < max_episode_steps):
            if hasattr(agent, 'compute_skill'):
                if step % skill_horizon == 0:
                    skill = agent.compute_skill(observation, goal)
                action = actor_fn(
                    observations=observation,
                    goals=goal,
                    skill=skill,
                    temperature=eval_temperature,
                )
            else:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            episode_lengths.append(step)
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    if episode_lengths:
        stats['episode_length'] = np.mean(episode_lengths)

    return stats, trajs, renders


def evaluate_skills(agent, env, num_skills=8, eval_temperature=0.0, max_episode_steps=None, seed=0):
    """Measure skill-conditioned coverage and alignment using evaluation-only rollouts."""
    rng = np.random.default_rng(seed)
    latent_dim = agent.config['latent_dim']
    skills = rng.normal(size=(num_skills, latent_dim)).astype(np.float32)
    skills /= np.maximum(np.linalg.norm(skills, axis=-1, keepdims=True), 1e-8)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed + 1))

    if max_episode_steps is None:
        max_episode_steps = getattr(env.spec, 'max_episode_steps', None) or 1000

    intrinsic_returns = []
    intrinsic_returns_per_step = []
    alignments = []
    endpoints = []
    coverages = []
    occupied = set()
    per_skill_metrics = {}
    for skill_id, skill in enumerate(skills):
        observation, _ = env.reset(options=dict(task_id=1, render_goal=False))
        start = np.asarray(observation)
        state = start
        total_intrinsic = 0.0
        states_xy = []
        skill_occupied = set()
        done = False
        for _ in range(max_episode_steps):
            action = actor_fn(
                observations=state,
                skill=skill,
                temperature=eval_temperature,
            )
            action = np.asarray(action)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = agent.compute_intrinsic_reward(state, next_state, skill)
            total_intrinsic += float(np.asarray(reward))
            state = next_state
            if np.asarray(state).shape[-1] >= 2:
                xy = np.asarray(state)[:2]
                states_xy.append(xy)
                cell = tuple(np.floor(xy / 1.0).astype(np.int32))
                occupied.add(cell)
                skill_occupied.add(cell)
            if done:
                break

        phi_start = agent.representation_network.select('phi')(
            start
        )
        phi_end = agent.representation_network.select('phi')(
            state
        )
        displacement = np.asarray(phi_end - phi_start)
        alignments.append(float(np.dot(displacement, skill) / max(np.linalg.norm(displacement), 1e-8)))
        intrinsic_returns.append(total_intrinsic)
        intrinsic_returns_per_step.append(total_intrinsic / max(len(states_xy), 1))
        coverages.append(len(skill_occupied))
        endpoints.append(np.asarray(state)[:2])
        per_skill_metrics[f'skill_eval/skill_{skill_id:02d}_intrinsic_return'] = float(total_intrinsic)
        per_skill_metrics[f'skill_eval/skill_{skill_id:02d}_alignment'] = alignments[-1]
        per_skill_metrics[f'skill_eval/skill_{skill_id:02d}_coverage_bins'] = float(len(skill_occupied))

    endpoint_array = np.asarray(endpoints)
    if len(endpoint_array) > 1:
        pairwise = endpoint_array[:, None, :] - endpoint_array[None, :, :]
        upper = np.triu_indices(len(endpoint_array), k=1)
        endpoint_diversity = float(np.linalg.norm(pairwise, axis=-1)[upper].mean())
    else:
        endpoint_diversity = 0.0

    metrics = {
        'skill_eval/mean_directional_intrinsic_return': float(np.mean(intrinsic_returns)),
        'skill_eval/mean_directional_intrinsic_reward_per_step': float(np.mean(intrinsic_returns_per_step)),
        'skill_eval/mean_directional_alignment': float(np.mean(alignments)),
        'skill_eval/endpoint_diversity': endpoint_diversity,
        'skill_eval/mean_skill_coverage_1m_bins': float(np.mean(coverages)),
        'skill_eval/state_coverage_1m_bins': float(len(occupied)),
    }
    metrics.update(per_skill_metrics)
    return metrics

def visualize(agent, epoch, env, save_dir):
    goals = []
    start_states = []
    for i in range(5):
        observation, info = env.reset(options=dict(task_id=i+1, render_goal=False))
        goal = info.get('goal')
        goals.append(goal)
        start_states.append(observation)
    env = env.unwrapped

    def xy_to_ij_numpy(xy):
        maze_unit = env._maze_unit
        i = jnp.floor((xy[1] + env._offset_y + 0.5 * maze_unit) / maze_unit).astype(int)
        j = jnp.floor((xy[0] + env._offset_x + 0.5 * maze_unit) / maze_unit).astype(int)
        return i, j

    if 'medium' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([7, 7])
    elif 'large' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([8, 11])
    elif 'giant' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([11, 15])
    else:
        raise NotImplementedError(f'Unknown maze type {env._maze_type}')

    fig, axes = plt.subplots(1, 5, figsize=(50, 10))

    for j in range(5):
        goal = goals[j]
        observations_list = []
        value_list = []
        dist_list = []

        xgrid = np.linspace(xmin, xmax, 200, endpoint=False)
        ygrid = np.linspace(ymin, ymax, 200, endpoint=False)

        obs_config = start_states[j][2:]  # keep the configuration part of the observation fixed

        X, Y = np.meshgrid(xgrid, ygrid)
        observations = np.stack([X.ravel(), Y.ravel()], axis=-1)
        observations = np.concatenate([observations, np.repeat(obs_config[None, :], observations.shape[0], axis=0)], axis=-1)
        B = 1000

        for i in range(200*200//B):
            obs = observations[i * B:(i + 1) * B]
            values = agent.compute_value(obs, np.repeat(goal[None, :], obs.shape[0], axis=0))
            dist = np.linalg.norm(obs[:, :2] - goal[None, :2], axis=-1)
            observations_list.append(obs)
            value_list.append(values)
            dist_list.append(dist)

        observations = jnp.concatenate(observations_list, axis=0)
        values = jnp.concatenate(value_list, axis=0).flatten()
        dists = jnp.concatenate(dist_list, axis=0).flatten()

        x, y = observations[:, 0], observations[:, 1]
        xi, yi = xy_to_ij_numpy([x, y])
        is_obstacle = env.maze_map[xi, yi] == 1
        values = values.at[is_obstacle].set(1000.0)

        filter_idx = values < 999.0

        observations = observations[filter_idx]
        values = values[filter_idx]
        dists = dists[filter_idx]

        vmin = values.min()
        vmax = values.max()
        print(f"goal {j+1} value range: {vmin} to {vmax}")

        # Value function scatter plot
        ax = axes[j]
        sc = ax.scatter(observations[:, 0], observations[:, 1], c=values, cmap='viridis', s=1.0)
        ax.set_title(f'Goal {j+1}')
        plt.colorbar(sc, ax=ax, label='Value')

    fig.suptitle(f'Value Function, epoch {epoch}')
    fig.savefig(os.path.join(save_dir, f"epoch{epoch}_value.png"), dpi=300)
    plt.close(fig)
