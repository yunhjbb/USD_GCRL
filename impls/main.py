import json
import os
import random
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, OfflineMETRADataset
from utils.env_utils import make_env_and_datasets, make_env_only
from utils.evaluation import evaluate, evaluate_skills, save_failed_eval_trajectories, visualize
from utils.flax_utils import restore_agent, save_agent, save_representation
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.metra_goal_map import save_metra_goal_map
from utils.online_metra import collect_and_update, create_metra_replay_buffer

FLAGS = flags.FLAGS

flags.DEFINE_string('project', 'LAVL', 'Project name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_bool('eval_only', False, 'Evaluate a restored checkpoint once without training.')
flags.DEFINE_bool('resume_training', False, 'Continue training after the restored checkpoint epoch.')
flags.DEFINE_bool('log_wandb', False, 'Whether to log to Weights & Biases.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the value function.')
flags.DEFINE_bool('metra_goal_map', False, 'Save a METRA representation distance map at each evaluation.')
flags.DEFINE_bool('save_representation', False, 'Save METRA phi parameters at each evaluation.')
flags.DEFINE_bool('save_failed_eval_trajectories', False, 'Plot and save failed evaluation paths when observations contain XY positions.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_integer('eval_max_episode_steps', None, 'Optional evaluation episode horizon.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_integer('skill_eval_horizon', 1, 'Number of evaluation steps to hold a goal-induced skill direction.')
flags.DEFINE_integer('skill_eval_episodes', 8, 'Number of random skills to evaluate for offline METRA.')
flags.DEFINE_bool('hilp_test_time_planning', False, 'Use dataset waypoints for HILP-style offline METRA evaluation.')
flags.DEFINE_integer('planning_num_candidates', 2048, 'Number of offline states sampled as planning waypoints.')
flags.DEFINE_integer('planning_recursions', 1, 'Number of midpoint refinements used by test-time planning.')

config_flags.DEFINE_config_file('agent', 'agents/lavl.py', lock_config=False)

def main(_):
    if FLAGS.eval_only and (FLAGS.restore_path is None or FLAGS.restore_epoch is None):
        raise ValueError('--eval_only requires both --restore_path and --restore_epoch.')
    if FLAGS.resume_training and (FLAGS.restore_path is None or FLAGS.restore_epoch is None):
        raise ValueError('--resume_training requires both --restore_path and --restore_epoch.')
    if FLAGS.resume_training and FLAGS.eval_only:
        raise ValueError('--resume_training cannot be combined with --eval_only.')
    if FLAGS.resume_training and FLAGS.restore_epoch >= FLAGS.train_steps:
        raise ValueError('--train_steps must be greater than --restore_epoch when resuming.')
    if FLAGS.planning_num_candidates < 1 or FLAGS.planning_recursions < 1:
        raise ValueError('Planning candidate count and recursion count must be positive.')
    # Set up logger.
    exp_name_str = get_exp_name(FLAGS.seed)
    if FLAGS.log_wandb:
        setup_wandb(project=FLAGS.project, group=FLAGS.env_name, name=exp_name_str)

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.project, FLAGS.env_name, exp_name_str)
    os.makedirs(save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    is_offline_metra = config['agent_name'] == 'offline_metra'
    is_online_metra = config['agent_name'] == 'online_metra'
    if is_online_metra:
        # Online METRA needs only the simulator; do not download/load the OGBench dataset.
        env = make_env_only(FLAGS.env_name)
        initial_observation, _ = env.reset(seed=FLAGS.seed)
        train_dataset = val_dataset = None
    else:
        env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name, frame_stack=config['frame_stack'], reset_env=not is_offline_metra
        )
        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
            'OfflineMETRADataset': OfflineMETRADataset,
        }[config['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset), config)
        if val_dataset is not None and not is_offline_metra:
            val_dataset = dataset_class(Dataset.create(**val_dataset), config)
        elif is_offline_metra:
            val_dataset = None

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if is_online_metra:
        if not hasattr(env.action_space, 'low'):
            raise ValueError('Online METRA currently requires a continuous OGBench action space.')
        example_observations = np.asarray(initial_observation, dtype=np.float32)[None]
        example_actions = np.asarray(env.action_space.sample(), dtype=np.float32)[None]
    else:
        example_batch = train_dataset.sample(1)
        if is_offline_metra:
            example_batch = example_batch['skill']
        if config['discrete']:
            # Fill with the maximum action to let the agent know the action space size.
            example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    if is_online_metra:
        agent = agent_class.create(FLAGS.seed, example_observations, example_actions, config)
        replay_buffer = None if FLAGS.eval_only else create_metra_replay_buffer(
            initial_observation, env.action_space.sample(), config
        )
    else:
        agent = agent_class.create(
            FLAGS.seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    start_iteration = FLAGS.restore_epoch + 1 if FLAGS.resume_training else 1
    iterations = [0] if FLAGS.eval_only else range(start_iteration, FLAGS.train_steps + 1)
    for i in tqdm.tqdm(iterations, smoothing=0.1, dynamic_ncols=True):
        # Evaluate a restored checkpoint without taking an optimization step.
        if not FLAGS.eval_only:
            if is_online_metra:
                agent, update_info = collect_and_update(
                    agent, env, replay_buffer, config,
                    seed=FLAGS.seed + i * config['traj_batch_size'],
                )
            elif is_offline_metra:
                update_infos = []
                for _ in range(config['updates_per_step']):
                    batch = train_dataset.sample(config['batch_size'])
                    agent, step_info = agent.update(batch)
                    update_infos.append(step_info)
                update_info = {
                    key: jnp.mean(jnp.stack([step_info[key] for step_info in update_infos]))
                    for key in update_infos[0]
                }
            else:
                batch = train_dataset.sample(config['batch_size'])
                agent, update_info = agent.update(batch)

        # Log metrics.
        if not FLAGS.eval_only and i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.log_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_only or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    skill_horizon=FLAGS.skill_eval_horizon,
                    max_episode_steps=FLAGS.eval_max_episode_steps,
                    planning_candidates=(
                        train_dataset.dataset['observations'][
                            np.random.choice(
                                train_dataset.size,
                                size=min(FLAGS.planning_num_candidates, train_dataset.size),
                                replace=False,
                            )
                        ]
                        if is_offline_metra and FLAGS.hilp_test_time_planning
                        else None
                    ),
                    planning_recursions=FLAGS.planning_recursions,
                )
                if FLAGS.save_failed_eval_trajectories:
                    save_failed_eval_trajectories(
                        trajs, env, save_dir, i, task_name, task_info=task_infos[task_id - 1]
                    )
                renders.extend(cur_renders)
                metric_names = ['success', 'episode_length']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if is_offline_metra and FLAGS.skill_eval_episodes > 0:
                eval_metrics.update(
                    evaluate_skills(
                        eval_agent,
                        env,
                        num_skills=FLAGS.skill_eval_episodes,
                        eval_temperature=FLAGS.eval_temperature,
                        seed=FLAGS.seed + i,
                    )
                )

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            if FLAGS.log_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            if (is_online_metra or is_offline_metra) and FLAGS.metra_goal_map:
                map_source = replay_buffer if is_online_metra else train_dataset.dataset
                save_metra_goal_map(eval_agent, map_source, save_dir, i, seed=FLAGS.seed)
            if (is_online_metra or is_offline_metra) and FLAGS.save_representation:
                save_representation(eval_agent, save_dir, i)

        if (i % FLAGS.eval_interval == 0) and FLAGS.visualize:
            visualize(agent, i, env, save_dir)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
