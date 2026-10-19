import unittest

import jax
import jax.numpy as jnp
import numpy as np

from agents.offline_metra import (
    high_level_awr_weights,
    high_level_directional_advantage,
    intrinsic_reward,
    lagrange_dual_loss,
    normalize,
    representation_objective,
    sample_unit_sphere,
)
from utils.datasets import Dataset, OfflineMETRADataset


class OfflineMETRATests(unittest.TestCase):
    def test_high_level_advantage_and_awr_weights_favor_goal_progress(self):
        phi_s = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        phi_goal = jnp.array([[2.0, 0.0], [2.0, 0.0]])
        phi_subgoal = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
        advantage = high_level_directional_advantage(phi_s, phi_goal, phi_subgoal)
        np.testing.assert_allclose(advantage, [1.0, -1.0])
        weights = high_level_awr_weights(advantage, beta=1.0, max_weight=100.0)
        self.assertGreater(float(weights[0]), float(weights[1]))
        np.testing.assert_allclose(jnp.mean(weights), 1.0)

    def test_representation_objective_matches_directional_progress_formula(self):
        phi_s = jnp.array([[0.0, 0.0]])
        phi_next = jnp.array([[1.0, 0.0]])
        phi_goal = jnp.array([[1.0, 1.0]])
        loss, info = representation_objective(phi_s, phi_next, phi_goal, 2.0, 1e-3)

        progress = 1.0 / jnp.sqrt(2.0)
        expected_objective = progress + 2.0 * 1e-3
        np.testing.assert_allclose(loss, -expected_objective, rtol=1e-6)
        np.testing.assert_allclose(info['progress'], progress, rtol=1e-6)

    def test_representation_gradient_treats_goal_direction_as_fixed(self):
        phi_s = jnp.array([[0.1, -0.3]])
        phi_next = jnp.array([[0.8, 0.2]])
        phi_goal = jnp.array([[0.5, 1.1]])

        def full_loss(state):
            return representation_objective(state, phi_next, phi_goal, 0.0, 1e-3)[0]

        def detached_direction_loss(state):
            direction = jax.lax.stop_gradient(normalize(phi_goal - state))
            delta = phi_next - state
            progress = jnp.sum(delta * direction, axis=-1)
            return -jnp.mean(progress)

        actual_grad = jax.grad(full_loss)(phi_s)
        detached_grad = jax.grad(detached_direction_loss)(phi_s)
        np.testing.assert_allclose(actual_grad, detached_grad, rtol=1e-6, atol=1e-6)

    def test_dual_gradient_moves_lambda_in_correct_direction(self):
        positive_constraint_grad = jax.grad(lagrange_dual_loss)(jnp.log(3.0), 0.2)
        violated_constraint_grad = jax.grad(lagrange_dual_loss)(jnp.log(3.0), -0.2)
        self.assertGreater(float(positive_constraint_grad), 0.0)
        self.assertLess(float(violated_constraint_grad), 0.0)

    def test_intrinsic_reward_is_detached(self):
        def reward_from_scale(scale):
            phi_s = jnp.array([[0.0, 1.0]]) * scale
            phi_next = jnp.array([[2.0, 3.0]]) * scale
            return intrinsic_reward(phi_s, phi_next, jnp.array([[1.0, 0.0]])).sum()

        self.assertEqual(float(jax.grad(reward_from_scale)(1.0)), 0.0)

    def test_unit_sphere_samples_are_normalized(self):
        skills = sample_unit_sphere(jax.random.PRNGKey(0), 128, 2)
        np.testing.assert_allclose(jnp.linalg.norm(skills, axis=-1), 1.0, atol=1e-6)

    def test_representation_goals_use_k_steps_or_episode_end(self):
        observations = np.arange(8, dtype=np.float32)[:, None]
        data = Dataset.create(
            observations=observations,
            actions=np.zeros((8, 1), dtype=np.float32),
            terminals=np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32),
            valids=np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32),
        )
        sampler = OfflineMETRADataset(data, {'rep_future_steps': 2})
        batch = sampler.sample(256)['representation']
        states = batch['observations'][:, 0]
        goals = batch['future_goals'][:, 0]
        high_goals = batch['high_actor_goals'][:, 0]
        high_targets = batch['high_actor_targets'][:, 0]
        episode_ends = np.where(states < 4, 3, 7)
        np.testing.assert_array_equal(goals, np.minimum(states + 2, episode_ends))
        self.assertTrue(np.all(goals > states))
        np.testing.assert_array_equal(high_targets, np.minimum(states + 2, high_goals))
        self.assertTrue(np.all(high_goals > states))
        same_episode = ((states < 4) & (goals < 4)) | ((states >= 4) & (goals >= 4))
        self.assertTrue(np.all(same_episode))


if __name__ == '__main__':
    unittest.main()
