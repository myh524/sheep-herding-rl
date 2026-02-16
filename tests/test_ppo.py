"""
Unit tests for simplified PPO components
"""

import unittest
import numpy as np
import torch
from gym import spaces

from onpolicy.utils.ppo_buffer import PPOReplayBuffer
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic


class MockArgs:
    def __init__(self):
        self.episode_length = 10
        self.n_rollout_threads = 2
        self.hidden_size = 32
        self.recurrent_N = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.use_gae = True
        self.use_popart = False
        self.use_valuenorm = False
        self.use_proper_time_limits = False
        self.gain = 0.01
        self.use_orthogonal = True
        self.use_policy_active_masks = True
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = False
        self.use_ReLU = True
        self.use_feature_normalization = True
        self.layer_N = 1
        self.stacked_frames = 1


class TestPPOBuffer(unittest.TestCase):
    def setUp(self):
        self.args = MockArgs()
        self.obs_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.act_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def test_buffer_shape_no_agents_dim(self):
        buffer = PPOReplayBuffer(self.args, self.obs_space, self.act_space)
        
        expected_obs_shape = (self.args.episode_length + 1, self.args.n_rollout_threads, 8)
        self.assertEqual(buffer.obs.shape, expected_obs_shape)
        
        expected_value_shape = (self.args.episode_length + 1, self.args.n_rollout_threads, 1)
        self.assertEqual(buffer.value_preds.shape, expected_value_shape)
        
        expected_action_shape = (self.args.episode_length, self.args.n_rollout_threads, 4)
        self.assertEqual(buffer.actions.shape, expected_action_shape)

    def test_buffer_insert(self):
        buffer = PPOReplayBuffer(self.args, self.obs_space, self.act_space)
        
        obs = np.random.randn(self.args.n_rollout_threads, 8).astype(np.float32)
        rnn_states = np.zeros((self.args.n_rollout_threads, 1, 32), dtype=np.float32)
        actions = np.random.randn(self.args.n_rollout_threads, 4).astype(np.float32)
        action_log_probs = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
        values = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
        rewards = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
        masks = np.ones((self.args.n_rollout_threads, 1), dtype=np.float32)
        
        buffer.insert(obs, rnn_states, rnn_states, actions, action_log_probs, values, rewards, masks)
        
        self.assertEqual(buffer.step, 1)
        np.testing.assert_array_almost_equal(buffer.obs[1], obs)
        np.testing.assert_array_almost_equal(buffer.actions[0], actions)

    def test_compute_returns(self):
        buffer = PPOReplayBuffer(self.args, self.obs_space, self.act_space)
        
        for step in range(self.args.episode_length):
            obs = np.random.randn(self.args.n_rollout_threads, 8).astype(np.float32)
            rnn_states = np.zeros((self.args.n_rollout_threads, 1, 32), dtype=np.float32)
            actions = np.random.randn(self.args.n_rollout_threads, 4).astype(np.float32)
            action_log_probs = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
            values = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
            rewards = np.ones((self.args.n_rollout_threads, 1), dtype=np.float32)
            masks = np.ones((self.args.n_rollout_threads, 1), dtype=np.float32)
            
            buffer.insert(obs, rnn_states, rnn_states, actions, action_log_probs, values, rewards, masks)
        
        next_value = np.zeros((self.args.n_rollout_threads, 1), dtype=np.float32)
        buffer.compute_returns(next_value)
        
        self.assertEqual(buffer.returns.shape, (self.args.episode_length + 1, self.args.n_rollout_threads, 1))

    def test_feed_forward_generator(self):
        buffer = PPOReplayBuffer(self.args, self.obs_space, self.act_space)
        
        for step in range(self.args.episode_length):
            obs = np.random.randn(self.args.n_rollout_threads, 8).astype(np.float32)
            rnn_states = np.zeros((self.args.n_rollout_threads, 1, 32), dtype=np.float32)
            actions = np.random.randn(self.args.n_rollout_threads, 4).astype(np.float32)
            action_log_probs = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
            values = np.random.randn(self.args.n_rollout_threads, 1).astype(np.float32)
            rewards = np.ones((self.args.n_rollout_threads, 1), dtype=np.float32)
            masks = np.ones((self.args.n_rollout_threads, 1), dtype=np.float32)
            
            buffer.insert(obs, rnn_states, rnn_states, actions, action_log_probs, values, rewards, masks)
        
        next_value = np.zeros((self.args.n_rollout_threads, 1), dtype=np.float32)
        buffer.compute_returns(next_value)
        
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        count = 0
        for sample in buffer.feed_forward_generator(advantages, num_mini_batch=1):
            count += 1
            self.assertEqual(len(sample), 11)
        
        self.assertEqual(count, 1)


class TestPPOActorCritic(unittest.TestCase):
    def setUp(self):
        self.args = MockArgs()
        self.obs_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.act_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.device = torch.device('cpu')

    def test_network_init(self):
        policy = PPOActorCritic(self.args, self.obs_space, self.act_space, self.device)
        
        self.assertIsInstance(policy.actor, torch.nn.Module)
        self.assertIsInstance(policy.critic, torch.nn.Module)

    def test_get_actions(self):
        policy = PPOActorCritic(self.args, self.obs_space, self.act_space, self.device)
        
        obs = torch.randn(2, 8)
        rnn_states_actor = torch.zeros(2, 1, 32)
        rnn_states_critic = torch.zeros(2, 1, 32)
        masks = torch.ones(2, 1)
        
        values, actions, action_log_probs, new_rnn_actor, new_rnn_critic = \
            policy.get_actions(obs, rnn_states_actor, rnn_states_critic, masks)
        
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(action_log_probs.shape, (2, 1))

    def test_get_values(self):
        policy = PPOActorCritic(self.args, self.obs_space, self.act_space, self.device)
        
        obs = torch.randn(2, 8)
        rnn_states_critic = torch.zeros(2, 1, 32)
        masks = torch.ones(2, 1)
        
        values = policy.get_values(obs, rnn_states_critic, masks)
        
        self.assertEqual(values.shape, (2, 1))

    def test_evaluate_actions(self):
        policy = PPOActorCritic(self.args, self.obs_space, self.act_space, self.device)
        
        obs = torch.randn(2, 8)
        rnn_states_actor = torch.zeros(2, 1, 32)
        rnn_states_critic = torch.zeros(2, 1, 32)
        actions = torch.randn(2, 4)
        masks = torch.ones(2, 1)
        
        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            obs, rnn_states_actor, rnn_states_critic, actions, masks
        )
        
        self.assertEqual(values.shape, (2, 1))
        self.assertEqual(action_log_probs.shape, (2, 1))
        self.assertEqual(dist_entropy.shape, ())


if __name__ == '__main__':
    unittest.main()
