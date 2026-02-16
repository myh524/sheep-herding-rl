"""
PPO Replay Buffer for Single-Agent Training
Simplified version without num_agents dimension
"""

import torch
import numpy as np
from typing import Optional, Tuple, Generator


class PPOReplayBuffer:
    """
    Buffer to store training data for single-agent PPO.
    
    Data shape: (episode_length, n_rollout_threads, ...)
    No num_agents dimension compared to SharedReplayBuffer.
    """
    
    def __init__(self, args, obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        obs_shape = self._get_shape_from_obs_space(obs_space)
        act_shape = self._get_shape_from_act_space(act_space)

        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def _get_shape_from_obs_space(self, obs_space):
        if hasattr(obs_space, 'shape'):
            return obs_space.shape
        return obs_space

    def _get_shape_from_act_space(self, act_space):
        if act_space.__class__.__name__ == "Discrete":
            return 1
        elif hasattr(act_space, 'shape'):
            return act_space.shape[0]
        return act_space

    def insert(
        self,
        obs: np.ndarray,
        rnn_states_actor: np.ndarray,
        rnn_states_critic: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        value_preds: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        bad_masks: np.ndarray = None,
        active_masks: np.ndarray = None,
        available_actions: np.ndarray = None,
    ) -> None:
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value: np.ndarray, value_normalizer=None):
        if self._use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_popart or self._use_valuenorm:
                    delta = (
                        self.rewards[step]
                        + self.gamma
                        * value_normalizer.denormalize(self.value_preds[step + 1])
                        * self.masks[step + 1]
                        - value_normalizer.denormalize(self.value_preds[step])
                    )
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = (
                        self.rewards[step]
                        + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = (
                    self.returns[step + 1] * self.gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def feed_forward_generator(
        self,
        advantages: np.ndarray,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator:
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires batch size ({batch_size}) >= num_mini_batch ({num_mini_batch})"
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        else:
            available_actions = None
            
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]

            yield (
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )
