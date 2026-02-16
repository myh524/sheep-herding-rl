"""
Simplified PPO Actor-Critic Networks for Sheep Herding
Single-agent version without share_obs distinction
"""

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.bounded_act import BoundedACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class PPOActor(nn.Module):
    """
    Actor network for single-agent PPO.
    
    Outputs formation parameters:
    - radius_mean: Standing radius around flock center [0, 20]
    - radius_std: Radius variation [0, 5]
    - angle_mean: Gathering angle [-pi, pi]
    - concentration: How concentrated the formation is [0, 1]
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu")
    ) -> None:
        super(PPOActor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = BoundedACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None
    ) -> Tuple[Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy


class PPOCritic(nn.Module):
    """
    Critic network for single-agent PPO.
    
    Takes observation and outputs value estimate.
    No share_obs needed for single-agent.
    """

    def __init__(
        self,
        args,
        obs_space,
        device=torch.device("cpu")
    ) -> None:
        super(PPOCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks) -> Tuple[Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states


class PPOActorCritic(nn.Module):
    """
    Combined Actor-Critic for single-agent PPO.
    
    Simplified interface without share_obs distinction.
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu")
    ):
        super(PPOActorCritic, self).__init__()

        self.actor = PPOActor(args, obs_space, action_space, device)
        self.critic = PPOCritic(args, obs_space, device)

        self.to(device)

    def get_actions(
        self,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks) -> Tensor:
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        values, _ = self.critic(obs, rnn_states_critic, masks)
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )

        return values, action_log_probs, dist_entropy


class ImprovedActorCritic(nn.Module):
    """
    Improved network architecture with deeper layers, LayerNorm and Dropout.
    Compatible interface with PPOActorCritic for seamless integration.
    """

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
        hidden_size: int = 256,
    ):
        super(ImprovedActorCritic, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]
        action_dim = action_space.shape[0]

        self._use_feature_normalization = args.use_feature_normalization
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        self.action_mean = nn.Linear(hidden_size // 2, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self._use_orthogonal = args.use_orthogonal
        if self._use_orthogonal:
            self._init_orthogonal()

        action_low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        self.register_buffer('action_low', action_low)
        self.register_buffer('action_high', action_high)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        self.to(device)

    def _init_orthogonal(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def _get_action_dist(self, actor_features: torch.Tensor) -> torch.distributions.Normal:
        action_mean = self.action_mean(actor_features)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def _to_action_space(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.action_scale + self.action_bias

    def _from_action_space(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.action_bias) / self.action_scale

    def get_actions(
        self,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        critic_features = self.critic(obs)
        values = critic_features

        actor_features = self.actor(obs)
        action_dist = self._get_action_dist(actor_features)

        if deterministic:
            z = action_dist.mean
        else:
            z = action_dist.rsample()

        tanh_actions = torch.tanh(z)
        actions = self._to_action_space(tanh_actions)

        log_prob = action_dist.log_prob(z).sum(-1, keepdim=True)
        log_det_jacobian = torch.log(1 - tanh_actions.pow(2) + 1e-6).sum(-1, keepdim=True)
        action_log_probs = log_prob - log_det_jacobian

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        critic_features = self.critic(obs)
        return critic_features

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._use_feature_normalization:
            obs = self.feature_norm(obs)

        critic_features = self.critic(obs)
        values = critic_features

        actor_features = self.actor(obs)
        action_dist = self._get_action_dist(actor_features)

        tanh_actions = self._from_action_space(action)
        tanh_actions = tanh_actions.clamp(-1 + 1e-6, 1 - 1e-6)
        z = 0.5 * torch.log((1 + tanh_actions) / (1 - tanh_actions))

        log_prob = action_dist.log_prob(z).sum(-1, keepdim=True)
        log_det_jacobian = torch.log(1 - tanh_actions.pow(2) + 1e-6).sum(-1, keepdim=True)
        action_log_probs = log_prob - log_det_jacobian

        dist_entropy = action_dist.entropy().sum(-1)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            if len(dist_entropy.shape) != len(active_masks.shape):
                active_masks = active_masks.squeeze(-1)
            dist_entropy = (dist_entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist_entropy.mean()

        return values, action_log_probs, dist_entropy
