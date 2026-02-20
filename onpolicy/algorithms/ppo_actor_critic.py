"""
PPO Actor-Critic Networks for Sheep Herding
Supports both continuous and discrete action spaces
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Categorical, Normal
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
    Supports both continuous and discrete action spaces.
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

        if action_space.__class__.__name__ == "Discrete":
            self.action_type = "discrete"
            self.action_dim = action_space.n
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            self.action_out = init_(nn.Linear(self.hidden_size, self.action_dim))
        else:
            self.action_type = "continuous"
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

        if self.action_type == "discrete":
            logits = self.action_out(actor_features)
            dist = Categorical(logits=logits)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
            action_log_probs = dist.log_prob(actions).unsqueeze(-1)
            actions = actions.unsqueeze(-1)
        else:
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

        if self.action_type == "discrete":
            logits = self.action_out(actor_features)
            dist = Categorical(logits=logits)
            
            action_flat = action.squeeze(-1).long()
            action_log_probs = dist.log_prob(action_flat).unsqueeze(-1)
            dist_entropy = dist.entropy()
            
            if active_masks is not None and self._use_policy_active_masks:
                if len(dist_entropy.shape) != len(active_masks.shape):
                    active_masks = active_masks.squeeze(-1)
                dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.squeeze(-1).sum()
            else:
                dist_entropy = dist_entropy.mean()
        else:
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
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=1.0)

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
    Supports both continuous and discrete action spaces.
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
        
        self.action_type = self.actor.action_type

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
    
    def check_initialization(self):
        """
        验证网络初始化是否合理
        
        检查项：
        1. 权重标准差是否在合理范围
        2. 偏置是否初始化为0
        3. 输出层权重是否较小
        """
        print("\n" + "="*60)
        print("网络初始化验证")
        print("="*60)
        
        issues = []
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 跳过LayerNorm权重（标准差为0是正常的）
                if 'feature_norm' in name or 'LayerNorm' in name or name.endswith('.2.weight'):
                    continue
                
                std = param.data.std().item()
                mean = param.data.mean().item()
                
                # 检查输出层权重（使用更宽松的标准）
                if 'fc_mean' in name or 'v_out' in name:
                    if std > 0.1:
                        issues.append(f"⚠️  {name}: 输出层权重标准差过大 ({std:.6f}), 建议使用小gain")
                    else:
                        print(f"✓ {name}: 输出层权重标准差合理 ({std:.6f})")
                else:
                    # 检查隐藏层权重标准差
                    if std < 0.01:
                        issues.append(f"⚠️  {name}: 权重标准差过小 ({std:.6f})")
                    elif std > 2.0:
                        issues.append(f"⚠️  {name}: 权重标准差过大 ({std:.6f})")
                        
            elif 'bias' in name:
                # 跳过LayerNorm偏置
                if 'feature_norm' in name or 'LayerNorm' in name or name.endswith('.2.bias'):
                    continue
                    
                mean = param.data.mean().item()
                if abs(mean) > 0.01:
                    issues.append(f"⚠️  {name}: 偏置未初始化为0 ({mean:.6f})")
        
        if issues:
            print("\n发现以下问题:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✓ 所有初始化检查通过!")
        
        print("="*60)
        
        return len(issues) == 0
