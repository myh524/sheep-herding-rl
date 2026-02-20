"""
Bounded Action Layer for Sheep Herding High-Level Controller
Outputs formation parameters with proper bounds using tanh squashing
"""

import torch
import torch.nn as nn
from .util import init
from typing import Optional, Tuple


class TanhNormal(torch.distributions.Normal):
    """
    Tanh-normal distribution for bounded continuous actions.
    Uses tanh to squash actions to [-1, 1] then scales to action bounds.
    
    Fixed implementation with correct log_prob and entropy calculations.
    """

    def __init__(self, loc, scale, action_low=None, action_high=None):
        super(TanhNormal, self).__init__(loc, scale)
        self.action_low = action_low
        self.action_high = action_high

        if action_low is not None and action_high is not None:
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0
        else:
            self.action_scale = 1.0
            self.action_bias = 0.0
        
        self._eps = 1e-6

    def sample(self):
        z = super().rsample()
        return torch.tanh(z)

    def rsample(self):
        z = super().rsample()
        return torch.tanh(z)

    def log_probs(self, actions):
        """
        Compute log probability of actions in action space.
        
        Args:
            actions: Actions in the actual action space [action_low, action_high]
        
        Returns:
            Log probabilities
        """
        tanh_actions = self.from_action_space(actions)
        
        tanh_actions = tanh_actions.clamp(-1 + self._eps, 1 - self._eps)
        
        z = self._inverse_tanh(tanh_actions)
        
        log_prob = super().log_prob(z)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        log_det_jacobian = torch.log(torch.clamp(1 - tanh_actions.pow(2), min=1e-4))
        log_prob = log_prob - log_det_jacobian.sum(-1, keepdim=True)
        
        if torch.isnan(log_prob).any():
            log_prob = torch.where(torch.isnan(log_prob), torch.zeros_like(log_prob), log_prob)
        
        return log_prob

    def _inverse_tanh(self, y):
        y = y.clamp(-1 + self._eps, 1 - self._eps)
        z = 0.5 * torch.log((1 + y) / (1 - y))
        if torch.isnan(z).any():
            z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
        return z

    def mode(self):
        z = self.mean
        return torch.tanh(z)

    def entropy(self):
        """
        Compute entropy of the tanh-normal distribution.
        
        Note: This is an approximation. The exact entropy of tanh-normal
        is intractable, so we use the entropy of the base normal distribution
        as an approximation.
        """
        return super().entropy().sum(-1)

    def to_action_space(self, x):
        if self.action_low is not None:
            return x * self.action_scale + self.action_bias
        return x

    def from_action_space(self, x):
        if self.action_low is not None:
            return (x - self.action_bias) / self.action_scale
        return x


class BoundedDiagGaussian(nn.Module):
    """
    Bounded Diagonal Gaussian Distribution for continuous action space.
    Uses tanh squashing to enforce action bounds.
    
    改进的初始化策略：
    - 输出层使用小gain (0.01) 避免初始策略过于确定
    - 保持初始探索性
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        action_low: torch.Tensor = None,
        action_high: torch.Tensor = None,
        use_orthogonal: bool = True,
        gain: float = 0.01,  # 改进：从0.5改为0.01，避免初始策略过于确定
    ):
        super(BoundedDiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        self.register_buffer('action_low', action_low)
        self.register_buffer('action_high', action_high)

        if action_low is not None and action_high is not None:
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0
        else:
            self.action_scale = 1.0
            self.action_bias = 0.0

    def forward(self, x: torch.Tensor) -> TanhNormal:
        action_mean = self.fc_mean(x)
        
        if torch.isnan(action_mean).any():
            action_mean = torch.where(
                torch.isnan(action_mean),
                torch.zeros_like(action_mean),
                action_mean
            )

        zeros = torch.zeros(action_mean.size(), device=x.device, dtype=x.dtype)

        action_logstd = self.logstd(zeros)
        action_logstd = torch.clamp(action_logstd, min=-5.0, max=2.0)

        return TanhNormal(
            action_mean,
            action_logstd.exp(),
            self.action_low,
            self.action_high,
        )

    def to_action_space(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.action_scale + self.action_bias


class BoundedACTLayer(nn.Module):
    """
    Action layer for bounded continuous action spaces.
    Designed for the sheep herding high-level controller.

    Action outputs:
    - radius_mean: [0, 20] - Standing radius around flock center
    - radius_std: [0, 5] - Radius variation
    - angle_mean: [-pi, pi] - Gathering angle
    - concentration: [0, 1] - How concentrated the formation is
    """

    def __init__(
        self,
        action_space,
        inputs_dim: int,
        use_orthogonal: bool = True,
        gain: float = 0.5,
    ):
        super(BoundedACTLayer, self).__init__()

        action_dim = action_space.shape[0]

        action_low = torch.tensor(action_space.low, dtype=torch.float32)
        action_high = torch.tensor(action_space.high, dtype=torch.float32)

        self.action_out = BoundedDiagGaussian(
            inputs_dim,
            action_dim,
            action_low,
            action_high,
            use_orthogonal,
            gain,
        )

    def forward(
        self,
        x: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist = self.action_out(x)

        if deterministic:
            actions = action_dist.mode()
        else:
            actions = action_dist.sample()

        actions = action_dist.to_action_space(actions)
        action_log_probs = action_dist.log_probs(actions)

        return actions, action_log_probs

    def evaluate_actions(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        active_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist = self.action_out(x)

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy()

        if active_masks is not None:
            if len(dist_entropy.shape) != len(active_masks.shape):
                active_masks = active_masks.squeeze(-1)
            dist_entropy = (dist_entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist_entropy.mean()

        return action_log_probs, dist_entropy


class AddBias(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, 1, -1)

        return x + bias