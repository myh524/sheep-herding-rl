"""
High-Level Action Decoder for Sheep Herding
Decodes normalized actions [-1, 1] to formation parameters
"""

import numpy as np
from typing import Dict, Any, Optional


class HighLevelAction:
    """
    High-level action decoder
    
    Decodes neural network output (normalized actions [-1, 1]) to actual station parameters:
    - mu_r: station radius mean
    - sigma_r: station radius std
    - mu_theta: relative angle (relative to target direction)
    - kappa: von Mises concentration parameter
    
    Kappa semantics:
    - kappa = 0: uniform distribution, herders evenly spread around flock (360° surround)
    - kappa > 0: concentrated around mu_theta direction
    - kappa -> infinity: all herders at same angle (overlap at mu_theta)
    
    This design allows the policy to:
    1. Use kappa=0 for surrounding/containing the flock
    2. Use kappa>0 for concentrated pushing in one direction
    """
    
    def __init__(
        self,
        R_ref: float = 8.0,
        R_min: float = 1.0,
        R_max: float = 25.0,
    ):
        self.R_ref = R_ref
        self.R_min = R_min
        self.R_max = R_max
        
        self.log_kappa_min = np.log(0.01)
        self.log_kappa_max = np.log(20.0)
    
    def decode_action(
        self,
        raw_action: np.ndarray,
        target_direction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Decode raw action with constraints
        
        Args:
            raw_action: raw action array, range [-1, 1], shape (4,)
                - [0]: normalized radius mean
                - [1]: normalized radius std
                - [2]: normalized relative angle (relative to target direction)
                - [3]: normalized log_kappa
            target_direction: target direction angle (radians)
        
        Returns:
            Decoded action dictionary
        """
        raw_action = np.clip(raw_action, -1.0, 1.0)
        
        mu_r = (raw_action[0] + 1) * self.R_ref
        mu_r = np.clip(mu_r, self.R_ref * 0.5, self.R_ref * 2)
        
        sigma_r = (raw_action[1] + 1) * self.R_ref * 0.25
        sigma_r = np.clip(sigma_r, 0.0, self.R_ref * 0.5)
        
        mu_theta_rel = raw_action[2] * np.pi
        
        if target_direction is not None:
            mu_theta = target_direction + mu_theta_rel
            mu_theta = np.arctan2(np.sin(mu_theta), np.cos(mu_theta))
        else:
            mu_theta = mu_theta_rel
        
        log_kappa = raw_action[3] * 4
        log_kappa = np.clip(log_kappa, self.log_kappa_min, self.log_kappa_max)
        kappa = np.exp(log_kappa)
        
        return {
            'mu_r': float(mu_r),
            'sigma_r': float(sigma_r),
            'mu_theta': float(mu_theta),
            'mu_theta_rel': float(mu_theta_rel),
            'kappa': float(kappa),
        }
    
    def sample_herder_positions(
        self,
        num_herders: int,
        mu_r: float,
        sigma_r: float,
        mu_theta: float,
        kappa: float
    ) -> tuple:
        """
        Sample herder positions (angles and radii)
        
        Kappa semantics:
        - kappa ≈ 0: uniform distribution (herders spread 360° around flock)
        - kappa ≈ 1-5: moderately concentrated around mu_theta
        - kappa ≈ 10-20: highly concentrated around mu_theta
        - kappa -> infinity: all herders at mu_theta (overlap)
        
        Args:
            num_herders: number of herders
            mu_r: radius mean
            sigma_r: radius std
            mu_theta: angle mean (concentration center)
            kappa: concentration parameter
        
        Returns:
            (angles, radii): tuple of angle and radius arrays
        """
        angles = np.zeros(num_herders, dtype=np.float32)
        radii = np.zeros(num_herders, dtype=np.float32)
        
        for i in range(num_herders):
            if kappa < 0.1:
                angles[i] = np.random.uniform(-np.pi, np.pi)
            else:
                angles[i] = np.random.vonmises(mu_theta, kappa)
            
            effective_sigma = sigma_r
            if kappa > 1.0:
                effective_sigma = sigma_r * (1.0 + 0.5 * min(kappa / 10.0, 1.0))
            
            radii[i] = np.clip(
                np.random.normal(mu_r, effective_sigma),
                self.R_min,
                self.R_max
            )
        
        return angles, radii


class KappaScheduler:
    """
    Kappa parameter scheduler for training stability
    """
    
    def __init__(
        self,
        warmup_epochs: int = 100,
        kappa_init: float = 1.0,
        kappa_min: float = 0.01,
        kappa_max: float = 20.0,
    ):
        self.warmup_epochs = warmup_epochs
        self.kappa_init = kappa_init
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        
        self.log_kappa_min = np.log(kappa_min)
        self.log_kappa_max = np.log(kappa_max)
    
    def get_kappa(self, raw_log_kappa: float, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.kappa_init
        
        log_kappa = np.clip(raw_log_kappa, self.log_kappa_min, self.log_kappa_max)
        return float(np.exp(log_kappa))
