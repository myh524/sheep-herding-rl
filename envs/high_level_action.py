"""
High-Level Action Decoder for Sheep Herding
Decodes normalized actions [-1, 1] to formation parameters

Design Philosophy:
==================
Based on real sheep herding physics and practical engineering considerations:

1. INTUITIVE PARAMETERS:
   - wedge_center: Where to position the formation (relative to target direction)
   - wedge_width: How spread out the herders are (0=concentrated, 1=360° surround)
   - radius: Distance from flock center
   - asymmetry: Offset for flank maneuvers (-1=left flank, 0=center, 1=right flank)

2. DETERMINISTIC MAPPING:
   Same action → Same target positions (no random sampling)
   This prevents target "flickering" and stabilizes policy learning.

3. PHYSICAL INTUITION:
   - wedge_width=0: All herders at same angle (push mode)
   - wedge_width=0.5: Herders spread 180° (half surround)
   - wedge_width=1: Herders spread 360° uniformly (full surround)

Action Space (4D):
==================
[0] wedge_center: [-1, 1] → angle offset from target direction [-π, π]
[1] wedge_width:  [-1, 1] → spread factor [0, 1] (0=push, 1=surround)
[2] radius:       [-1, 1] → distance from flock [R_min, R_max]
[3] asymmetry:    [-1, 1] → flank offset [-wedge_width/2, +wedge_width/2]
"""

import numpy as np
from typing import Dict, Any, Optional


class HighLevelAction:
    """
    High-level action decoder with intuitive sheep herding semantics
    
    Action Space (4D continuous, each in [-1, 1]):
    - wedge_center: Formation center angle (relative to target direction)
    - wedge_width: Formation spread (0=concentrated push, 1=360° surround)
    - radius: Distance from flock center
    - asymmetry: Flank offset for directional steering
    """
    
    def __init__(
        self,
        R_ref: float = 8.0,
        R_min: float = 3.0,
        R_max: float = 20.0,
    ):
        self.R_ref = R_ref
        self.R_min = R_min
        self.R_max = R_max
    
    def decode_action(
        self,
        raw_action: np.ndarray,
        target_direction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Decode raw action to formation parameters
        
        Args:
            raw_action: raw action array, range [-1, 1], shape (4,)
                - [0]: wedge_center (formation center angle offset)
                - [1]: wedge_width (formation spread factor)
                - [2]: radius (distance from flock)
                - [3]: asymmetry (flank offset)
            target_direction: target direction angle (radians)
        
        Returns:
            Decoded action dictionary with physical parameters
        """
        raw_action = np.clip(raw_action, -1.0, 1.0)
        
        wedge_center_offset = raw_action[0] * np.pi
        if target_direction is not None:
            wedge_center = target_direction + wedge_center_offset
            wedge_center = np.arctan2(np.sin(wedge_center), np.cos(wedge_center))
        else:
            wedge_center = wedge_center_offset
        
        wedge_width = (raw_action[1] + 1) / 2
        
        radius = self.R_min + (raw_action[2] + 1) / 2 * (self.R_max - self.R_min)
        radius = np.clip(radius, self.R_min, self.R_max)
        
        asymmetry = raw_action[3] * 0.5
        
        return {
            'wedge_center': float(wedge_center),
            'wedge_center_offset': float(wedge_center_offset),
            'wedge_width': float(wedge_width),
            'radius': float(radius),
            'asymmetry': float(asymmetry),
        }
    
    def sample_herder_positions(
        self,
        num_herders: int,
        wedge_center: float,
        wedge_width: float,
        radius: float,
        asymmetry: float,
        **kwargs
    ) -> tuple:
        """
        Compute herder positions deterministically
        
        Formation Logic:
        1. wedge_width controls total spread:
           - 0.0: All herders at wedge_center (push mode)
           - 0.5: Herders spread 180° (half surround)
           - 1.0: Herders spread 360° uniformly around flock (full surround)
        
        2. asymmetry controls flank emphasis:
           - 0.0: Symmetric distribution around wedge_center
           - >0: Shift formation to right side (steer flock left)
           - <0: Shift formation to left side (steer flock right)
        
        Args:
            num_herders: number of herders
            wedge_center: formation center angle (radians)
            wedge_width: spread factor [0, 1]
            radius: distance from flock center
            asymmetry: flank offset [-0.5, 0.5]
        
        Returns:
            (angles, radii): tuple of angle and radius arrays
        """
        angles = np.zeros(num_herders, dtype=np.float32)
        radii = np.zeros(num_herders, dtype=np.float32)
        
        if wedge_width >= 1.0:
            for i in range(num_herders):
                angles[i] = wedge_center + i * (2 * np.pi / num_herders)
                angles[i] = np.arctan2(np.sin(angles[i]), np.cos(angles[i]))
                radii[i] = radius
        else:
            total_spread = wedge_width * 2 * np.pi
            
            center_offset = asymmetry * total_spread
            formation_center = wedge_center + center_offset
            formation_center = np.arctan2(np.sin(formation_center), np.cos(formation_center))
            
            if num_herders == 1:
                angles[0] = formation_center
                radii[0] = radius
            else:
                for i in range(num_herders):
                    fraction = i / (num_herders - 1)
                    
                    angle_offset = (fraction - 0.5) * total_spread
                    angles[i] = formation_center + angle_offset
                    angles[i] = np.arctan2(np.sin(angles[i]), np.cos(angles[i]))
                    
                    edge_factor = abs(fraction - 0.5) * 2
                    radii[i] = radius * (1.0 + edge_factor * 0.1)
                    radii[i] = np.clip(radii[i], self.R_min, self.R_max)
        
        return angles, radii
    
    def get_formation_mode(self, wedge_width: float) -> str:
        """
        Get human-readable formation mode name
        
        Args:
            wedge_width: spread factor [0, 1]
        
        Returns:
            Formation mode string
        """
        if wedge_width < 0.2:
            return "PUSH"
        elif wedge_width < 0.5:
            return "NARROW"
        elif wedge_width < 0.8:
            return "WIDE"
        elif wedge_width < 1.0:
            return "HALF_SURROUND"
        else:
            return "FULL_SURROUND"


class FormationAnalyzer:
    """
    Analyze and visualize formation parameters for debugging
    """
    
    @staticmethod
    def describe_formation(decoded_action: Dict[str, Any], num_herders: int) -> str:
        """
        Generate human-readable description of the formation
        
        Args:
            decoded_action: Decoded action dictionary
            num_herders: number of herders
        
        Returns:
            Description string
        """
        wedge_width = decoded_action['wedge_width']
        asymmetry = decoded_action['asymmetry']
        radius = decoded_action['radius']
        
        mode = HighLevelAction().get_formation_mode(wedge_width)
        
        if wedge_width >= 1.0:
            spread_deg = 360
        else:
            spread_deg = wedge_width * 360
        
        if abs(asymmetry) < 0.1:
            flank = "centered"
        elif asymmetry > 0:
            flank = f"right flank (+{asymmetry*100:.0f}%)"
        else:
            flank = f"left flank ({asymmetry*100:.0f}%)"
        
        return (
            f"Formation: {mode} mode\n"
            f"  - Spread: {spread_deg:.0f}°\n"
            f"  - Radius: {radius:.1f} units\n"
            f"  - Flank: {flank}\n"
            f"  - Herders: {num_herders}"
        )


class KappaScheduler:
    """
    Legacy compatibility - now controls wedge_width warmup
    """
    
    def __init__(
        self,
        warmup_epochs: int = 100,
        kappa_init: float = 0.5,
        **kwargs
    ):
        self.warmup_epochs = warmup_epochs
        self.wedge_width_init = kappa_init
    
    def get_wedge_width(self, raw_width: float, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.wedge_width_init
        return (raw_width + 1) / 2
