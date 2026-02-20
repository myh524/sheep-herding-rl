"""
Curriculum Learning Environment for Sheep Herding
Implements progressive difficulty scaling for better generalization
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from gym import spaces

from envs.sheep_flock import SheepFlockEnv
from envs.high_level_action import HighLevelAction


class CurriculumStage:
    """
    Curriculum learning stage definition
    
    Defines difficulty parameters for each stage:
    - num_sheep: Number of sheep
    - num_herders: Number of herders
    - world_size: World size
    - episode_length: Maximum episode length
    - target_success_rate: Success rate threshold for stage advancement
    - min_episodes: Minimum episodes before checking advancement
    """
    
    def __init__(
        self,
        name: str,
        num_sheep: int,
        num_herders: int,
        world_size: Tuple[float, float],
        episode_length: int,
        target_success_rate: float,
        min_episodes: int,
    ):
        self.name = name
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        self.world_size = world_size
        self.episode_length = episode_length
        self.target_success_rate = target_success_rate
        self.min_episodes = min_episodes


class CurriculumSheepFlockEnv(SheepFlockEnv):
    """
    Curriculum learning environment for sheep herding
    
    Features:
    1. Progressive difficulty scaling
    2. Randomized target and initial positions
    3. Multi-stage training support
    """
    
    DEFAULT_STAGES = [
        CurriculumStage(
            name='Stage 0: Simple',
            num_sheep=3,
            num_herders=3,
            world_size=(60.0, 60.0),
            episode_length=80,
            target_success_rate=0.8,
            min_episodes=50,
        ),
        CurriculumStage(
            name='Stage 1: Medium',
            num_sheep=5,
            num_herders=3,
            world_size=(60.0, 60.0),
            episode_length=100,
            target_success_rate=0.7,
            min_episodes=100,
        ),
        CurriculumStage(
            name='Stage 2: Target',
            num_sheep=10,
            num_herders=3,
            world_size=(60.0, 60.0),
            episode_length=150,
            target_success_rate=0.6,
            min_episodes=200,
        ),
    ]
    
    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
        start_stage: int = 0,
        dt: float = 0.1,
        random_seed: Optional[int] = None,
        auto_advance: bool = True,
        action_repeat: int = 5,
    ):
        """
        Initialize curriculum learning environment
        
        Args:
            stages: List of curriculum stages, uses default if None
            start_stage: Starting stage index
            dt: Time step
            random_seed: Random seed
            auto_advance: Whether to automatically advance to next stage
            action_repeat: Number of sub-steps to repeat each action
        """
        self.stages = stages if stages is not None else self.DEFAULT_STAGES
        self.current_stage_idx = start_stage
        self.auto_advance = auto_advance
        
        self.episode_history: List[bool] = []
        self.total_episodes = 0
        
        current_stage = self.stages[self.current_stage_idx]
        
        super().__init__(
            world_size=current_stage.world_size,
            num_sheep=current_stage.num_sheep,
            num_herders=current_stage.num_herders,
            episode_length=current_stage.episode_length,
            dt=dt,
            random_seed=random_seed,
            action_repeat=action_repeat,
        )
    
    def _get_current_stage(self) -> CurriculumStage:
        """Get current stage"""
        return self.stages[self.current_stage_idx]
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        
        if self._seed is not None:
            np.random.seed(self._seed)
            self._seed = None
        
        target_pos = np.array([
            self.world_size[0] * np.random.uniform(0.6, 0.9),
            self.world_size[1] * np.random.uniform(0.3, 0.7)
        ])
        
        self.scenario.reset(target_position=target_pos)
        self.prev_distance = self.scenario.get_distance_to_target()
        
        return self._get_obs()
    
    def _check_done(self) -> bool:
        """Check if episode is done"""
        if self.scenario.is_flock_at_target():
            return True
        
        if self.current_step >= self.episode_length:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        info = super()._get_info()
        info['stage_id'] = self.current_stage_idx
        info['stage_name'] = self._get_current_stage().name
        info['total_episodes'] = self.total_episodes
        info['success_rate'] = self._get_success_rate()
        return info
    
    def episode_end(self, success: bool):
        """
        Record episode end
        
        Args:
            success: Whether episode was successful
        """
        self.episode_history.append(success)
        self.total_episodes += 1
        
        if len(self.episode_history) > 100:
            self.episode_history = self.episode_history[-100:]
        
        if self.auto_advance:
            self._check_advance()
    
    def _get_success_rate(self) -> float:
        """Get recent success rate"""
        if not self.episode_history:
            return 0.0
        return sum(self.episode_history) / len(self.episode_history)
    
    def _check_advance(self):
        """Check if should advance to next stage"""
        stage = self._get_current_stage()
        
        if len(self.episode_history) >= stage.min_episodes:
            success_rate = self._get_success_rate()
            
            if success_rate >= stage.target_success_rate:
                self.advance_stage()
    
    def advance_stage(self) -> bool:
        """
        Advance to next stage
        
        Returns:
            Whether successfully advanced to next stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        self.current_stage_idx += 1
        new_stage = self._get_current_stage()
        
        self.world_size = new_stage.world_size
        self.num_sheep = new_stage.num_sheep
        self.num_herders = new_stage.num_herders
        self.episode_length = new_stage.episode_length
        
        self.scenario = type(self.scenario)(
            world_size=new_stage.world_size,
            num_sheep=new_stage.num_sheep,
            num_herders=new_stage.num_herders,
        )
        
        self.action_decoder = HighLevelAction(
            R_ref=np.linalg.norm(new_stage.world_size) / 6.0,
            R_min=1.0,
            R_max=np.linalg.norm(new_stage.world_size) / 2.0,
        )
        
        self._setup_spaces()
        
        self.episode_history = []
        
        print(f"Advanced to {new_stage.name}: "
              f"sheep={new_stage.num_sheep}, "
              f"herders={new_stage.num_herders}, "
              f"world={new_stage.world_size}, "
              f"episode_length={new_stage.episode_length}")
        
        return True
    
    def set_stage(self, stage_idx: int):
        """
        Set current stage
        
        Args:
            stage_idx: Stage index
        """
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            new_stage = self._get_current_stage()
            
            self.world_size = new_stage.world_size
            self.num_sheep = new_stage.num_sheep
            self.num_herders = new_stage.num_herders
            self.episode_length = new_stage.episode_length
            
            self.scenario = type(self.scenario)(
                world_size=new_stage.world_size,
                num_sheep=new_stage.num_sheep,
                num_herders=new_stage.num_herders,
            )
            
            self.action_decoder = HighLevelAction(
                R_ref=np.linalg.norm(new_stage.world_size) / 6.0,
                R_min=1.0,
                R_max=np.linalg.norm(new_stage.world_size) / 2.0,
            )
            
            self._setup_spaces()
            self.episode_history = []
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get curriculum learning info"""
        stage = self._get_current_stage()
        return {
            'current_stage': self.current_stage_idx,
            'stage_name': stage.name,
            'num_stages': len(self.stages),
            'num_sheep': stage.num_sheep,
            'num_herders': stage.num_herders,
            'world_size': stage.world_size,
            'episode_length': stage.episode_length,
            'target_success_rate': stage.target_success_rate,
            'min_episodes': stage.min_episodes,
            'success_rate': self._get_success_rate(),
            'total_episodes': self.total_episodes,
            'episodes_in_stage': len(self.episode_history),
        }


class RandomizedSheepFlockEnv(SheepFlockEnv):
    """
    支持随机化的羊群引导环境
    
    用于训练泛化能力，随机化:
    - 羊的数量
    - 初始位置
    - 目标位置
    - 羊的速度
    """
    
    def __init__(
        self,
        num_sheep_range: Tuple[int, int] = (5, 15),
        num_herders_range: Tuple[int, int] = (2, 4),
        world_size_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((40.0, 40.0), (60.0, 60.0)),
        sheep_speed_range: Tuple[float, float] = (0.5, 1.5),
        episode_length: int = 100,
        dt: float = 0.1,
        random_seed: Optional[int] = None,
        action_repeat: int = 5,
    ):
        """
        初始化随机化环境
        
        Args:
            num_sheep_range: 羊数量范围
            num_herders_range: 机械狗数量范围
            world_size_range: 世界大小范围
            sheep_speed_range: 羊速度范围
            episode_length: 每个episode的最大步数
            dt: 时间步长
            random_seed: 随机种子
            action_repeat: 每个动作重复执行的子步数
        """
        self.num_sheep_range = num_sheep_range
        self.num_herders_range = num_herders_range
        self.world_size_range = world_size_range
        self.sheep_speed_range = sheep_speed_range
        
        self._rng = np.random.RandomState(random_seed)
        
        initial_world_size = (
            (world_size_range[0][0] + world_size_range[1][0]) / 2,
            (world_size_range[0][1] + world_size_range[1][1]) / 2,
        )
        initial_num_sheep = (num_sheep_range[0] + num_sheep_range[1]) // 2
        initial_num_herders = (num_herders_range[0] + num_herders_range[1]) // 2
        
        super().__init__(
            world_size=initial_world_size,
            num_sheep=initial_num_sheep,
            num_herders=initial_num_herders,
            episode_length=episode_length,
            dt=dt,
            random_seed=random_seed,
            action_repeat=action_repeat,
        )
    
    def reset(self) -> np.ndarray:
        """重置环境，随机化参数"""
        self.current_step = 0
        
        self.num_sheep = self._rng.randint(
            self.num_sheep_range[0],
            self.num_sheep_range[1] + 1
        )
        self.num_herders = self._rng.randint(
            self.num_herders_range[0],
            self.num_herders_range[1] + 1
        )
        
        self.world_size = (
            self._rng.uniform(
                self.world_size_range[0][0],
                self.world_size_range[1][0]
            ),
            self._rng.uniform(
                self.world_size_range[0][1],
                self.world_size_range[1][1]
            ),
        )
        
        sheep_speed = self._rng.uniform(
            self.sheep_speed_range[0],
            self.sheep_speed_range[1]
        )
        
        self.scenario = type(self.scenario)(
            world_size=self.world_size,
            num_sheep=self.num_sheep,
            num_herders=self.num_herders,
            sheep_config={
                'max_speed': sheep_speed,
                'max_force': 0.1,
                'perception_radius': 5.0,
                'separation_radius': 2.0,
            },
        )
        
        self.action_decoder = HighLevelAction(
            R_ref=np.linalg.norm(self.world_size) / 6.0,
            R_min=1.0,
            R_max=np.linalg.norm(self.world_size) / 2.0,
        )
        
        self._setup_spaces()
        
        target_pos = np.array([
            self.world_size[0] * self._rng.uniform(0.7, 0.9),
            self.world_size[1] * self._rng.uniform(0.3, 0.7)
        ])
        
        self.scenario.reset(target_position=target_pos)
        self.prev_distance = self.scenario.get_distance_to_target()
        
        return self._get_obs()
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = super()._get_info()
        info['randomized_num_sheep'] = self.num_sheep
        info['randomized_num_herders'] = self.num_herders
        info['randomized_world_size'] = self.world_size
        return info
