"""
Curriculum Learning Environment for Sheep Herding
Implements progressive difficulty scaling for better generalization
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from gym import spaces

from envs.defaults import (
    DEFAULT_CURRICULUM_STAGE_SPECS,
    DEFAULT_DT,
    DEFAULT_FLOCK_EPISODE_LENGTH,
    DEFAULT_RANDOMIZED_NUM_HERDERS_RANGE,
    DEFAULT_RANDOMIZED_NUM_SHEEP_RANGE,
    DEFAULT_RANDOMIZED_SHEEP_SPEED_RANGE,
    DEFAULT_RANDOMIZED_WORLD_SIZE_RANGE,
    FORMATION_DELTA_COVERAGE_MAX,
    FORMATION_DELTA_RADIUS_MAX,
    FORMATION_DELTA_THETA_MAX_RAD,
    randomized_sheep_config,
)
from envs.sheep_flock import SheepFlockEnv
from envs.sheep_scenario import sample_random_target_position
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
        CurriculumStage(**spec) for spec in DEFAULT_CURRICULUM_STAGE_SPECS
    ]
    
    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
        start_stage: int = 0,
        dt: float = DEFAULT_DT,
        random_seed: Optional[int] = None,
        auto_advance: bool = True,
        use_herder_kinematics: bool = True,
        end_episode_when_at_target: bool = False,
        info_include_flock_state: bool = False,
        formation_delta_mode: bool = False,
        formation_delta_theta_max_rad: float = FORMATION_DELTA_THETA_MAX_RAD,
        formation_delta_radius_max: float = FORMATION_DELTA_RADIUS_MAX,
        formation_delta_coverage_max: float = FORMATION_DELTA_COVERAGE_MAX,
        high_level_interval: Optional[int] = None,
        herder_motion: Optional[Dict[str, Any]] = None,
        herder_physics_legacy: bool = False,
        enforce_disk_boundary: bool = True,
        low_level_model_dir: Optional[str] = None,
        low_level_substeps: Optional[int] = None,
        low_level_device: str = "cpu",
        low_level_world_size: Optional[float] = None,
        low_level_num_obstacles: int = 1,
        low_level_max_speed: float = 2.0,
        low_level_max_edge_dist: Optional[float] = None,
        low_level_use_shepherd: bool = True,
        low_level_deterministic: bool = True,
    ):
        """
        Initialize curriculum learning environment
        
        Args:
            stages: List of curriculum stages, uses default if None
            start_stage: Starting stage index
            dt: Time step
            random_seed: Random seed
            auto_advance: Whether to automatically advance to next stage
            use_herder_kinematics: 与 SheepFlockEnv 相同；阶段切换时保留该设置
            end_episode_when_at_target: 与 SheepFlockEnv 相同
            info_include_flock_state: 与 SheepFlockEnv 相同
            formation_delta_mode, formation_delta_*: 与 SheepFlockEnv 相同
            high_level_interval: 与 SheepFlockEnv 相同
            herder_motion, herder_physics_legacy: 与 SheepFlockEnv 相同
            low_level_*: 与 SheepFlockEnv 相同（注意阶段切换时 num_herders 变化需与低层 checkpoint 一致）
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
            use_herder_kinematics=use_herder_kinematics,
            end_episode_when_at_target=end_episode_when_at_target,
            info_include_flock_state=info_include_flock_state,
            formation_delta_mode=formation_delta_mode,
            formation_delta_theta_max_rad=formation_delta_theta_max_rad,
            formation_delta_radius_max=formation_delta_radius_max,
            formation_delta_coverage_max=formation_delta_coverage_max,
            high_level_interval=high_level_interval,
            herder_motion=herder_motion,
            herder_physics_legacy=herder_physics_legacy,
            enforce_disk_boundary=enforce_disk_boundary,
            low_level_model_dir=low_level_model_dir,
            low_level_substeps=low_level_substeps,
            low_level_device=low_level_device,
            low_level_world_size=low_level_world_size,
            low_level_num_obstacles=low_level_num_obstacles,
            low_level_max_speed=low_level_max_speed,
            low_level_max_edge_dist=low_level_max_edge_dist,
            low_level_use_shepherd=low_level_use_shepherd,
            low_level_deterministic=low_level_deterministic,
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
        
        target_pos = sample_random_target_position(self.world_size)
        
        self.scenario.reset(target_position=target_pos)
        self._sync_prev_d_hat_after_reset()
        self._reward_components = {}
        self._reset_formation_integrator()

        return self._get_obs()

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
            use_herder_kinematics=self.scenario.use_herder_kinematics,
            herder_motion=dict(self.herder_motion),
            enforce_disk_boundary=self.enforce_disk_boundary,
        )
        
        self.action_decoder = HighLevelAction()
        
        self._setup_spaces()
        self._reset_formation_integrator()

        self.episode_history = []
        self._low_bridge = None

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
                use_herder_kinematics=self.scenario.use_herder_kinematics,
                herder_motion=dict(self.herder_motion),
                enforce_disk_boundary=self.enforce_disk_boundary,
            )
            
            self.action_decoder = HighLevelAction()
            
            self._setup_spaces()
            self._reset_formation_integrator()
            self.episode_history = []
            self._low_bridge = None

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
        num_sheep_range: Tuple[int, int] = DEFAULT_RANDOMIZED_NUM_SHEEP_RANGE,
        num_herders_range: Tuple[int, int] = DEFAULT_RANDOMIZED_NUM_HERDERS_RANGE,
        world_size_range: Tuple[Tuple[float, float], Tuple[float, float]] = DEFAULT_RANDOMIZED_WORLD_SIZE_RANGE,
        sheep_speed_range: Tuple[float, float] = DEFAULT_RANDOMIZED_SHEEP_SPEED_RANGE,
        episode_length: int = DEFAULT_FLOCK_EPISODE_LENGTH,
        dt: float = DEFAULT_DT,
        random_seed: Optional[int] = None,
        use_herder_kinematics: bool = True,
        end_episode_when_at_target: bool = False,
        info_include_flock_state: bool = False,
        formation_delta_mode: bool = False,
        formation_delta_theta_max_rad: float = FORMATION_DELTA_THETA_MAX_RAD,
        formation_delta_radius_max: float = FORMATION_DELTA_RADIUS_MAX,
        formation_delta_coverage_max: float = FORMATION_DELTA_COVERAGE_MAX,
        high_level_interval: Optional[int] = None,
        herder_motion: Optional[Dict[str, Any]] = None,
        herder_physics_legacy: bool = False,
        enforce_disk_boundary: bool = True,
        low_level_model_dir: Optional[str] = None,
        low_level_substeps: Optional[int] = None,
        low_level_device: str = "cpu",
        low_level_world_size: Optional[float] = None,
        low_level_num_obstacles: int = 1,
        low_level_max_speed: float = 2.0,
        low_level_max_edge_dist: Optional[float] = None,
        low_level_use_shepherd: bool = True,
        low_level_deterministic: bool = True,
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
            use_herder_kinematics: 与 SheepFlockEnv 相同
            end_episode_when_at_target: 与 SheepFlockEnv 相同
            info_include_flock_state: 与 SheepFlockEnv 相同
            formation_delta_mode, formation_delta_*: 与 SheepFlockEnv 相同
            high_level_interval: 与 SheepFlockEnv 相同
            herder_motion, herder_physics_legacy: 与 SheepFlockEnv 相同
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
            use_herder_kinematics=use_herder_kinematics,
            end_episode_when_at_target=end_episode_when_at_target,
            info_include_flock_state=info_include_flock_state,
            formation_delta_mode=formation_delta_mode,
            formation_delta_theta_max_rad=formation_delta_theta_max_rad,
            formation_delta_radius_max=formation_delta_radius_max,
            formation_delta_coverage_max=formation_delta_coverage_max,
            high_level_interval=high_level_interval,
            herder_motion=herder_motion,
            herder_physics_legacy=herder_physics_legacy,
            enforce_disk_boundary=enforce_disk_boundary,
            low_level_model_dir=low_level_model_dir,
            low_level_substeps=low_level_substeps,
            low_level_device=low_level_device,
            low_level_world_size=low_level_world_size,
            low_level_num_obstacles=low_level_num_obstacles,
            low_level_max_speed=low_level_max_speed,
            low_level_max_edge_dist=low_level_max_edge_dist,
            low_level_use_shepherd=low_level_use_shepherd,
            low_level_deterministic=low_level_deterministic,
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
            sheep_config=randomized_sheep_config(sheep_speed),
            use_herder_kinematics=self.scenario.use_herder_kinematics,
            herder_motion=dict(self.herder_motion),
            enforce_disk_boundary=self.enforce_disk_boundary,
        )
        
        self.action_decoder = HighLevelAction()
        
        self._setup_spaces()
        self._low_bridge = None

        target_pos = sample_random_target_position(self.world_size, rng=self._rng)
        
        self.scenario.reset(target_position=target_pos)
        self._sync_prev_d_hat_after_reset()
        self._reward_components = {}
        self._reset_formation_integrator()

        return self._get_obs()

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = super()._get_info()
        info['randomized_num_sheep'] = self.num_sheep
        info['randomized_num_herders'] = self.num_herders
        info['randomized_world_size'] = self.world_size
        return info
