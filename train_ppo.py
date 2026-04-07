"""
Simplified PPO Training Script for Sheep Herding
Single-agent version without num_agents dimension
"""

import argparse
import os
from collections import defaultdict
import torch
import numpy as np
from datetime import datetime
from typing import Tuple

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic
from onpolicy.utils.ppo_buffer import PPOReplayBuffer
from onpolicy.utils.reward_normalizer import RunningMeanStd


class SimpleLogger:
    """增强版训练日志记录器，支持多指标记录和JSON格式导出；可选 TensorBoard。"""
    def __init__(self, save_folder, use_tensorboard: bool = False):
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)
        self.log_file = open(os.path.join(save_folder, 'training_log.txt'), 'w')
        self.json_log_file = open(os.path.join(save_folder, 'training_metrics.json'), 'w')
        self.metrics_history = []
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(save_folder, 'tb')
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print(f"TensorBoard log dir: {tb_dir}")
            except ImportError:
                print("未安装 tensorboard，跳过。安装: pip install tensorboard")
                print("警告: 已传 --use_tensorboard 但无法创建 SummaryWriter，TensorBoard 将无曲线。")

    def log(self, metrics, *, file_stdout: bool = True, tensorboard: bool = True):
        """file_stdout: 打印并写入 training_log / JSON 历史；tensorboard: 写入事件文件（可更高频）。"""
        if file_stdout:
            parts = []
            for k, v in metrics.items():
                if k.startswith('reward/'):
                    continue
                if isinstance(v, float):
                    parts.append(f'{k}: {v:.6f}')
                else:
                    parts.append(f'{k}: {v}')
            msg = ' | '.join(parts)
            print(msg)
            self.log_file.write(msg + '\n')
            self.log_file.flush()
            self.metrics_history.append(metrics)

        if tensorboard and self.tb_writer is not None:
            step = int(np.asarray(metrics.get('total_steps', 0)).item())
            for k, v in metrics.items():
                if k == 'success_rate' and isinstance(v, str):
                    s = v.strip().rstrip('%')
                    try:
                        self.tb_writer.add_scalar('train/success_rate', float(s) / 100.0, step)
                    except ValueError:
                        pass
                    continue
                try:
                    if isinstance(v, (bool, np.bool_)):
                        continue
                    fv = float(np.asarray(v).item())
                except (TypeError, ValueError, OverflowError):
                    continue
                tag = k if k.startswith('reward/') else f'train/{k}'
                self.tb_writer.add_scalar(tag, fv, step)
            self.tb_writer.flush()

    def save_json(self):
        """保存完整的训练历史到JSON文件"""
        import json
        with open(os.path.join(self.save_folder, 'training_metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def close(self):
        self.save_json()
        self.log_file.close()
        self.json_log_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='sheep_herding')
    parser.add_argument('--scenario_name', type=str, default='default')

    parser.add_argument('--num_sheep', type=int, default=10)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument('--episode_length', type=int, default=150)
    parser.add_argument('--world_size', type=float, nargs=2, default=[50.0, 50.0])
    
    parser.add_argument('--use_curriculum', action='store_true', default=False,
                        help='Use curriculum learning for training')
    parser.add_argument('--use_randomized', action='store_true', default=False,
                        help='Use randomized environment for generalization training')
    parser.add_argument('--start_stage', type=int, default=0,
                        help='Starting stage for curriculum learning')
    parser.add_argument(
        '--herder_teleport',
        action='store_true',
        default=False,
        help='机械狗每步直接置于编队目标点，关闭运动学（先验队形效果；后续可再训低层运动）',
    )
    parser.add_argument(
        '--end_episode_when_at_target',
        action='store_true',
        default=False,
        help='羊群质心距目标 < 阈值时立刻结束 episode（旧行为）；默认关闭，跑满 episode_length 以学习滞留控制',
    )

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_env_steps', type=int, default=1000000)
    parser.add_argument('--n_rollout_threads', type=int, default=1)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate schedule')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate for cosine/linear decay')
    parser.add_argument('--ppo_epoch', type=int, default=10)
    parser.add_argument('--num_mini_batch', type=int, default=4)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--clip_param_final', type=float, default=0.1,
                        help='Final clip parameter for clip annealing')
    parser.add_argument('--use_clip_annealing', action='store_true', default=False,
                        help='Enable clip parameter annealing during training')
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--use_adaptive_gae', action='store_true', default=False)
    parser.add_argument('--gae_lambda_min', type=float, default=0.9)
    parser.add_argument('--gae_lambda_max', type=float, default=0.995)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--use_ReLU', action='store_true', default=True)
    parser.add_argument('--network_architecture', type=str, default='mlp', choices=['mlp', 'improved_mlp', 'resnet', 'lstm'],
                        help='Network architecture to use: mlp, improved_mlp, resnet, lstm')
    parser.add_argument('--use_layer_norm', action='store_true', default=True)
    parser.add_argument('--use_dropout', action='store_true', default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--use_orthogonal', action='store_true', default=True)
    parser.add_argument('--use_feature_normalization', action='store_true', default=True)
    parser.add_argument('--use_popart', action='store_true', default=False)
    parser.add_argument('--use_valuenorm', action='store_true', default=False)
    parser.add_argument('--use_gae', action='store_true', default=True)
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=True)
    parser.add_argument('--use_clipped_value_loss', action='store_true', default=True)
    parser.add_argument('--use_reward_normalization', action='store_true', default=True)

    parser.add_argument('--use_entropy_decay', action='store_true', default=True)
    parser.add_argument('--initial_entropy_coef', type=float, default=0.05)
    parser.add_argument('--final_entropy_coef', type=float, default=0.001)
    parser.add_argument('--use_cosine_lr', action='store_true', default=True)
    
    parser.add_argument('--use_kl_penalty', action='store_true', default=True)
    parser.add_argument('--target_kl', type=float, default=0.015)
    parser.add_argument('--kl_coef', type=float, default=0.2)
    parser.add_argument('--kl_coef_multiplier', type=float, default=1.5)
    parser.add_argument('--kl_coef_min', type=float, default=0.01)
    parser.add_argument('--kl_coef_max', type=float, default=2.0)

    parser.add_argument('--recurrent_N', type=int, default=1)
    parser.add_argument('--use_naive_recurrent_policy', action='store_true', default=False)
    parser.add_argument('--use_recurrent_policy', action='store_true', default=False)

    parser.add_argument('--gain', type=float, default=0.5)
    parser.add_argument('--use_policy_active_masks', action='store_true', default=True)
    parser.add_argument('--stacked_frames', type=int, default=1)

    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='sheep_herding')
    parser.add_argument(
        '--use_tensorboard',
        action='store_true',
        default=False,
        help='写入 TensorBoard 事件到 run 目录下的 tb/，便于边训边看曲线',
    )
    parser.add_argument(
        '--tensorboard_log_interval',
        type=int,
        default=1,
        help='每 N 个 PPO macro 写一次 TensorBoard（默认 1=每个 macro 一个点）；不影响 log_interval',
    )

    parser.add_argument('--algorithm_name', type=str, default='ppo')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False)

    args = parser.parse_args()
    return args


def make_train_env(args, seed_offset: int = 0):
    """Create one env instance; seed_offset makes parallel envs use different RNG streams."""
    rs = None if args.seed is None else int(args.seed) + int(seed_offset)
    use_herder_kinematics = not args.herder_teleport
    end_on_target = args.end_episode_when_at_target
    if args.use_curriculum:
        from envs import CurriculumSheepFlockEnv
        env = CurriculumSheepFlockEnv(
            start_stage=args.start_stage,
            random_seed=rs,
            use_herder_kinematics=use_herder_kinematics,
            end_episode_when_at_target=end_on_target,
        )
    elif args.use_randomized:
        from envs import RandomizedSheepFlockEnv
        env = RandomizedSheepFlockEnv(
            episode_length=args.episode_length,
            random_seed=rs,
            use_herder_kinematics=use_herder_kinematics,
            end_episode_when_at_target=end_on_target,
        )
    else:
        from envs import SheepFlockEnv
        env = SheepFlockEnv(
            world_size=tuple(args.world_size),
            num_sheep=args.num_sheep,
            num_herders=args.num_herders,
            episode_length=args.episode_length,
            random_seed=rs,
            use_herder_kinematics=use_herder_kinematics,
            end_episode_when_at_target=end_on_target,
        )
    return env


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOTrainer:
    """Simplified PPO Trainer for single-agent sheep herding"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(args.seed)

        n_threads = args.n_rollout_threads
        self.envs = [make_train_env(args, seed_offset=i) for i in range(n_threads)]
        self.env = self.envs[0]
        self.num_herders = args.num_herders

        # Create policy based on selected architecture
        if args.network_architecture == 'improved_mlp':
            from onpolicy.algorithms.ppo_actor_critic import ImprovedActorCritic
            self.policy = ImprovedActorCritic(
                args,
                self.env.observation_space,
                self.env.action_space,
                self.device,
            ).to(self.device)
        else:
            self.policy = PPOActorCritic(
                args,
                self.env.observation_space,
                self.env.action_space,
                self.device,
            ).to(self.device)

        self.entropy_coef = args.initial_entropy_coef
        self.initial_entropy_coef = args.initial_entropy_coef
        self.final_entropy_coef = args.final_entropy_coef
        
        # Clip parameter for PPO
        self.clip_param = args.clip_param
        self.clip_param_final = args.clip_param_final
        self.use_clip_annealing = args.use_clip_annealing

        self.use_kl_penalty = args.use_kl_penalty
        self.target_kl = args.target_kl
        self.kl_coef = args.kl_coef
        self.kl_coef_multiplier = args.kl_coef_multiplier
        self.kl_coef_min = args.kl_coef_min
        self.kl_coef_max = args.kl_coef_max
        self.kl_divergence = 0.0

        self.use_adaptive_gae = args.use_adaptive_gae
        self.gae_lambda = args.gae_lambda
        self.gae_lambda_min = args.gae_lambda_min
        self.gae_lambda_max = args.gae_lambda_max
        self.value_pred_error = 0.0

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=args.lr,
        )

        self.buffer = PPOReplayBuffer(
            args,
            self.env.observation_space,
            self.env.action_space,
        )
        # Override the gae_lambda in buffer to use our dynamic value
        if hasattr(self.buffer, 'gae_lambda'):
            self.buffer.gae_lambda = self.gae_lambda

        from onpolicy.utils.valuenorm import ValueNorm
        self.value_normalizer = ValueNorm(1) if args.use_valuenorm else None

        self.reward_normalizer = RunningMeanStd() if args.use_reward_normalization else None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(
            'results',
            args.env_name,
            args.scenario_name,
            args.algorithm_name,
            f'seed{args.seed}',
            timestamp,
        )
        if args.experiment_name:
            run_dir = os.path.join(run_dir, args.experiment_name)

        self.run_dir = run_dir
        self.save_dir = os.path.join(run_dir, 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = SimpleLogger(run_dir, use_tensorboard=args.use_tensorboard)

        self.total_num_steps = 0
        self.episode = 0

    def _reset_all_envs(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs], axis=0)

    def _expand_actions_for_herders(self, action: np.ndarray, num_herders: int) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        actions = np.zeros((num_herders, a.shape[0]), dtype=np.float32)
        for i in range(num_herders):
            actions[i] = a.copy()
        return actions

    def _step_all_envs(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step each env with its own action. On done, curriculum hooks + reset.
        Returns stacked obs (N, obs_dim), rewards (N, 1), dones (N,), infos list.
        """
        n = self.args.n_rollout_threads
        obs_list: list = []
        rewards = np.zeros((n, 1), dtype=np.float32)
        dones = np.zeros((n,), dtype=bool)
        infos: list = []

        for i, env in enumerate(self.envs):
            expanded = self._expand_actions_for_herders(actions[i], env.num_herders)
            o, r, d, info = env.step(expanded)
            if d:
                if hasattr(env, 'episode_end'):
                    succ = False
                    if isinstance(info, dict):
                        succ = bool(info.get('is_success', False))
                    env.episode_end(succ)
                o = env.reset()
            obs_list.append(np.asarray(o, dtype=np.float32).reshape(-1))
            rewards[i, 0] = float(r)
            dones[i] = bool(d)
            infos.append(info)

        return np.stack(obs_list, axis=0), rewards, dones, infos

    def collect(self, step):
        self.policy.eval()

        n_rollout_threads = self.args.n_rollout_threads

        with torch.no_grad():
            value, action, action_log_prob, rnn_states, rnn_states_critic = \
                self.policy.get_actions(
                    self.buffer.obs[step].reshape(-1, *self.buffer.obs.shape[2:]),
                    self.buffer.rnn_states[step].reshape(-1, *self.buffer.rnn_states.shape[2:]),
                    self.buffer.rnn_states_critic[step].reshape(-1, *self.buffer.rnn_states_critic.shape[2:]),
                    self.buffer.masks[step].reshape(-1, *self.buffer.masks.shape[2:]),
                    deterministic=False,
                )

        values = value.cpu().numpy().reshape(n_rollout_threads, 1)
        actions = action.cpu().numpy().reshape(n_rollout_threads, -1)
        action_log_probs = action_log_prob.cpu().numpy().reshape(n_rollout_threads, 1)
        rnn_states_out = rnn_states.cpu().numpy().reshape(n_rollout_threads, *rnn_states.shape[1:])
        rnn_states_critic_out = rnn_states_critic.cpu().numpy().reshape(n_rollout_threads, *rnn_states_critic.shape[1:])

        return values, actions, action_log_probs, rnn_states_out, rnn_states_critic_out

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data

        n_rollout_threads = self.args.n_rollout_threads

        if isinstance(dones, bool):
            dones_array = np.array([dones] * n_rollout_threads, dtype=bool)
        elif isinstance(dones, np.ndarray):
            if dones.ndim == 0:
                dones_array = np.array([bool(dones)] * n_rollout_threads, dtype=bool)
            else:
                dones_array = dones
        else:
            dones_array = np.array([bool(dones)] * n_rollout_threads, dtype=bool)

        masks = np.ones((n_rollout_threads, 1), dtype=np.float32)
        masks[dones_array] = 0.0

        rnn_states[dones_array] = np.zeros(
            (dones_array.sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_array] = np.zeros(
            (dones_array.sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32,
        )

        obs_batch = np.asarray(obs, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = np.tile(obs_batch[np.newaxis, :], (n_rollout_threads, 1))

        if isinstance(rewards, (int, float)):
            rewards_batch = np.array([rewards] * n_rollout_threads).reshape(n_rollout_threads, 1)
        else:
            rewards_batch = np.array(rewards).reshape(n_rollout_threads, 1)

        if self.reward_normalizer is not None:
            self.reward_normalizer.update(rewards_batch)
            rewards_batch = self.reward_normalizer.normalize(rewards_batch)

        self.buffer.insert(
            obs_batch,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards_batch,
            masks,
        )

    def compute_returns(self):
        self.policy.eval()

        with torch.no_grad():
            next_values = self.policy.get_values(
                self.buffer.obs[-1].reshape(-1, *self.buffer.obs.shape[2:]),
                self.buffer.rnn_states_critic[-1].reshape(-1, *self.buffer.rnn_states_critic.shape[2:]),
                self.buffer.masks[-1].reshape(-1, *self.buffer.masks.shape[2:]),
            )
            next_values = next_values.cpu().numpy().reshape(self.args.n_rollout_threads, 1)

        self.buffer.compute_returns(next_values, self.value_normalizer)

    def train(self):
        self.policy.train()

        advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
        # Enhanced advantage normalization with clipping
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -10.0, 10.0)

        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        total_grad_norm = 0
        valid_updates = 0

        for _ in range(self.args.ppo_epoch):
            data_generator = self.buffer.feed_forward_generator(
                advantages,
                self.args.num_mini_batch,
            )

            for sample in data_generator:
                obs_batch, rnn_states_batch, \
                    rnn_states_critic_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, \
                    active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch = sample

                obs_batch = np.clip(obs_batch, -100.0, 100.0)
                obs_batch = np.nan_to_num(obs_batch, nan=0.0, posinf=100.0, neginf=-100.0)

                values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    active_masks_batch,
                )

                old_action_log_probs_batch = torch.from_numpy(old_action_log_probs_batch).to(self.device)
                adv_targ = torch.from_numpy(adv_targ).to(self.device)
                value_preds_batch = torch.from_numpy(value_preds_batch).to(self.device)
                return_batch = torch.from_numpy(return_batch).to(self.device)

                if torch.isnan(action_log_probs).any() or torch.isnan(values).any():
                    continue

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio = torch.clamp(ratio, 0.0, 10.0)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                   1 + self.clip_param) * adv_targ

                action_loss = -torch.min(surr1, surr2).mean()

                if self.args.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(
                            -self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Calculate KL divergence if using KL penalty
                if self.use_kl_penalty:
                    # KL divergence = E[log(pi_old/pi_new)] = E[log_prob_old - log_prob_new]
                    # Should always be non-negative; use abs() to ensure numerical stability
                    kl_div = torch.clamp(old_action_log_probs_batch - action_log_probs, min=0.0).mean()
                    self.kl_divergence = kl_div.item()
                    # Add KL penalty to loss
                    kl_loss = kl_div * self.kl_coef
                    # Calculate value prediction error for adaptive GAE
                    if self.use_adaptive_gae:
                        self.value_pred_error = torch.abs(values - return_batch).mean().item()
                    loss = value_loss * self.args.value_loss_coef + \
                           action_loss - \
                           dist_entropy * self.entropy_coef + \
                           kl_loss
                else:
                    # Calculate value prediction error for adaptive GAE
                    if self.use_adaptive_gae:
                        self.value_pred_error = torch.abs(values - return_batch).mean().item()
                    loss = value_loss * self.args.value_loss_coef + \
                           action_loss - \
                           dist_entropy * self.entropy_coef

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                
                # 计算梯度范数（在裁剪之前）
                grad_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.args.max_grad_norm)
                
                if torch.isnan(torch.tensor(grad_norm)):
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += action_loss.item()
                total_entropy += dist_entropy.item()
                total_grad_norm += grad_norm
                valid_updates += 1

        if valid_updates > 0:
            return {
                'total_loss': total_loss / valid_updates,
                'value_loss': total_value_loss / valid_updates,
                'policy_loss': total_policy_loss / valid_updates,
                'entropy': total_entropy / valid_updates,
                'grad_norm': total_grad_norm / valid_updates,
            }
        return {
            'total_loss': 0.0,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'entropy': 0.0,
            'grad_norm': 0.0,
        }

    def save(self):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_num_steps': self.total_num_steps,
            'episode': self.episode,
        }
        torch.save(checkpoint, os.path.join(
            self.save_dir, f'model_{self.total_num_steps}.pt'))

    def run(self):
        n_threads = self.args.n_rollout_threads
        ep_len = self.args.episode_length
        steps_per_rollout = ep_len * n_threads
        macro_episodes = max(1, int(self.args.num_env_steps) // steps_per_rollout)

        for episode in range(macro_episodes):
            if self.args.use_linear_lr_decay:
                self.adjust_lr(episode, macro_episodes)
            self.adjust_entropy_coef(episode, macro_episodes)
            self.adjust_clip_param(episode, macro_episodes)

            self.buffer.obs[0] = self._reset_all_envs().copy()
            self.buffer.step = 0

            rnn_states = np.zeros(
                (n_threads, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
            rnn_states_critic = np.zeros(
                (n_threads, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
            masks = np.ones((n_threads, 1), dtype=np.float32)

            last_dones = np.zeros((n_threads,), dtype=bool)
            last_infos: list = []
            reward_comp_sums = defaultdict(float)
            reward_comp_n = 0

            for step in range(ep_len):
                values, actions, action_log_probs, new_rnn_states, new_rnn_states_critic = \
                    self.collect(step)

                obs, rewards, dones, infos = self._step_all_envs(actions)

                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    rc = info.get('reward_components')
                    if not rc:
                        continue
                    for name, val in rc.items():
                        try:
                            reward_comp_sums[name] += float(val)
                        except (TypeError, ValueError):
                            pass
                    reward_comp_n += 1

                data = (
                    obs, rewards, dones, infos, values, actions,
                    action_log_probs, new_rnn_states, new_rnn_states_critic
                )
                self.insert(data)

                self.total_num_steps += n_threads
                last_dones = dones
                last_infos = infos

            for i, env in enumerate(self.envs):
                if hasattr(env, 'episode_end') and not last_dones[i]:
                    info = last_infos[i] if i < len(last_infos) else {}
                    succ = bool(info.get('is_success', False)) if isinstance(info, dict) else False
                    env.episode_end(succ)

            self.compute_returns()
            train_metrics = self.train()
            self.adjust_kl_coef()
            self.adjust_gae_lambda()
            self.buffer.after_update()

            self.episode = episode

            avg_reward = np.mean(self.buffer.rewards) * self.args.episode_length
            current_lr = self.optimizer.param_groups[0]['lr']

            log_data = {
                'episode': episode,
                'total_steps': self.total_num_steps,
                'avg_reward': avg_reward,
                'train_loss': train_metrics['total_loss'],
                'value_loss': train_metrics['value_loss'],
                'policy_loss': train_metrics['policy_loss'],
                'entropy': train_metrics['entropy'],
                'grad_norm': train_metrics['grad_norm'],
                'lr': current_lr,
                'entropy_coef': self.entropy_coef,
                'clip_param': self.clip_param,
            }

            if self.use_kl_penalty:
                log_data['kl_divergence'] = self.kl_divergence
                log_data['kl_coef'] = self.kl_coef

            if self.use_adaptive_gae:
                log_data['gae_lambda'] = self.gae_lambda
                log_data['value_pred_error'] = self.value_pred_error
            else:
                log_data['gae_lambda'] = self.gae_lambda

            if hasattr(self.env, 'get_curriculum_info'):
                curriculum_info = self.env.get_curriculum_info()
                log_data['stage'] = curriculum_info['current_stage']
                log_data['success_rate'] = f"{curriculum_info['success_rate']:.2%}"

            if reward_comp_n > 0:
                for name, s in reward_comp_sums.items():
                    log_data[f'reward/{name}'] = s / reward_comp_n

            tb_step = (
                self.args.use_tensorboard
                and self.logger.tb_writer is not None
                and episode % self.args.tensorboard_log_interval == 0
            )
            file_step = episode % self.args.log_interval == 0

            if file_step and tb_step:
                self.logger.log(log_data, file_stdout=True, tensorboard=True)
            elif file_step:
                self.logger.log(log_data, file_stdout=True, tensorboard=False)
            elif tb_step:
                self.logger.log(log_data, file_stdout=False, tensorboard=True)

            if episode % self.args.save_interval == 0:
                self.save()

        self.save()
        print("Training completed!")
        self.logger.close()

    def adjust_lr(self, episode, episodes):
        """Adjust learning rate with warmup and decay."""
        total_steps = episode * self.args.episode_length * self.args.n_rollout_threads
        
        # Warmup phase
        if total_steps < self.args.lr_warmup_steps:
            warmup_factor = total_steps / self.args.lr_warmup_steps
            lr = self.args.lr * warmup_factor
        else:
            # Decay phase
            if self.args.use_cosine_lr:
                # Cosine annealing from lr to lr_min
                progress = (total_steps - self.args.lr_warmup_steps) / (self.args.num_env_steps - self.args.lr_warmup_steps)
                lr = self.args.lr_min + 0.5 * (self.args.lr - self.args.lr_min) * (1 + np.cos(np.pi * progress))
            else:
                # Linear decay
                progress = (total_steps - self.args.lr_warmup_steps) / (self.args.num_env_steps - self.args.lr_warmup_steps)
                lr = self.args.lr * max(0.01, 1 - progress)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_entropy_coef(self, episode, episodes):
        if self.args.use_entropy_decay:
            self.entropy_coef = self.initial_entropy_coef * (0.5 * (1 + np.cos(np.pi * episode / episodes))) + \
                               self.final_entropy_coef * (0.5 * (1 - np.cos(np.pi * episode / episodes)))

    def adjust_clip_param(self, episode, episodes):
        """Anneal clip parameter during training for more stable updates."""
        if self.use_clip_annealing:
            progress = episode / episodes
            self.clip_param = self.args.clip_param - \
                             (self.args.clip_param - self.clip_param_final) * progress

    def adjust_kl_coef(self):
        if self.use_kl_penalty and self.kl_divergence > 0:
            if self.kl_divergence > self.target_kl * 1.5:
                # Increase KL penalty if divergence is too high
                self.kl_coef = min(self.kl_coef * self.kl_coef_multiplier, self.kl_coef_max)
            elif self.kl_divergence < self.target_kl * 0.5:
                # Decrease KL penalty if divergence is too low
                self.kl_coef = max(self.kl_coef / self.kl_coef_multiplier, self.kl_coef_min)

    def adjust_gae_lambda(self):
        if self.use_adaptive_gae and self.value_pred_error > 0:
            # Adjust GAE lambda based on value prediction error
            if self.value_pred_error > 5.0:
                # Higher error, use lower lambda to reduce variance
                self.gae_lambda = max(self.gae_lambda * 0.99, self.gae_lambda_min)
            elif self.value_pred_error < 1.0:
                # Lower error, use higher lambda to reduce bias
                self.gae_lambda = min(self.gae_lambda * 1.01, self.gae_lambda_max)
            # Update buffer's gae_lambda
            if hasattr(self.buffer, 'gae_lambda'):
                self.buffer.gae_lambda = self.gae_lambda


def main():
    args = parse_args()
    print(f"Training with args: {args}")

    trainer = PPOTrainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
