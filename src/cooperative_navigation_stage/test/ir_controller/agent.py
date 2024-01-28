from model import Actor, Critic
from buffer import TrajectoryBuffer, ReplayMemoryBuffer
from logger import Logger
from datetime import datetime
import torch
import torch.nn as nn
import gazebo_cooperative_navigation_test_ir_env as gazebo_env

import numpy as np
import torch.optim as optim
import random
from torch.multiprocessing import Manager
from torch.optim.lr_scheduler import LambdaLR
import os
import json
import time


class PPOAgent:
    def __init__(self, env_name, n_iteration, n_states, action_bounds, n_actions, actor_lr, critic_lr, gamma, gae_lambda, clip_epsilon, buffer_size, batch_size, epoch, collect_step, entropy_coefficient,device):
        self.env_name = env_name
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir_name = f"./ppo_{self.env_name}/{self.start_time}"
        # ディレクトリを作成
        os.makedirs(self.dir_name, exist_ok=True)

        # モデルを保存するディレクトリを作成
        self.model_path = f"{self.dir_name}/models"
        os.makedirs(self.model_path, exist_ok=True)

        # hyperparameters
        self.n_iteration = n_iteration # イテレーション数
        self.actor_lr = actor_lr # アクターの学習率
        self.critic_lr = critic_lr # クリティックの学習率
        self.gamma = gamma # 割引率
        self.gae_lambda = gae_lambda # GAEのλ
        self.clip_epsilon = clip_epsilon # クリッピングするときのε
        self.buffer_size = buffer_size # バッファサイズ
        self.epoch = epoch # エポック数
        self.batch_size = batch_size # バッチサイズ
        self.collect_step = collect_step # 1回学習を行うあたりに必要なステップ数
        self.n_states = n_states # 状態の次元数
        self.action_bounds = action_bounds # 行動の最大値と最小値
        self.n_actions = n_actions # 行動の次元数
        self.entropy_coefficient = entropy_coefficient # エントロピー項の係数
        self.device = device # CPUかGPUか

        # 標準化のための変数
        # self.state_mean = np.zeros(self.n_states)
        # self.state_std = np.ones(self.n_states)
        # self.state_count = 0


        self.iteration = 0 # 現在のイテレーション数

        self.actor = Actor(n_states=self.n_states,n_actions=self.n_actions).to(self.device)
        self.critic = Critic(n_states=self.n_states).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

        self.trajectory_buffer = TrajectoryBuffer(device=self.device)
        self.replay_memory_buffer = ReplayMemoryBuffer(self.buffer_size, self.batch_size, self.device)

        self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)
        self.critic_scheduler = LambdaLR(self.actor_optimizer, lr_lambda=self.scheduler)

        self.logger = Logger(self.dir_name, self.n_actions)
    def get_action(self, id, state, local_actor, logger_dict, collect_action_flag):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_mean, action_std = local_actor(state)

        # ガウス分布を作成
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        logger_entropy = action_distribution.entropy().cpu().numpy()

        # Logger
        logger_dict["entropies"].append(action_distribution.entropy().cpu().numpy().mean())
       
        # ガウス分布から行動をサンプリング
        action = action_distribution.sample()
        
        # ガウス分布でサンプリングときの確率密度の対数を計算
        log_prob_old = action_distribution.log_prob(action)


        # LOGGER 環境id=0のロボットのみ
        if id == 0 and  collect_action_flag is True:
            logger_dict["action_means"].append(action_mean.cpu().numpy()[0])
            logger_dict["action_stds"].append(action_std.cpu().numpy()[0])
            logger_dict["actions"].append(action.cpu().numpy()[0])

        action_2d = np.atleast_2d(action.cpu().numpy()) # 2次元配列に変換
        log_prob_old_2d = np.atleast_2d(log_prob_old.cpu().numpy()) # 2次元配列に変換
        return action_2d, log_prob_old_2d, logger_dict
    
    def data_collection(self, id ,seed, share_memory_actor):
        # print(f"Process {id} : Started collecting data")
        # print("self.state_mean", self.state_mean)
        # print("self.state_std", self.state_std)

        # 子プロセスで再度シード値を設定
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 環境のインスタンスを作成
        env = gazebo_env.GazeboEnvironment(id=id)

        # データ収集のステップ数を初期化
        collect_step_count = 0

        # 一連のエピソードデータを格納するリスト
        trajectory = []

        # logger dictionary
        logger_dict = {
            "total_rewards": [],
            "total_baseline_rewards": [],
            "entropies": [],
            "action_means": [],
            "action_stds": [],
            "actions": [],
            "angle_to_goals": [],
            "pheromone_mean": [],
            "pheromone": [],
            "pheromone_left": [],
            "pheromone_right": [],
            "ir_left": [],
            "ir_right": [],
            "step_count": []
        }

        collect_action_flag = True
        try:
            while True:
                state = env.reset(seed=random.randint(0,100000))
                if id == 0:
                    print("initial state", state)
                
                # 1エピソードごとに格納するリスト
                episode_data = [[] for _ in range(env.robot_num)]
                done = [False] * env.robot_num
                total_reward = [0] * env.robot_num
                total_baseline_reward = [0] * env.robot_num
                total_steps = [0] * env.robot_num

                while not all(done): # すべてのエージェントが終了するまで
                    # NNから行動を獲得
                    action, log_prob_old, logger_dict = self.get_action(id, state, share_memory_actor, logger_dict, collect_action_flag)

                    # 環境に行動を入力し、次の状態、報酬、終了判定などを取得
                    next_state, reward, terminated, baseline_reward, info = env.step(action)
                    # ロボットごとに処理
                    for i in range(env.robot_num):
                        if done[i] is True:
                            # dummy state
                            state[i] = [0] * self.n_states
                            continue

                        total_steps[i] += 1 # 1エピソードのステップ数をカウント
                        total_reward[i] += reward[i] # 1エピソードの報酬をカウント
                        total_baseline_reward[i] += baseline_reward[i] # 1エピソードのベースライン報酬をカウント

                        # 1ステップのデータをエピソードデータに格納
                        episode_data[i].append((state[i], action[i], log_prob_old[i], reward[i], next_state[i], terminated[i]))

                        # データ収集のステップ数を+1 
                        # 複数のロボット共通でカウント
                        collect_step_count += 1

                        # 状態を更新
                        state[i] = next_state[i]


                        # LOGGER アクションのログは、環境id=0のロボットのみ
                        if id == 0 and i == 0 and collect_action_flag:
                            # TODO actionのログ
                            logger_dict["angle_to_goals"].append(info[0]["angle_to_goal"])
                            logger_dict["pheromone_mean"].append(info[0]["pheromone_mean"])
                            logger_dict["pheromone"].append(info[0]["pheromone_value"])
                            logger_dict["pheromone_left"].append(info[0]["pheromone_left_value"])
                            logger_dict["pheromone_right"].append(info[0]["pheromone_right_value"])
                            logger_dict["ir_left"].append(info[0]["ir_left_value"])
                            logger_dict["ir_right"].append(info[0]["ir_right_value"])
                            

                        
                        # ループの終了判定. すべてのエージェントが終了したら終了
                        if terminated[i] is True:
                            done[i] = True 
                            if i == 0:
                                collect_action_flag = False                       
                        
                            print(f"HERO_{id * env.robot_num + i} Reward : {total_reward[i]}, Step : {total_steps[i]}")
                            if total_steps[i] > 1:
                                trajectory.append(episode_data[i])
                                logger_dict["total_rewards"].append(total_reward[i])
                                logger_dict["total_baseline_rewards"].append(total_baseline_reward[i])
                                logger_dict["step_count"].append((info[i]["done_category"],total_steps[i]))
                
                # データ収集のステップ数が一定数を超えたら終了
                if collect_step_count >= self.collect_step:
                    break

            env.shutdown()
            print(f"Process {id} is Finished")
            return (trajectory, logger_dict)
        except Exception as e:
            print(f"ERROR : Process {id} is Finished")
            print(e)
            env.shutdown()
            return (None, None)
        
    def compute_advantages_and_add_to_buffer(self):
        for trajectory in self.trajectory_buffer.buffer:
            local_critic = self.create_critics_copy()
            states, actions, log_prob_olds, rewards, next_states, dones = self.trajectory_buffer.get_per_trajectory(trajectory)
            with torch.no_grad():
                # print("state_values", states)
                state_values = local_critic(states).squeeze()
                # print("state_values", state_values)

                # print("next_states", next_states)
                next_state_values = local_critic(next_states).squeeze()
                # print("next_state_values", next_state_values)

            
            advantages = []
            advantage = 0
            for t in reversed(range(0, len(rewards))):
                # エピソードの終了した場合の価値は0にする
                non_terminal = 1 - dones[t].item()

                # TD誤差を計算
                # print("non_terminal", non_terminal)
                # print("reward", rewards[t])
                # print("next_state_values", next_state_values[t])
                # print("state_values", state_values[t])
                delta = rewards[t] + self.gamma * next_state_values[t] * non_terminal - state_values[t]
                

                # Advantage関数を計算
                advantage = (delta + self.gamma * self.gae_lambda * non_terminal * advantage).detach()
                advantages.append(advantage)
            advantages = advantages[::-1]

            for idx, state in enumerate(states):
                self.replay_memory_buffer.add_replay_memory(state, actions[idx], log_prob_olds[idx],rewards[idx], next_states[idx], dones[idx], advantages[idx])
                self.logger.advantage_history.append(advantages[idx].item())

    def compute_ppo_loss(self, state, action, log_prob_old, advantage):
        advantage.unsqueeze_(-1)
        action_mean, action_std = self.actor(state)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        log_prob_new = action_distribution.log_prob(action)
        ratio = torch.exp(log_prob_new - log_prob_old)
        unprocessed_loss = ratio * advantage
        clipped_loss = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        loss =  -torch.min(unprocessed_loss, clipped_loss).mean() - self.entropy_coefficient * action_distribution.entropy().mean()
        return loss

    def compute_value_loss(self, state, reward, next_state, done):
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)

        target = reward + (1 - done) * self.gamma * next_state_value
        loss_fn = nn.MSELoss()
        loss = loss_fn(state_value, target)
        return loss
    
    def update_state_normalization(self, episode_data):
        """
        収集したエピソードデータから各状態の平均と標準偏差を計算し更新する。

        Args:
            episodes (list): 収集したエピソードデータのリスト。
        """
        # 状態の統計量を更新
        for episode in episode_data:
            for step in episode:
                state, _, _, _, _, _ = step
                self.update_state_normalization_online(state)

        self.finalize_state_normalization()

    def update_state_normalization_online(self, new_state):
        self.state_count += 1
        delta = new_state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = new_state - self.state_mean
        self.state_std += delta * delta2

    def finalize_state_normalization(self):
        if self.state_count > 1:
            self.state_std = np.sqrt(self.state_std / (self.state_count - 1))
        else:
            self.state_std = np.ones(self.n_states)
    
    def optimize(self, actor_loss, critic_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def schedule_lr(self):
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.logger.lr_actor_history.append(self.actor_scheduler.get_last_lr()[0])
        self.logger.lr_critic_history.append(self.critic_scheduler.get_last_lr()[0])
    def scheduler(self, step):
        return max(1.0 - float(step / self.n_iteration), 0)
    def save_weights(self, iteration):
        filepath = os.path.join(self.model_path, f'{iteration}_weights.pth')

        torch.save({"current_policy_state_dict": self.actor.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                    "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
                    "iteration": iteration,
                    # "state_rms_mean": state_rms.mean,
                    # "state_rms_var": state_rms.var,
                    # "state_rms_count": state_rms.count,
                    },filepath)
        

    
    def load_weights(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint["current_policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler_state_dict"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        self.iteration = checkpoint["iteration"]
        # state_rms_mean = checkpoint["state_rms_mean"]
        # state_rms_var = checkpoint["state_rms_var"]
        # return iteration

    def set_to_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def set_to_train_mode(self):
        self.actor.train()
        self.critic.train()

    def create_actor_copy(self):
        # 新しい Actor インスタンスを作成し、現在のモデルの状態をコピー
        # actor_copy = Actor(n_states=self.n_states, n_actions=self.n_actions)
        actor_copy = Actor(n_states=self.n_states, n_actions=self.n_actions).to("cpu")
        actor_copy.load_state_dict(self.actor.state_dict())
        return actor_copy
    
    def create_critics_copy(self):
        # 新しい Critic インスタンスを作成し、現在のモデルの状態をコピー
        critic_copy = Critic(n_states=self.n_states)
        critic_copy.load_state_dict(self.critic.state_dict())
        return critic_copy
    

    def save_setting_config(self):
    # ハイパーパラメータの辞書を作成
    
        hyperparameters = {
            'env_name': self.env_name,
            'n_iteration': self.n_iteration,
            'n_states': self.n_states,
            'action_bounds': self.action_bounds,
            'n_actions': self.n_actions,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'collect_step': self.collect_step,
            'entropy_coefficient': self.entropy_coefficient,
            'device': str(self.device),
            'actor_network_architecture': self.actor.get_architecture(),
            'critic_network_architecture': self.critic.get_architecture(),
            # ここに他の必要なパラメータを追加
        }

        # 環境に特有のパラメータ（環境名やバージョンなど）
        # 学習プロセスの設定
        env_params = {
            'robot_action_parameters': {
                'max_linear_velocity': 0.2,  # 仮の値
                'max_angular_velocity': 1.0  # 仮の値
            },
            'goal_position': 'random',  # ゴール位置はランダムに設定
            'reward_calculation': {
                'collision_penalty': -30.0,
                'goal_reward': 30.0,
                'distance_reward_weight': {
                    'positive': 4.0,
                    'negative': 6.0
                },
                'angular_velocity_penalty': -1.0,
                'time_penalty': -1.0
            },
            'timeout': 40.0  # エピソードのタイムアウト時間
        }

        # 乱数生成のシード値
        seed_params = {
            'experiment_seed': 1023,  # あるいは他の適切な変数

        }

        # 使用したデバイスの情報
        device_params = {
            'device': str(self.device),
            # GPUモデルやメモリサイズなど、必要に応じて追加
        }

        # 最適化アルゴリズムの情報
        optimization_params = {
            'optimizer': 'Adam',
            'actor_optimizer_params': self.actor_optimizer.state_dict(),
            'critic_optimizer_params': self.critic_optimizer.state_dict()
        }
        # 実験の日時
        experiment_time_params = {
            'start_time': self.start_time,
        }

        # すべての情報を1つの辞書に統合
        all_params = {
            'hyperparameters': hyperparameters,
            'environment_params': env_params,
            'seed_params': seed_params,
            'device_params': device_params,
            'optimization_params': optimization_params,
            'experiment_time_params': experiment_time_params
        }
        # JSON ファイルに保存
        filepath = os.path.join(self.dir_name, 'setting.json')
        with open(filepath, 'w') as json_file:
            json.dump(all_params, json_file, indent=4)

        print(f"Experiment settings saved to {filepath}")
