from collections import deque
import random
import numpy as np
import torch

class TrajectoryBuffer:
    # 初期化
    def __init__(self, device):
        self.buffer = []
        self.device = device

    # バッファにトラジェクトリを追加
    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def __len__(self):
        return len(self.buffer)

    # 与えられたトラジェクトリの要素を分解して返す
    def get_per_trajectory(self, trajectory):
        state = torch.tensor(
            np.stack([x[0] for x in trajectory]), dtype=torch.float32
        )
        action = torch.tensor(
            np.array([x[1] for x in trajectory]), dtype=torch.float32
        )
        log_prob_old = torch.tensor(
            np.array([x[2] for x in trajectory]), dtype=torch.float32
        )
        reward = torch.tensor(
            np.array([x[3] for x in trajectory]), dtype=torch.float32
        )
        next_state = torch.tensor(
            np.stack([x[4] for x in trajectory]), dtype=torch.float32
        )
        done = torch.tensor(
            np.array([x[5] for x in trajectory]), dtype=torch.int32
        )

        return state, action, log_prob_old, reward, next_state, done
        
    def reset(self):
        self.buffer.clear()


class ReplayMemoryBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device


    def add_replay_memory(
        self, state, action, log_prob_old, reward, next_state, done, advantage
    ):
        data = (state, action, log_prob_old, reward, next_state, done, advantage)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    # 情報の確認必須
    def get_minibatch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.stack([x[0] for x in data]).to(self.device)
        action = torch.cat([x[1].unsqueeze(0) for x in data]).to(self.device)
        log_prob_old = torch.cat([x[2].unsqueeze(0) for x in data]).to(self.device)
        reward = torch.cat([x[3].unsqueeze(0) for x in data]).to(self.device)
        reward.unsqueeze_(1)

        next_state = torch.stack([x[4] for x in data]).to(self.device)
        done = torch.cat([x[5].unsqueeze(0) for x in data]).to(self.device)
        done.unsqueeze_(1)
        advantage = torch.cat([x[6].unsqueeze(0) for x in data]).to(self.device)
        return state, action, log_prob_old, reward, next_state, done, advantage
    def reset(self):
        self.buffer.clear()
