## 经验池

import torch
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################    经验池    ####################################
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 核心缓冲区，自动移除旧经验

    def push_batch(self, obs, action, logs, reward, next_obs):
        """将多个经验存储到缓冲区"""
        experience = (
            obs.clone().detach().to(DEVICE),
            action.clone().detach().to(DEVICE),
            logs.clone().detach().to(DEVICE),
            reward.clone().detach().to(DEVICE),
            next_obs.clone().detach().to(DEVICE)
        )
        self.buffer.append(experience)
    
    def prepare_all_data(self):
        """返回整个缓冲区的数据"""
        return {
            'obs': torch.stack([exp[0] for exp in self.buffer]),
            'actions': torch.stack([exp[1] for exp in self.buffer]),
            'log_probs': torch.stack([exp[2] for exp in self.buffer]),
            'rewards': torch.stack([exp[3] for exp in self.buffer]),
            'next_obs': torch.stack([exp[4] for exp in self.buffer])
        }

    def clear(self):
        """清空所有存储的经验"""
        self.buffer.clear()

