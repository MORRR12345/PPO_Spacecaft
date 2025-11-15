# 奖励函数

# pyright: reportPrivateImportUsage=false
import torch
from torch import amp
    
def compute_reward(env):
    """ 计算奖励（原来）"""
    with torch.no_grad(), amp.autocast(device_type='cuda', enabled=True):
        #* 准备数据
        dx = env.pos - env.target_pos

        #* 编队奖励
        error = torch.mean(torch.norm(dx, dim=-1), dim=-1)/env.map_size # [N, n, 3] -> [N, n] -> [N]
        goal_reward = 10.0 * (torch.exp(- error) - 1) - 1.0 * error # [N] -> [N]

        fuel_reward = torch.mean(env.fuel[:,:,0], dim=-1) - 1.0 # [N,n,1] -> [N]

        reward = torch.stack([goal_reward, fuel_reward], dim=-1)  # [N] -> [N, 2]

    return reward