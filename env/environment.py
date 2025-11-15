# 主体环境

import torch

from env.dynamic import update_pos
from env.reward import compute_reward

MAP_SIZE = 1000 # 1km
POS_SCALE = 0.01
VEL_SCALE = 1

TOTAL_TIME = 1000  # 1800s = 30min 仿真时间
SIM_TIME = 1  # 2s 仿真时间间隔
DEC_STEP = 10  # 决策间隔步数 #* N=1时为连续推力

MAX_A = 0.1  # 最大加速度m/s^2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpaceEnv:
    def __init__(self, num_env, num_agent):
        self.num_env = num_env
        self.num_agent = num_agent

        self.obs_dim = 10 # 位置3、速度3、燃料1、目标位置3
        self.action_dim = 3

        self.total_time = TOTAL_TIME
        self.sim_time = SIM_TIME
        self.dec_step = DEC_STEP
        self.max_step = int(self.total_time/(self.dec_step*SIM_TIME))

        self.map_size = MAP_SIZE
        self.max_a = MAX_A

        self.pos = torch.zeros(num_env, num_agent, 3, device=DEVICE, dtype=torch.float32)
        self.vel = torch.zeros(num_env, num_agent, 3, device=DEVICE, dtype=torch.float32)
        self.fuel = torch.ones(num_env, num_agent, 1, device=DEVICE, dtype=torch.float32)
        self.target_pos = torch.zeros(num_env, num_agent, 3, device=DEVICE, dtype=torch.float32)

    def get_dim(self):
        return self.obs_dim, self.action_dim, self.obs_dim*self.num_agent, self.num_agent

    def get_obss(self):
        return torch.cat((self.pos*POS_SCALE, self.vel*VEL_SCALE, self.fuel, self.target_pos*POS_SCALE), dim=-1)
    
    def reset(self):
        """ 重置环境 """
        self.pos = (2 * torch.rand(self.num_env, self.num_agent, 3, device=DEVICE) - 1) * self.map_size
        self.vel.zero_()
        self.fuel[:] = 1.0
        self.target_pos = (2 * torch.rand(self.num_env, self.num_agent, 3, device=DEVICE) - 1) * self.map_size

        self.time = 0

        return self.get_obss(), False


    def step(self, action):
        """ 执行一步 """
        #* 1.消耗燃料
        self.fuel -= torch.norm(action, dim=-1, keepdim=True)*self.dec_step*self.sim_time/self.total_time
        #* 2.执行动作
        self.pos, self.vel = update_pos(self, action, space=False)
        #* 3.计算奖励
        reward = compute_reward(self)
        #* 4.更新步数和计算done
        self.time += self.dec_step*self.sim_time
        done = self.time > self.total_time
        return self.get_obss(), reward, done