# 航天器动力学部分
# pyright: reportPrivateImportUsage=false
import torch
from torch import amp
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EARTH_MU = torch.tensor(3.986004418e14, device=DEVICE)  # 地球引力常数：m^3/s^2
EARTH_R = torch.tensor(6.371e6, device=DEVICE)  # 地球半径：6371km
EARTH_J2 = torch.tensor(1.08263e-3, device=DEVICE)  # 地球J2摄动常数

RADIUS = torch.tensor(7.4e6, device=DEVICE) # 轨道半径：7400km
OMEGA = torch.sqrt(EARTH_MU / RADIUS**3)  # 轨道速度：rad/s 约为0.001(3141s、52min一个周期)
RADIUS_VEC = torch.tensor([0.0, 0.0, RADIUS], device=DEVICE)

##############################轨道动力学##############################
def update_pos(env, action, space = True):
    """更新位置和速度"""
    with torch.no_grad(), amp.autocast(device_type='cuda', enabled=True):
        acc = env.max_a*clip_tensor(action, 0, 1) # 最大加速度约束

        env.vel += acc*env.dec_step*env.sim_time
        if space:
            for _ in range(env.dec_step): # 10
                # 轨道动力学, 非线性且加入J2扰动
                space_acc = get_space_acc(env.pos, env.vel, J2_noise=True)

                env.vel += space_acc*env.sim_time 
                env.pos += env.vel*env.sim_time
        else:
            env.pos += env.vel*env.dec_step*env.sim_time
            
    return env.pos, env.vel

############################################################

def clip_tensor(tensor, min, max):
    """裁剪张量到指定范围"""
    tensor_norm = torch.norm(tensor, dim=-1).unsqueeze(-1)
    tensor_ = tensor.clone()*(torch.clip(tensor_norm, min, max)/(tensor_norm+1e-8))
    return tensor_

def get_space_acc(pos, vel, J2_noise=True):
    """计算空间环境下的加速度"""

    ri = torch.norm(pos + RADIUS_VEC, dim=-1)  # 计算每个位置向量的模长
    mu_ri3 = EARTH_MU / (ri**3)

    with torch.no_grad(), amp.autocast(device_type='cuda'):
        #* 计算引力加速度
        space_acc = pos.new_empty(pos.shape)
        # ddx = -2*w*dz + w^2*x - (mu/ri^3)*x
        # ddy = -(mu/ri3)*y 
        # ddz = 2*w*dx + w^2*z - (mu/ri^3)*(r0+z) + mu/r0^2
        space_acc[:,:,0] = - 2*OMEGA*vel[:,:,2] + OMEGA**2*pos[:,:,0] - mu_ri3*pos[:,:,0]
        space_acc[:,:,1] = - mu_ri3*pos[:,:,1]
        space_acc[:,:,2] =   2*OMEGA*vel[:,:,0] + OMEGA**2*pos[:,:,2] - mu_ri3*(RADIUS + pos[:,:,2]) + EARTH_MU/(RADIUS**2)

        if J2_noise:
            #* 计算J2摄动加速度
            J2_acc = pos.new_empty(pos.shape) # zeros_like
            J2_Re2 = EARTH_J2 * (EARTH_R**2) # J2*Re^2
            mu_ri5 = EARTH_MU / (ri**5) # mu/ri^5
            yi2_ri2 = (pos[:,:,1]**2) / (ri**2) # y^2/ri^2

            # J2_x = (mu/ri^5)*x * [1.5 * J2*Re^2 * (5*y^2/ri^2-1)]
            # J2_y = (mu/ri^5)*y * [1.5 * J2*Re^2 * (5*y^2/ri^2-3)]
            # J2_z = (mu/ri^5)*(r0+z) * [1.5 * J2*Re^2 * (5*y^2/ri^2-1)] + 1.5 * (mu/ri^4) * J2*Re^2
            J2_acc[:,:,0] = mu_ri5*pos[:,:,0] * (1.5 * J2_Re2 *(5*yi2_ri2 - 1))
            J2_acc[:,:,1] = mu_ri5*pos[:,:,1] * (1.5 * J2_Re2 *(5*yi2_ri2 - 3))
            J2_acc[:,:,2] = mu_ri5*(RADIUS+pos[:,:,2]) * (1.5 * J2_Re2 *(5*yi2_ri2 - 1)) + 1.5 * (EARTH_MU/(ri**4)) * J2_Re2

            space_acc += J2_acc
    
    return space_acc