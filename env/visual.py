# 可视化实现

import numpy as np
import matplotlib.pyplot as plt # type: ignore
THRESHOLD = 0.01 # 允许最大估计误差

class Visualizer:
    """三维环境可视化渲染器"""
    def __init__(self, map_size, num_agents, max_time):
        self.map_size = map_size
        self.num_agents = num_agents
        self.max_time = max_time

        #* 1.多个窗口
        self.fig_curves = plt.figure(figsize=(6, 6))
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+100+100")   # type: ignore 左上角

        self.fig_3d = plt.figure(figsize=(6, 6))
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+800+100")   # type: ignore 左上角

        #* 2.多个子图
        self.ax_reward = self.fig_curves.add_subplot(211) # 奖励子图
        self.ax_action = self.fig_curves.add_subplot(212) # 动作子图
        self.fig_curves.subplots_adjust(hspace=0.4)  # 增加垂直间距
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d') # 3D子图

        #* 3.设置坐标轴范围和标签
        self.ax_3d.set_xlim(-map_size, map_size)
        self.ax_3d.set_ylim(-map_size, map_size)
        self.ax_3d.set_zlim(-map_size, map_size)
        self.ax_3d.set_xlabel('X(m)', fontsize=10)
        self.ax_3d.set_ylabel('Y(m)', fontsize=10)
        self.ax_3d.set_zlabel('Z(m)', fontsize=10)
        # self.ax_3d.set_title('Spacecraft Trajectory', fontsize=12)
        self.ax_3d.grid(True)
        self.ax_3d.view_init(elev=20, azim=60) # 设置视角 elev=25, azim=30
        self.ax_3d.tick_params(axis='both', which='major', labelsize=8) # 设置数字大小

        #* 4.初始变量
        # 生成 [0, 255] 范围内的随机 RGB 数组（shape: num_colors × 3）
        rgb_array = np.random.randint(100, 256, size=(num_agents, 3))
        self.history_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_array] # 转换为 HEX 颜色

        plt.ion()  # 启用交互模式
        plt.tight_layout()
        plt.draw()  # 初始绘制
        plt.show()

    def init_scatter(self, agent_pos, target_pos): # 关闭景深：depthshade=False
        Initial_scatter = self.ax_3d.scatter(agent_pos[:,0], agent_pos[:,1], agent_pos[:,2],  
                                             c="#000000", s=12, marker='o', label="Initial Pos") 
        target_scatter = self.ax_3d.scatter(target_pos[:,0], target_pos[:,1], target_pos[:,2],  
                                            s=14, marker='o', edgecolors="#F96464", facecolors='none', linewidth=0.6, label="Target Pos") 
        agent_scatter = self.ax_3d.scatter(agent_pos[:,0], agent_pos[:,1], agent_pos[:,2], 
                                           c="#004CFF", s=4, marker='d', label="Spacecraft")  #  zorder=5,
        self.ax_3d.legend(loc='upper right', fontsize=8)
        return agent_scatter

    #***************************************************************************************#
    def move_scatter(self, scatter, pos):
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])

    def show_history(self, pos1, pos2):
        """绘制历史轨迹"""
        for i in range(pos1.shape[0]):
            self.ax_3d.plot(
                    [pos1[i,0], pos2[i,0]],
                    [pos1[i,1], pos2[i,1]],
                    [pos1[i,2], pos2[i,2]],
                    color=self.history_colors[i], linewidth=1
                )
        return None

    def show_reward(self, reward, time):
        self.ax_reward.set_xlabel('Time(s)', fontsize=10)
        self.ax_reward.set_ylabel('Reward', fontsize=10)
        # self.ax_reward.set_title('Reward Curve', fontsize=12)
        self.ax_reward.grid(True)

        self.ax_reward.plot(time, reward, linewidth=1, label="Reward")
        self.ax_reward.set_xlim(0, time[-1])
        self.ax_reward.legend(loc='lower right', fontsize=8)

    def show_action(self, action):
        self.ax_action.set_xlabel('Time(s)', fontsize=10)
        self.ax_action.set_ylabel('Action(m/s)', fontsize=10)
        # self.ax_action.set_title('Action curve', fontsize=12)
        self.ax_action.grid(True)

        action_type = ['△Vx', '△Vy', '△Vz']
        for i in range(len(action_type)):
            self.ax_action.plot(range(len(action)), action[:,i], linewidth=1, label=action_type[i])
        self.ax_action.set_xlim(0, self.max_time)
        self.ax_action.legend(loc='upper right', fontsize=8)

    def show_collision(self, agent_scatter, warn_mask, safe_mask):
        """显示碰撞状态"""
        colors = np.array(["#004CFF"] * len(warn_mask))  # 默认绿色
        colors[warn_mask] = "#FDFFA2"  # 警告黄色
        colors[safe_mask] = "#FF5252"  # 危险红
        agent_scatter.set_color(colors)
    
    def clear_all(self):
        """清除所有添加的图形元素"""
        for line in self.ax_3d.lines:
            line.remove()   
        for collection in self.ax_3d.collections:
            collection.remove()

    def save(self, path):
        self.fig_3d.savefig(f"{path}/picture/play_3d.pdf", format="pdf", bbox_inches="tight", pad_inches=0.3)
        self.fig_curves.savefig(f"{path}/picture/play_curves.pdf", format="pdf", bbox_inches="tight")