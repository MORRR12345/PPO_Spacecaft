# 🚀PPO_Spacecaft
基于航天器动力学的简单PPO框架
## 📁文件结构
```
├── .vscode/                     ## vscode编译器文件夹
│   └── settings.json             # 编译器配置文件夹
├── agent/                       ## 智能体主体
│   ├── _init_.py                 # 空白
│   ├── agent.py                  # 智能体类
│   ├── actor.py                  # actor
│   ├── critic.py                 # critic
│   └── buffer.py                 # 经验池
├── env/                         ## 环境主体
│   ├── _init_.py                 # 空白
│   ├── environment.py            # 环境类
│   ├── dynamic.py                # 动力学
│   ├── reward.py                 # 奖励函数
│   └── visual.py                 # 可视化
├── other/                       ## 一些工具
│   ├── clear.py                  # 清理model文件夹
│   ├── show.py                   # 再一次绘制训练结果曲线
│   └── test.py                   # 用于你测试一些简单的程序
├── model/                       ## 模型等数据保存
│   └── Train_data_11-15-18-56/   # 保存文件夹, 数字为月-日-时-分
│       ├── actor_pth/            # actor训练好的参数
│       ├── critic_pth/           # critic训练好的参数
│       ├── data/                 # loss等曲线数据
│       └── picture/              # 可视化保存的照片
├── train.py                     ## 训练
├── play.py                      ## 演示
└── tool.py                      ## 工具
```
## 🔨需要的环境
- python3.8以上(推荐3.11)
- pytorch(根据自己显卡下载版本)
- numpy(一般不用安装, 会自带)
- matplotlib
- tqdm
## 🎮使用
- 先运行`python train.py`, 训练网络, 训练好的数据会自动存在`model/`文件夹里
- 然后运行`python play.py`, 查看效果
