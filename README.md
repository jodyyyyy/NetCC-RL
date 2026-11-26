# NetCC-RL: Reinforcement Learning for Continuous Control of Complex Networks

面向复杂网络连续控制的强化学习研究代码。当前用于课题组内部研究、算法复现和后续开发；论文及完整实验数据尚未发布。请保持仓库私有，未经权利人确认不要公开或转发第三方源码及拓扑数据。

## 快速导航

- [中文代码与文件指南](docs/code-guide.md)：每类文件的用途、修改算法和实验的入口。
- [详细复现协议（英文）](docs/reproduction.md)：训练预算、算法更新频率、评估和 checkpoint 兼容。
- [实验结果说明](results/README.md)：如何保存、分析和补充训练数据。
- [第三方来源及版权](THIRD_PARTY_NOTICES.md)：必须保留的来源与许可说明。

## 1. 安装（Linux，Python 3.10）

推荐独立环境。已验证的训练版本为 PyTorch 2.4.0；以下是 CPU 安装例子。GPU 用户先安装与驱动兼容的 PyTorch 2.4 CUDA wheel，再执行最后一行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch==2.4.0+cpu'
python -m pip install -c constraints-training.txt '.[baselines,analysis,test]'
```

`baselines` 安装 SB3（PPO/SAC/TD3）；`analysis` 安装绘图和 TensorBoard；`test` 安装 pytest。其余算法的必要依赖在基础包中。以下命令从仓库根目录执行。

## 2. 第一次运行：先短测，再正式训练

```bash
python -m netcc_rl.catalog --validate
python -m netcc_rl.train --config configs/ablation/sis/E0.yaml --smoke --output runs/sis-e0-smoke
python -m netcc_rl.evaluate --config configs/ablation/sis/E0.yaml --checkpoint runs/sis-e0-smoke/final.pt --episodes 1
```

对比算法示例：

```bash
python -m netcc_rl.train --config configs/comparison/sis-ppo.yaml --smoke --output runs/sis-ppo-smoke
python -m netcc_rl.evaluate --config configs/comparison/sis-ppo.yaml --checkpoint runs/sis-ppo-smoke/checkpoint.json --episodes 1
```

正式训练示例（需要可用 GPU，1M 真实环境步，不是短测）：

```bash
python -m netcc_rl.train --config configs/ablation/sis/E8.yaml --device cuda:0 --seed 0 --output runs/sis-E8-seed0
```

每次训练使用新输出目录。消融正式配置默认启用 compile + CUDA Graph；要显式使用 eager，指定 `--eager --device cpu` 或 `--eager --device cuda:0`。短测自动使用缩小预算，不能当正式结果。对比算法使用自己的更新机制；统一入口不等于统一 learner。详细参数见 `--help`。

## 3. 实验内容

| 环境 | 节点 | 固定执行器 |
| --- | ---: | ---: |
| Kuramoto | 100 | 50 |
| SIS | 100 | 50 |
| Swing (IEEE39-derived core38) | 38 | 19 |

SIS 共同协议：动作 [-1,1]，干预强度 (a+1)/2，控制成本系数 0.1。历史 CMAT SIS 使用 [-1,0.2] 和零成本，单独保留在 `configs/historical/`，不能与共同协议直接比较。

36 组消融由以下 12 个 arm × 三环境组成：

| Arm | Encoder | Dynamics | Actor |
| --- | --- | --- | --- |
| E0 | GATv2 | D1 | Shared |
| E1 | Node-wise MLP | D1 | Shared |
| E2 | GATv2 | D0 | Shared |
| E3 | GATv2 | D1 | Global |
| E4 | Node-wise MLP | D0 | Shared |
| E5 | Node-wise MLP | D1 | Global |
| E6 | GATv2 | D0 | Global |
| E7 | Node-wise MLP | D0 | Global |
| E8 | Global MLP-512 | D1 | Shared |
| E9 | Global MLP-512 | D0 | Shared |
| E10 | Global MLP-512 | D1 | Global |
| E11 | Global MLP-512 | D0 | Global |

GATv2 使用 Mean+Max readout；Node-wise MLP 匹配节点编码器参数量，不使用图邻接；Global MLP 直接展平原生特征，隐藏层固定512，输出64（不是严格参数量匹配）。D1 为 H=3 模型预测路径，D0 为真实下一观测的一步目标；Shared Actor 使用 [局部原生状态,z]，Global Actor 仅使用 z。E0 是完整算法，不重复计数。

24 组对比：PPO、SAC、TD3、MAPPO、HAPPO、MAT、JointPPO、CMAT × 三环境。DMPO 不在范围内。正式预算为每组 seed 0、1M 真实环境步。

## 4. 开发与测试

先看 `configs/`、`netcc_rl/train.py` 和 [文件指南](docs/code-guide.md)，不需要一开始阅读所有底层代码。改算法后至少执行：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q
```

GPU 加速及跨运行时恢复的完整验收尚未完成。正式配置和已运行实验身份保持冻结；新的研究变体建议新增配置/分支，不直接覆盖历史协议。

## 5. 数据与引用

本仓库暂不包含训练数据、模型权重或未发表论文。后续按 [results/README.md](results/README.md) 记录配置、源码版本、seed、协议和数据哈希。单训练 seed 的实验只支持描述性比较。以下为现有算法参考，不代表 NetCC-RL 论文已发表。

- Hansen, Su and Wang. [TD-MPC2](https://arxiv.org/abs/2310.16828).
- Brody, Alon and Yahav. [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).
- Schulman et al. [PPO](https://arxiv.org/abs/1707.06347). Haarnoja et al. [SAC](https://arxiv.org/abs/1801.01290). Fujimoto, van Hoof and Meger. [TD3](https://arxiv.org/abs/1802.09477).
- Yu et al. [MAPPO](https://arxiv.org/abs/2103.01955). Kuba et al. [HAPPO](https://arxiv.org/abs/2109.11251). Wen et al. [MAT](https://arxiv.org/abs/2205.14953). Zhao et al. [CMAT](https://arxiv.org/abs/2604.13472).
- The upstream code and data attributions are in [Third-party notices](THIRD_PARTY_NOTICES.md).
