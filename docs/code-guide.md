# NetCC-RL 中文代码与文件指南

本指南供后续研究和课题组成员接手使用。建议先运行 README 的短测，再按任务阅读源码。不要为了目录简洁删除冻结配置、拓扑、依赖或兼容代码。

## 1. 根目录：日常从哪里开始

| 路径 | 用途 | 是否需要常看 |
| --- | --- | --- |
| README.md | 安装、最小运行例子、实验范围 | 是 |
| configs/ablation/ | 三环境各 E0–E11；每个 YAML 是 arm 选择器 | 是 |
| configs/comparison/ | 八算法 × 三环境完整比较 profile | 是 |
| configs/historical/sis-cmat.yaml | 原 CMAT SIS 的非共同协议，供旧结果复现 | 仅旧结果 |
| netcc_rl/ | Python 算法与环境实现 | 改算法时 |
| scripts/ | 统计、TensorBoard、旧数据与权重转换 | 按需 |
| tests/ | 正确性回归测试 | 改代码后 |
| docs/ | 中文导航及详细复现协议 | 按需 |
| results/README.md | 未来结果归档格式，当前不是实际结果 | 保存数据时 |
| pyproject.toml | pip 安装、依赖、包资源和版本定义 | 通常不改 |
| constraints.txt | 复现依赖版本约束 | 安装时 |
| MANIFEST.in | 源码发行包应包含的配置、文档和脚本 | 打包时 |
| .gitignore | Git 忽略规则 | 通常不改 |
| LICENSE / THIRD_PARTY_NOTICES.md | 内部使用状态及第三方版权 | 保留 |

暂未启用的 GitHub Actions 模板和缺少作者的 CITATION.cff 已从工作树移除；旧版本仍可从 Git 历史找回。现在使用本地 pytest；论文作者及正式许可确定后再补完整引用文件。不要删除第三方许可。

## 2. 公共入口与配置

| 文件 | 负责内容 |
| --- | --- |
| netcc_rl/train.py | 训练 CLI、设备/seed、短测与算法分派 |
| netcc_rl/evaluate.py | checkpoint 加载、评估 episode/seed、JSON 输出 |
| netcc_rl/cli.py | 共用参数解析、对比配置校验、后端分派 |
| netcc_rl/catalog.py | 列举与校验60个正式配置身份，不启动训练 |
| netcc_rl/matrix.py | 消融配置展开、冻结参数检查及参数量统计 |
| netcc_rl/ablation.yaml | E0–E11 的 G/D/A 因子定义，Global MLP 宽度512 |
| netcc_rl/comparison_output.py | 对比算法统一日志、结果格式及 checkpoint 索引 |
| netcc_rl/__init__.py | Python 包初始化 |

消融 YAML 很短是正常的：实际公共超参数来自 proposed/config.yaml、proposed/configs/defaults/ 和 ablation.yaml，由 matrix.py 展开。不要把所有字段重复拷入36个 YAML；冻结身份检查可能拒绝改动。新的非复现实验应独立增加配置和身份。

## 3. 主算法：netcc_rl/algorithms/proposed/

| 文件/模块 | 负责内容 |
| --- | --- |
| agent.py（`NetCCAgent`） | 动作接口、学习编排及 checkpoint 读写 |
| train.py / evaluate.py / runner.py | 主算法入口、采集与评估循环 |
| config.py / config.yaml | 公共设置与身份校验 |
| configs/defaults/ | 三个环境的冻结基础配置 |
| models/architecture.py | 解析宽度、节点、拓扑与模型身份 |
| models/network_world_model.py / world_model.py | 世界模型与各网络组合 |
| models/graph_networks.py / gatv2_encoder.py / node_features.py | PyG 编码、图结构和原生节点特征 |
| models/layers.py / ablation/variants.py | 网络层及 Node-wise MLP、Global MLP、Global Actor |
| ablation/model_free.py | D0 的真实下一观测 Bellman 目标 |
| learning/learner_update.py / learner_runtime.py / compile.py | 更新顺序、CUDA Graph 和 TorchInductor |
| replay/ | 单步 Replay、采样、批量采集 |
| utils/ | Gaussian 与数值运算、初始化、预算、日志与计时 |
| LICENSE.TDMPC2 | 上游许可 |

修改 Encoder 时，从 `models/graph_networks.py` 和 `ablation/variants.py` 追踪入口。共享 Actor 在 `models/graph_networks.py`，全局 Actor 在 `ablation/variants.py`。D1 loss 和目标网络更新从 `agent.py` 与 `learning/learner_update.py` 开始，D0 从 `ablation/model_free.py` 开始。

## 4. 对比算法：netcc_rl/algorithms/baselines/

| 算法 | 主要入口 |
| --- | --- |
| PPO / SAC / TD3 | sb3_agent.py、sb3_config.py（使用 Stable-Baselines3） |
| MAPPO | mappo.py、mappo_batched.py、vendor/onpolicy/ |
| HAPPO | happo.py、baseline_networks.py、baseline_config.py |
| MAT / JointPPO | joint_policy.py、joint_train.py |
| CMAT | cmat/run.py、cmat/joint_policy.py、cmat/joint_train.py、cmat/gpu_batch.py |

七个非 CMAT 算法由 baselines/run.py 组织训练；CMAT 使用独立 runner。共享目录不是重复算法：environment.py/environment_profile.py 负责环境绑定，phase_features.py 处理相位特征，neighborhoods.py 处理局部观测范围，trajectories.py、batched.py、runtime.py、profiling.py 等支持数据、执行和诊断；registry.py 定义该 runner 支持的算法；configs/*.json 保存既有正式 profile；vendor/ 和 licenses/ 保存实际使用的上游代码、来源与许可。

CMAT 子目录中的 task_adapter.py、task_joint.py、phase_features.py 等是其独立适配链，不能只因名称和外层相似就删除或合并。各算法的优化更新频率和信息范围不相同。

## 5. 环境：netcc_rl/environments/

- __init__.py：环境创建入口。
- mpcsc/：Kuramoto 与 SIS 的配置、动力学、环境封装、固定执行器和 GPU 批量模拟。adapter.py 提供 NumpyEnvAdapter、TensorEnvAdapter。
- mpcsc/dynamics/kuramoto.py、sis.py：分别是两种物理动力学；random_streams.py 为随机流支持。
- graph/：Swing 及图拓扑、冻结环境读取；data/ 中有三环境的冻结快照、CMAT 来源数据和 core38 拓扑。
- 哈希命名的 JSON 与 index.json 是复现实验资产，不是缓存或垃圾文件；修改可能改变 checkpoint 身份。
- LICENSE.MPCSC 等版权文件必须保留。

## 6. 分析脚本

| 脚本 | 用途 |
| --- | --- |
| analyze_ablation.py | 消融评估曲线、AUC、Late 窗口和因子统计 |
| analyze_evaluation.py | 独立100回合评估统计 |
| tensorboard_export.py | 从 metrics.jsonl 导出 TensorBoard 事件 |
| migrate_comparison_checkpoint.py | 将旧对比 checkpoint 迁移到新索引格式 |
| convert_run.py | 将旧 CMAT 运行输出转成统一结果格式 |

## 7. 历史配置与 checkpoint

`compat/ablation.py` 负责 v4 权重和旧配置。新权重写 v5，模型参数键和张量形状沿用原版。旧配置的环境/arm、因子、基础配置摘要和架构摘要按明确映射读取；在加载模型前校验摘要、环境快照、模型尺寸和参数键。v4 和 v5 加载均仅恢复模型权重。

历史 PPO/SAC/TD3 使用原运行目录的 `config.json` 校验环境、执行器、配置和预算。兼容层仅替换归档中已知的旧 PhaseExtractor 路径，保留归档里的策略参数。MAPPO 的正式权重可能记录最后一个 100 步的部分回合，兼容层只在预算确实推出该长度时识别它。CMAT 映射旧适配器版本字段并继续检查完整适配器身份。旧 SIS 协议迁移使用 `configs/historical/sis-cmat.yaml`。

```bash
python -m netcc_rl.evaluate --config configs/ablation/sis/E0.yaml \
  --checkpoint PATH_TO_FINAL_PT --episodes 1
python scripts/migrate_comparison_checkpoint.py --config configs/comparison/sis-ppo.yaml \
  --checkpoint PATH_TO_FINAL_ZIP --output RUNS_MIGRATED_SIS_PPO
python -m netcc_rl.evaluate --config configs/comparison/sis-ppo.yaml \
  --checkpoint RUNS_MIGRATED_SIS_PPO/checkpoint.json --episodes 1
```

保留源文件，选择新的输出目录，只加载来源可信的权重。跨版本精确续训未验证。

## 8. 测试与后续研究

test_ablation.py 检查消融配置、模型及环境/学习合同；test_comparison.py 检查对比接口；test_acceleration.py 检查加速路径。日常先跑 CPU pytest，再根据设备情况运行 GPU 专项。尚未完成的 GPU 验收不应被描述为已通过。

结果默认写入 runs/，不要把 checkpoint、大日志和 TensorBoard 事件直接提交到 Git。使用新目录、记录 commit、config、seed 和环境协议。当前论文未发表；仓库供内部协作，不能视作已取得第三方公开再分发权或已采用开源许可证。
