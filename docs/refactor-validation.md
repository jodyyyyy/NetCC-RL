# 目录重构与回归说明

基准是 NetCC-RL `d45e63526e03b96b721dc15d51e148af7ca80d2b`。本次接续 `3650c4f`，代码重构提交为 `24b798a`。

## 命名迁移

| 旧名称 | 新名称 |
| --- | --- |
| `proposed/tdmpc2.py` · `TDMPC2` | `proposed/agent.py` · `NetCCAgent` |
| `environments/mpcsc/tdmpc2.py` | `environments/mpcsc/adapter.py` |
| `TDMPC2NumpyAdapter` · `TDMPC2TensorAdapter` | `NumpyEnvAdapter` · `TensorEnvAdapter` |
| `proposed/configs/exp016/` | `proposed/configs/defaults/` |
| `common/adaptive_architecture.py` | `models/architecture.py` |
| `common/adaptive_world_model.py` · `AdaptiveWorldModel` | `models/network_world_model.py` · `NetworkWorldModel` |
| `common/graph_world_model.py` | `models/graph_networks.py` |
| `common/exp017.py` | `ablation/variants.py`、`ablation/model_free.py`、`learning/compile.py` |
| `baselines/fidelity_config.py` · `fidelity_networks.py` | `baseline_config.py` · `baseline_networks.py` |
| `baselines/sb3.py` · `sb3_settings.py` | `sb3_agent.py` · `sb3_config.py` |
| `baselines/roster.py` | `registry.py` |
| `baselines/onpolicy.py` · `onpolicy_fast.py` | `mappo.py` · `mappo_batched.py` |
| `constraints-training.txt` | `constraints.txt` |
| `analyze_exp017.py` · `normalize_comparison_run.py` · `summarize_independent_eval.py` | `analyze_ablation.py` · `convert_run.py` · `analyze_evaluation.py` |
| `tests/test_exp017.py` | `tests/test_ablation.py` |
| `exp017_{arm,environment,factors}` | `ablation_{arm,environment,factors}` |
| `exp017_base_sha256` · `exp017_identity_sha256` | `base_config_sha256` · `ablation_identity_sha256` |
| `exp016_arm: G2` · `exp016_adaptive_v1` | `graph_readout_variant: mean_max` · `adaptive_width_v1` |
| `exp016_three_environments` · `exp017_factorial_gaussian_v1` | `frozen_vector_v1` · `ablation_gaussian_v2` |
| `exp015_task_adapter_v1` | `continuous_task_adapter_v1` |

`common/` 内其余组件按职责移入 `models/`、`learning/`、`replay/`、`utils/`。旧序列化名称保留在兼容层与回归测试，历史来源路径保留在冻结资产索引，上游 TD-MPC2 和 MAPPO 专名及许可保留在原文件。以下命令列出任意文件的改名前后路径。

```bash
git diff -M --name-status d45e6352 24b798a
```

## 移除与兼容范围

删除 Turbo-GNN 实现、校验清单、构建脚本、SM75 补丁、专用 Git 属性和配置字段。三个冻结拓扑资产移除 CSR、度数分桶和相关元数据，保留物理快照、候选 COO、batch shape、物理边和执行器顺序。显式请求旧后端会产生错误。三套历史向量采集默认值改用同一 PyG 路径，正式 36 组原本已使用 PyG。

新 proposed checkpoint 写入 v5。原 v4 权重张量键、形状及张量含义保持一致。探索性旧 Global MLP 宽度与正式 512 不兼容。SB3 读取器映射两个旧 PhaseExtractor 模块路径，校验原策略参数；迁移 FlowActor 历史对比权重需要原运行的 `config.json`。CMAT 历史标签迁移保留完整任务、动作和拓扑身份。跨版本精确续训未验证。

## 执行方式

```bash
python -m netcc_rl.catalog --validate
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -q
python -m netcc_rl.train --config configs/ablation/sis/E9.yaml --smoke --output PATH_TO_SMOKE
python -m netcc_rl.evaluate --config configs/ablation/sis/E9.yaml \
  --checkpoint PATH_TO_SMOKE/final.pt --episodes 1
python -m build
```

在两个独立进程运行基准与候选，固定 Python 哈希种子、PyTorch 随机种子、环境初态及 Replay 起点。Proposed 对照使用各环境前 150 步轨迹、固定策略噪声和两次真实 Replay 更新，并重新加载各自保存的权重。基线每种算法选一个环境，保留可触发更新所需的预算，比较初始权重、轨迹、动作、损失、有限梯度与 Adam 更新。模型参数摘要包括目标网络。

## 当前结果

- 36 组消融的正式配置与 CPU 解析结果逐字段核对，24 组对比配置及历史 CMAT SIS 也通过核对。差异限于命名、已移除后端选项与版本化摘要。核对了三个冻结拓扑中保存的物理、COO 和批量形状数据。
- 36 组消融的短程对照全部通过。两次更新的动作、损失、梯度、Adam 状态、模型及目标网络权重摘要逐位一致。8 种基线每种抽查一个环境，算法更新的固定输入摘要一致。另执行了 24 组对比的正式最小更新短测及 checkpoint 重载评估。
- 历史权重通过严格加载和单回合评估，共 24 组原消融、5 组已有权重的 Global MLP-512 消融和 24 组对比。其余 7 组 Global MLP-512 在检查时尚无最终权重。负例覆盖错误 arm、同形状的另一环境、旧基础配置摘要、非 512 Global MLP、错误 SB3 运行 manifest 及错误 SIS 协议。还验证了旧 NetCC-RL PPO 归档和初版 CMAT 输出转换。
- Pytest 全集按测试节点分六批运行，106 通过，72 个 GPU 用例跳过。独立 CPU 环境分别安装 wheel 和 sdist，均通过依赖检查与仓库外配置枚举。wheel 完成 SIS E9 短测和评估，sdist 完成 SIS CMAT 短测和评估。旧消融分析与独立评估汇总可读取历史行。原基准本身有一条测试仍期待禁止使用旧目录名。候选改为测试实际合同，即训练需要显式输出目录。
- 初次验收时八张 GPU 均已有训练进程。之后获准共享物理 GPU 7 做短程 CUDA 补验。两轮验收均未启动正式训练。

## 短程 CUDA 补验

同一张物理卡上分别运行 `d45e6352` 与候选代码。每组使用 B512、同一组真实回放起点和固定随机种子，各执行四次 learner 更新。两版均通过原有的 eager 对照容差后，再配对比较加速路径的动作、Q、损失、梯度范数、模型和目标网络权重及 Adam 状态。测试保留 CUDA Graph 捕获和缓存复用断言。

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \\
NETCC_RL_CUDA_ACCEPTANCE=1 NETCC_RL_COMPILE_ACCEPTANCE=1 NETCC_RL_FULL_BATCH_ACCEPTANCE=1 \\
OMP_NUM_THREADS=1 python -m pytest -q tests/test_acceleration.py \\
  -k 'E0-swing or E9-sis or E10-kuramoto'
```

2026-11-26 在物理 GPU 7 上使用 PyTorch 2.4.0 和 CUDA 12.1。基准与候选各通过三组 B512 测试，24 次 learner 更新均完成。三组都捕获并复用了 CUDA Graph。

- SIS E9 与 Kuramoto E10 的记录张量前后完全一致。
- Swing E0 的最大绝对差分别为动作 `1.82e-6`、Q `5.96e-8`、损失或梯度范数 `1.48e-5`、模型含目标网络权重 `5.95e-7`、Adam 状态 `5.12e-8`，均在原容差内。
- 共享卡监测到短时热限频，最高采样温度 87°C。现有训练进程保持运行。本次仅记录数值与加速路径正确性。
