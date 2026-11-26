# NetCC-RL

NetCC-RL is a Python library for reinforcement learning on continuous-control tasks in complex networks. It includes three frozen environments, a graph-based continuous actor–critic, and eight RL/MARL comparisons.

## Install

Use Python 3.10 and a virtual environment. The reproduction profiles were exercised with PyTorch 2.4.0. Install the CPU wheel first if running on CPU. For CUDA, use the corresponding PyTorch 2.4 wheel for your driver and runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch==2.4.0+cpu'
python -m pip install -c constraints-training.txt '.[baselines,analysis,test]'
```

The base install covers the proposed learner and CMAT. The `baselines` extra installs SB3 for PPO, SAC and TD3. The `analysis` extra installs plotting and TensorBoard tools. The `test` extra installs pytest. Run these examples from the source directory containing `configs/`. Installed modules can run from any directory when you pass the paths to the configs and outputs.

## Train and evaluate

```bash
python -m netcc_rl.catalog --validate
python -m netcc_rl.train --config configs/ablation/sis/E0.yaml --smoke --output runs/sis-e0-smoke
python -m netcc_rl.evaluate --config configs/ablation/sis/E0.yaml --checkpoint runs/sis-e0-smoke/final.pt --episodes 1
python -m netcc_rl.train --config configs/comparison/sis-cmat.yaml --smoke --output runs/sis-cmat-smoke
python -m netcc_rl.evaluate --config configs/comparison/sis-cmat.yaml --checkpoint runs/sis-cmat-smoke/checkpoint.json --episodes 1
```

The same train and evaluation commands accept every profile in `configs/comparison/`. Select `kuramoto`, `sis` or `swing` and one of `ppo`, `sac`, `td3`, `mappo`, `happo`, `mat`, `jointppo` or `cmat`. All comparison runs write the same filenames and result schema. `checkpoint.json` resolves each algorithm's native weight format. Use `--seed` for training and `--eval-seed` for evaluation. Training requires a fresh output path.

The smoke mode performs real environment steps and learner updates with a reduced budget. The formal profiles specify 1,000,000 real environment transitions and training seed 0. Use `--device cuda:0` to select visible CUDA device 0. The proposed formal profiles use PyTorch compilation and CUDA Graphs. To request eager execution explicitly, pass `--eager --device cpu` or `--eager --device cuda:0`.

## Environments and protocols

- **Kuramoto** uses 100 coupled oscillators and 50 fixed actuators. Policies that encode phases use node-major sine and cosine features.
- **SIS** uses 100 nodes and 50 fixed actuators. In the common protocol, action `a` in `[-1,1]` maps to control intensity `(a+1)/2`. The reward subtracts 0.1 times mean control intensity from the infection reward. This makes `-1` the no-intervention action.
- **Swing** uses the 38-node core extracted from the IEEE 39-bus topology and 19 fixed actuators. The simulator uses its frozen Swing dynamics and actuator model.

`configs/comparison/sis-cmat.yaml` uses the common SIS protocol. The historical CMAT SIS experiment used zero control cost and bounded actions to `[-1,0.2]`. Its config is under `configs/historical/` and its returns belong to that protocol. CMAT uses 50 deterministic latent-consensus iterations, a B32 collector and episode-major Adam updates. Evaluation uses deterministic actions for all comparisons. The proposed evaluation profile uses seeded stochastic actions.

## Ablations

The 36 ablations cross 12 settings with all three environments. E0 is the proposed profile and appears once per environment.

- **G0** matches the node embedding parameter count of **G1**, the two-layer four-head GATv2 encoder with Mean+Max global readout.
- **G2** flattens native node features and uses a hidden width of 512 and a 64-dimensional latent. Input widths for Kuramoto, SIS and Swing are 200, 100 and 190. Complete encoder sizes are 136,768, 85,568 and 131,648, respectively. Relative to G1, the differences are +8.59%, −31.99% and +4.20%.
- **D0** uses one-step Bellman targets from real next observations. **D1** uses a horizon-3 global MLP world model with five Q heads and two-hot value targets. Each policy call executes one action.
- **A0** shares a Gaussian actor across actuators and conditions it on the native local state and global latent. **A1** uses a centralized diagonal Gaussian conditioned on the global latent.

The matrix source declares the factor assignment for E0–E11. The eight baseline learners preserve their separate policy information, return or GAE calculation, and optimizer cadence. See the protocol and checkpoint details in [Reproduction](docs/reproduction.md).

## Results and references

Version 0.1.0 is a code-only candidate. The results schema and reproduction commands are in [Results](results/README.md). A single training seed supports descriptive comparisons. Episode-level intervals describe the evaluation distribution for a fixed checkpoint.

- Hansen, Su and Wang. [TD-MPC2](https://arxiv.org/abs/2310.16828).
- Brody, Alon and Yahav. [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).
- Schulman et al. [PPO](https://arxiv.org/abs/1707.06347). Haarnoja et al. [SAC](https://arxiv.org/abs/1801.01290). Fujimoto, van Hoof and Meger. [TD3](https://arxiv.org/abs/1802.09477).
- Yu et al. [MAPPO](https://arxiv.org/abs/2103.01955). Kuba et al. [HAPPO](https://arxiv.org/abs/2109.11251). Wen et al. [MAT](https://arxiv.org/abs/2205.14953). Zhao et al. [CMAT](https://arxiv.org/abs/2604.13472).
- The upstream code and data attributions are in [Third-party notices](THIRD_PARTY_NOTICES.md).
