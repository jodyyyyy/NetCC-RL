# Seed 0 results, 2026-11-27

This snapshot contains 36 completed ablations and 24 completed baseline runs. The unified SIS CMAT run completed 1,000,000 training transitions and 500,010 optimizer updates. Its protocol uses the common SIS reward with a control-cost coefficient of 0.1 and action bounds of `[-1, 1]`.

Each completed run used training seed 0 and 1,000,000 real environment transitions. Evaluation transitions are separate from that budget. All returns are cumulative episode rewards. Episode variation and the independent evaluation intervals describe fixed checkpoints trained once. Comparisons over training seeds require additional runs.

| File | Contents |
| --- | --- |
| [runs.csv](runs.csv) | One row per ablation or included baseline. Contains status, training/evaluation metadata and final return statistics. |
| [final-episodes.csv](final-episodes.csv) | Final evaluation returns by run and seed. Transition counts are available for ablations and CMAT. Termination flags are available for ablations. |
| [ablation-curve.csv](ablation-curve.csv) | Evaluation return means at recorded training steps. |
| [independent-summary.csv](independent-summary.csv) | E0–E7 fixed-checkpoint evaluation means, sample SDs and 95% Student t intervals. |
| [independent-episodes.csv](independent-episodes.csv) | Episode-level records for the independent evaluation. |
| [SHA256SUMS](SHA256SUMS) | SHA-256 hashes for the five CSV files. |

The ablation curves use stochastic actions and 20 evaluation episodes per point (seeds 202600–202619), every 50,000 training transitions from 0 to 1,000,000. The completed baseline results use deterministic actions and 50 evaluation episodes per run (seeds 202600–202649). See the `eval_mode` and `eval_episodes` columns when comparing results. The independent E0–E7 evaluation uses stochastic actions and 100 fresh seeds (203000–203099) per checkpoint. Its `steps` column counts evaluation-episode length. Sample SD uses the number of evaluation episodes minus one as its divisor.

Run IDs use `<environment>-<method>-seed0`. The proposed protocol labels in `runs.csv` distinguish the original E0–E7 matrix from the fixed-width E8–E11 runs. Baseline protocol labels map to the corresponding YAML files in [configs/comparison](../../configs/comparison). The [experiment overview](../../README.md#3-实验内容) maps the ablation arms to the encoder, dynamics and actor. E8–E11 use a global MLP with 512 hidden units and 64 output units. The model and actor parameter counts come from each completed ablation's training summary. These counts vary across settings.

The source hashes in `runs.csv` identify the training configuration and final evaluation. For Kuramoto and Swing CMAT, the configuration hash covers canonical JSON for the recorded training profile, environment adapter identity, collector batch size, budget and seed. The CMAT evaluation hash covers the original final line of the evaluation JSONL. Blank fields indicate pending results or fields unavailable in the original evaluation output. This snapshot exports evaluation data from existing runs. The experiment entry points are documented in [reproduction.md](../../docs/reproduction.md).
