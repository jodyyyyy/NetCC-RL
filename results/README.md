# Results

The [2026-11-26 seed 0 snapshot](seed0-20261126/README.md) contains final evaluation returns for completed ablations and seven baselines on SIS, Kuramoto and Swing, plus an independent evaluation of E0–E7.

Record the source version, configuration and environment hashes, training seed, real-step budget, evaluation settings and artifact hashes when adding new run archives. Keep the protocol identifier on each run. The comparison CLI records `algorithm`, `environment` and `protocol` in evaluation and metric rows. Evaluation records include the episode seed, return, transition count and episode boundary.

For ablation curves, place runs in directories named `<environment>-<arm>-seed0` under a shared output directory. Run the code-only analysis and TensorBoard export as follows.

```bash
python scripts/analyze_ablation.py --run-root runs --output analysis --set all
python scripts/tensorboard_export.py --input runs/sis-E0-seed0/metrics.jsonl --output tensorboard/sis-E0
tensorboard --logdir tensorboard
```

The analyzer lists pending runs and computes statistics only after the required evaluation sequence is present. For independent evaluation, save each 100-episode JSON file under `<environment>-<arm>-seed0.json` and pass its directory to `scripts/analyze_evaluation.py`. Episode-level mean, sample standard deviation and Student-t interval describe each fixed checkpoint.
