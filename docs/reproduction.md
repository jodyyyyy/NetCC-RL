# Reproduction

## Learner cadence

The proposed learner samples real transitions from episode-scoped replay. Its formal profiles use batch size 512, 5,000 warmup transitions, and UTD 1. D0 learns from the real next state. D1 uses masked model rollouts with up to three real-transition targets.

The PPO profile collects 150 transitions before each update and runs its final 100-transition partial rollout at the formal budget. SAC and TD3 keep their 5,000-transition warmup, one-step collection cadence, and delayed policy updates where applicable. MAPPO, MAT and JointPPO collect one episode per outer update. HAPPO collects ten episodes per update and flushes the final partial batch. CMAT collects 32 episodes per wave and processes the packed episodes in slot order, completing each episode's Adam updates before the next. The formal tail has ten 150-transition episodes and one 100-transition segment. CMAT's total budget is 500,010 Adam steps.

The common SIS protocol uses the same physical snapshot and driver order across all comparisons. Its normalized action bounds and reward are also used by the proposed learner. Algorithm-specific collector RNG and policy/optimizer cadence follow their own profiles.

## Evaluation

Formal training evaluates the proposed learner every 50,000 transitions with 20 episodes starting at seed 202600. Comparison learners evaluate the final checkpoint with 50 episodes starting at 202600. CMAT also evaluates at the first completed collection wave at or after each 50,000-transition boundary. For independent proposed evaluation, use `--episodes 100 --eval-seed 203000`. The analyzer uses the trapezoidal integral over the full step axis divided by its length. Late3 and Late5 refer to the last three or five scheduled evaluations.

## Checkpoints

Pass a comparison run's `checkpoint.json` to the evaluator. The index records the payload hash and native format. The proposed learner checks the architecture, environment, critic support and Gaussian semantics before loading. Proposed checkpoints are saved in format v5. The v4 reader validates the original architecture digest and frozen base-config digest before mapping metadata fields. Both versions use the same model keys and require a strict tensor layout match.

Comparison checkpoint identity includes the full selected config. Native PyTorch optimizer snapshots from older runners use runner-specific containers. Run `python scripts/migrate_comparison_checkpoint.py --config CONFIG --checkpoint OLD_CHECKPOINT --output NEW_DIR` for historical comparison weights. For legacy CMAT weights, the converter migrates the saved `learner` mapping into a dictionary with the exact historical comparison config and `real_steps`, then validate the adapter identity and incremental backend through the evaluator. For the initial NetCC-RL schema, the evaluator recognizes the legacy config embedded in the checkpoint. Run `python scripts/convert_run.py --config CONFIG --run LEGACY_RUN --output NEW_DIR` to export a completed legacy CMAT run into the common comparison schema. Legacy training metrics lack per-episode summaries, so their `episodes` arrays are empty. Evaluation episodes are preserved. For historical SB3 archives, retain the run's sibling `config.json`. The migrator verifies the original manifest and maps the serialized `PhaseExtractor` class path in the archive. The original policy settings must match the selected formal profile. Load only trusted PyTorch and SB3 archives.

Proposed checkpoints restore model weights. CMAT writes a separate `last-good.pt` with RNG, Adam state, simulator state and pending trajectories for same-runtime recovery with `--resume`. The final evaluation checkpoint serves inference. Restart validation across runtime versions is pending.

## Execution modes

Formal proposed configurations bind PyG 2.6.1 with the explicit FP32 compile and CUDA Graph paths. CPU and eager CUDA runs are available separately. The former Turbo-GNN backend was removed. Configurations that request it explicitly fail with a clear error. Frozen PyG COO layouts, physical snapshots, and driver order are preserved.
