"""Versioned source profiles and separately applied bounded smoke overrides."""
from copy import deepcopy
import json
from pathlib import Path

DEFAULT = Path(__file__).with_name('configs') / 'formal-hyperparameters-v2.json'


def load_profile(algorithm, *, smoke=False, path=None):
    spec = json.loads(Path(path or DEFAULT).read_text())
    if spec.get('schema_version') != 2:
        raise ValueError('source fidelity config v2 required')
    from netcc_rl.algorithms.baselines.registry import require_active
    require_active(algorithm,training=True)
    entry = spec['algorithms'][algorithm]
    config = deepcopy(entry['formal_runtime'])
    if smoke:
        config.update(deepcopy(entry['smoke_only_overrides']))
    config.update(algorithm=algorithm, profile_mode='bounded_smoke' if smoke else 'source_profile',
                  config_version=2, formal_training_authorized=True)
    return config
