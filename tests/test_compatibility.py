from copy import deepcopy
from pathlib import Path

import pytest

from netcc_rl.cli import read_config
from netcc_rl.compat.ablation import normalize_record
from netcc_rl.compat.comparison import cmat_config
from netcc_rl.identity import digest


def test_historical_ablation_identity_is_checked_before_mapping():
    old_arch = dict(version="exp016_adaptive_v1", exp017_environment="sis", exp017_arm="E0")
    legacy = dict(exp017_identity_sha256=digest(old_arch), architecture_resolved=old_arch)
    resolved = normalize_record(legacy)
    assert resolved["architecture_resolved"] == {
        "version": "adaptive_width_v1", "ablation_environment": "sis", "ablation_arm": "E0"
    }
    assert resolved["ablation_identity_sha256"] == digest(resolved["architecture_resolved"])
    assert legacy["architecture_resolved"] == old_arch
    wrong = deepcopy(legacy)
    wrong["architecture_resolved"]["exp017_arm"] = "E1"
    with pytest.raises(ValueError, match="digest differs"):
        normalize_record(wrong)


@pytest.mark.parametrize("text, match", [
    ("exp017_arm: E0\nablation_arm: E0\n", "Duplicate"),
    ("ablation_arm: E0\nablation_arm: E1\n", "Duplicate"),
    ("graph_backend: turbo\n", "Unsupported graph_backend"),
])
def test_conflicting_fields_and_removed_backend_are_rejected(tmp_path, text, match):
    path = tmp_path / "profile.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        read_config(path)


def test_cmat_legacy_schema_migration_preserves_protocol():
    config = read_config(Path(__file__).resolve().parent / "fixtures/legacy_cmat_sis.yaml")
    legacy = deepcopy(config)
    legacy["implementation"]["adapter_identity"]["version"] = "exp015_task_adapter_v1"
    assert cmat_config(legacy) == config
    assert legacy["implementation"]["adapter_identity"]["version"] == "exp015_task_adapter_v1"
    assert config["action_bounds"] == [-1, 0.2]
