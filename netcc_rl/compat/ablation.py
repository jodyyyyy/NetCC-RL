"""Translate legacy ablation names after validating their original identities."""

from collections.abc import Mapping

from netcc_rl.identity import digest

_LEGACY_KEYS = {
    "exp016_arm": "graph_readout_variant",
    "exp017_arm": "ablation_arm",
    "exp017_environment": "ablation_environment",
    "exp017_factors": "ablation_factors",
    "exp017_base_sha256": "base_config_sha256",
    "exp017_identity_sha256": "ablation_identity_sha256",
}
_LEGACY_VALUES = {
    "model_width_profile": {"exp016_adaptive_v1": "adaptive_width_v1"},
    "version": {"exp016_adaptive_v1": "adaptive_width_v1"},
    "collector_protocol": {"exp016_three_environments": "frozen_vector_v1"},
    "mainline_contract": {"exp017_factorial_gaussian_v1": "ablation_gaussian_v2"},
    "actor_local": {
        "native sin/cos + G2 global latent": "native sin/cos + Mean+Max global latent",
        "native state + G2 global latent": "native state + Mean+Max global latent",
    },
}
_LEGACY_BASE_HASHES = {
    "kuramoto": "7f84c43348139228e305f894e6f29a903b45431f1c801cf6a2cc80dc615eb1eb",
    "sis": "bc74088e8c0d0d6bb335168446204aac34203acc69f38e8e9b51a3542da4a778",
    "swing": "96d5d952fedb58022b67ab8cbca07c9acdca5fca469635eec3de6a4671977de3",
}
# These serialized v4 defaults affect only the removed backend.
_INACTIVE_BACKEND_DEFAULTS = {
    "turbo_symmetric_csr": False, "turbo_fused_projection": False,
    "turbo_save_output_for_backward": False, "turbo_grad_A_reduce_row_chunk_size": 512,
    "turbo_forward_light_warps": 1, "turbo_backward_light_warps": 1,
    "turbo_forward_heavy_warps": 8, "turbo_backward_heavy_warps": 8,
    "turbo_forward_quantile": 0.99, "turbo_backward_quantile": 0.99,
}


def normalize_record(record, *, legacy_allowed=True):
    """Accept old names only as a single, unambiguous representation."""
    if not isinstance(record, Mapping):
        raise ValueError("Expected a configuration or metadata mapping")
    result = dict(record)
    if result.get("graph_backend") == "turbo":
        raise ValueError("Unsupported graph_backend 'turbo'; this build supports PyG")
    old_digest = result.get("exp017_identity_sha256")
    old_arch = result.get("architecture_resolved")
    if "ablation_identity_sha256" in result:
        if not isinstance(old_arch, Mapping) or digest(old_arch) != result["ablation_identity_sha256"]:
            raise ValueError("Architecture identity digest differs")
    if "exp017_identity_sha256" in result:
        if not isinstance(old_arch, Mapping) or digest(old_arch) != old_digest:
            raise ValueError("Historical architecture identity digest differs")
    for old, new in _LEGACY_KEYS.items():
        if old not in result:
            continue
        if not legacy_allowed or new in result:
            raise ValueError(f"Duplicate or unsupported legacy identity field {old!r}")
        result[new] = result.pop(old)
    if "exp016_arm" in record:
        if result["graph_readout_variant"] != "G2":
            raise ValueError("Unsupported historical graph readout variant")
        result["graph_readout_variant"] = "mean_max"
    for key, aliases in _LEGACY_VALUES.items():
        value = result.get(key)
        if isinstance(value, str) and value in aliases:
            if not legacy_allowed:
                raise ValueError(f"Unsupported legacy identity value for {key}")
            result[key] = aliases[value]
    if "exp017_base_sha256" in record:
        from netcc_rl.matrix import MATRIX
        name = result.get("ablation_environment")
        if name not in _LEGACY_BASE_HASHES or result["base_config_sha256"] != _LEGACY_BASE_HASHES[name]:
            raise ValueError("Historical base configuration digest differs")
        result["base_config_sha256"] = MATRIX["base_config_sha256"][name]
    for key in tuple(result):
        if not key.startswith("turbo_"):
            continue
        value = result[key]
        default = _INACTIVE_BACKEND_DEFAULTS.get(key)
        if (not legacy_allowed or result.get("graph_backend") != "pyg"
                or type(value) is not type(default) or value != default):
            raise ValueError(f"Unsupported removed backend option {key!r}")
        del result[key]
    if old_arch is not None:
        result["architecture_resolved"] = normalize_record(old_arch, legacy_allowed=legacy_allowed)
        if old_digest is not None:
            result["ablation_identity_sha256"] = digest(result["architecture_resolved"])
    return result


def checkpoint_records(checkpoint):
    if checkpoint.get("kind") != "model_weights" or checkpoint.get("format_version") not in (4, 5):
        raise ValueError("Unsupported proposed checkpoint format")
    legacy = checkpoint["format_version"] == 4
    return (
        normalize_record(checkpoint.get("metadata"), legacy_allowed=legacy),
        normalize_record(checkpoint.get("config"), legacy_allowed=legacy),
    )
