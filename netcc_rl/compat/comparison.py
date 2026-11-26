"""Comparison metadata compatibility for historical environment identities."""

from netcc_rl.identity import digest


def adapter_identity(value):
    """Map the version label while retaining all environment and action fields."""
    if value.get("version") == "exp015_task_adapter_v1":
        return {**value, "version": "continuous_task_adapter_v1"}
    return value


def cmat_config(config):
    if "implementation" in config:
        implementation = config["implementation"]
        return {**config, "implementation": {
            **implementation, "adapter_identity": adapter_identity(implementation["adapter_identity"])
        }}
    if "adapter_identity" in config:
        return {**config, "adapter_identity": adapter_identity(config["adapter_identity"])}
    return config


def validate_historical_serial_manifest(saved, config):
    semantics = saved["action_semantics"]
    if config["algorithm"] == "ppo" and semantics == "Gaussian stored unclipped; native environment clips":
        semantics = "Gaussian stored unclipped; SB3 clips at environment boundary"
    if (digest(saved["snapshot"]) != config["snapshot_sha256"]
            or saved["drivers"] != config["drivers"]
            or saved["profile"] != config["profile"]
            or semantics != config["action_semantics"]
            or (saved["seed"], saved["budget"]) != (config["seed"], config["steps"])):
        raise ValueError("Historical comparison run environment or profile differs")


def historical_serial_profile(profile, config):
    if config["algorithm"] != "mappo":
        return profile
    length = config["profile"]["episode_length"]
    # The historical runner serializes its args after the final partial episode.
    final_length = config["steps"] % length or length
    if profile.get("episode_length") == final_length:
        return {**profile, "episode_length": length}
    return profile
