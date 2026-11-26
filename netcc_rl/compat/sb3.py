"""Read the feature extractor from trusted older SB3 archives."""

import base64
import io
import json
import pickle
from zipfile import ZipFile


class _PolicyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "PhaseExtractor" and module in {
            "netcc_rl.algorithms.baselines.sb3", "FlowActor.exp018.sb3",
        }:
            from netcc_rl.algorithms.baselines.sb3_agent import PhaseExtractor
            return PhaseExtractor
        return super().find_class(module, name)


def load_agent(agent_class, path, device, expected_policy):
    from netcc_rl.algorithms.baselines.sb3_agent import PhaseExtractor

    with ZipFile(path) as archive:
        encoded = json.loads(archive.read("data"))["policy_kwargs"][":serialized:"]
    policy = _PolicyUnpickler(io.BytesIO(base64.b64decode(encoded, validate=True))).load()
    expected = dict(net_arch=expected_policy, features_extractor_class=PhaseExtractor)
    if agent_class.__name__ == "SAC":
        expected["use_sde"] = False
    if policy != expected:
        raise ValueError("SB3 checkpoint policy architecture differs")
    return agent_class.load(path, device=device, custom_objects={"policy_kwargs": policy})
