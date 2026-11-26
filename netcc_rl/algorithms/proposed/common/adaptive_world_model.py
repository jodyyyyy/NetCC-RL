"""G2 global GAT context with native local inputs for the shared policy."""

from .world_model import WorldModel
from .node_features import node_features


class AdaptiveWorldModel(WorldModel):
    def encode_actor_features(self, obs):
        return node_features(obs, self.cfg), self.encode(obs, None)
