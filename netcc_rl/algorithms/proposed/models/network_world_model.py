"""Mean+Max global GAT context with native local inputs for the shared policy."""

from netcc_rl.algorithms.proposed.models.world_model import WorldModel
from netcc_rl.algorithms.proposed.models.node_features import node_features


class NetworkWorldModel(WorldModel):
    def encode_actor_features(self, obs):
        return node_features(obs, self.cfg), self.encode(obs, None)
