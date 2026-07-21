from torch import nn

from models.deep_learning._networks import RAFTNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class RAFTPriceRegressor(TorchSequencePriceRegressor):
    source = "https://github.com/archon159/RAFT"

    def _build_network(self, input_size: int) -> nn.Module:
        return RAFTNetwork(input_size, self.hidden_size)

