from torch import nn

from models.deep_learning._networks import LiquidNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class LNNPriceRegressor(TorchSequencePriceRegressor):
    source = "https://arxiv.org/abs/2006.04439"

    def _build_network(self, input_size: int) -> nn.Module:
        return LiquidNetwork(input_size, self.hidden_size)

