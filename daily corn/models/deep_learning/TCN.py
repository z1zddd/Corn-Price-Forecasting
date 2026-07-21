from torch import nn

from models.deep_learning._networks import TemporalConvNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class TCNPriceRegressor(TorchSequencePriceRegressor):
    source = "https://arxiv.org/abs/1803.01271"

    def _build_network(self, input_size: int) -> nn.Module:
        return TemporalConvNetwork(input_size, self.hidden_size)

