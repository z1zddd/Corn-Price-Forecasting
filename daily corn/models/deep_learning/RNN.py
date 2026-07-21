from torch import nn

from models.deep_learning._networks import RecurrentNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class RNNPriceRegressor(TorchSequencePriceRegressor):
    source = "https://pytorch.org/docs/stable/generated/torch.nn.RNN.html"

    def _build_network(self, input_size: int) -> nn.Module:
        return RecurrentNetwork(input_size, self.hidden_size)

