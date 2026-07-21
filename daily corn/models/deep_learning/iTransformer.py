from torch import nn

from models.deep_learning._networks import InvertedTransformerNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class ITransformerPriceRegressor(TorchSequencePriceRegressor):
    source = "https://github.com/thuml/iTransformer"

    def _build_network(self, input_size: int) -> nn.Module:
        del input_size
        return InvertedTransformerNetwork(self.lookback, self.hidden_size)
