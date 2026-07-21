from torch import nn

from models.deep_learning._networks import TransformerNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class SimpleTMPriceRegressor(TorchSequencePriceRegressor):
    source = "independent Transformer price-regression adaptation"

    def _build_network(self, input_size: int) -> nn.Module:
        return TransformerNetwork(input_size, self.hidden_size, int(self.params.get("num_heads", 2)))

