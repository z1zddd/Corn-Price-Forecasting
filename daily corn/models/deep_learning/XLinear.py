from torch import nn

from models.deep_learning._networks import XLinearNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class XLinearPriceRegressor(TorchSequencePriceRegressor):
    source = "independent XLinear paper-based price-regression adaptation"

    def _build_network(self, input_size: int) -> nn.Module:
        return XLinearNetwork(self.lookback, input_size, self.hidden_size)

