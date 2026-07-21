from torch import nn

from models.deep_learning._networks import MAFSNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class MAFSPriceRegressor(TorchSequencePriceRegressor):
    source = "MAFS paper-aligned feature-selection price-regression adaptation"

    def _build_network(self, input_size: int) -> nn.Module:
        return MAFSNetwork(input_size, self.hidden_size)

