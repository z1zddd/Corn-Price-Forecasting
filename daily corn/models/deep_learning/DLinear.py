from torch import nn

from models.deep_learning._networks import DLinearNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class DLinearPriceRegressor(TorchSequencePriceRegressor):
    source = "https://github.com/cure-lab/LTSF-Linear"

    def _build_network(self, input_size: int) -> nn.Module:
        return DLinearNetwork(
            self.lookback,
            input_size,
            kernel_size=int(self.params.get("kernel_size", 25)),
            individual=bool(self.params.get("individual", False)),
        )
