from torch import nn

from models.deep_learning._networks import PatchTransformerNetwork
from models.deep_learning._sequence_adapter import TorchSequencePriceRegressor


class PatchTSTPriceRegressor(TorchSequencePriceRegressor):
    source = "https://github.com/yuqinie98/PatchTST"

    def _build_network(self, input_size: int) -> nn.Module:
        return PatchTransformerNetwork(
            self.lookback, input_size, self.hidden_size, int(self.params.get("patch_size", 2))
        )

