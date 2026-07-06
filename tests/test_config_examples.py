from pathlib import Path

import pytest

from commodity_backtest.config.loader import load_config
from commodity_backtest.config.schema import validate_config


def test_package_imports():
    import commodity_backtest

    assert commodity_backtest.__version__ == "0.1.0"


def test_template_config_loads():
    cfg = load_config(Path("configs/template.yaml"))

    assert cfg["commodity"]["name"] == "template"
    assert cfg["data"]["feature_cols"] == "auto_numeric"
    assert cfg["target"]["mode"] == "classification"


def test_corn_config_validates():
    cfg = load_config(Path("configs/corn.yaml"))

    validate_config(cfg)

    assert cfg["commodity"]["name"] == "corn"
    assert cfg["train_window"]["mode"] == "expanding"


def test_additional_commodity_configs_validate():
    for path in [Path("configs/soybean.yaml"), Path("configs/rebar.yaml")]:
        cfg = load_config(path)
        validate_config(cfg)
        assert cfg["data"]["feature_cols"] == "auto_numeric"
        assert cfg["target"]["mode"] == "classification"


def test_invalid_lookback_rejected():
    cfg = load_config(Path("configs/template.yaml"))
    cfg["lookback"]["default"] = 60
    cfg["train_window"]["min_train_periods"] = 60

    with pytest.raises(ValueError, match="lookback.default must be smaller"):
        validate_config(cfg)


def test_invalid_lookback_candidate_rejected():
    cfg = load_config(Path("configs/template.yaml"))
    cfg["lookback"]["default"] = 3
    cfg["lookback"]["candidates"] = [3, 12]
    cfg["train_window"]["min_train_periods"] = 12

    with pytest.raises(ValueError, match="lookback.candidates"):
        validate_config(cfg)