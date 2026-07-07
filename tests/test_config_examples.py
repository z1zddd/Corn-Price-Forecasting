from pathlib import Path

import pytest

from config.loader import load_config
from config.schema import validate_config


def test_cli_imports():
    import cli

    assert callable(cli.main)


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
    for path in [
        Path("configs/soybean.yaml"),
        Path("configs/rebar.yaml"),
        Path("configs/corn_dual_stream_lstm.yaml"),
        Path("configs/corn_official_pool_57_h1_no_news.yaml"),
        Path("configs/corn_official_pool_57_h2_no_news.yaml"),
        Path("configs/corn_official_pool_57_h1_with_news.yaml"),
        Path("configs/corn_official_pool_57_h2_with_news.yaml"),
    ]:
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
