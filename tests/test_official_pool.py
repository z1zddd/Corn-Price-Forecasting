import numpy as np

from models.deployment_ensemble import BEST_DEPLOYMENT_MODEL_POOL_NAME
from models.official_pool import OFFICIAL_57_MODEL_NAMES, build_official_model_pool, expand_model_pool, format_input
from models.registry import create_model, expand_model_configs


def test_official_pool_has_expected_57_names():
    specs = build_official_model_pool()

    assert [spec.name for spec in specs] == OFFICIAL_57_MODEL_NAMES
    assert len(specs) == 57


def test_official_pool_expands_to_framework_model_configs():
    expanded = expand_model_pool("official_57")

    assert len(expanded) == 57
    assert expanded[0] == {"name": "ada_boost_tree", "type": "official_pool", "enabled": True}
    assert expanded[-1]["name"] == "xgboost_gbtree"
    assert expand_model_configs("official_57") == expanded


def test_best_deployment_pool_expands_to_registered_ensemble_models():
    expanded = expand_model_pool(BEST_DEPLOYMENT_MODEL_POOL_NAME)

    assert [row["name"] for row in expanded] == [
        "corn_h1_forward_replacement_ap_hard_vote",
        "corn_h2_forward_replacement_ba_hard_vote",
    ]
    assert all(row["type"] == "deployment_ensemble" for row in expanded)
    assert len(expand_model_pool("official_57_plus_best_deployment")) == 59


def test_registry_creates_all_official_pool_adapters_without_optional_imports():
    for model_name in OFFICIAL_57_MODEL_NAMES:
        model = create_model({"name": model_name, "type": "official_pool"})

        assert model.spec.name == model_name
        assert hasattr(model, "fit_with_targets")
        assert hasattr(model, "predict_regression")


def test_official_pool_tabular_model_fits_predicts_and_regresses():
    x = np.array(
        [
            [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]],
            [[3.0, 2.0, 1.0], [0.3, 0.2, 0.1]],
            [[1.5, 2.5, 3.5], [0.2, 0.3, 0.4]],
            [[3.5, 2.5, 1.5], [0.4, 0.3, 0.2]],
            [[2.0, 3.0, 4.0], [0.2, 0.4, 0.6]],
            [[4.0, 3.0, 2.0], [0.6, 0.4, 0.2]],
            [[2.5, 3.5, 4.5], [0.3, 0.5, 0.7]],
            [[4.5, 3.5, 2.5], [0.7, 0.5, 0.3]],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.04, -0.02], dtype=float)
    model = create_model({"name": "random_forest_shallow", "type": "official_pool"})

    model.fit_with_targets(x[:6], y[:6], returns[:6])
    prob = model.predict_proba(x[6:])
    pred = model.predict(x[6:])
    raw = model.predict_regression(x[6:])

    assert prob.shape == (2,)
    assert raw.shape == (2,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    assert set(pred.tolist()).issubset({0, 1})


def test_official_pool_input_layouts():
    x = np.zeros((2, 3, 4), dtype=float)

    assert format_input(x, "tabular_flat").shape == (2, 12)
    assert format_input(x, "keras_sequence").shape == (2, 4, 3)
    assert format_input(x, "aeon_collection").shape == (2, 3, 4)
    assert format_input(x, "aeon_collection_pad10").shape == (2, 3, 10)
