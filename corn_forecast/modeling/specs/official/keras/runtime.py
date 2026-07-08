"""Keras sequence estimators for the official model pool."""

from __future__ import annotations

import numpy as np


class KerasSequenceClassifier:
    """Keras sequence classifier built from official TensorFlow/Keras layers."""

    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(self.architecture, np.asarray(x).shape[1:], self.params, "classification", self.seed)
        class_weight = None
        unique, counts = np.unique(y.astype(int), return_counts=True)
        if len(unique) == 2 and counts.min() > 0:
            total = float(counts.sum())
            class_weight = {int(cls): total / (2.0 * float(count)) for cls, count in zip(unique, counts)}
        self.model_.fit(
            np.asarray(x, dtype=np.float32),
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
            class_weight=class_weight,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras classifier is not fitted")
        return np.clip(np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class KerasSequenceRegressor:
    """Keras sequence regressor built from official TensorFlow/Keras layers."""

    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(self.architecture, np.asarray(x).shape[1:], self.params, "regression", self.seed)
        self.model_.fit(
            np.asarray(x, dtype=np.float32),
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras regressor is not fitted")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1)


def import_keras():
    configure_tensorflow_runtime()
    import tensorflow as tf
    from tensorflow import keras

    configure_tensorflow_runtime()
    return tf, keras


def configure_tensorflow_runtime() -> None:
    try:
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass


def build_keras_sequence_model(architecture: str, input_shape: tuple[int, ...], params: dict[str, object], task: str, seed: int):
    tf, keras = import_keras()
    _ = tf
    inputs = keras.Input(shape=input_shape)
    x = inputs
    architecture = architecture.lower()
    if architecture in {"lstm", "gru", "bilstm", "bigru"}:
        layer_name = "GRU" if "gru" in architecture else "LSTM"
        layer_cls = getattr(keras.layers, layer_name)
        bidirectional = architecture.startswith("bi") or bool(params.get("bidirectional", False))
        units = as_int_list(params.get("units", 32))
        dropout = float(params.get("dropout", 0.0))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
        for idx, unit in enumerate(units):
            recurrent = layer_cls(
                int(unit),
                activation=str(params.get("activation", "tanh")),
                recurrent_activation=str(params.get("recurrent_activation", "sigmoid")),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=idx < len(units) - 1,
            )
            x = keras.layers.Bidirectional(recurrent)(x) if bidirectional else recurrent(x)
    elif architecture == "tcn":
        from tcn import TCN

        x = TCN(
            nb_filters=int(params.get("nb_filters", 16)),
            kernel_size=int(params.get("kernel_size", 2)),
            nb_stacks=int(params.get("nb_stacks", 1)),
            dilations=tuple(int(v) for v in params.get("dilations", (1,))),
            padding=str(params.get("padding", "causal")),
            use_skip_connections=bool(params.get("use_skip_connections", True)),
            dropout_rate=float(params.get("dropout_rate", 0.0)),
            return_sequences=False,
            activation=str(params.get("activation", "relu")),
            use_batch_norm=bool(params.get("use_batch_norm", False)),
            use_layer_norm=bool(params.get("use_layer_norm", False)),
            name=f"tcn_{seed % 100000}",
        )(x)
    else:
        raise ValueError(f"Unsupported Keras sequence architecture: {architecture}")

    for unit in as_int_list(params.get("dense_units", [])):
        x = keras.layers.Dense(int(unit), activation=str(params.get("dense_activation", "relu")))(x)
        if float(params.get("dense_dropout", 0.0)) > 0:
            x = keras.layers.Dropout(float(params.get("dense_dropout", 0.0)))(x)
    if task == "classification":
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        loss = str(params.get("loss", "binary_crossentropy"))
        metrics = ["accuracy"]
    elif task == "regression":
        outputs = keras.layers.Dense(1, activation="linear")(x)
        loss = str(params.get("loss", "mse"))
        metrics = ["mse"]
    else:
        raise ValueError(f"Unknown Keras task: {task}")
    model = keras.Model(inputs=inputs, outputs=outputs)
    learning_rate = float(params.get("learning_rate", 1e-3))
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=float(params.get("weight_decay", 1e-4)))
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def as_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]
