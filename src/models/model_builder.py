from tensorflow import keras


def build_model(config: dict, num_classes: int) -> keras.Model:
    """Build a transfer learning model using EfficientNet as a backbone."""
    backbone = config["model"].get("backbone", "efficientnetb0").lower()
    input_shape = tuple(config["model"].get("input_shape", [224, 224, 3]))
    dropout_rate = float(config["model"].get("dropout_rate", 0.3))
    dense_units = int(config["model"].get("dense_units", 256))
    learning_rate = float(config["model"].get("learning_rate", 1e-4))
    activation = config["model"].get("activation", "softmax")
    pretrained = config["model"].get("pretrained", True)

    if backbone == "efficientnetb0":
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    outputs = keras.layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = keras.Model(inputs, outputs, name="cancer_detection_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
