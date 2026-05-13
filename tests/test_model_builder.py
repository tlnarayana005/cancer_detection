from src.models.model_builder import build_model


def test_build_model():
    config = {
        "model": {
            "backbone": "efficientnetb0",
            "input_shape": [224, 224, 3],
            "dropout_rate": 0.2,
            "dense_units": 128,
            "learning_rate": 1e-4,
            "activation": "softmax",
            "pretrained": False,
        }
    }
    model = build_model(config, num_classes=2)
    assert model.output_shape == (None, 2)
    assert model.inputs[0].shape[1:] == (224, 224, 3)
