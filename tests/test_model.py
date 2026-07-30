from __future__ import annotations

from adengine.model import CTRModel, sigmoid


def test_sigmoid_bounds():
    assert sigmoid(-1000) == 0.0
    assert sigmoid(1000) == 1.0
    assert 0.4 < sigmoid(0.0) < 0.6


def test_cold_model_predicts_half():
    model = CTRModel(num_features=8)
    assert model.predict([1.0] + [0.0] * 7) == 0.5


def test_model_learns_positive_feature():
    model = CTRModel(num_features=4, lr=0.5)
    # Always-on feature 0 predicts click when present.
    data = [([1.0, 0.0, 0.0, 0.0], 1) for _ in range(40)]
    data += [([0.0, 1.0, 0.0, 0.0], 0) for _ in range(40)]
    model.train(data, epochs=20)
    assert model.predict([1.0, 0.0, 0.0, 0.0]) > 0.7
    assert model.predict([0.0, 1.0, 0.0, 0.0]) < 0.3


def test_train_materializes_generator_across_epochs():
    model = CTRModel(num_features=2, lr=0.2)

    def gen():
        yield ([1.0, 0.0], 1)
        yield ([0.0, 1.0], 0)

    loss = model.train(gen(), epochs=3)
    assert loss >= 0.0
    assert model.weights[0] != 0.0
