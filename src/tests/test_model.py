
import numpy as np
from sklearn.dummy import DummyRegressor

from src.train import evaluate_model


def test_evaluate_model_returns_expected_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    model = DummyRegressor(strategy="constant", constant=2.0)
    model.fit(np.arange(3).reshape(-1, 1), y_true)

    metrics = evaluate_model(model, np.arange(3).reshape(-1, 1), y_true)

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert np.isclose(metrics["MAE"], 2.0 / 3.0)
    assert np.isclose(metrics["RMSE"], np.sqrt(2.0 / 3.0))
    assert np.isfinite(metrics["R2"])


def test_perfect_predictions_have_zero_error():
    class PerfectModel:
        def predict(self, X):
            return np.asarray(X).reshape(-1)

    y_true = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_model(PerfectModel(), y_true.reshape(-1, 1), y_true)

    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert np.isclose(metrics["R2"], 1.0)
