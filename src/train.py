
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    return {
        "MAE": mean_absolute_error(y, predictions),
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions),
    }


def main():
    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("Baseline results:")
    print(evaluate_model(baseline, X_test, y_test))

    print("\nRandom Forest results:")
    print(evaluate_model(model, X_test, y_test))

    output_path = Path("models/house_price_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"\nSaved trained model to {output_path}")


if __name__ == "__main__":
    main()
