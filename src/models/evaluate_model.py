import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import os


def evaluate_model(X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[float, float, float]:
    """Evaluate trained model and save predictions + metrics."""
    if X_test.empty or y_test.empty:
        raise ValueError("Input dataframe is empty.")

    model = joblib.load("models/trained_model.pkl")
    y_pred = model.predict(X_test)
    y_true = y_test.squeeze()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    os.makedirs("data", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    pd.DataFrame({"prediction": y_pred}).to_csv("data/predictions.csv", index=False)
    scores = {"mse": mse, "rmse": rmse, "r2": r2}
    with open("metrics/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    return mse, rmse, r2


if __name__ == "__main__":
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")
    mse, rmse, r2 = evaluate_model(X_test, y_test)
    print(f"Mean squared error: {mse}")
    print(f"Root mean squared error: {rmse}")
    print(f"R-squared score: {r2}")