import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    """Train final model with best hyperparameters and save it."""
    if X_train.empty or y_train.empty:
        raise ValueError("Input dataframe is empty.")

    # Load the best hyperparameters
    best_params = joblib.load("models/best_params.pkl")

    # Train the model
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train.squeeze())

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/trained_model.pkl")

    return model

if __name__ == "__main__":
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    model = train_model(X_train, y_train)
    print("Model trained successfully.")