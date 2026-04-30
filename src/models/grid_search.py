import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def grid_search(X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """Run grid search and save best hyperparameters."""
    if X_train.empty or y_train.empty:
        raise ValueError("Input dataframe is empty.")

    # Define the model
    model = RandomForestRegressor()

    # Define the parameters
    parameters = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train.squeeze())

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid_search.best_params_, "models/best_params.pkl")

    return grid_search.best_params_

if __name__ == "__main__":
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    best_params = grid_search(X_train, y_train)
    print("Best params:", best_params)