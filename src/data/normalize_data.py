import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


def normalize_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize train and test feature data."""
    if X_train.empty or X_test.empty:
        raise ValueError("Input dataframe is empty.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test must have the same columns.")

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed_data", exist_ok=True)

    # Save the scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Save the data to csv
    X_train_scaled.to_csv("data/processed_data/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("data/processed_data/X_test_scaled.csv", index=False)

    # Check the shapes of the data
    print(f"X_train scaled shape: {X_train_scaled.shape}")
    print(f"X_test scaled shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")
    normalize_data(X_train, X_test)