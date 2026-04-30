import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if "silica_concentrate" not in df.columns:
        raise ValueError("Column 'silica_concentrate' not found in dataframe.")

    # Remove the date column
    df = df.drop(columns=["date"])

    # Split the data into train and test sets
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the processed data directory if it doesn't exist
    os.makedirs("data/processed_data", exist_ok=True)

    # Save the data to csv
    X_train.to_csv("data/processed_data/X_train.csv", index=False)
    X_test.to_csv("data/processed_data/X_test.csv", index=False)
    y_train.to_csv("data/processed_data/y_train.csv", index=False)
    y_test.to_csv("data/processed_data/y_test.csv", index=False)

    # Check the shapes of the data
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv("data/raw_data/raw.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    if X_train.shape[0] > 0 and X_test.shape[0] > 0 and y_train.shape[0] > 0 and y_test.shape[0] > 0:
        print("Data split successfully.")
    else:
        print("Data split failed.")
        raise ValueError("Data split failed.")