import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_raw(df: pd.DataFrame) -> bool:
    """Validate raw data with basic quality checks."""

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if df.isna().any().any():
        raise ValueError("Input dataframe contains missing values.")

    if df.duplicated().any():
        raise ValueError("Input dataframe contains duplicated rows.")

    non_numeric_cols = [
        col for col in df.columns if col != "date" and not is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")

    return True


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw_data/raw.csv")
    if validate_raw(raw_df):
        print("Raw data is valid.")
    else:
        print("Raw data is invalid.")