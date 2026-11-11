"""Loading, cleaning and splitting the Telco churn data."""
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_and_clean(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and apply the row-independent cleaning steps.

    Nothing here is *learned* from the data (no means, no encodings), so it is
    safe to run before the train/test split.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'Telco Customer Churn' from "
            "Kaggle and place the CSV there."
        )
    df = pd.read_csv(path)

    # customerID is an identifier, not a feature.
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text; blank strings belong to brand-new
    # customers (tenure == 0) who have not been billed yet -> 0.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = total.isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "Blank TotalCharges with tenure > 0"
    df["TotalCharges"] = total.fillna(0.0)

    # Target as 0/1.
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df


def split(df: pd.DataFrame):
    """Stratified 80/20 split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )


def load_train_test():
    """Convenience wrapper used by every script so the split is always identical."""
    return split(load_and_clean())


if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Rows: {len(df)}  |  overall churn rate: {df[config.TARGET].mean():.4f}")
    print(f"Train: {len(X_train)} rows, churn rate {y_train.mean():.4f}")
    print(f"Test:  {len(X_test)} rows, churn rate {y_test.mean():.4f}")
