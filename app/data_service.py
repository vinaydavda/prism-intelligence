
from pathlib import Path

import pandas as pd


# Project data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Required columns for each dataset
REQUIRED_COLUMNS = {
    "financials": {
        "symbol",
        "period",
        "revenue_cr",
        "net_profit_cr",
        "operating_margin_pct",
        "eps",
    },
    "prices": {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    },
    "alternative_signals": {
        "date",
        "symbol",
        "metric",
        "value",
        "unit",
        "source",
    },
}


def load_dataset(filename: str, dataset_name: str) -> pd.DataFrame:
    """Load and validate a CSV dataset."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required = REQUIRED_COLUMNS[dataset_name]
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name}: Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name}: Dataset is empty"
        )

    return df


def load_financials(symbol: str) -> pd.DataFrame:
    """Load company financial information."""

    df = load_dataset(
        "financials.csv",
        "financials"
    )

    numeric_columns = [
        "revenue_cr",
        "net_profit_cr",
        "operating_margin_pct",
        "eps",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if df[["symbol", "period"] + numeric_columns].isnull().any().any():
        raise ValueError("Financial data contains missing values")

    df["symbol"] = df["symbol"].str.upper()

    return df[df["symbol"] == symbol.upper()].copy()


def load_prices(symbol: str) -> pd.DataFrame:
    """Load historical stock prices."""

    df = load_dataset(
        "prices.csv",
        "prices"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if df[["date", "symbol"] + numeric_columns].isnull().any().any():
        raise ValueError("Price data contains missing values")

    df["symbol"] = df["symbol"].str.upper()

    df = df[df["symbol"] == symbol.upper()].copy()

    return df.sort_values("date").reset_index(drop=True)


def load_alternative_signals(symbol: str) -> pd.DataFrame:
    """Load alternative company activity data."""

    df = load_dataset(
        "alternative_signals.csv",
        "alternative_signals"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="raise"
    )

    if df[["date", "symbol", "metric", "value", "source"]].isnull().any().any():
        raise ValueError("Alternative data contains missing values")

    df["symbol"] = df["symbol"].str.upper()

    df = df[df["symbol"] == symbol.upper()].copy()

    return df.sort_values(
        ["metric", "date"]
    ).reset_index(drop=True)


def get_company_data(symbol: str) -> dict:
    """Load all datasets for a company."""

    financials = load_financials(symbol)
    prices = load_prices(symbol)
    alternative = load_alternative_signals(symbol)

    if financials.empty:
        raise ValueError(f"No financial data found for {symbol}")

    if prices.empty:
        raise ValueError(f"No price data found for {symbol}")

    if alternative.empty:
        raise ValueError(f"No alternative data found for {symbol}")

    return {
        "symbol": symbol.upper(),
        "financials": financials,
        "prices": prices,
        "alternative": alternative,
    }
