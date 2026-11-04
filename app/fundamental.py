
import pandas as pd


def calculate_growth(current: float, previous: float):
    """Calculate percentage growth between two periods."""

    if previous == 0:
        return None

    return round(
        ((current - previous) / abs(previous)) * 100,
        2
    )


def analyze_fundamentals(financials: pd.DataFrame) -> dict:
    """
    Analyze company financial performance.

    Calculates:
    - Revenue growth
    - Net profit growth
    - Operating margin
    """

    if financials.empty:
        raise ValueError("Financial data is empty")

    required_columns = [
        "period",
        "revenue_cr",
        "net_profit_cr",
        "operating_margin_pct",
        "eps",
    ]

    missing = set(required_columns) - set(financials.columns)

    if missing:
        raise ValueError(
            f"Missing financial columns: {sorted(missing)}"
        )

    df = financials.copy()

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

    if df[required_columns].isnull().any().any():
        raise ValueError("Financial data contains missing values")

    df = df.sort_values("period").reset_index(drop=True)

    latest = df.iloc[-1]

    result = {
        "period": latest["period"],
        "revenue_cr": float(latest["revenue_cr"]),
        "net_profit_cr": float(latest["net_profit_cr"]),
        "operating_margin_pct": float(
            latest["operating_margin_pct"]
        ),
        "eps": float(latest["eps"]),
        "revenue_growth_pct": None,
        "profit_growth_pct": None,
        "margin_change_pct_points": None,
        "growth_period": "QoQ",
    }

    if len(df) >= 2:
        previous = df.iloc[-2]

        result["revenue_growth_pct"] = calculate_growth(
            latest["revenue_cr"],
            previous["revenue_cr"]
        )

        result["profit_growth_pct"] = calculate_growth(
            latest["net_profit_cr"],
            previous["net_profit_cr"]
        )

        result["margin_change_pct_points"] = round(
            latest["operating_margin_pct"]
            - previous["operating_margin_pct"],
            2
        )

    return result
