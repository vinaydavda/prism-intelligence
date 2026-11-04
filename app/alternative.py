
import pandas as pd


def calculate_growth(current: float, previous: float):
    """Calculate percentage growth between two observations."""

    if previous == 0:
        return None

    return round(
        ((current - previous) / abs(previous)) * 100,
        2
    )


def analyze_alternative(
    alternative_data: pd.DataFrame
) -> dict:
    """
    Analyze alternative company data.

    Calculates growth for:
    - App downloads
    - Website traffic
    - Job postings
    """

    required_columns = {
        "date",
        "symbol",
        "metric",
        "value",
        "source",
    }

    missing = required_columns - set(alternative_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if alternative_data.empty:
        raise ValueError("Alternative data is empty")

    df = alternative_data.copy()

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

    if (df["value"] < 0).any():
        raise ValueError("Alternative data cannot contain negative values")

    df["symbol"] = df["symbol"].str.upper()

    if df.duplicated(["date", "symbol", "metric"]).any():
        raise ValueError("Duplicate alternative observations found")

    results = []

    # Analyze each alternative metric separately
    for metric, group in df.groupby("metric"):

        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < 2:
            continue

        previous = group.iloc[-2]
        latest = group.iloc[-1]

        growth = calculate_growth(
            latest["value"],
            previous["value"]
        )

        if growth is None:
            direction = "UNDEFINED"
        elif growth > 0:
            direction = "POSITIVE"
        elif growth < 0:
            direction = "NEGATIVE"
        else:
            direction = "NEUTRAL"

        results.append({
            "metric": metric,
            "previous_value": float(previous["value"]),
            "latest_value": float(latest["value"]),
            "growth_pct": growth,
            "direction": direction,
            "previous_date": previous["date"].strftime("%Y-%m-%d"),
            "latest_date": latest["date"].strftime("%Y-%m-%d"),
            "source": latest["source"],
        })

    if not results:
        raise ValueError(
            "Insufficient observations to calculate signals"
        )

    # Separate customer interest from hiring activity
    customer_metrics = {
        "app_downloads",
        "web_traffic",
    }

    customer_signals = [
        item for item in results
        if item["metric"] in customer_metrics
    ]

    hiring_signals = [
        item for item in results
        if item["metric"] == "job_postings"
    ]

    def get_group_trend(signals):
        if not signals:
            return "INSUFFICIENT_DATA"

        directions = [item["direction"] for item in signals]

        if "UNDEFINED" in directions:
            return "INSUFFICIENT_DATA"

        if all(d == "POSITIVE" for d in directions):
            return "POSITIVE"

        if all(d == "NEGATIVE" for d in directions):
            return "NEGATIVE"

        if all(d == "NEUTRAL" for d in directions):
            return "NEUTRAL"

        return "MIXED"

    return {
        "symbol": str(df["symbol"].iloc[0]),
        "customer_interest_signal": get_group_trend(
            customer_signals
        ),
        "hiring_signal": get_group_trend(
            hiring_signals
        ),
        "signals": results,
    }
