
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Wilder's smoothing."""

    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0), 50
    )

    return rsi


def analyze_technical(prices: pd.DataFrame) -> dict:
    """Calculate technical indicators for a stock."""

    required_columns = {"date", "close", "symbol"}

    missing = required_columns - set(prices.columns)

    if missing:
        raise ValueError(
            f"Missing price columns: {sorted(missing)}"
        )

    if prices.empty:
        raise ValueError("Price data is empty")

    df = prices.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="raise")

    if df[["date", "close"]].isnull().any().any():
        raise ValueError("Price data contains missing values")

    if (df["close"] <= 0).any():
        raise ValueError("Closing prices must be positive")

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate price dates found")

    if len(df) < 50:
        raise ValueError(
            "At least 50 trading sessions are required"
        )

    # Simple Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # Relative Strength Index
    df["rsi_14"] = calculate_rsi(df["close"], period=14)

    # MACD
    ema_12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    latest = df.iloc[-1]

    if latest["close"] > latest["sma_20"] > latest["sma_50"]:
        trend = "BULLISH"
    elif latest["close"] < latest["sma_20"] < latest["sma_50"]:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    if latest["rsi_14"] >= 70:
        rsi_status = "OVERBOUGHT"
    elif latest["rsi_14"] <= 30:
        rsi_status = "OVERSOLD"
    else:
        rsi_status = "NEUTRAL"

    macd_status = (
        "BULLISH"
        if latest["macd"] > latest["macd_signal"]
        else "BEARISH"
    )

    return {
        "symbol": str(latest["symbol"]),
        "as_of": latest["date"].strftime("%Y-%m-%d"),
        "close": round(float(latest["close"]), 2),
        "sma_20": round(float(latest["sma_20"]), 2),
        "sma_50": round(float(latest["sma_50"]), 2),
        "rsi_14": round(float(latest["rsi_14"]), 2),
        "rsi_status": rsi_status,
        "macd": round(float(latest["macd"]), 4),
        "macd_signal": round(float(latest["macd_signal"]), 4),
        "macd_histogram": round(
            float(latest["macd_histogram"]), 4
        ),
        "macd_status": macd_status,
        "trend": trend,
    }
