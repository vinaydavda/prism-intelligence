
from typing import Optional


# Overall component weights
COMPONENT_WEIGHTS = {
    "fundamental": 0.35,
    "technical": 0.35,
    "alternative": 0.30,
}

margin_benchmark = 15
growth_multiplier = 2.5
rsi_multiplier = 1.5


def clamp(value: float, minimum=0, maximum=100) -> float:
    """Keep a score within the 0-100 range."""

    return max(minimum, min(value, maximum))


def growth_score(growth: Optional[float]) -> Optional[float]:
    """Convert percentage growth into a 0-100 score."""

    if growth is None:
        return None

    # Note: Here growth multiplier (2.5) controls how quickly the score changes with growth.
    return round(clamp(50 + (growth * growth_multiplier)), 2)


def weighted_average(metrics: dict) -> Optional[float]:
    """Calculate weighted average, ignoring unavailable metrics."""

    available = {
        name: (score, weight)
        for name, (score, weight) in metrics.items()
        if score is not None
    }

    if not available:
        return None

    total_weight = sum(weight for _, weight in available.values())

    return round(
        sum(score * weight for score, weight in available.values())
        / total_weight,
        2
    )


def calculate_fundamental_score(data: dict) -> Optional[float]:
    """Calculate fundamental intelligence score."""

    margin = data.get("operating_margin_pct")

    # Note: We selected 15% as our assumed benchmark for a maximum score.
    margin_score = (
        clamp(margin / margin_benchmark * 100)
        if margin is not None
        else None
    )

    return weighted_average({
        "revenue": (growth_score(data.get("revenue_growth_pct")), 0.35),
        "profit": (growth_score(data.get("profit_growth_pct")), 0.35),
        "margin": (margin_score, 0.30),
    })


def calculate_technical_score(data: dict) -> Optional[float]:
    """Calculate technical intelligence score."""

    trend = data.get("trend")
    rsi = data.get("rsi_14")
    macd_status = data.get("macd_status")

    trend_scores = {
        "BULLISH": 100,
        "NEUTRAL": 50,
        "BEARISH": 0,
    }

    trend_score = trend_scores.get(trend)

    rsi_score = (
        clamp(100 - abs(rsi - 55) * rsi_multiplier)
        if rsi is not None
        else None
    )

    macd_scores = {
        "BULLISH": 100,
        "BEARISH": 0,
    }

    macd_score = macd_scores.get(macd_status)

    return weighted_average({
        "trend": (trend_score, 0.40),
        "rsi": (rsi_score, 0.30),
        "macd": (macd_score, 0.30),
    })


def calculate_alternative_score(data: dict) -> Optional[float]:
    """Calculate alternative intelligence score."""

    signals = {
        item["metric"]: growth_score(item.get("growth_pct"))
        for item in data.get("signals", [])
    }

    return weighted_average({
        "app_downloads": (signals.get("app_downloads"), 0.40),
        "web_traffic": (signals.get("web_traffic"), 0.40),
        "job_postings": (signals.get("job_postings"), 0.20),
    })


def calculate_confidence(available_components: int) -> str:
    """Determine confidence based on data availability."""

    if available_components == 3:
        return "HIGH"

    if available_components == 2:
        return "MEDIUM"

    return "LOW"


def calculate_composite_score(
    fundamental: dict,
    technical: dict,
    alternative: dict,
) -> dict:
    """Combine all intelligence scores."""

    component_scores = {
        "fundamental": calculate_fundamental_score(fundamental),
        "technical": calculate_technical_score(technical),
        "alternative": calculate_alternative_score(alternative),
    }

    available = {
        name: score
        for name, score in component_scores.items()
        if score is not None
    }

    if not available:
        raise ValueError("No valid analysis scores available")

    # Renormalize weights when a component is unavailable
    total_weight = sum(
        COMPONENT_WEIGHTS[name]
        for name in available
    )

    overall_score = sum(
        score * COMPONENT_WEIGHTS[name] / total_weight
        for name, score in available.items()
    )

    overall_score = round(overall_score, 2)

    if overall_score >= 65:
        signal = "POSITIVE"
    elif overall_score <= 35:
        signal = "NEGATIVE"
    else:
        signal = "NEUTRAL"

    return {
        "overall_score": overall_score,
        "signal": signal,
        "confidence": calculate_confidence(len(available)),
        "component_scores": component_scores,
        "available_components": list(available.keys()),
        "weights_used": {
            name: round(
                COMPONENT_WEIGHTS[name] / total_weight,
                4
            )
            for name in available
        },
    }
