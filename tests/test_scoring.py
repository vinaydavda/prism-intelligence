import pytest

from app.scoring import (
    clamp,
    growth_score,
    weighted_average,
    calculate_fundamental_score,
    calculate_confidence,
)


def test_clamp():
    assert clamp(120) == 100
    assert clamp(-10) == 0
    assert clamp(65) == 65


def test_growth_score():
    assert growth_score(0) == 50
    assert growth_score(10) == 75
    assert growth_score(20) == 100
    assert growth_score(-10) == 25
    assert growth_score(None) is None


def test_weighted_average():
    result = weighted_average({
        "revenue": (80, 0.5),
        "profit": (60, 0.5),
    })

    assert result == 70


def test_weighted_average_ignores_missing_scores():
    result = weighted_average({
        "revenue": (80, 0.5),
        "profit": (None, 0.5),
    })

    assert result == 80


def test_weighted_average_with_no_scores():
    assert weighted_average({
        "revenue": (None, 0.5),
        "profit": (None, 0.5),
    }) is None


def test_fundamental_score():
    data = {
        "revenue_growth_pct": 10,
        "profit_growth_pct": 20,
        "operating_margin_pct": 7.5,
    }

    score = calculate_fundamental_score(data)

    assert score == 76.25


def test_confidence():
    assert calculate_confidence(3) == "HIGH"
    assert calculate_confidence(2) == "MEDIUM"
    assert calculate_confidence(1) == "LOW"
