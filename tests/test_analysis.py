
import pandas as pd
import pytest

from app.fundamental import analyze_fundamentals
from app.technical import analyze_technical
from app.alternative import analyze_alternative


def test_fundamental_rejects_empty_data():
    with pytest.raises(ValueError):
        analyze_fundamentals(pd.DataFrame())


def test_fundamental_rejects_missing_columns():
    data = pd.DataFrame({
        "period": ["2026-Q1"],
        "revenue_cr": [5000],
    })

    with pytest.raises(ValueError):
        analyze_fundamentals(data)


def test_technical_rejects_insufficient_data():
    data = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20),
        "close": [250.0] * 20,
    })

    with pytest.raises(ValueError):
        analyze_technical(data)


def test_technical_rejects_empty_data():
    with pytest.raises(ValueError):
        analyze_technical(pd.DataFrame())


def test_alternative_rejects_empty_data():
    with pytest.raises(ValueError):
        analyze_alternative(pd.DataFrame())


def test_alternative_rejects_negative_values():
    data = pd.DataFrame({
        "date": ["2026-11-01"],
        "symbol": ["ZOMATO"],
        "metric": ["app_downloads"],
        "value": [-100],
        "unit": ["count"],
        "source": ["synthetic"],
    })

    with pytest.raises(ValueError):
        analyze_alternative(data)
