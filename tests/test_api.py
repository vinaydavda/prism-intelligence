import app.main as main

from fastapi.testclient import TestClient

client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_endpoint(monkeypatch):
    fake_analysis = {
        "symbol": "ZOMATO",
        "scoring": {
            "overall_score": 78.39,
            "signal": "POSITIVE",
        },
    }

    monkeypatch.setattr(
        main,
        "run_company_analysis",
        lambda symbol: fake_analysis,
    )

    response = client.get("/analyze/ZOMATO")

    assert response.status_code == 200
    assert response.json()["symbol"] == "ZOMATO"
    assert response.json()["scoring"]["overall_score"] == 78.39


def test_analyze_endpoint_handles_invalid_data(monkeypatch):
    def raise_error(symbol):
        raise ValueError("Financial data is empty")

    monkeypatch.setattr(
        main,
        "run_company_analysis",
        raise_error,
    )

    response = client.get("/analyze/ZOMATO")

    assert response.status_code == 400
    assert response.json()["detail"] == "Financial data is empty"


def test_ask_endpoint(monkeypatch):
    fake_analysis = {
        "symbol": "ZOMATO",
        "scoring": {"overall_score": 78.39},
    }

    fake_report = {
        "model": "gpt-4.1-mini",
        "question": "Why is the score positive?",
        "report": {
            "direct_answer": "The score reflects positive signals.",
            "recommendation": "NOT_APPLICABLE",
        },
    }

    monkeypatch.setattr(
        main,
        "run_company_analysis",
        lambda symbol: fake_analysis,
    )

    monkeypatch.setattr(
        main,
        "generate_analysis_report",
        lambda question, analysis_data: fake_report,
    )

    response = client.post(
        "/ask",
        json={
            "symbol": "ZOMATO",
            "question": "Why is the score positive?",
        },
    )

    assert response.status_code == 200
    assert response.json()["report"]["direct_answer"] == (
        "The score reflects positive signals."
    )


def test_ask_endpoint_validates_question():
    response = client.post(
        "/ask",
        json={
            "symbol": "ZOMATO",
            "question": "Hi",
        },
    )

    assert response.status_code == 422
