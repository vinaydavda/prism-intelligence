
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data_service import get_company_data
from app.fundamental import analyze_fundamentals
from app.technical import analyze_technical
from app.alternative import analyze_alternative

from app.scoring import calculate_composite_score
from app.llm_analyst import generate_analysis_report


app = FastAPI(
    title="Prism Intelligence",
    description="Financial intelligence API powered by quantitative analysis and LLMs.",
    version="1.0.0",
)


# Request model for the /ask endpoint
class AskRequest(BaseModel):
    symbol: str = Field(default="ZOMATO", min_length=1)
    question: str = Field(min_length=3, max_length=1000)


def run_company_analysis(symbol: str) -> dict:
    """
    Runs the complete quantitative analysis pipeline
    for a company and calculates its intelligence score.
    """

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Company symbol cannot be empty")

    # 1. Load datasets
    company_data = get_company_data(symbol)

    # 2. Fundamental analysis
    fundamental = analyze_fundamentals(
        company_data["financials"]
    )

    # 3. Technical analysis
    technical = analyze_technical(
        company_data["prices"]
    )

    # 4. Alternative data analysis
    alternative = analyze_alternative(
        company_data["alternative"]
    )

    # 5. Calculate composite intelligence score
    scoring = calculate_composite_score(
        fundamental,
        technical,
        alternative,
    )

    # 6. Prepare structured analysis context
    analysis_data = {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_type": "SYNTHETIC DEMONSTRATION DATA",
        "fundamental": fundamental,
        "technical": technical,
        "alternative": alternative,
        "scoring": scoring,
    }

    return analysis_data


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Prism Intelligence",
    }


@app.get("/analyze/{symbol}")
def analyze_company(symbol: str):
    """
    Returns quantitative analysis without calling the LLM.
    """

    try:
        return run_company_analysis(symbol)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Runs quantitative analysis and asks the LLM
    to answer the user's question.
    """

    try:
        # Run quantitative analysis
        analysis_data = run_company_analysis(request.symbol)

        # Generate LLM response
        result = generate_analysis_report(
            question=request.question,
            analysis_data=analysis_data,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM analysis failed: {str(error)}",
        )
