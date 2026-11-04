
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")


def get_openai_client():
    """Create an OpenAI client."""

    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    return OpenAI(api_key=API_KEY)


SYSTEM_PROMPT = """
You are Prism's financial market intelligence analyst.

Your job is to explain structured financial and market analysis
provided by the Python quantitative engine.

Rules:

1. Use only the data provided in the analysis context.
2. Never invent financial metrics, market prices or facts.
3. Clearly distinguish observed data from interpretation.
4. Explain why the overall intelligence score was generated.
5. Identify positive factors and negative factors.
6. Mention missing data and limitations.
7. Alternative data marked as synthetic must be identified
   as synthetic, not real company information.
8. Do not present analysis as guaranteed investment advice.
9. If the data is insufficient, explicitly say so.
10. Keep the explanation clear and concise.

Return a structured JSON object with these fields:

{
    "direct_answer": "Direct answer to the user's question",
    "recommendation": "BUY / SELL / HOLD / INSUFFICIENT_DATA",
    "recommendation_reasoning": "Explanation",
    "executive_summary": "Short overall explanation",
    "fundamental_analysis": "Explanation of financial metrics",
    "technical_analysis": "Explanation of technical indicators",
    "alternative_analysis": "Explanation of alternative signals",
    "positive_factors": ["factor 1"],
    "negative_factors": ["factor 1"],
    "risks": ["risk 1"],
    "data_limitations": ["limitation 1"]
}

Return valid JSON only. Do not use Markdown code fences.
"""


def generate_analysis_report(
    question: str,
    analysis_data: dict
) -> dict:
    """
    Generate an LLM-based financial intelligence report.

    Args:
        question: User's natural language question.
        analysis_data: Results from Python analysis modules.

    Returns:
        Dictionary containing the LLM-generated report.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    client = get_openai_client()

    context = json.dumps(
        analysis_data,
        indent=2,
        default=str
    )

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=f"""
        ANALYSIS DATA:

        {context}

        USER QUESTION:

        {question}
        """
    )

    answer = response.output_text.strip()

    if not answer:
        raise ValueError("LLM returned an empty response")

    try:
        report = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid JSON"
        ) from exc

    if not isinstance(report, dict):
        raise ValueError("LLM response must be a JSON object")

    required_fields = {
        "executive_summary",
        "fundamental_analysis",
        "technical_analysis",
        "alternative_analysis",
        "positive_factors",
        "negative_factors",
        "risks",
        "data_limitations",
    }

    missing_fields = required_fields - set(report)

    if missing_fields:
        raise ValueError(
            f"LLM response missing fields: {sorted(missing_fields)}"
        )

    return {
        "model": MODEL,
        "question": question,
        "report": report,
    }
