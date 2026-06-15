import json
import os

import streamlit as st

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a sharp e-commerce data analyst writing for a D2C brand's founder. "
    "You will be given a JSON object of pre-computed metrics from their Shopify order data. "
    "Write a concise executive summary in markdown:\n"
    "- 3-5 bullet points on what's going well\n"
    "- 3-5 bullet points on the biggest risks/problems (be specific with numbers)\n"
    "- 2-4 concrete, prioritized action items\n"
    "Be direct and numbers-driven. Do not restate every metric, only the most important ones. "
    "Keep the whole response under 300 words."
)


def get_api_key() -> str | None:
    ANTHROPIC_API_KEY="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"
    return os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)


@st.cache_data(show_spinner=False)
def generate_summary(metrics: dict, api_key_present: bool) -> str:
    """Calls Claude (Haiku) with a compact metrics JSON to produce an
    executive summary. Cached on the metrics dict to avoid repeat spend."""
    api_key = get_api_key()
    if not api_key:
        return (
            "**AI summary unavailable** — set the `ANTHROPIC_API_KEY` environment "
            "variable and reload the app to enable Claude-generated insights."
        )

    try:
        import anthropic
    except ImportError:
        return "**AI summary unavailable** — the `anthropic` package is not installed."

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Here are the metrics:\n```json\n{json.dumps(metrics, default=str, indent=2)}\n```",
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"**AI summary failed**: {e}"
