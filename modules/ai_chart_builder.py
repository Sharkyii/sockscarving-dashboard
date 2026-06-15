import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ai_insights import get_api_key

MODEL = "claude-haiku-4-5"

DATAFRAME_DOCS = {
    "orders": "one row per order (all statuses) — order/date/geo/payment/discount/RTO/cancellation fields",
    "paid": "subset of `orders` where Financial Status == 'paid' and Paid at is known — adds Order Seq, Is New Customer Order, Paid Month",
    "li": (
        "one row per line item, paid orders only — adds Category, Product Family, Style Tags, Revenue (qty * price). "
        "Pack Size is the pack size printed on that SKU (often 1 for 'Pack of 1-Pair' items that are actually sold "
        "as part of a multi-buy bundle). Effective Pack Size rolls up all 'Pack of 1-Pair' items bought in the same "
        "order into the real bundle size (e.g. 6 separate 1-pair items in one order = Effective Pack Size 6) — "
        "prefer Effective Pack Size for any 'how many pairs do customers buy at once' question"
    ),
    "li_all": "one row per line item, all orders — same columns as `li` but includes unpaid/RTO orders",
    "customers": "one row per customer (Billing Name + Zip key) — Orders, Total_Spend, AOV, Repeat_Buyer, Tier",
}

CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|```\s*$", re.MULTILINE)
IMPORT_LINE_RE = re.compile(r"^[ \t]*(?:import\s+.+|from\s+\S+\s+import\s+.+)$\n?", re.MULTILINE)

SAFE_BUILTINS = {
    name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
        "list", "max", "min", "range", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "isinstance", "type", "print",
    ]
}


def _describe_df(df: pd.DataFrame, name: str, max_cols: int = 35) -> str:
    if df.empty:
        return f"- `{name}`: EMPTY (not available in the uploaded data)"
    lines = [f"- `{name}` ({DATAFRAME_DOCS.get(name, '')}): {len(df):,} rows"]
    for col in list(df.columns)[:max_cols]:
        dtype = str(df[col].dtype)
        if dtype in ("object", "category") and df[col].nunique(dropna=True) <= 12:
            sample = [str(v) for v in df[col].dropna().unique()[:6]]
            lines.append(f"    - `{col}` ({dtype}), values e.g. {sample}")
        else:
            lines.append(f"    - `{col}` ({dtype})")
    return "\n".join(lines)


def build_schema_context(data: dict) -> str:
    parts = [_describe_df(data[name], name) for name in ["orders", "paid", "li", "li_all", "customers"]]
    return "\n".join(parts)


def _system_prompt(schema_context: str) -> str:
    return (
        "You are a Python data-visualization assistant for a Streamlit dashboard analyzing "
        "Shopify order data for a D2C socks brand. The following pandas DataFrames are already "
        "loaded in scope:\n\n"
        f"{schema_context}\n\n"
        "Given the user's request, write Python code that produces ONE of:\n"
        "- a Plotly figure assigned to a variable named `fig` (use `px` or `go`), or\n"
        "- a small pandas DataFrame/Series assigned to a variable named `result_df` (for tables), or both.\n\n"
        "Rules:\n"
        "- Only use the provided DataFrames (`orders`, `paid`, `li`, `li_all`, `customers`), `pd`, `np`, `px`, `go`.\n"
        "- Do not import anything, read/write files, use exec/eval, or access the network.\n"
        "- Always filter/groupby defensively (drop NaNs where relevant) and limit large result tables to top ~20 rows.\n"
        "- `Paid at` and `Created at` are already timezone-aware pandas datetime columns -- use `.dt` accessors "
        "directly on them (e.g. `.dt.month_name()`, `.dt.to_period('M')`). Never pass them through `pd.to_datetime(...)` "
        "again, and never call `pd.to_datetime` on a non-date column (e.g. `Fulfillment Status`, `Financial Status` "
        "are plain category strings, not dates).\n"
        "- For 'by month across all years combined' / seasonality requests, group by `.dt.month` or "
        "`.dt.month_name()` on `Paid at` (not `.dt.to_period('M')`, which keeps years separate).\n"
        "- Output ONLY raw Python code. No markdown fences, no explanation, no comments about what you're doing. "
        "Keep the code compact enough to fit well within the token limit -- avoid overly verbose `go.Figure` "
        "styling when `px` would do."
    )


@st.cache_data(show_spinner=False)
def generate_chart_code(prompt: str, schema_context: str, api_key_present: bool) -> str:
    api_key = get_api_key()
    if not api_key:
        return ""

    try:
        import anthropic
    except ImportError:
        return ""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1536,
        system=_system_prompt(schema_context),
        messages=[{"role": "user", "content": prompt}],
    )
    code = response.content[0].text
    code = CODE_FENCE_RE.sub("", code).strip()
    # The model is told `pd`/`np`/`px`/`go` are already in scope and not to
    # import anything, but it occasionally adds redundant `import ...` lines
    # anyway. The execution sandbox has no `__import__` builtin, so any such
    # line raises ImportError and kills the whole script -- strip them.
    return IMPORT_LINE_RE.sub("", code).strip()


def execute_chart_code(code: str, data: dict):
    """Runs AI-generated code in a restricted namespace. Returns
    (fig_or_None, result_df_or_None, error_or_None)."""
    exec_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "orders": data["orders"],
        "paid": data["paid"],
        "li": data["li"],
        "li_all": data["li_all"],
        "customers": data["customers"],
    }
    exec_locals: dict = {}
    try:
        exec(code, exec_globals, exec_locals)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    fig = exec_locals.get("fig", exec_globals.get("fig"))
    result_df = exec_locals.get("result_df", exec_globals.get("result_df"))

    if fig is None and result_df is None:
        return None, None, "Code ran but did not produce `fig` or `result_df`."

    return fig, result_df, None
