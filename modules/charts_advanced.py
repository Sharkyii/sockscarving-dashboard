import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import advanced_analytics as aa
from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, DANGER, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "advanced.ipynb")


def render(data: dict):
    if data["paid"].empty or data["customers"].empty:
        st.info("Upload paid order data to see advanced analytics (RFM, churn, forecasting, CLV).")
        return

    run_notebook(NOTEBOOK, {
        "data": data, "st": st, "px": px, "go": go, "pd": pd, "aa": aa,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "DANGER": DANGER, "PALETTE": PALETTE,
    })
