import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, DANGER, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "financial.ipynb")


def render(data: dict):
    orders = data["orders"]
    if orders.empty or "Total" not in orders.columns:
        st.info("Upload order export files with financial columns ('Total', 'Discount Amount', 'Payment Method', etc.) to see financial analytics.")
        return

    run_notebook(NOTEBOOK, {
        "data": data, "st": st, "px": px, "go": go, "pd": pd,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "DANGER": DANGER, "PALETTE": PALETTE,
    })
