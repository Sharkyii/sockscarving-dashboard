import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "customers.ipynb")


def render(data: dict):
    paid = data["paid"]
    if paid.empty or "Customer Key" not in paid.columns:
        st.info("Upload order export files with 'Billing Name'/'Billing Zip' and paid orders to see customer analytics.")
        return

    run_notebook(NOTEBOOK, {
        "data": data, "st": st, "px": px, "go": go, "pd": pd,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "PALETTE": PALETTE,
    })
