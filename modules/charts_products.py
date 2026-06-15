import os

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import RTO_TRACKING_START
from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, DANGER, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "products.ipynb")


def render(data: dict):
    li = data["li"]
    if li.empty:
        st.info("Upload order export files with line-item columns ('Lineitem name', 'Lineitem quantity', 'Lineitem price') to see product analytics.")
        return

    run_notebook(NOTEBOOK, {
        "data": data, "st": st, "px": px, "pd": pd,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "DANGER": DANGER, "PALETTE": PALETTE,
        "RTO_TRACKING_START": RTO_TRACKING_START,
    })
