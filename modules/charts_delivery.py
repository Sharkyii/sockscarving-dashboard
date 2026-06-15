import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.data_processing import RTO_TRACKING_START
from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, DANGER, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "delivery.ipynb")


def render(data: dict):
    orders = data["orders"]
    if orders.empty or "Is Shipped" not in orders.columns:
        st.info("Upload order export files to see delivery analytics.")
        return

    shipped = orders[orders["Is Shipped"]]
    if shipped.empty:
        st.info("No shipped orders found (Fulfillment Status == 'fulfilled') in the uploaded data.")
        return

    run_notebook(NOTEBOOK, {
        "data": data, "st": st, "px": px, "go": go, "pd": pd,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "DANGER": DANGER, "PALETTE": PALETTE,
        "RTO_TRACKING_START": RTO_TRACKING_START,
    })
