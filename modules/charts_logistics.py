import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.notebook_runner import run_notebook
from modules.theme import kpi_card, ACCENT1, ACCENT2, SUCCESS, DANGER, WARNING, PALETTE

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "ipynb", "logistics.ipynb")


def render(courier_data: dict):
    if not courier_data:
        st.info(
            "Upload Shiprocket shipment export(s) (or select files from the local `export/` folder) "
            "in the sidebar under 'Courier / Shipment Data' to see courier performance, RTO root "
            "causes, NDR analysis, and delivery SLA timing."
        )
        return

    run_notebook(NOTEBOOK, {
        "courier_data": courier_data, "st": st, "px": px, "go": go, "pd": pd,
        "kpi_card": kpi_card, "ACCENT1": ACCENT1, "ACCENT2": ACCENT2,
        "SUCCESS": SUCCESS, "DANGER": DANGER, "WARNING": WARNING, "PALETTE": PALETTE,
    })
