import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- Shared color palette (dark, minimal, professional) ----
BG = "#0B0E14"
CARD_BG = "#12161D"
CARD_BORDER = "#232833"
ACCENT1 = "#5B8DEF"   # muted blue (primary)
ACCENT2 = "#7C93B8"   # slate blue-gray (secondary)
SUCCESS = "#3FB97D"   # muted green
WARNING = "#C9A14A"   # muted amber
DANGER = "#D9695F"    # muted red
TEXT = "#E6E8EC"
MUTED = "#878E9C"

# Restrained, monochrome-leaning categorical palette
PALETTE = [ACCENT1, ACCENT2, "#A7B6CC", SUCCESS, MUTED, "#3D4A63"]


def register_plotly_theme():
    template = go.layout.Template(pio.templates["plotly_dark"])
    template.layout.colorway = PALETTE
    template.layout.paper_bgcolor = CARD_BG
    template.layout.plot_bgcolor = CARD_BG
    template.layout.font = dict(color=TEXT, family="Inter, sans-serif")
    template.layout.legend = dict(bgcolor="rgba(0,0,0,0)")
    template.layout.margin = dict(l=40, r=20, t=50, b=40)
    template.layout.xaxis = dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER)
    template.layout.yaxis = dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER)
    pio.templates["socks_dark"] = template
    pio.templates.default = "socks_dark"


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid {CARD_BORDER};
        }}
        h1, h2, h3, h4 {{
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.01em;
            color: {TEXT};
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 700;
            color: {TEXT};
            margin-bottom: 0;
        }}
        .hero-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }}
        .kpi-card {{
            background-color: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-left: 3px solid {ACCENT1};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            height: 100%;
        }}
        .kpi-label {{
            color: {MUTED};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 650;
            color: {TEXT};
        }}
        .kpi-delta-pos {{ color: {SUCCESS}; font-size: 0.85rem; font-weight: 600; }}
        .kpi-delta-neg {{ color: {DANGER}; font-size: 0.85rem; font-weight: 600; }}
        .section-card {{
            background-color: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }}
        .ai-box {{
            background-color: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-left: 3px solid {ACCENT1};
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            line-height: 1.55;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {CARD_BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 6px 6px 0 0;
            padding: 8px 18px;
            color: {MUTED};
        }}
        .stTabs [aria-selected="true"] {{
            color: {TEXT};
            border-bottom: 2px solid {ACCENT1};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, positive: bool = True):
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
