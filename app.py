import os

import pandas as pd
import streamlit as st

from modules import theme, data_processing as dp, ai_insights, ai_chart_builder as acb, auth
from modules import charts_orders, charts_delivery, charts_products, charts_customers, charts_financial, charts_advanced

st.set_page_config(
    page_title="SocksCarving Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.register_plotly_theme()
theme.inject_css()

if not auth.check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar - data input
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Data Source")

uploaded_files = st.sidebar.file_uploader(
    "Upload Shopify order export(s) — CSV or XLSX, any number of files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Or load local files** (for testing)")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
candidate_files = [
    f for f in ["orders.csv", "customer_data.csv"]
    if os.path.exists(os.path.join(PROJECT_ROOT, f))
]
selected_local = st.sidebar.multiselect(
    "Local files in project root",
    options=candidate_files,
    default=candidate_files,
)

st.sidebar.markdown("---")
api_key_present = bool(ai_insights.get_api_key())
if api_key_present:
    st.sidebar.success("Claude API key detected — AI summaries enabled.")
else:
    st.sidebar.warning("Set `ANTHROPIC_API_KEY` env var to enable AI summaries.")

# ---------------------------------------------------------------------------
# Load & process data
# ---------------------------------------------------------------------------

raw = pd.DataFrame()

if uploaded_files:
    bundle = [(f.name, f.getvalue()) for f in uploaded_files]
    with st.spinner("Loading and merging uploaded files..."):
        raw = dp.load_and_merge(bundle)
elif selected_local:
    paths = [os.path.join(PROJECT_ROOT, f) for f in selected_local]
    paths_with_mtime = [(p, os.path.getmtime(p)) for p in paths]
    with st.spinner("Loading local files..."):
        raw = dp.load_local_files(paths_with_mtime)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">SocksCarving Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload your order exports to get a full breakdown across orders, '
    'delivery, products, customers, and finance — with AI-generated insights.</div>',
    unsafe_allow_html=True,
)

if raw.empty:
    st.markdown(
        '<div class="section-card">'
        'Upload one or more Shopify order export files (CSV/XLSX) in the sidebar, '
        'or pick a local file to get started. Multiple files are merged and deduplicated automatically.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Crunching the numbers..."):
    data = dp.build_datasets(raw)

st.caption(f"Loaded **{len(raw):,}** rows from **{len(uploaded_files) if uploaded_files else len(selected_local)}** file(s) "
           f"→ **{len(data['orders']):,}** unique orders.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_orders, tab_delivery, tab_products, tab_customers, tab_financial, tab_advanced, tab_ai = st.tabs(
    ["Overview", "Orders", "Delivery", "Products", "Customers", "Financial & Marketing", "Advanced Analytics", "Ask AI for a Chart"]
)

with tab_overview:
    metrics = dp.build_overview_metrics(data)

    orders, paid = data["orders"], data["paid"]
    cols = st.columns(4)
    kpi_defs = []
    if "total_orders" in metrics:
        kpi_defs.append(("Total Orders", f"{metrics['total_orders']:,}"))
    if "paid_orders" in metrics:
        kpi_defs.append(("Paid Orders", f"{metrics['paid_orders']:,}"))
    if "total_revenue" in metrics:
        kpi_defs.append(("Total Revenue", f"₹{metrics['total_revenue']:,.0f}"))
    if "avg_order_value" in metrics:
        kpi_defs.append(("Avg Order Value", f"₹{metrics['avg_order_value']:,.0f}"))
    if "delivery_rate_pct" in metrics:
        kpi_defs.append(("Delivery Rate", f"{metrics['delivery_rate_pct']:.1f}%"))
    if "rto_rate_pct" in metrics:
        kpi_defs.append(("RTO Rate", f"{metrics['rto_rate_pct']:.1f}%"))
    if "unique_customers" in metrics:
        kpi_defs.append(("Unique Customers", f"{metrics['unique_customers']:,}"))
    if "repeat_customer_rate_pct" in metrics:
        kpi_defs.append(("Repeat Customer Rate", f"{metrics['repeat_customer_rate_pct']:.1f}%"))

    for i, (label, value) in enumerate(kpi_defs):
        with cols[i % 4]:
            theme.kpi_card(label, value)
        if i % 4 == 3 and i != len(kpi_defs) - 1:
            cols = st.columns(4)

    st.markdown("###")
    st.subheader("AI Executive Summary")
    st.caption(f"Generated by `{ai_insights.MODEL}` from the metrics above.")

    if st.button("Generate AI Summary", type="primary", disabled=not api_key_present):
        with st.spinner("Asking Claude..."):
            summary = ai_insights.generate_summary(metrics, api_key_present)
        st.session_state["ai_summary"] = summary

    if "ai_summary" in st.session_state:
        st.markdown(f'<div class="ai-box">{st.session_state["ai_summary"]}</div>', unsafe_allow_html=True)
    elif not api_key_present:
        st.info("Set the `ANTHROPIC_API_KEY` environment variable to enable AI-generated summaries (uses Claude Haiku — low cost).")

with tab_orders:
    charts_orders.render(data)

with tab_delivery:
    charts_delivery.render(data)

with tab_products:
    charts_products.render(data)

with tab_customers:
    charts_customers.render(data)

with tab_financial:
    charts_financial.render(data)

with tab_advanced:
    charts_advanced.render(data)

with tab_ai:
    st.subheader("Ask AI for a Chart")
    st.caption(
        f"Describe the chart or table you want in plain English. `{acb.MODEL}` will write the "
        "Plotly/pandas code against your loaded data and add it to this page."
    )

    if not api_key_present:
        st.info("Set the `ANTHROPIC_API_KEY` environment variable to enable this feature.")
    else:
        with st.form("ai_chart_form", clear_on_submit=True):
            user_prompt = st.text_area(
                "What do you want to see?",
                placeholder="e.g. 'Show me a pie chart of orders by Payment Type' or "
                "'Bar chart of total revenue per Shipping Province, top 10'",
                height=80,
            )
            submitted = st.form_submit_button("Generate Chart", type="primary")

        if submitted and user_prompt.strip():
            schema_context = acb.build_schema_context(data)
            with st.spinner("Asking Claude to write the chart..."):
                code = acb.generate_chart_code(user_prompt.strip(), schema_context, api_key_present)

            entry = {"prompt": user_prompt.strip(), "code": code}
            if not code:
                entry["error"] = "No code returned by the AI."
            else:
                fig, result_df, error = acb.execute_chart_code(code, data)
                entry["fig"] = fig
                entry["result_df"] = result_df
                entry["error"] = error

            st.session_state.setdefault("ai_charts", []).append(entry)

    ai_charts = st.session_state.get("ai_charts", [])
    if not ai_charts:
        if api_key_present:
            st.markdown(
                '<div class="section-card">No custom charts yet — describe what you want above and '
                'click "Generate Chart". Each chart you create will be added below and stick around '
                'while you explore the dashboard.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("###")
        for i, entry in enumerate(reversed(ai_charts)):
            real_idx = len(ai_charts) - 1 - i
            st.markdown(f"**\"{entry['prompt']}\"**")
            if entry.get("error"):
                st.error(entry["error"])
            if entry.get("fig") is not None:
                st.plotly_chart(entry["fig"], width="stretch", key=f"ai_chart_{real_idx}")
            if entry.get("result_df") is not None:
                st.dataframe(entry["result_df"], width="stretch")
            with st.expander("Show generated code"):
                st.code(entry["code"], language="python")
            if st.button("Remove", key=f"ai_chart_remove_{real_idx}"):
                ai_charts.pop(real_idx)
                st.rerun()
            st.markdown("---")
