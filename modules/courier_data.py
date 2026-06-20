import io

import numpy as np
import pandas as pd
import streamlit as st

# Only these columns are used anywhere in the courier/logistics analytics.
# Restricting at load time keeps memory usage low for these large
# (100-300MB+) Shiprocket shipment exports.
USEFUL_COLUMNS = {
    "Order ID", "Channel", "Status", "Master SKU", "Product Name", "Product Category",
    "Product Quantity", "Channel Created At", "Address City", "Address State",
    "Address Pincode", "Payment Method", "Order Total", "Weight (KG)",
    "Courier Company", "Master Courier", "Pickup Scheduled Date", "Order Picked Up Date",
    "Order Shipped Date", "EDD", "Delayed Reason", "Order Delivered Date",
    "RTO Initiated Date", "RTO Delivered Date", "COD Remittance Date",
    "COD Payble Amount", "Remitted Amount", "Zone", "Attempt Count", "RTO Risk",
    "RTO Reason", "Order Risk", "Address Risk", "Latest NDR Reason",
    "Cancellation Reason",
}

# Shiprocket exports occasionally ship with a feature (risk scoring) not yet
# present for older batches, in which case the column is filled with its own
# header text as a placeholder for every row instead of being left blank.
_PLACEHOLDER_RISK_VALUES = {"RTO Risk", "Order Risk", "Address Risk"}

_ZONE_LABELS = {
    "z_a": "Zone A (Local)",
    "z_b": "Zone B (Regional)",
    "z_c": "Zone C (Metro-to-Metro)",
    "z_d": "Zone D (Rest of India)",
    "z_e": "Zone E (Remote/NE/J&K)",
    "z_e2": "Zone E (Remote/NE/J&K)",
}

_COURIER_KEYWORDS = [
    ("Blue Dart", "Blue Dart"),
    ("Bluedart", "Blue Dart"),
    ("Delhivery", "Delhivery"),
    ("Xpressbees", "Xpressbees"),
    ("Shadowfax", "Shadowfax"),
    ("Amazon", "Amazon"),
    ("SRX Premium", "SRX Premium"),
    ("Ekart", "Ekart"),
    ("DTDC", "DTDC"),
    ("Ecom Express", "Ecom Express"),
    ("India Post", "India Post"),
]


@st.cache_data(show_spinner=False)
def load_courier_exports(paths_with_mtime: list[tuple[str, float]]) -> pd.DataFrame:
    """Load Shiprocket shipment/courier export CSVs directly from disk."""
    frames = []
    for path, _mtime in paths_with_mtime:
        frames.append(pd.read_csv(path, low_memory=False, usecols=lambda c: c in USEFUL_COLUMNS))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates()


@st.cache_data(show_spinner=False)
def load_and_merge_courier(file_bundle: list[tuple[str, bytes]]) -> pd.DataFrame:
    """Load any number of uploaded Shiprocket export CSVs and merge them."""
    frames = []
    for name, content in file_bundle:
        frames.append(pd.read_csv(io.BytesIO(content), low_memory=False,
                                   usecols=lambda c: c in USEFUL_COLUMNS))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.drop_duplicates()


def _parse_messy_datetime(s: pd.Series) -> pd.Series:
    """Some export batches corrupt datetime columns into bare time-duration
    fragments (e.g. '00:00.0') or huge garbage integers instead of leaving
    them blank. Genuine values always contain '/' (M/D/YYYY[ H:MM[:SS]]);
    anything else is treated as missing rather than mis-parsed."""
    plausible = s.where(s.astype(str).str.contains("/", na=False))
    return pd.to_datetime(plausible, errors="coerce")


def _clean_risk(s: pd.Series) -> pd.Series:
    return s.where(~s.isin(_PLACEHOLDER_RISK_VALUES))


def _normalize_courier(row_courier_company, row_master_courier) -> str:
    if isinstance(row_master_courier, str) and row_master_courier.strip():
        return row_master_courier.strip()
    text = str(row_courier_company)
    for keyword, label in _COURIER_KEYWORDS:
        if keyword.lower() in text.lower():
            return label
    return "Other" if text and text.lower() != "nan" else np.nan


_OUTCOME_DELIVERED = "Delivered"
_OUTCOME_RTO = "RTO"
_OUTCOME_CANCELLED = "Cancelled"
_OUTCOME_LOST = "Lost/Damaged"
_OUTCOME_IN_PROGRESS = "In Transit / Other"


def _outcome(status: str) -> str:
    s = str(status).upper().replace("_", " ").strip()
    if s == "DELIVERED":
        return _OUTCOME_DELIVERED
    if "RTO" in s:
        return _OUTCOME_RTO
    if "CANCEL" in s:
        return _OUTCOME_CANCELLED
    if s in {"LOST", "DESTROYED", "DISPOSED OFF", "UNTRACEABLE", "MISROUTED"}:
        return _OUTCOME_LOST
    return _OUTCOME_IN_PROGRESS


@st.cache_data(show_spinner=False)
def build_courier_dataset(raw: pd.DataFrame) -> dict:
    """Clean the raw Shiprocket export into a typed dataframe with derived
    delivery-performance fields, ready for charting."""
    if raw.empty:
        return {}

    df = raw.copy()

    df["Order Date"] = pd.to_datetime(df["Channel Created At"], errors="coerce")
    df = df[df["Order Date"].notna()]
    df["Order Month"] = df["Order Date"].dt.to_period("M")

    for col in ["Order Picked Up Date", "Order Delivered Date", "RTO Initiated Date",
                "RTO Delivered Date", "EDD"]:
        if col in df.columns:
            df[col] = _parse_messy_datetime(df[col].astype(str))
    # COD Remittance Date and Pickup Scheduled Date are date-only/clean in
    # every batch observed so far -- parse directly.
    for col in ["COD Remittance Date", "Pickup Scheduled Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Drop the handful of 1/1/1900 placeholder timestamps.
    df.loc[df["Order Delivered Date"] < pd.Timestamp("2023-01-01"), "Order Delivered Date"] = pd.NaT

    df["Order Total"] = pd.to_numeric(df["Order Total"], errors="coerce")
    df["Weight (KG)"] = pd.to_numeric(df["Weight (KG)"], errors="coerce")
    df["COD Payble Amount"] = pd.to_numeric(df.get("COD Payble Amount"), errors="coerce")
    df["Remitted Amount"] = pd.to_numeric(df.get("Remitted Amount"), errors="coerce")
    df["Attempt Count"] = pd.to_numeric(df.get("Attempt Count"), errors="coerce")

    df["Payment Method"] = df["Payment Method"].str.lower().map(
        lambda x: "COD" if x == "cod" else ("Prepaid" if x == "prepaid" else x)
    )

    df["Zone Label"] = df["Zone"].map(_ZONE_LABELS).fillna(df["Zone"])

    df["Courier"] = [
        _normalize_courier(cc, mc)
        for cc, mc in zip(df.get("Courier Company", pd.Series(dtype=str)),
                           df.get("Master Courier", pd.Series(dtype=str)))
    ]

    for col in ["RTO Risk", "Order Risk", "Address Risk"]:
        if col in df.columns:
            df[col] = _clean_risk(df[col])

    df["Outcome"] = df["Status"].map(_outcome)
    df["Is_Delivered"] = df["Outcome"] == _OUTCOME_DELIVERED
    df["Is_RTO"] = df["Outcome"] == _OUTCOME_RTO
    df["Is_Cancelled"] = df["Outcome"] == _OUTCOME_CANCELLED

    df["Delivery_Days"] = np.where(
        df["Is_Delivered"] & df["Order Delivered Date"].notna(),
        (df["Order Delivered Date"] - df["Order Date"]).dt.total_seconds() / 86400,
        np.nan,
    )
    df.loc[(df["Delivery_Days"] < 0) | (df["Delivery_Days"] > 60), "Delivery_Days"] = np.nan

    rto_days = (df["RTO Delivered Date"] - df["RTO Initiated Date"]).dt.total_seconds() / 86400
    df["RTO_Resolution_Days"] = rto_days.where((rto_days >= 0) & (rto_days <= 90))

    remit_lag = (df["COD Remittance Date"] - df["Order Delivered Date"]).dt.total_seconds() / 86400
    df["COD_Remittance_Lag_Days"] = remit_lag.where((remit_lag >= 0) & (remit_lag <= 120))

    overall_peak_month = df["Order Month"].max()
    earliest_month = df["Order Month"].min()

    return {
        "shipments": df,
        "date_range": (earliest_month, overall_peak_month),
        "edd_usable": False,  # EDD column is corrupted in nearly all rows across all batches observed.
    }
