import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

TOP_PROVINCES_N = 8
TOP_CATEGORIES_N = 8
CHURN_WINDOW_DAYS = 180


# ---------------------------------------------------------------------------
# RFM Segmentation
# ---------------------------------------------------------------------------

def _rfm_segment(row) -> str:
    # 94%+ of customers here are one-time buyers, so F_Score (clipped to 5)
    # is ~1 for almost everyone -- a quintile-based F score has no spread.
    # Segment on raw Frequency (1 / 2 / 3+) crossed with Recency and Monetary
    # instead, so the buckets actually reflect this brand's behavior.
    f, r, m = row["Frequency"], row["R_Score"], row["M_Score"]
    if f >= 3:
        return "Champions" if r >= 3 else "At-Risk Champions"
    if f == 2:
        return "Loyal Customers" if r >= 3 else "At-Risk Repeat Buyers"
    # f == 1
    if r >= 4:
        return "Big Spenders (One-Time)" if m >= 4 else "New Customers"
    if r == 3:
        return "Promising"
    if r == 2:
        return "Fading One-Timers"
    return "Lost"


@st.cache_data(show_spinner=False)
def compute_rfm(data: dict) -> pd.DataFrame:
    paid = data["paid"]
    if paid.empty or "Customer Key" not in paid.columns:
        return pd.DataFrame()

    snapshot = paid["Paid at"].max() + pd.Timedelta(days=1)
    rfm = paid.groupby("Customer Key").agg(
        Recency=("Paid at", lambda x: (snapshot - x.max()).days),
        Frequency=("Name", "nunique"),
        Monetary=("Total", "sum"),
    ).reset_index()

    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["F_Score"] = rfm["Frequency"].clip(upper=5)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["Segment"] = rfm.apply(_rfm_segment, axis=1)
    return rfm


# ---------------------------------------------------------------------------
# Churn / Win-back Risk Model
# ---------------------------------------------------------------------------

def _churn_features(hist: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    feat = hist.groupby("Customer Key").agg(
        Recency=("Paid at", lambda x: (ref_date - x.max()).days),
        Frequency=("Name", "nunique"),
        Monetary=("Total", "sum"),
        Tenure=("Paid at", lambda x: (ref_date - x.min()).days),
        Discount_Rate=("Has Discount", "mean"),
        RTO_Rate=("Any RTO", "mean"),
        Province=("Shipping Province", "first"),
    ).reset_index()
    feat["AOV"] = feat["Monetary"] / feat["Frequency"]
    cod = (hist["Payment Type"] == "COD").groupby(hist["Customer Key"]).mean()
    feat = feat.merge(cod.rename("COD_Rate"), on="Customer Key", how="left")
    return feat


def _product_features(li: pd.DataFrame, cutoff: pd.Timestamp = None) -> pd.DataFrame:
    """Per-customer product-mix and seasonality features derived from line items."""
    if li.empty:
        return pd.DataFrame(columns=["Customer Key", "Avg_Item_Price", "Avg_Pack_Size",
                                      "Last_Purchase_Month", "Primary_Category"])

    li_hist = li.copy()
    li_hist["Paid at"] = pd.to_datetime(li_hist["Paid at"], errors="coerce", utc=True)
    if cutoff is not None:
        li_hist = li_hist[li_hist["Paid at"] <= cutoff]
    li_hist["Customer Key"] = (
        li_hist["Billing Name"].astype(str).str.strip().str.lower()
        + " | " + li_hist["Billing Zip"].astype(str).str.strip()
    )

    pack_size_col = "Effective Pack Size" if "Effective Pack Size" in li_hist.columns else "Pack Size"
    agg = li_hist.groupby("Customer Key").agg(
        Avg_Item_Price=("Lineitem price", "mean"),
        Avg_Pack_Size=(pack_size_col, "mean"),
        Last_Purchase_Month=("Paid at", lambda x: x.max().month),
    )

    cat_rev = li_hist.groupby(["Customer Key", "Category"])["Revenue"].sum().reset_index()
    primary_idx = cat_rev.groupby("Customer Key")["Revenue"].idxmax()
    primary_cat = cat_rev.loc[primary_idx].set_index("Customer Key")["Category"]
    agg["Primary_Category"] = primary_cat

    agg["Month_Sin"] = np.sin(2 * np.pi * agg["Last_Purchase_Month"] / 12)
    agg["Month_Cos"] = np.cos(2 * np.pi * agg["Last_Purchase_Month"] / 12)
    return agg.reset_index()


def _churn_encode(feat: pd.DataFrame, top_provinces: list, top_categories: list) -> pd.DataFrame:
    feat = feat.copy()
    feat["Province_Grp"] = feat["Province"].where(feat["Province"].isin(top_provinces), "Other")
    prov_dummies = pd.get_dummies(feat["Province_Grp"], prefix="Prov")
    for p in top_provinces + ["Other"]:
        col = f"Prov_{p}"
        if col not in prov_dummies.columns:
            prov_dummies[col] = 0
    prov_dummies = prov_dummies[[f"Prov_{p}" for p in top_provinces + ["Other"]]]

    feat["Category_Grp"] = feat["Primary_Category"].where(feat["Primary_Category"].isin(top_categories), "Other")
    cat_dummies = pd.get_dummies(feat["Category_Grp"], prefix="Cat")
    for c in top_categories + ["Other"]:
        col = f"Cat_{c}"
        if col not in cat_dummies.columns:
            cat_dummies[col] = 0
    cat_dummies = cat_dummies[[f"Cat_{c}" for c in top_categories + ["Other"]]]

    feature_cols = [
        "Recency", "Frequency", "Monetary", "Tenure", "AOV", "Discount_Rate", "COD_Rate", "RTO_Rate",
        "Avg_Item_Price", "Avg_Pack_Size", "Month_Sin", "Month_Cos",
    ]
    X = pd.concat([
        feat[feature_cols].fillna(0).reset_index(drop=True),
        prov_dummies.reset_index(drop=True),
        cat_dummies.reset_index(drop=True),
    ], axis=1)
    return X


@st.cache_data(show_spinner=False)
def compute_churn_model(data: dict, window_days: int = CHURN_WINDOW_DAYS) -> dict:
    paid = data["paid"]
    li = data.get("li", pd.DataFrame())
    if paid.empty or "Customer Key" not in paid.columns:
        return {}

    snapshot = paid["Paid at"].max() + pd.Timedelta(days=1)
    cutoff = snapshot - pd.Timedelta(days=window_days)

    hist = paid[paid["Paid at"] <= cutoff]
    future = paid[paid["Paid at"] > cutoff]
    if hist["Customer Key"].nunique() < 50:
        return {}

    train_feat = _churn_features(hist, cutoff)
    train_prod = _product_features(li, cutoff=cutoff)
    train_feat = train_feat.merge(train_prod, on="Customer Key", how="left")
    returned = set(future["Customer Key"].unique())
    train_feat["Returned"] = train_feat["Customer Key"].isin(returned).astype(int)

    top_provinces = train_feat["Province"].value_counts().head(TOP_PROVINCES_N).index.tolist()
    top_categories = train_feat["Primary_Category"].value_counts().head(TOP_CATEGORIES_N).index.tolist()

    X = _churn_encode(train_feat, top_provinces, top_categories)
    y = train_feat["Returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)

    auc = None
    if y_test.nunique() > 1:
        proba_test = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Score every customer "as of today"
    live_feat = _churn_features(paid, snapshot)
    live_prod = _product_features(li, cutoff=snapshot)
    live_feat = live_feat.merge(live_prod, on="Customer Key", how="left")
    X_live = _churn_encode(live_feat, top_provinces, top_categories)
    live_feat["Return_Probability"] = clf.predict_proba(X_live)[:, 1]
    live_feat["Churn_Risk_Pct"] = (1 - live_feat["Return_Probability"]) * 100
    live_feat["Risk_Tier"] = pd.qcut(
        live_feat["Return_Probability"].rank(method="first"), 3,
        labels=["High Risk", "Medium Risk", "Low Risk"]
    )
    # Churn_Risk_Pct is squashed into ~87-100% for everyone because the
    # brand's overall reorder rate is only ~2% -- 1-p barely moves even
    # though p itself varies 10x+ between segments. Churn_Risk_Percentile
    # re-expresses risk as this customer's rank relative to all others
    # (0 = most likely to return, 100 = least likely), giving a metric with
    # real spread for comparing segments.
    live_feat["Churn_Risk_Percentile"] = (1 - live_feat["Return_Probability"].rank(pct=True)) * 100

    return {
        "auc": auc,
        "feature_importance": importances,
        "scored_customers": live_feat[
            ["Customer Key", "Recency", "Frequency", "Monetary", "AOV", "Return_Probability",
             "Churn_Risk_Pct", "Churn_Risk_Percentile", "Risk_Tier"]
        ],
        "overall_return_rate": y.mean(),
        "window_days": window_days,
    }


# ---------------------------------------------------------------------------
# Demand Forecasting
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_forecast(data: dict, periods: int = 3, top_n: int = 8) -> dict:
    li = data["li"]
    if li.empty or "Order Month" not in li.columns:
        return {}

    monthly = li.groupby(["Order Month", "Category"])["Lineitem quantity"].sum().reset_index()

    full_range = pd.period_range(li["Order Month"].min(), li["Order Month"].max(), freq="M")
    # The last month is usually still in progress (partial data), so exclude it
    # from the training history and forecast it instead, alongside future months.
    hist_range = full_range[:-1] if len(full_range) > 1 else full_range
    recent_months = hist_range[-6:] if len(hist_range) >= 6 else hist_range

    # Rank by recent (last 6 complete months) revenue so currently-active
    # categories are forecast, not all-time bestsellers that have since been
    # discontinued.
    recent_li = li[li["Order Month"].isin(recent_months)]
    top_categories = (
        recent_li.groupby("Category")["Revenue"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    )

    results = {}
    for cat in top_categories:
        series = monthly[monthly["Category"] == cat].set_index("Order Month")["Lineitem quantity"]
        series = series.reindex(hist_range, fill_value=0).astype(float)
        ts = series.copy()
        ts.index = ts.index.to_timestamp()

        recent_avg = ts.tail(6).mean()
        overall_peak = ts.max()
        is_dead = overall_peak > 0 and recent_avg < 0.15 * overall_peak

        try:
            if len(ts) >= 24 and not is_dead:
                model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal="add",
                                              seasonal_periods=12, initialization_method="estimated")
            elif len(ts) >= 4:
                model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None,
                                              initialization_method="estimated")
            else:
                raise ValueError("series too short")
            fit = model.fit()
            forecast = fit.forecast(periods)
        except Exception:
            avg = ts.tail(3).mean()
            future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            forecast = pd.Series([avg] * periods, index=future_idx)

        # Floor active categories at 20% of their recent average so a sharp
        # one-month dip doesn't crash the forecast to zero for a product
        # that's still selling.
        floor = 0.2 * recent_avg if not is_dead else 0.0
        forecast = forecast.clip(lower=floor)
        results[cat] = {"history": ts, "forecast": forecast}

    return results


# ---------------------------------------------------------------------------
# Customer Lifetime Value
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_clv(data: dict) -> dict:
    paid = data["paid"]
    li = data["li"]
    customers = data["customers"]
    if customers.empty or paid.empty:
        return {}

    customers = customers.copy()
    customers["Tenure_Days"] = (customers["Last_Order"] - customers["First_Order"]).dt.days

    repeat = customers[customers["Repeat_Buyer"]]
    avg_lifespan_days = float(repeat["Tenure_Days"].mean()) if not repeat.empty else 365.0
    if not repeat.empty:
        annual_freq_repeat = repeat["Orders"] / repeat["Tenure_Days"].clip(lower=1) * 365
        avg_annual_freq_repeat = float(annual_freq_repeat.median())
    else:
        avg_annual_freq_repeat = 1.0
    repeat_rate = float(customers["Repeat_Buyer"].mean())

    customers["Annual_Frequency"] = np.where(
        customers["Repeat_Buyer"],
        (customers["Orders"] / customers["Tenure_Days"].clip(lower=1) * 365).clip(upper=avg_annual_freq_repeat * 3),
        np.nan,
    )

    customers["Predicted_CLV"] = np.where(
        customers["Repeat_Buyer"],
        customers["AOV"] * customers["Annual_Frequency"] * (avg_lifespan_days / 365),
        customers["AOV"] * repeat_rate * avg_annual_freq_repeat * (avg_lifespan_days / 365),
    )

    # Primary location per customer
    if "Shipping Province" in paid.columns:
        cust_loc = paid.sort_values("Paid at").groupby("Customer Key")["Shipping Province"].first()
        customers = customers.merge(cust_loc.rename("Shipping Province"), on="Customer Key", how="left")

    # Primary product category per customer (by revenue)
    if not li.empty and "Category" in li.columns:
        li_cust = li.copy()
        li_cust["Customer Key"] = (
            li_cust["Billing Name"].astype(str).str.strip().str.lower()
            + " | " + li_cust["Billing Zip"].astype(str).str.strip()
        )
        cat_rev = li_cust.groupby(["Customer Key", "Category"])["Revenue"].sum().reset_index()
        primary_idx = cat_rev.groupby("Customer Key")["Revenue"].idxmax()
        primary_cat = cat_rev.loc[primary_idx, ["Customer Key", "Category"]].rename(columns={"Category": "Primary Category"})
        customers = customers.merge(primary_cat, on="Customer Key", how="left")

    by_location = pd.DataFrame()
    if "Shipping Province" in customers.columns:
        by_location = customers.groupby("Shipping Province").agg(
            Customers=("Customer Key", "count"),
            Avg_Historical_CLV=("Total_Spend", "mean"),
            Avg_Predicted_CLV=("Predicted_CLV", "mean"),
            Total_Predicted_CLV=("Predicted_CLV", "sum"),
        ).round(2).sort_values("Total_Predicted_CLV", ascending=False).head(15)

    by_category = pd.DataFrame()
    if "Primary Category" in customers.columns:
        by_category = customers.groupby("Primary Category").agg(
            Customers=("Customer Key", "count"),
            Avg_Historical_CLV=("Total_Spend", "mean"),
            Avg_Predicted_CLV=("Predicted_CLV", "mean"),
            Total_Predicted_CLV=("Predicted_CLV", "sum"),
        ).round(2).sort_values("Total_Predicted_CLV", ascending=False).head(15)

    return {
        "customers": customers,
        "by_location": by_location,
        "by_category": by_category,
        "avg_lifespan_days": avg_lifespan_days,
        "repeat_rate": repeat_rate,
        "avg_annual_freq_repeat": avg_annual_freq_repeat,
    }


# ---------------------------------------------------------------------------
# Unified Customer Segments (RFM + Churn Risk + CLV)
# ---------------------------------------------------------------------------

SEGMENT_ACTIONS = {
    "Champions": "Reward & upsell -- loyalty perks, early access to new drops",
    "At-Risk Champions": "Urgent win-back -- your best repeat buyers are going cold",
    "Loyal Customers": "Nurture -- bundles, personalized recommendations",
    "At-Risk Repeat Buyers": "Win-back -- remind them of past favorites + offer",
    "Big Spenders (One-Time)": "Convert to repeat -- second-purchase incentive",
    "New Customers": "Onboard -- welcome series, encourage a 2nd order",
    "Promising": "Re-engage before they go cold -- limited-time offer",
    "Fading One-Timers": "Low-cost retargeting email",
    "Lost": "Deprioritize -- exclude from paid retargeting",
}

# Segments whose customers are valuable enough that a churn-risk flag is worth acting on.
HIGH_VALUE_SEGMENTS = {"Champions", "At-Risk Champions", "Loyal Customers", "Big Spenders (One-Time)"}


@st.cache_data(show_spinner=False)
def compute_customer_segments(data: dict) -> dict:
    """Merge RFM segments with churn risk and predicted CLV into one table."""
    rfm = compute_rfm(data)
    if rfm.empty:
        return {}

    merged = rfm[["Customer Key", "Recency", "Frequency", "Monetary",
                   "R_Score", "F_Score", "M_Score", "Segment"]].copy()

    churn = compute_churn_model(data)
    if churn:
        merged = merged.merge(
            churn["scored_customers"][["Customer Key", "Return_Probability", "Churn_Risk_Pct",
                                        "Churn_Risk_Percentile", "Risk_Tier"]],
            on="Customer Key", how="left",
        )
    else:
        merged["Churn_Risk_Pct"] = np.nan
        merged["Churn_Risk_Percentile"] = np.nan
        merged["Risk_Tier"] = np.nan

    clv = compute_clv(data)
    if clv:
        merged = merged.merge(
            clv["customers"][["Customer Key", "Predicted_CLV", "Shipping Province", "Primary Category"]],
            on="Customer Key", how="left",
        )
    else:
        merged["Predicted_CLV"] = np.nan

    merged["Recommended Action"] = merged["Segment"].map(SEGMENT_ACTIONS)

    segment_summary = merged.groupby("Segment").agg(
        Customers=("Customer Key", "count"),
        Total_Revenue=("Monetary", "sum"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Avg_Churn_Risk_Pct=("Churn_Risk_Pct", "mean"),
        Avg_Churn_Risk_Percentile=("Churn_Risk_Percentile", "mean"),
        Avg_Predicted_CLV=("Predicted_CLV", "mean"),
        Total_Predicted_CLV=("Predicted_CLV", "sum"),
    ).round(2)
    segment_summary["Recommended Action"] = segment_summary.index.map(SEGMENT_ACTIONS)
    segment_summary = segment_summary.sort_values("Total_Predicted_CLV", ascending=False)

    watchlist = pd.DataFrame()
    if merged["Risk_Tier"].notna().any():
        watchlist = merged[
            merged["Segment"].isin(HIGH_VALUE_SEGMENTS) & (merged["Risk_Tier"] == "High Risk")
        ].sort_values("Predicted_CLV", ascending=False).head(20)

    return {
        "customers": merged,
        "segment_summary": segment_summary,
        "watchlist": watchlist,
        "has_churn": bool(churn),
        "has_clv": bool(clv),
    }
