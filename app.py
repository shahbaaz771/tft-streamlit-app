import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Favorita Forecast Dashboard", layout="wide")

ARTIFACT_DIR = Path("artifacts")


@st.cache_data
def load_data():
    meta = pd.read_parquet(ARTIFACT_DIR / "meta.parquet")
    forecast_detail = pd.read_parquet(ARTIFACT_DIR / "forecast_detail.parquet")
    history = pd.read_parquet(ARTIFACT_DIR / "history.parquet")

    trending_path = ARTIFACT_DIR / "trending_scores.parquet"
    if trending_path.exists():
        trending_scores = pd.read_parquet(trending_path)
    else:
        trending_scores = pd.DataFrame()

    with open(ARTIFACT_DIR / "config.json", "r") as f:
        config = json.load(f)

    for df in [meta, forecast_detail, history, trending_scores]:
        if "store_nbr" in df.columns:
            df["store_nbr"] = df["store_nbr"].astype(str)
        if "item_nbr" in df.columns:
            df["item_nbr"] = df["item_nbr"].astype(str)
        if "family" in df.columns:
            df["family"] = df["family"].astype(str)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    return meta, forecast_detail, history, trending_scores, config


def safe_show_columns(df: pd.DataFrame, preferred_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in preferred_cols if c in df.columns]
    return df[cols] if cols else df


def first_existing_column(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def sort_trending_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    score_desc_candidates = [
        "trend_score",
        "trending_score",
        "score",
        "weighted_score",
        "growth_score",
        "uplift",
        "lift",
    ]
    rank_asc_candidates = [
        "rank",
        "trend_rank",
    ]

    for col in score_desc_candidates:
        if col in df.columns:
            return df.sort_values(col, ascending=False).reset_index(drop=True)

    for col in rank_asc_candidates:
        if col in df.columns:
            return df.sort_values(col, ascending=True).reset_index(drop=True)

    return df.reset_index(drop=True)


meta, forecast_detail, history, trending_scores, config = load_data()

st.title("Favorita Forecast Dashboard")
st.caption("Store → Replenish/Promote → Family → Item → Trend + Forecast")

if "store_nbr" not in meta.columns:
    st.error("meta.parquet does not contain 'store_nbr'.")
    st.stop()

stores = sorted(meta["store_nbr"].dropna().unique().tolist())
if not stores:
    st.error("No stores found in meta.parquet.")
    st.stop()

store_choice = st.selectbox("Select Store", stores)

store_meta = meta[meta["store_nbr"] == store_choice].copy()

if store_meta.empty:
    st.warning("No data found for this store.")
    st.stop()

st.subheader(f"Store {store_choice} recommendations")

repl_sort_col = first_existing_column(
    store_meta,
    ["pred_sum_nextH", "pred_avg_nextH", "forecast_sum", "forecast_avg"],
)

promo_sort_col = first_existing_column(
    store_meta,
    [
        "promo_uplift_units_est",
        "promo_uplift_est",
        "promo_score",
        "promo_ratio",
        "pred_sum_nextH",
    ],
)

repl_cols_preferred = [
    "store_nbr",
    "item_nbr",
    "family",
    "pred_sum_nextH",
    "pred_avg_nextH",
    "forecast_sum",
    "forecast_avg",
]

promo_cols_preferred = [
    "store_nbr",
    "item_nbr",
    "family",
    "promo_ratio",
    "promo_uplift_units_est",
    "promo_uplift_est",
    "promo_score",
    "pred_sum_nextH",
]

if repl_sort_col is not None:
    repl = (
        store_meta.sort_values(repl_sort_col, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
else:
    repl = store_meta.head(10).reset_index(drop=True)

if promo_sort_col is not None:
    promo = (
        store_meta.sort_values(promo_sort_col, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
else:
    promo = store_meta.head(10).reset_index(drop=True)

repl = safe_show_columns(repl, repl_cols_preferred)
promo = safe_show_columns(promo, promo_cols_preferred)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 Replenish")
    st.dataframe(repl, use_container_width=True)

with col2:
    st.markdown("### Top 10 Promote")
    st.dataframe(promo, use_container_width=True)

if "family" in store_meta.columns:
    families = sorted(store_meta["family"].dropna().astype(str).unique().tolist())
else:
    families = []

if not families:
    st.warning("No family values found for this store.")
    st.stop()

family_choice = st.selectbox("Select Family", families)

store_family_meta = store_meta[store_meta["family"] == family_choice].copy()

if store_family_meta.empty:
    st.warning("No items found for this family in this store.")
    st.stop()

st.markdown(f"### Trending Score Table — Family: {family_choice}")

trending_view = trending_scores.copy()

if not trending_view.empty:
    if "store_nbr" in trending_view.columns:
        trending_view = trending_view[trending_view["store_nbr"] == store_choice]
    if "family" in trending_view.columns:
        trending_view = trending_view[trending_view["family"] == family_choice]

trending_view = sort_trending_table(trending_view)

if trending_view.empty:
    st.info("No trending score data found for this store-family combination.")
else:
    trending_preferred_cols = [
        "store_nbr",
        "family",
        "product_type",
        "item_nbr",
        "trend_score",
        "trending_score",
        "score",
        "rank",
        "trend_rank",
        "sales_last_7",
        "sales_prev_7",
        "growth_rate",
    ]
    trending_display = safe_show_columns(trending_view, trending_preferred_cols)
    st.dataframe(trending_display, use_container_width=True)

family_items = sorted(store_family_meta["item_nbr"].dropna().astype(str).unique().tolist())

if not family_items:
    st.warning("No items found for this selected family.")
    st.stop()

item_choice = st.selectbox("Select Item", family_items)

item_forecast = forecast_detail.copy()
if "store_nbr" in item_forecast.columns:
    item_forecast = item_forecast[item_forecast["store_nbr"] == store_choice]
if "item_nbr" in item_forecast.columns:
    item_forecast = item_forecast[item_forecast["item_nbr"] == item_choice]

item_history = history.copy()
if "store_nbr" in item_history.columns:
    item_history = item_history[item_history["store_nbr"] == store_choice]
if "item_nbr" in item_history.columns:
    item_history = item_history[item_history["item_nbr"] == item_choice]

st.subheader(f"Forecast Plot — Store {store_choice} | Family {family_choice} | Item {item_choice}")

if item_forecast.empty:
    st.info("No forecast details found for this store-item.")
else:
    if "date" in item_forecast.columns:
        item_forecast = item_forecast.sort_values("date")
    if "date" in item_history.columns:
        item_history = item_history.sort_values("date")

    max_encoder_length = int(config.get("max_encoder_length", 60))
    if not item_history.empty:
        item_history = item_history.tail(max_encoder_length)

    fig, ax = plt.subplots(figsize=(10, 4))

    if not item_history.empty and {"date", "unit_sales"}.issubset(item_history.columns):
        ax.plot(item_history["date"], item_history["unit_sales"], label="History (actual)")

    if {"date", "actual_sales"}.issubset(item_forecast.columns):
        ax.plot(item_forecast["date"], item_forecast["actual_sales"], label="Horizon (actual)")

    forecast_col = first_existing_column(
        item_forecast,
        ["forecast_sales", "predicted_sales", "pred_sales", "yhat"],
    )
    if forecast_col is not None and "date" in item_forecast.columns:
        ax.plot(item_forecast["date"], item_forecast[forecast_col], label="Horizon (forecast)")

    if "date" in item_forecast.columns and len(item_forecast) > 0:
        forecast_start = item_forecast["date"].min()
        ax.axvline(forecast_start, linestyle="--", label="Forecast start")

    ax.set_title(f"Store {store_choice} | Item {item_choice}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.markdown("### Forecast details")
    forecast_cols_preferred = [
        "store_nbr",
        "item_nbr",
        "family",
        "date",
        "actual_sales",
        "forecast_sales",
        "predicted_sales",
        "pred_sales",
        "yhat",
    ]
    forecast_display = safe_show_columns(item_forecast, forecast_cols_preferred)
    st.dataframe(forecast_display, use_container_width=True)
