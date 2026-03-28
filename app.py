import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

st.set_page_config(page_title="Favorita Forecast Dashboard", layout="wide")

ARTIFACT_DIR = Path("artifacts")


@st.cache_data
def load_data():
    meta = pd.read_parquet(ARTIFACT_DIR / "meta.parquet")
    forecast_detail = pd.read_parquet(ARTIFACT_DIR / "forecast_detail.parquet")
    history = pd.read_parquet(ARTIFACT_DIR / "history.parquet")
    trending_scores = pd.read_parquet(ARTIFACT_DIR / "trending_scores.parquet")

    with open(ARTIFACT_DIR / "config.json", "r") as f:
        config = json.load(f)

    # Standardize dtypes
    if "store_nbr" in meta.columns:
        meta["store_nbr"] = meta["store_nbr"].astype(str)
    if "item_nbr" in meta.columns:
        meta["item_nbr"] = meta["item_nbr"].astype(str)

    if "store_nbr" in forecast_detail.columns:
        forecast_detail["store_nbr"] = forecast_detail["store_nbr"].astype(str)
    if "item_nbr" in forecast_detail.columns:
        forecast_detail["item_nbr"] = forecast_detail["item_nbr"].astype(str)
    if "date" in forecast_detail.columns:
        forecast_detail["date"] = pd.to_datetime(forecast_detail["date"])

    if "store_nbr" in history.columns:
        history["store_nbr"] = history["store_nbr"].astype(str)
    if "item_nbr" in history.columns:
        history["item_nbr"] = history["item_nbr"].astype(str)
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"])

    if "store_nbr" in trending_scores.columns:
        trending_scores["store_nbr"] = trending_scores["store_nbr"].astype(str)

    return meta, forecast_detail, history, trending_scores, config


def safe_show_columns(df, preferred_cols):
    cols = [c for c in preferred_cols if c in df.columns]
    return df[cols] if cols else df


def get_store_trending_scores(trending_scores, store_choice):
    """
    Return a store-filtered trending table if store_nbr exists.
    Otherwise return the whole table.
    """
    if trending_scores.empty:
        return trending_scores

    if "store_nbr" in trending_scores.columns:
        return trending_scores[trending_scores["store_nbr"] == store_choice].copy()

    return trending_scores.copy()


def sort_trending_table(df):
    """
    Try to sort trending table by the most likely ranking column if present.
    """
    sort_candidates = [
        "trend_score",
        "trending_score",
        "score",
        "rank",
        "trend_rank",
        "lift",
        "uplift",
    ]

    for col in sort_candidates:
        if col in df.columns:
            ascending = col in {"rank", "trend_rank"}
            return df.sort_values(col, ascending=ascending).reset_index(drop=True)

    return df.reset_index(drop=True)


# Load data
meta, forecast_detail, history, trending_scores, config = load_data()

st.title("Favorita Forecast Dashboard")
st.caption("Store-level replenish, promote, trending, and forecast view")

# -------------------------
# Sidebar / top controls
# -------------------------
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

# -------------------------
# Replenish / Promote tables
# -------------------------
st.subheader(f"Store {store_choice} recommendations")

repl_cols_preferred = [
    "store_nbr",
    "item_nbr",
    "family",
    "pred_sum_nextH",
    "pred_avg_nextH",
]

promo_cols_preferred = [
    "store_nbr",
    "item_nbr",
    "family",
    "promo_ratio",
    "promo_uplift_units_est",
    "pred_sum_nextH",
]

repl = (
    store_meta.sort_values("pred_sum_nextH", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
repl = safe_show_columns(repl, repl_cols_preferred)

promo = (
    store_meta.sort_values("promo_uplift_units_est", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
promo = safe_show_columns(promo, promo_cols_preferred)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 10 Replenish")
    st.dataframe(repl, use_container_width=True)

with col2:
    st.markdown("### Top 10 Promote")
    st.dataframe(promo, use_container_width=True)

# -------------------------
# Trending table
# -------------------------
trending_view = get_store_trending_scores(trending_scores, store_choice)
trending_view = sort_trending_table(trending_view)

st.markdown("### Trending Score Table")

if trending_view.empty:
    st.info("No trending score data found for this store.")
else:
    # Try to show the most useful columns first if they exist
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

# -------------------------
# Item selector
# -------------------------
top_items = []
if "item_nbr" in repl.columns:
    top_items.extend(repl["item_nbr"].dropna().astype(str).tolist())
if "item_nbr" in promo.columns:
    top_items.extend(promo["item_nbr"].dropna().astype(str).tolist())

top_items = pd.Series(top_items).drop_duplicates().tolist()

all_items = []
if "item_nbr" in store_meta.columns:
    all_items = sorted(store_meta["item_nbr"].dropna().astype(str).unique().tolist())

item_options = top_items + [x for x in all_items if x not in top_items]

if not item_options:
    st.warning("No items found for this store.")
    st.stop()

item_choice = st.selectbox("Select Item", item_options)

# -------------------------
# Item detail data
# -------------------------
item_forecast = forecast_detail[
    (forecast_detail["store_nbr"] == store_choice) &
    (forecast_detail["item_nbr"] == item_choice)
].copy()

item_history = history[
    (history["store_nbr"] == store_choice) &
    (history["item_nbr"] == item_choice)
].copy()

st.subheader(f"Store {store_choice} | Item {item_choice}")

if item_forecast.empty:
    st.info("No forecast details found for this store-item.")
else:
    # Sort
    if "date" in item_forecast.columns:
        item_forecast = item_forecast.sort_values("date")
    if "date" in item_history.columns:
        item_history = item_history.sort_values("date")

    # Limit history shown on plot
    max_encoder_length = int(config.get("max_encoder_length", 60))
    if not item_history.empty:
        item_history = item_history.tail(max_encoder_length)

    fig, ax = plt.subplots(figsize=(10, 4))

    if not item_history.empty and "unit_sales" in item_history.columns:
        ax.plot(item_history["date"], item_history["unit_sales"], label="History (actual)")

    if "actual_sales" in item_forecast.columns:
        ax.plot(item_forecast["date"], item_forecast["actual_sales"], label="Horizon (actual)")

    if "forecast_sales" in item_forecast.columns:
        ax.plot(item_forecast["date"], item_forecast["forecast_sales"], label="Horizon (forecast)")

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
    ]
    forecast_display = safe_show_columns(item_forecast, forecast_cols_preferred)
    st.dataframe(forecast_display, use_container_width=True)
