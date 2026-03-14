import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

st.set_page_config(page_title="Store Forecast Dashboard", layout="wide")

ARTIFACT_DIR = Path("artifacts")

@st.cache_data
def load_data():
    meta = pd.read_parquet(ARTIFACT_DIR / "meta.parquet")
    forecast_detail = pd.read_parquet(ARTIFACT_DIR / "forecast_detail.parquet")
    history = pd.read_parquet(ARTIFACT_DIR / "history.parquet")

    with open(ARTIFACT_DIR / "config.json", "r") as f:
        config = json.load(f)

    meta["store_nbr"] = meta["store_nbr"].astype(str)
    meta["item_nbr"] = meta["item_nbr"].astype(str)

    forecast_detail["store_nbr"] = forecast_detail["store_nbr"].astype(str)
    forecast_detail["item_nbr"] = forecast_detail["item_nbr"].astype(str)
    forecast_detail["date"] = pd.to_datetime(forecast_detail["date"])

    history["store_nbr"] = history["store_nbr"].astype(str)
    history["item_nbr"] = history["item_nbr"].astype(str)
    history["date"] = pd.to_datetime(history["date"])

    return meta, forecast_detail, history, config


meta, forecast_detail, history, config = load_data()

st.title("Store Forecast Dashboard")

stores = sorted(meta["store_nbr"].dropna().unique().tolist())
store_choice = st.selectbox("Select Store", stores)

store_meta = meta[meta["store_nbr"] == store_choice].copy()

if store_meta.empty:
    st.warning("No data found for this store.")
    st.stop()

# Top replenish
repl = (
    store_meta.sort_values("pred_sum_nextH", ascending=False)
    .head(10)[["store_nbr", "item_nbr", "family", "pred_sum_nextH", "pred_avg_nextH"]]
    .reset_index(drop=True)
)

# Top promote
promo = (
    store_meta.sort_values("promo_uplift_units_est", ascending=False)
    .head(10)[["store_nbr", "item_nbr", "family", "promo_ratio", "promo_uplift_units_est", "pred_sum_nextH"]]
    .reset_index(drop=True)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Replenish")
    st.dataframe(repl, use_container_width=True)

with col2:
    st.subheader("Top 10 Promote")
    st.dataframe(promo, use_container_width=True)

top_items = pd.concat([repl["item_nbr"], promo["item_nbr"]]).drop_duplicates().tolist()
all_items = sorted(store_meta["item_nbr"].dropna().unique().tolist())
item_options = top_items + [x for x in all_items if x not in top_items]

item_choice = st.selectbox("Select Item", item_options)

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
    # limit history to recent period for plotting
    if not item_history.empty:
        item_history = item_history.sort_values("date").tail(config["max_encoder_length"])

    fig, ax = plt.subplots(figsize=(10, 4))

    if not item_history.empty:
        ax.plot(item_history["date"], item_history["unit_sales"], label="History (actual)")

    ax.plot(item_forecast["date"], item_forecast["actual_sales"], label="Horizon (actual)")
    ax.plot(item_forecast["date"], item_forecast["forecast_sales"], label="Horizon (forecast)")

    forecast_start = item_forecast["date"].min()
    ax.axvline(forecast_start, linestyle="--", label="Forecast start")

    ax.set_title(f"Store {store_choice} | Item {item_choice}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Forecast details")
    st.dataframe(item_forecast, use_container_width=True)
