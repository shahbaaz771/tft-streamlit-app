import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Favorita Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = Path("artifacts")


@st.cache_data
def load_data():
    meta = pd.read_parquet(ARTIFACT_DIR / "meta.parquet")
    forecast_detail = pd.read_parquet(ARTIFACT_DIR / "forecast_detail.parquet")
    history = pd.read_parquet(ARTIFACT_DIR / "history.parquet")

    trending_path = ARTIFACT_DIR / "trending_scores.parquet"
    trending_scores = pd.read_parquet(trending_path) if trending_path.exists() else pd.DataFrame()

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
    rank_asc_candidates = ["rank", "trend_rank"]

    for col in score_desc_candidates:
        if col in df.columns:
            return df.sort_values(col, ascending=False).reset_index(drop=True)

    for col in rank_asc_candidates:
        if col in df.columns:
            return df.sort_values(col, ascending=True).reset_index(drop=True)

    return df.reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def plot_family_pie(df: pd.DataFrame, title: str):
    if df.empty or "family" not in df.columns:
        st.info(f"No family data available for {title}.")
        return

    family_counts = (
        df["family"]
        .fillna("NA")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    family_counts.columns = ["family", "count"]

    if family_counts.empty:
        st.info(f"No family data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        family_counts["count"],
        labels=family_counts["family"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig)


meta, forecast_detail, history, trending_scores, config = load_data()

st.title("Favorita Forecast Dashboard")
st.caption("Interactive dashboard for store-level replenish, promote, trending, and forecast analysis.")

if "store_nbr" not in meta.columns:
    st.error("meta.parquet does not contain 'store_nbr'.")
    st.stop()

stores = sorted(meta["store_nbr"].dropna().unique().tolist())
if not stores:
    st.error("No stores found in meta.parquet.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Filters")

store_choice = st.sidebar.selectbox("Store", stores)

store_meta = meta[meta["store_nbr"] == store_choice].copy()
if store_meta.empty:
    st.warning("No data found for this store.")
    st.stop()

if "family" not in store_meta.columns:
    st.error("meta.parquet does not contain 'family'.")
    st.stop()

families = sorted(store_meta["family"].dropna().astype(str).unique().tolist())
family_options = ["All"] + families
family_choice = st.sidebar.selectbox("Family", family_options, index=0)

if family_choice == "All":
    store_family_meta = store_meta.copy()
else:
    store_family_meta = store_meta[store_meta["family"] == family_choice].copy()

if store_family_meta.empty:
    st.warning("No items found for this family in this store.")
    st.stop()

family_items = sorted(store_family_meta["item_nbr"].dropna().astype(str).unique().tolist())
item_options = ["All"] + family_items
item_choice = st.sidebar.selectbox("Item (for forecast plot)", item_options, index=0)

view_mode = st.sidebar.radio(
    "Recommendation View",
    options=["Replenish", "Promote"],
    index=0,
)
top_n = st.sidebar.slider("Top N recommendations", min_value=5, max_value=20, value=10)
show_forecast_details = st.sidebar.checkbox("Show forecast details table", value=True)

# -----------------------------
# Build recommendation tables
# -----------------------------
repl_sort_col = first_existing_column(
    store_meta,
    ["pred_sum_nextH", "pred_avg_nextH", "forecast_sum", "forecast_avg"],
)

promo_sort_col = first_existing_column(
    store_meta,
    [
        "final_promote_score",
        "pred_rec_score",
        "promo_ratio",
        "promo_uplift_units_est",
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
    "final_promote_score",
    "pred_rec_score",
    "promo_ratio",
    "promo_uplift_units_est",
    "pred_sum_nextH",
]

if repl_sort_col is not None:
    repl = (
        store_meta.sort_values(repl_sort_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
else:
    repl = store_meta.head(top_n).reset_index(drop=True)

if promo_sort_col is not None:
    promo = (
        store_meta.sort_values(promo_sort_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
else:
    promo = store_meta.head(top_n).reset_index(drop=True)

repl = safe_show_columns(repl, repl_cols_preferred)
promo = safe_show_columns(promo, promo_cols_preferred)

# -----------------------------
# Trending table
# -----------------------------
trending_view = trending_scores.copy()
if not trending_view.empty:
    if "store_nbr" in trending_view.columns:
        trending_view = trending_view[trending_view["store_nbr"] == store_choice]
    if family_choice != "All" and "family" in trending_view.columns:
        trending_view = trending_view[trending_view["family"] == family_choice]

trending_view = sort_trending_table(trending_view)

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
    "count",
    "coverage",
    "sales_last_7",
    "sales_prev_7",
    "growth_rate",
]
trending_display = safe_show_columns(trending_view, trending_preferred_cols)

# -----------------------------
# KPI cards
# -----------------------------
k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Items in Store", f"{len(store_meta):,}")

with k2:
    if "pred_sum_nextH" in store_meta.columns:
        st.metric("Total Forecast Demand", f"{store_meta['pred_sum_nextH'].sum():,.1f}")
    else:
        st.metric("Total Forecast Demand", "N/A")

with k3:
    if family_choice == "All":
        st.metric("Items in Selected Family", f"{len(store_meta):,}")
    else:
        st.metric("Items in Selected Family", f"{len(store_family_meta):,}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Recommendations", "Trending", "Forecast Plot", "Forecast Details"]
)

with tab1:
    st.subheader(f"Store {store_choice} recommendations")

    if view_mode == "Both":
        p1, p2 = st.columns(2)

        with p1:
            st.markdown(f"### Replenish Family Distribution (Top {top_n})")
            plot_family_pie(repl, "Replenish by Family")

        with p2:
            st.markdown(f"### Promote Family Distribution (Top {top_n})")
            plot_family_pie(promo, "Promote by Family")

        st.divider()

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### Top {top_n} Replenish")
            st.dataframe(repl, use_container_width=True)
            st.download_button(
                "Download Replenish CSV",
                data=to_csv_bytes(repl),
                file_name=f"replenish_store_{store_choice}.csv",
                mime="text/csv",
            )

        with c2:
            st.markdown(f"### Top {top_n} Promote")
            st.dataframe(promo, use_container_width=True)
            st.download_button(
                "Download Promote CSV",
                data=to_csv_bytes(promo),
                file_name=f"promote_store_{store_choice}.csv",
                mime="text/csv",
            )

    elif view_mode == "Replenish":
        st.markdown(f"### Replenish Family Distribution (Top {top_n})")
        plot_family_pie(repl, "Replenish by Family")
        st.divider()
        st.markdown(f"### Top {top_n} Replenish")
        st.dataframe(repl, use_container_width=True)
        st.download_button(
            "Download Replenish CSV",
            data=to_csv_bytes(repl),
            file_name=f"replenish_store_{store_choice}.csv",
            mime="text/csv",
        )

    elif view_mode == "Promote":
        st.markdown(f"### Promote Family Distribution (Top {top_n})")
        plot_family_pie(promo, "Promote by Family")
        st.divider()
        st.markdown(f"### Top {top_n} Promote")
        st.dataframe(promo, use_container_width=True)
        st.download_button(
            "Download Promote CSV",
            data=to_csv_bytes(promo),
            file_name=f"promote_store_{store_choice}.csv",
            mime="text/csv",
        )

with tab2:
    family_label = family_choice if family_choice != "All" else "All Families"
    st.subheader(f"Trending Score Table — {family_label}")

    if trending_display.empty:
        st.info("No trending score data found for this selection.")
    else:
        st.dataframe(trending_display, use_container_width=True)
        st.download_button(
            "Download Trending CSV",
            data=to_csv_bytes(trending_display),
            file_name=f"trending_store_{store_choice}_{family_label.replace(' ', '_')}.csv",
            mime="text/csv",
        )

with tab3:
    if item_choice == "All":
        st.subheader("Forecast Plot")
        st.info("Select a specific item from the sidebar to view the forecast plot.")
    else:
        item_forecast = forecast_detail.copy()
        if "store_nbr" in item_forecast.columns:
            item_forecast = item_forecast[item_forecast["store_nbr"] == store_choice]
        if "item_nbr" in item_forecast.columns:
            item_forecast = item_forecast[item_forecast["item_nbr"] == item_choice]
        if "date" in item_forecast.columns:
            item_forecast = item_forecast.sort_values("date")

        item_history = history.copy()
        if "store_nbr" in item_history.columns:
            item_history = item_history[item_history["store_nbr"] == store_choice]
        if "item_nbr" in item_history.columns:
            item_history = item_history[item_history["item_nbr"] == item_choice]
        if "date" in item_history.columns:
            item_history = item_history.sort_values("date")

        max_encoder_length = int(config.get("max_encoder_length", 60))
        if not item_history.empty:
            item_history = item_history.tail(max_encoder_length)

        forecast_col = first_existing_column(
            item_forecast,
            ["forecast_sales", "predicted_sales", "pred_sales", "yhat"],
        )

        st.subheader(
            f"Forecast Plot — Store {store_choice} | "
            f"{'Family ' + family_choice + ' | ' if family_choice != 'All' else ''}"
            f"Item {item_choice}"
        )

        if item_forecast.empty:
            st.info("No forecast details found for this store-item.")
        else:
            fig, ax = plt.subplots(figsize=(11, 4.5))

            if not item_history.empty and {"date", "unit_sales"}.issubset(item_history.columns):
                ax.plot(item_history["date"], item_history["unit_sales"], label="History (actual)")

            if {"date", "actual_sales"}.issubset(item_forecast.columns):
                ax.plot(item_forecast["date"], item_forecast["actual_sales"], label="Horizon (actual)")

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

with tab4:
    st.subheader("Forecast Details")

    if item_choice == "All":
        st.info("Select a specific item from the sidebar to view forecast details.")
    else:
        item_forecast = forecast_detail.copy()
        if "store_nbr" in item_forecast.columns:
            item_forecast = item_forecast[item_forecast["store_nbr"] == store_choice]
        if "item_nbr" in item_forecast.columns:
            item_forecast = item_forecast[item_forecast["item_nbr"] == item_choice]
        if "date" in item_forecast.columns:
            item_forecast = item_forecast.sort_values("date")

        if item_forecast.empty:
            st.info("No forecast details found for this store-item.")
        elif show_forecast_details:
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
            st.download_button(
                "Download Forecast Details CSV",
                data=to_csv_bytes(forecast_display),
                file_name=f"forecast_store_{store_choice}_item_{item_choice}.csv",
                mime="text/csv",
            )
        else:
            st.info("Enable 'Show forecast details table' from the sidebar to view this section.")
