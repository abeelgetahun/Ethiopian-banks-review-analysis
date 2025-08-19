import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.visualization import create_visualizations


def load_data():
    base = Path(__file__).resolve().parent
    task2_dir = base / "data" / "task2"
    processed_dir = base / "data" / "processed"

    final_path = task2_dir / "final_results.csv"
    if final_path.exists():
        final_df = pd.read_csv(final_path)
    else:
        # Fallback to processed with minimal columns
        proc_csv = processed_dir / "bank_reviews_processed.csv"
        final_df = pd.read_csv(proc_csv) if proc_csv.exists() else pd.DataFrame()
    return final_df


def main():
    st.set_page_config(page_title="Ethiopian Banks App Reviews", layout="wide")
    st.title("Ethiopian Banking Apps – Customer Feedback Intelligence")
    st.caption("Reliable, transparent insights for product and risk teams")

    df = load_data()
    if df.empty:
        st.warning("No data found. Please run the pipeline to generate analysis outputs.")
        return

    # Filters
    banks = sorted(df["bank"].dropna().unique().tolist()) if "bank" in df else []
    left, mid, right = st.columns(3)
    with left:
        selected_banks = st.multiselect("Banks", banks, default=banks)
    with mid:
        min_rating, max_rating = st.slider("Rating range", 1, 5, (1, 5))
    with right:
        method = st.selectbox("Sentiment method", ["ensemble", "transformer", "vader", "textblob"])

    # Filter data
    if "bank" in df:
        df = df[df["bank"].isin(selected_banks)]
    if "rating" in df:
        df = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Reviews", f"{len(df):,}")
    with k2:
        pos_pct = 0
        if "sentiment_label" in df and len(df) > 0:
            pos_pct = 100 * (df["sentiment_label"].str.lower() == "positive").mean()
        st.metric("Positive %", f"{pos_pct:.1f}%")
    with k3:
        avg_rating = df["rating"].mean() if "rating" in df and len(df) > 0 else 0
        st.metric("Avg rating", f"{avg_rating:.2f}")
    with k4:
        banks_n = df["bank"].nunique() if "bank" in df else 0
        st.metric("Banks", f"{banks_n}")

    # Plots
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Construct a minimal data dict for visualization utilities if possible
    data_dict = {}
    data_dict["final"] = df
    # Optional: theme/keywords data if available
    task2_dir = Path("data/task2")
    themes_csv = task2_dir / "bank_themes.csv"
    keywords_csv = task2_dir / "bank_keywords.csv"
    if themes_csv.exists():
        data_dict["themes"] = pd.read_csv(themes_csv)
    if keywords_csv.exists():
        data_dict["keywords"] = pd.read_csv(keywords_csv)

    try:
        create_visualizations(data_dict, str(figures_dir))
    except Exception:
        pass

    c1, c2 = st.columns(2)
    img_paths = [
        figures_dir / "sentiment_distribution.png",
        figures_dir / "rating_distribution.png",
        figures_dir / "themes_heatmap.png",
        figures_dir / "sentiment_by_theme.png",
    ]
    for i, p in enumerate(img_paths):
        if p.exists():
            (c1 if i % 2 == 0 else c2).image(str(p), use_container_width=True)

    st.subheader("Samples")
    st.dataframe(df.head(50))


if __name__ == "__main__":
    main()
