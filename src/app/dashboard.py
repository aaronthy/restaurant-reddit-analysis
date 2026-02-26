import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from analysis.dashboard_data import (
    load_data,
    get_category_counts,
    get_avg_rating_by_category,
    get_low_rating_counts,
    get_filtered_reviews,
    get_business_insights,
)

st.set_page_config(page_title="Restaurant Operations Dashboard", layout="wide")
st.title("Restaurant Operations Dashboard")

# Load
df = load_data()

# KPI cards
insights = get_business_insights(df)
col1, col2 = st.columns(2)
col1.metric("Total Reviews", insights["total_reviews"])
if "avg_overall" in insights:
    col2.metric("Avg Rating", insights["avg_overall"])

# Charts
st.header("Review Categories")
st.bar_chart(get_category_counts(df))

st.header("Average Rating by Category")
avg = get_avg_rating_by_category(df)
if avg is not None:
    st.bar_chart(avg)
else:
    st.info("No 'stars' column found, so average ratings are unavailable.")

st.header("Top Problem Areas (Low Ratings)")
neg = get_low_rating_counts(df, threshold=2)
if neg is not None:
    st.bar_chart(neg)
else:
    st.info("No 'stars' column found, so low-rating analysis is unavailable.")

# Explore table
st.header("Explore Reviews")
categories = sorted(df["label_clean"].unique())
category = st.selectbox("Select Category", categories)

table = get_filtered_reviews(df, category_clean=category, limit=30)
st.dataframe(table, use_container_width=True)

# Business insights (display only)
st.header("Business Insights")
st.subheader("Key Findings")

if "worst_category" in insights:
    st.write(
        f"Lowest rated category: **{insights['worst_category']}** "
        f"({insights['worst_score']} stars)"
    )

if "top_issue" in insights:
    st.write(f"Most frequent complaint area (≤2 stars): **{insights['top_issue']}**")

st.subheader("Recommendation")
if "top_issue" in insights:
    st.write(
        f"Prioritize improvements in **{insights['top_issue']}**—it appears most often "
        "in low-rating reviews."
    )
else:
    st.write("Prioritize improvements based on the categories with the lowest ratings.")

st.set_page_config(layout="wide")
