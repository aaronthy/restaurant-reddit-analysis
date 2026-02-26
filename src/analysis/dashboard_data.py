import pandas as pd


def load_data(path: str = "data/yelp_labeled.csv") -> pd.DataFrame:
    """Load labeled Yelp review data and add cleaned label for display."""
    df = pd.read_csv(path)

    # Ensure required column exists
    if "label" not in df.columns:
        raise ValueError("Expected column 'label' in dataset.")

    # Clean labels for nicer charts/UI
    df["label_clean"] = (
        df["label"]
        .astype(str)
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    return df


def get_category_counts(df: pd.DataFrame) -> pd.Series:
    """Counts of reviews per category (clean labels)."""
    return df["label_clean"].value_counts().sort_values(ascending=False)


def get_avg_rating_by_category(df: pd.DataFrame) -> pd.Series | None:
    """Average stars per category (clean labels), if stars column exists."""
    if "stars" not in df.columns:
        return None
    return (
        df.groupby("label_clean")["stars"]
        .mean()
        .sort_values(ascending=False)
    )


def get_low_rating_counts(df: pd.DataFrame, threshold: int = 2) -> pd.Series | None:
    """Counts of low-rating (<= threshold) reviews by category, if stars exists."""
    if "stars" not in df.columns:
        return None
    neg = df[df["stars"] <= threshold]
    if neg.empty:
        return None
    return neg["label_clean"].value_counts().sort_values(ascending=False)


def get_filtered_reviews(
    df: pd.DataFrame,
    category_clean: str,
    limit: int = 30
) -> pd.DataFrame:
    """Return a small table of reviews for a selected category."""
    cols = [c for c in ["text", "stars", "useful", "label_clean"] if c in df.columns]
    out = df[df["label_clean"] == category_clean][cols].head(limit).copy()
    return out


def get_business_insights(df: pd.DataFrame) -> dict:
    """
    Compute dashboard KPIs/insights (no Streamlit here).
    Returns a dict with keys like total_reviews, avg_overall, worst_category, etc.
    """
    insights: dict = {"total_reviews": int(len(df))}

    if "stars" in df.columns and len(df) > 0:
        insights["avg_overall"] = round(float(df["stars"].mean()), 2)

        avg_by_cat = df.groupby("label_clean")["stars"].mean()
        insights["worst_category"] = str(avg_by_cat.idxmin())
        insights["worst_score"] = round(float(avg_by_cat.min()), 2)

        neg = df[df["stars"] <= 2]
        if not neg.empty:
            insights["top_issue"] = str(neg["label_clean"].value_counts().idxmax())

    return insights