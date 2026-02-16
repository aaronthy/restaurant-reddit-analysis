import pandas as pd
from textblob import TextBlob

df = pd.read_csv("data/yelp_sample.csv")
df["text"] = df["text"].fillna("").str.lower()

# Categories
CATEGORIES = {
    "service": ["slow", "wait", "waiting", "rude", "ignored", "server", "staff", "service"],
    "food_quality": ["cold", "bland", "taste", "burnt", "raw", "undercooked", "overcooked", "flavor", "tasteless"],
    "cleanliness": ["dirty", "smell", "hair", "bugs", "unclean", "sticky"],
    "price_value": ["expensive", "overpriced", "price", "value", "portion", "cost"],
    "order_accuracy": ["wrong", "incorrect", "missing", "forgot", "mistake"],
    "management": ["manager", "management", "owner", "policy", "refund", "complaint"],
    "ambience": ["loud", "noisy", "crowded", "music", "atmosphere"],
    "staffing": ["understaffed", "short staff", "busy", "overworked", "not enough staff"],
    "waiting_time": ["line", "waited", "long wait", "queue"]
}

def match_categories(text):
    hits = []
    for cat, keywords in CATEGORIES.items():
        if any(k in text for k in keywords):
            hits.append(cat)
    return hits if hits else ["other"]

df["categories"] = df["text"].apply(match_categories)

# sentiment
df["sentiment"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity)

# explode
x = df.explode("categories")

# KPIs
freq = x["categories"].value_counts()
avg_stars = x.groupby("categories")["stars"].mean()
useful_sum = x.groupby("categories")["useful"].sum()
sentiment = x.groupby("categories")["sentiment"].mean()

# combine into one table
kpi = pd.DataFrame({
    "frequency": freq,
    "avg_stars": avg_stars,
    "useful": useful_sum,
    "sentiment": sentiment
}).fillna(0)

# normalize (0-1 scale)
kpi["freq_norm"] = kpi["frequency"] / kpi["frequency"].max()
kpi["useful_norm"] = kpi["useful"] / kpi["useful"].max()

# failure score (higher = worse)
kpi["failure_score"] = (
    kpi["freq_norm"] * 0.4 +
    kpi["useful_norm"] * 0.3 +
    (5 - kpi["avg_stars"]) / 5 * 0.2 +
    (-kpi["sentiment"]) * 0.1
)

kpi = kpi.sort_values("failure_score", ascending=False)

print("\n=== KPI Summary ===")
print(kpi.round(3))

# show top issues
print("\n=== Top Failure Drivers ===")
print(kpi.head(5)[["failure_score", "frequency", "avg_stars", "sentiment"]])

import matplotlib.pyplot as plt

# remove "other"
top = kpi[kpi.index != "other"].head(5)

top["failure_score"].plot(kind="bar")

plt.title("Top Restaurant Failure Drivers")
plt.ylabel("Failure Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# examples
top_cats = kpi.head(3).index.tolist()

print("\n=== Example Reviews (Top Issues) ===")
for cat in top_cats:
    print(f"\n--- {cat} ---")
    examples = x[x["categories"] == cat].sort_values("stars").head(3)
    for _, r in examples.iterrows():
        print(f"[{int(r['stars'])}★] {str(r['text'])[:200]}...")

