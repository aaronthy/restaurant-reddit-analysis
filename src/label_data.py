import pandas as pd

INPUT = "data/yelp_sample.csv"
OUTPUT = "data/yelp_labeled.csv"

df = pd.read_csv(INPUT)
df["text"] = df["text"].fillna("").str.lower()

CATEGORIES = {
    "service": ["slow", "wait", "waiting", "rude", "ignored", "server", "staff", "service"],
    "food_quality": ["cold", "bland", "taste", "burnt", "raw", "undercooked", "overcooked", "flavor", "tasteless"],
    "cleanliness": ["dirty", "smell", "hair", "bugs", "unclean", "sticky"],
    "price_value": ["expensive", "overpriced", "price", "value", "portion", "cost"],
    "order_accuracy": ["wrong", "incorrect", "missing", "forgot", "mistake"],
    "management": ["manager", "management", "owner", "policy", "refund", "complaint"],
    "ambience": ["loud", "noisy", "crowded", "music", "atmosphere"],
    "staffing": ["understaffed", "short staff", "busy", "overworked", "not enough staff"],
    "waiting_time": ["line", "waited", "long wait", "queue"],
}

def label_one(text):
    best_cat = "other"
    best_score = 0

    for cat, keys in CATEGORIES.items():
        score = sum(1 for k in keys if k in text)
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


df["label"] = df["text"].apply(label_one)

# Optional: drop "other" to make training cleaner
train_df = df[df["label"] != "other"].copy()

print("Label distribution (training):")
print(train_df["label"].value_counts())

train_df.to_csv(OUTPUT, index=False)
print(f"\nSaved labeled data to {OUTPUT}")
