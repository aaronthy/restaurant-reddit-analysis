import os
import csv
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample


# =========================
# CONFIG
# =========================
DATA_PATH = "data/yelp_labeled.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

MAX_PER_CLASS = 800
MIN_PER_CLASS = 62


# =========================
# BALANCING FUNCTION
# =========================
def balance_dataframe(df, label_col="label"):
    parts = []

    for label, group in df.groupby(label_col):
        n = len(group)

        if n < MIN_PER_CLASS:
            group_bal = resample(
                group,
                replace=True,
                n_samples=MIN_PER_CLASS,
                random_state=RANDOM_STATE,
            )
        elif n > MAX_PER_CLASS:
            group_bal = resample(
                group,
                replace=False,
                n_samples=MAX_PER_CLASS,
                random_state=RANDOM_STATE,
            )
        else:
            group_bal = group

        parts.append(group_bal)

    df_bal = pd.concat(parts, ignore_index=True)
    df_bal = df_bal.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df_bal


# =========================
# MAIN TRAINING
# =========================
def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    print("Loaded:", DATA_PATH)
    print("Columns:", df.columns.tolist())

    print("\nOriginal distribution:")
    print(df["label"].value_counts())

    # Balance data
    df_bal = balance_dataframe(df)

    print("\nBalanced distribution:")
    print(df_bal["label"].value_counts())

    # Split
    X = df_bal["text"]
    y = df_bal["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {acc:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    version_path = os.path.join(
        MODEL_DIR, f"text_classifier_{timestamp}.joblib"
    )
    latest_path = os.path.join(
        MODEL_DIR, "text_classifier.joblib"
    )

    joblib.dump(model, version_path)
    joblib.dump(model, latest_path)

    print(f"\nSaved version: {version_path}")
    print(f"Updated latest model: {latest_path}")

    # =========================
    # SAVE METRICS
    # =========================
    metrics_path = os.path.join(MODEL_DIR, "metrics.csv")

    macro_f1 = classification_report(
        y_test, y_pred, output_dict=True
    )["macro avg"]["f1-score"]

    row = [timestamp, acc, macro_f1]

    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "accuracy", "macro_f1"])

        writer.writerow(row)

    print(f"Saved metrics to: {metrics_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()



