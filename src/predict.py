import joblib

MODEL_PATH = "models/text_classifier.joblib"
model = joblib.load(MODEL_PATH)

tests = [
    "Service was slow and the staff was rude.",
    "Food was cold and bland, totally tasteless.",
    "Overpriced for the portion size.",
    "They forgot half my order and it was incorrect.",
    "management is bad"
]

for t in tests:
    label = model.predict([t.lower()])[0]
    print(f"\nText: {t}\nPredicted label: {label}")
