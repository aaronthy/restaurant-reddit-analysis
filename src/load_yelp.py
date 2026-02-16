import pandas as pd

# use your local path
file_path = "C:/Users/aaron/yelp/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json"
print("Loading data...")

# only load first 5000 rows
df = pd.read_json(file_path, lines=True, nrows=5000)

print("Loaded!")

# keep only useful columns
df = df[["text", "stars", "useful"]]

print("\nSample data:")
print(df.head())

print("\nShape:", df.shape)

# save smaller dataset
df.to_csv("data/yelp_sample.csv", index=False)

print("\nSaved sample to data/yelp_sample.csv")
