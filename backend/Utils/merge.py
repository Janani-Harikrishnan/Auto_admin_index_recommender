import pandas as pd

# Load your two CSV files
df1 = pd.read_csv("synthetic_olist_queries_augmented.csv")
df2 = pd.read_csv("synthetic_queries.csv")

# Merge them
merged_df = pd.concat([df1, df2], ignore_index=True)

# Optional: shuffle the rows
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged CSV
merged_df.to_csv("synthetic_queries_merged.csv", index=False)
