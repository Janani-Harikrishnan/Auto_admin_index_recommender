import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt

# ---------- Helper ----------
def clean_column_name(col: str) -> str:
    """Normalize column names (same function you already use)."""
    col = col.lower().strip()
    col = re.sub(r'[^a-z0-9_]', '', col)
    return col

# ---------- Load dataset ----------
df = pd.read_csv("merged_queries_with_indexes.csv")

# Fill any missing index labels with empty string
df["target_indexes"] = df["target_indexes"].fillna("")

# Keep only queries that have at least one target index
mask_nonempty = df["target_indexes"].str.strip() != ""
df = df[mask_nonempty].copy()
print(f"✅ Kept {len(df)} queries with non-empty target_indexes.\n")

# ---------- Extract & clean labels ----------
# Convert comma-separated index columns into a list per query
df["label_list"] = df["target_indexes"].apply(
    lambda x: [clean_column_name(t.strip()) for t in str(x).split(",") if t.strip()]
)

# Flatten all labels into a single list
all_labels = [label for sublist in df["label_list"] for label in sublist]

# ---------- Frequency Analysis ----------
label_counts = Counter(all_labels)
total_unique = len(label_counts)

print(f"🔹 Total unique index labels: {total_unique}")
print("\n🔸 Top 20 most frequent index columns:")
for lbl, cnt in label_counts.most_common(20):
    print(f"  {lbl:<25} {cnt}")

# ---------- Convert to DataFrame for plotting ----------
label_df = pd.DataFrame(label_counts.items(), columns=["column_name", "frequency"])
label_df = label_df.sort_values("frequency", ascending=False)

# ---------- Visualization ----------
plt.figure(figsize=(12, 6))
plt.bar(label_df["column_name"][:20], label_df["frequency"][:20])
plt.xticks(rotation=45, ha='right')
plt.xlabel("Index Column Name")
plt.ylabel("Frequency")
plt.title("Top 20 Most Frequent Index Targets in Query Log")
plt.tight_layout()
plt.show()

# ---------- Identify Imbalance ----------
# You can tune the threshold depending on your data
threshold = max(5, 0.05 * len(df))
underrepresented = label_df[label_df["frequency"] < threshold]["column_name"].tolist()

print(f"\n⚠️ Underrepresented columns (appear < {threshold} times):")
print(underrepresented)

# ---------- Optional: Save for future balancing ----------
label_df.to_csv("label_frequency_summary.csv", index=False)
print("\n✅ Saved label frequencies → label_frequency_summary.csv")

# ---------- Suggestions ----------
print("\n💡 Next Steps:")
print("1️⃣ Generate or duplicate queries involving the underrepresented columns above.")
print("2️⃣ Balance your dataset so each key column appears roughly similar number of times.")
print("3️⃣ Retrain your ML model after balancing to improve accuracy and generalization.")
