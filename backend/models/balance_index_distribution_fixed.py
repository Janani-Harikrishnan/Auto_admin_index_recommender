import pandas as pd
import random
import os
import re

# --------------------------------------
# 1️⃣ Configuration
# --------------------------------------
INPUT_FILE = "merged_queries_with_indexes.csv"
OUTPUT_FILE = INPUT_FILE
df = pd.read_csv(INPUT_FILE)
max_count = df["target_indexes"].value_counts().max()
TARGET_SIZE_PER_LABEL = int(max_count * 0.9)

tables = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at"],
    "customers": ["customer_id", "customer_unique_id", "customer_city", "customer_state"],
    "order_items": ["order_item_id", "order_id", "product_id", "seller_id", "price", "freight_value"],
    "products": ["product_id", "product_category_name", "product_name_lenght", "product_weight_g"],
    "sellers": ["seller_id", "seller_city", "seller_state"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"]
}

conditions = [
    "price > 100",
    "freight_value < 50",
    "product_category_name = 'electronics'",
    "customer_state = 'SP'",
    "order_status = 'delivered'",
    "seller_state = 'RJ'",
    "product_weight_g BETWEEN 500 AND 2000"
]

# --------------------------------------
# 2️⃣ Function to infer target index from query text
# --------------------------------------
def infer_target_index(query):
    """
    Heuristic: detect likely table-column combination from SQL query.
    """
    query_lower = query.lower()
    for table, cols in tables.items():
        for col in cols:
            pattern = rf"\b{col}\b"
            if re.search(pattern, query_lower):
                return f"{table}{col}"
    return "unknown"

# --------------------------------------
# 3️⃣ Generate queries
# --------------------------------------
def generate_queries_for_index(table, column, n=100):
    queries = []
    for _ in range(n):
        q_type = random.choice(["select", "aggregate", "join"])
        if q_type == "select":
            cols = random.sample(tables[table], k=min(3, len(tables[table])))
            query = f"SELECT {', '.join(cols)} FROM {table} WHERE {column} IS NOT NULL"
            if random.random() < 0.5:
                query += f" AND {random.choice(conditions)}"
            query += f" LIMIT {random.randint(50, 200)};"
        elif q_type == "aggregate":
            func = random.choice(["COUNT", "SUM", "AVG"])
            group_col = random.choice(tables[table])
            query = f"SELECT {group_col}, {func}({column}) FROM {table} GROUP BY {group_col};"
        else:
            # Optional join with related table
            related_table = random.choice([t for t in tables if t != table])
            join_col = random.choice(tables[related_table])
            query = (
                f"SELECT {table}.{column}, {related_table}.{join_col} "
                f"FROM {table} JOIN {related_table} ON {table}.{column} = {related_table}.{join_col} "
                f"WHERE {random.choice(conditions)};"
            )
        # Now auto-generate label
        target_index = infer_target_index(query)
        queries.append({"query": query, "target_indexes": target_index})
    return queries

# --------------------------------------
# 4️⃣ Balance data
# --------------------------------------
def balance_data(input_file, target_size_per_label):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found!")

    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} queries from {input_file}")

    label_counts = df["target_indexes"].value_counts()
    underrepresented = label_counts[label_counts < target_size_per_label]
    print(f"Found {len(underrepresented)} underrepresented labels")

    synthetic_rows = []
    for label, count in underrepresented.items():
        missing = target_size_per_label - count
        if "_" in label or label == "unknown":
            continue

        parts = []
        for table in tables:
            if label.startswith(table):
                parts = [table, label[len(table):]]
                break
        if not parts:
            continue

        table, column = parts
        if column not in tables.get(table, []):
            continue

        print(f"🔹 Generating {missing} queries for {label}")
        synthetic_rows.extend(generate_queries_for_index(table, column, n=missing))

    if not synthetic_rows:
        print("✅ Dataset already balanced enough.")
        return df

    df_new = pd.DataFrame(synthetic_rows)
    df_final = pd.concat([df, df_new], ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"\n📈 Added {len(df_new)} new queries")
    print(f"💾 Saved balanced dataset to {OUTPUT_FILE}")
    print(f"Total queries after balancing: {len(df_final)}")

    return df_final

# --------------------------------------
# 5️⃣ Run
# --------------------------------------
if __name__ == "__main__":
    balance_data(INPUT_FILE, TARGET_SIZE_PER_LABEL)
