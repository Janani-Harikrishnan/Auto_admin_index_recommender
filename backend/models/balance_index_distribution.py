import pandas as pd
import random
import os
import re

INPUT_FILE = "merged_queries_with_indexes.csv"
OUTPUT_FILE = INPUT_FILE

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


# -----------------------------------------------------------
# Helper: Smartly split table+column from label (like "ordersorder_id")
# -----------------------------------------------------------
def split_label(label: str):
    """Return (table, column) tuple or (None, None) if not matched"""
    for table, cols in tables.items():
        # Try to find the column part by matching known columns
        for col in cols:
            if label == f"{table}{col}" or label.endswith(col):
                return table, col
    return None, None


# -----------------------------------------------------------
# Query generator for a given (table, column)
# -----------------------------------------------------------
def generate_queries_for_index(table, column, n=100):
    queries = []
    for _ in range(n):
        q_type = random.choice(["select", "aggregate"])
        if q_type == "select":
            cols = random.sample(tables[table], k=min(3, len(tables[table])))
            query = f"SELECT {', '.join(cols)} FROM {table} WHERE {column} IS NOT NULL"
            if random.random() < 0.5:
                query += f" AND {random.choice(conditions)}"
            query += f" LIMIT {random.randint(50, 200)};"
        else:
            func = random.choice(["COUNT", "SUM", "AVG"])
            group_col = random.choice(tables[table])
            query = f"SELECT {group_col}, {func}({column}) FROM {table} GROUP BY {group_col};"
        queries.append({"query": query, "target_indexes": f"{table}{column}"})
    return queries


# -----------------------------------------------------------
# Balance dataset by generating missing examples
# -----------------------------------------------------------
def balance_data(input_file, target_size_per_label=4000):
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} queries from {input_file}")

    label_counts = df["target_indexes"].value_counts()
    underrepresented = label_counts[label_counts < target_size_per_label]
    print(f"Found {len(underrepresented)} underrepresented labels")

    synthetic_rows = []
    for label, count in underrepresented.items():
        missing = target_size_per_label - count
        table, column = split_label(label)
        if not table or not column:
            continue

        print(f"🔹 Generating {missing} queries for {label}")
        synthetic_rows.extend(generate_queries_for_index(table, column, n=missing))

    if synthetic_rows:
        df_new = pd.DataFrame(synthetic_rows)
        df_final = pd.concat([df, df_new], ignore_index=True)
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n📈 Added {len(df_new)} new queries across {len(underrepresented)} labels")
        print(f"💾 Saved balanced dataset to {OUTPUT_FILE}")
        print(f"Total queries after balancing: {len(df_final)}")
    else:
        print("✅ Dataset already balanced enough (no new queries generated).")

if __name__ == "__main__":
    balance_data(INPUT_FILE, target_size_per_label=4000)
