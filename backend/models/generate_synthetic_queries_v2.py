import pandas as pd
import random
import os

# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
NUM_QUERIES = 10000
OUTPUT_FILE = "merged_queries_with_indexes.csv"

tables = [
    "orders",
    "customers",
    "order_items",
    "products",
    "sellers",
    "geolocation"
]

columns = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at"],
    "customers": ["customer_id", "customer_unique_id", "customer_city", "customer_state"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
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

joins = [
    ("orders", "order_items", "order_id"),
    ("orders", "customers", "customer_id"),
    ("order_items", "products", "product_id"),
    ("order_items", "sellers", "seller_id")
]

# ------------------------------------------
# QUERY GENERATORS
# ------------------------------------------
def generate_simple_query():
    table = random.choice(tables)
    selected_cols = random.sample(columns[table], k=random.randint(1, min(3, len(columns[table]))))
    query = f"SELECT {', '.join(selected_cols)} FROM {table}"

    if random.random() < 0.7:
        query += f" WHERE {random.choice(conditions)}"
    if random.random() < 0.5:
        query += f" LIMIT {random.randint(10, 200)}"

    query += ";"

    # Infer likely target index from WHERE clause or table
    idx = [f"{table}{selected_cols[0]}"]
    return query, ",".join(idx)

def generate_join_query():
    (t1, t2, join_col) = random.choice(joins)
    cols1 = random.sample(columns[t1], k=random.randint(1, 2))
    cols2 = random.sample(columns[t2], k=random.randint(1, 2))
    query = f"""
        SELECT {', '.join([f'{t1}.{c}' for c in cols1])}, {', '.join([f'{t2}.{c}' for c in cols2])}
        FROM {t1}
        JOIN {t2} ON {t1}.{join_col} = {t2}.{join_col}
    """.strip()

    if random.random() < 0.7:
        query += f" WHERE {random.choice(conditions)}"
    query += f" LIMIT {random.randint(20, 500)};"

    idx = [f"{t1}{join_col}", f"{t2}{join_col}"]
    return query, ",".join(idx)

def generate_aggregate_query():
    table = random.choice(list(columns.keys()))
    numeric_cols = [c for c in columns[table] if any(k in c for k in ["price", "value", "weight", "lenght", "id"])]
    if not numeric_cols:
        numeric_cols = [random.choice(columns[table])]

    col = random.choice(numeric_cols)
    agg_func = random.choice(["SUM", "AVG", "COUNT"])
    group_col = random.choice(columns[table])

    query = f"SELECT {group_col}, {agg_func}({col}) FROM {table} GROUP BY {group_col}"
    query += ";"

    idx = [f"{table}{group_col}"]
    return query, ",".join(idx)

# ------------------------------------------
# MAIN SYNTHETIC GENERATION
# ------------------------------------------
def generate_synthetic_queries(num_queries=NUM_QUERIES):
    queries = []
    for _ in range(num_queries):
        qtype = random.choice(["simple", "join", "aggregate"])
        if qtype == "simple":
            query, idx = generate_simple_query()
        elif qtype == "join":
            query, idx = generate_join_query()
        else:
            query, idx = generate_aggregate_query()
        queries.append({"query": query.strip(), "target_indexes": idx})
    return pd.DataFrame(queries)

# ------------------------------------------
# SAVE OR APPEND TO CSV
# ------------------------------------------
def main():
    df_new = generate_synthetic_queries()
    print(f"✅ Generated {len(df_new)} synthetic queries")

    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        merged_df = pd.concat([df_existing, df_new], ignore_index=True)
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"📄 Appended to existing file: {OUTPUT_FILE}")
        print(f"Total rows after append: {merged_df.shape[0]}")
    else:
        df_new.to_csv(OUTPUT_FILE, index=False)
        print(f"🆕 Created new file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
