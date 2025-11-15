import random
import pandas as pd
import os

# -------------------------
# 1. Define database schema
# -------------------------

tables = {
    "Customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    "Orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
    "Products": ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght",
                 "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    "Sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "OrderItems": ["order_item_id", "order_id", "product_id", "seller_id", "shipping_limit_date", "price",
                   "freight_value"],
    "Payments": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
    "Reviews": ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
                "review_creation_date", "review_answer_timestamp"]
}

# Example values for filters
sample_values = {
    "order_status": ["delivered", "shipped", "canceled", "invoiced", "processing"],
    "customer_state": ["SP", "RJ", "MG", "RS", "BA"],
    "product_category_name": ["fashion", "electronics", "home", "beauty", "sports"],
    "payment_type": ["credit_card", "boleto", "voucher", "debit_card"],
    "review_score": [1, 2, 3, 4, 5]
}

# -------------------------
# 2. Functions to create queries
# -------------------------

def random_select_clause(table):
    cols = tables[table]
    n_cols = random.randint(1, min(4, len(cols)))
    return ", ".join(random.sample(cols, n_cols))

def random_where_clause(table):
    col = random.choice(tables[table])
    if col in sample_values:
        val = random.choice(sample_values[col])
        if isinstance(val, str):
            return f"{col} = '{val}'"
        else:
            return f"{col} = {val}"
    elif "date" in col:
        return f"{col} > '2023-01-01'"
    elif "price" in col or "weight" in col or "qty" in col:
        return f"{col} > {random.randint(1, 100)}"
    else:
        return f"{col} IS NOT NULL"

def random_join_clause():
    # Choose two tables with a common foreign key
    joins = [
        ("Orders", "Customers", "customer_id"),
        ("OrderItems", "Orders", "order_id"),
        ("OrderItems", "Products", "product_id"),
        ("OrderItems", "Sellers", "seller_id"),
        ("Payments", "Orders", "order_id"),
        ("Reviews", "Orders", "order_id")
    ]
    table1, table2, key = random.choice(joins)
    return table1, table2, key

def generate_query():
    query_type = random.choice(["simple_select", "join_select", "aggregate_select"])
    
    if query_type == "simple_select":
        table = random.choice(list(tables.keys()))
        select_clause = random_select_clause(table)
        where_clause = random_where_clause(table) if random.random() < 0.7 else ""
        query = f"SELECT {select_clause} FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        return query + ";"

    elif query_type == "join_select":
        t1, t2, key = random_join_clause()
        select_clause = f"{random_select_clause(t1)}, {random_select_clause(t2)}"
        where_clause = random_where_clause(t1) if random.random() < 0.7 else ""
        query = f"SELECT {select_clause} FROM {t1} JOIN {t2} ON {t1}.{key} = {t2}.{key}"
        if where_clause:
            query += f" WHERE {where_clause}"
        return query + ";"

    elif query_type == "aggregate_select":
        table = random.choice(list(tables.keys()))
        col = random.choice(tables[table])
        agg_func = random.choice(["COUNT", "SUM", "AVG", "MIN", "MAX"])
        select_clause = f"{agg_func}({col})"
        group_by = ""
        if random.random() < 0.5:
            group_by = random.choice(tables[table])
            query = f"SELECT {select_clause}, {group_by} FROM {table} GROUP BY {group_by}"
        else:
            query = f"SELECT {select_clause} FROM {table}"
        return query + ";"

# -------------------------
# 3. Generate synthetic queries
# -------------------------

num_queries = 1000  # Change as needed
synthetic_queries = []

for _ in range(num_queries):
    synthetic_queries.append(generate_query())

# -------------------------
# 4. Save to CSV
# -------------------------

output_file = "synthetic_queries.csv"
pd.DataFrame({"query": synthetic_queries}).to_csv(output_file, index=False)
print(f"{num_queries} synthetic queries saved to {output_file}")
