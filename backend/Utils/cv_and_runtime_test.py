"""
cv_and_runtime_test.py

Performs:
  1) K-fold cross-validation of your saved RandomForest multioutput model.
  2) Runtime comparison for sample queries on MySQL:
       - measure avg runtime before indexes
       - optionally create predicted indexes (heuristic mapping)
       - measure avg runtime after indexes
       - clean up created indexes (if created)

By default: DRY-RUN (APPLY_INDEXES=False). Set APPLY_INDEXES=True to actually create indexes.
"""

import time
import json
import re
import joblib
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import mysql.connector

# --------------------------
# User config (edit as needed)
# --------------------------
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "Ping#2123"
MYSQL_DB = "olist_db"

ARTIFACT_MODEL = "rf_index_model.pkl"
ARTIFACT_MLB = "mlb_encoder.pkl"
FEATURE_SCHEMA = "feature_schema.json"
DATA_CSV = "merged_queries_with_indexes.csv"

NUM_RUNTIME_QUERIES = 10
WARMUP_RUNS = 1
TIMED_RUNS = 3
APPLY_INDEXES = False
TMP_INDEX_PREFIX = "ml_reco_idx_"

# --------------------------
# --- Helper snippets included ---
# --------------------------

# --- 1️⃣ Fix ambiguous columns ---
def qualify_columns(sql, table_aliases):
    for table, alias in table_aliases.items():
        sql = sql.replace(f"{table}.", f"{alias}.")
        for col in ["order_id", "customer_id", "product_id"]:
            if f" {col}" in sql and f"{alias}.{col}" not in sql:
                sql = sql.replace(f" {col}", f" {alias}.{col}")
    return sql

# --- 2️⃣ Stable timing ---
def time_query(cursor, sql, repeats=7):
    times = []
    for _ in range(repeats):
        start = time.time()
        cursor.execute(sql)
        cursor.fetchall()
        times.append(time.time() - start)
    return float(np.median(times[1:]))  # drop first (warm-up)

# --- 3️⃣ Label cleaning & mapping ---
label_to_tablecol = {
    "ordersorder_id": ("orders", "order_id"),
    "orderitemsorder_id": ("orderitems", "order_id"),
    "paymentsorder_id": ("payments", "order_id"),
    "reviewsorder_id": ("reviews", "order_id"),
}

def clean_label(label):
    label = label.lower()
    label = re.sub(r'[^a-z0-9_]', '', label)
    label = label.replace("desc", "").replace("having", "")
    return label

def map_labels_to_indexes(pred_labels):
    mapped = []
    for lbl in pred_labels:
        lbl_clean = clean_label(lbl)
        if lbl_clean in label_to_tablecol:
            mapped.append(label_to_tablecol[lbl_clean])
    return mapped

# --------------------------
# Load artifacts + data
# --------------------------
print("Loading model and artifacts...")
model = joblib.load(ARTIFACT_MODEL)
mlb = joblib.load(ARTIFACT_MLB)
with open(FEATURE_SCHEMA, "r") as f:
    schema = json.load(f)
NUMERIC_FEATURES = schema["numeric"]
VOCAB = schema["columns_vocab"]

print("Loading dataset for CV and runtime sampling...")
df = pd.read_csv(DATA_CSV)
df['target_indexes'] = df['target_indexes'].fillna("")
df = df[df['target_indexes'].str.strip() != ""].reset_index(drop=True)
print(f"Rows available for experiments (non-empty labels): {len(df)}")

# Build feature matrix
def clean_column_name(col: str) -> str:
    col = str(col).lower().strip()
    col = re.sub(r'[^a-z0-9_]', '', col)
    return col

def extract_features_for_row(query: str):
    q = (query or "").lower()
    select_cols = []
    m = re.search(r'select\s+(.*?)\s+from', q, flags=re.S)
    if m:
        raw = [c.strip() for c in m.group(1).split(',')]
        for c in raw:
            c = re.sub(r'\s+as\s+.*$', '', c)
            func_m = re.match(r'([a-z_]+)\(([\w\.]+)\)', c)
            if func_m:
                func, inner = func_m.group(1), func_m.group(2)
                select_cols.append(clean_column_name(inner))
                select_cols.append(clean_column_name(func + inner))
            else:
                select_cols.append(clean_column_name(c))

    where_cols = [clean_column_name(c) for c in re.findall(r'where\s+([\w\.]+)', q)]
    join_cols = [clean_column_name(c) for c in re.findall(r'join\s+\w+\s+on\s+([\w\.]+)', q)]
    group_by_cols = []
    for gb in re.findall(r'group by\s+([\w\.,\s]+)', q):
        group_by_cols += [clean_column_name(c.strip()) for c in gb.split(',') if c.strip()]
    order_by_cols = []
    for ob in re.findall(r'order by\s+([\w\.,\s]+)', q):
        order_by_cols += [clean_column_name(c.strip()) for c in ob.split(',') if c.strip()]

    all_cols = sorted(set(select_cols + where_cols + join_cols + group_by_cols + order_by_cols))
    num_tables = len(re.findall(r'from\s+(\w+)|join\s+(\w+)', q))
    num_where_conditions = len(re.findall(r'\b(where|and|or)\b', q))
    num_join_columns = len(join_cols)
    num_group_order_columns = len(group_by_cols + order_by_cols)
    num_columns = len(all_cols)

    return {
        "num_tables": num_tables,
        "num_where_conditions": num_where_conditions,
        "num_join_columns": num_join_columns,
        "num_group_order_columns": num_group_order_columns,
        "num_columns": num_columns,
        "columns": all_cols,
    }

rows_features = [extract_features_for_row(q) for q in df['query']]
X_numeric = pd.DataFrame(rows_features)[NUMERIC_FEATURES]

def encode_columns(cols, vocab):
    s = set(cols)
    return [1 if v in s else 0 for v in vocab]

cols_binary = pd.DataFrame(
    [encode_columns(r['columns'], VOCAB) for r in rows_features],
    columns=[f"col_{c}" for c in VOCAB]
)
X_full = pd.concat([X_numeric.reset_index(drop=True), cols_binary.reset_index(drop=True)], axis=1)

# Build Y
def parse_label_list(s):
    return [clean_column_name(t.strip()) for t in str(s).split(',') if t.strip()]

y_lists = df['target_indexes'].apply(parse_label_list)
Y_full = mlb.transform(y_lists)

print("Feature matrix shape:", X_full.shape)
print("Label matrix shape:", Y_full.shape)

# --------------------------
# 1) K-Fold Cross-Validation
# --------------------------
print("\n=== K-Fold Cross-Validation (5-fold) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = defaultdict(list)
fold_idx = 0
for train_idx, val_idx in kf.split(X_full):
    fold_idx += 1
    Xtr, Xval = X_full.iloc[train_idx], X_full.iloc[val_idx]
    Ytr, Yval = Y_full[train_idx], Y_full[val_idx]

    base = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    clf = MultiOutputClassifier(base)
    clf.fit(Xtr, Ytr)
    Ypred = clf.predict(Xval)

    metrics['f1'].append(f1_score(Yval, Ypred, average='micro'))
    metrics['precision'].append(precision_score(Yval, Ypred, average='micro', zero_division=0))
    metrics['recall'].append(recall_score(Yval, Ypred, average='micro', zero_division=0))
    metrics['accuracy'].append(accuracy_score(Yval, Ypred))
    metrics['hamming_loss'].append(hamming_loss(Yval, Ypred))

    print(f"Fold {fold_idx}: f1={metrics['f1'][-1]:.4f} precision={metrics['precision'][-1]:.4f} recall={metrics['recall'][-1]:.4f} accuracy={metrics['accuracy'][-1]:.4f}")

def summarize_metric(name, arr):
    return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

print("\nCross-val summary (mean ± std):")
for key in ['f1', 'precision', 'recall', 'accuracy', 'hamming_loss']:
    print(f"{key:10}: {summarize_metric(key, metrics[key])}")

# --------------------------
# 2) Runtime comparison
# --------------------------
print("\n=== Runtime comparison on MySQL (sample queries) ===")
conn = mysql.connector.connect(
    host=MYSQL_HOST, port=MYSQL_PORT,
    user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB,
    autocommit=True
)
cursor = conn.cursor()
cursor.execute("SHOW TABLES")
tables = [row[0].lower() for row in cursor.fetchall()]
print("Tables found in DB:", tables)

def build_feature_vector_for_query(query):
    f = extract_features_for_row(query)
    numeric = {k: f[k] for k in NUMERIC_FEATURES}
    col_bits = encode_columns(f['columns'], VOCAB)
    numeric_df = pd.DataFrame([numeric])[NUMERIC_FEATURES]
    col_df = pd.DataFrame([col_bits], columns=[f"col_{c}" for c in VOCAB])
    return pd.concat([numeric_df, col_df], axis=1)

sample_df = df.head(NUM_RUNTIME_QUERIES)
results = []

for idx, row in sample_df.iterrows():
    query_text = row['query']
    print("\n-- Query #{} --".format(idx))
    print("SQL:", query_text)

    # Predict indexes
    Xnew = build_feature_vector_for_query(query_text)
    preds = model.predict(Xnew)
    predicted_labels = mlb.inverse_transform(preds)
    predicted_labels = predicted_labels[0] if len(predicted_labels) > 0 else []
    print("Predicted labels:", predicted_labels)

    # Map labels using snippet
    mapped_indexes = map_labels_to_indexes(predicted_labels)

    # Fix ambiguous columns
    sql = qualify_columns(query_text, {t.title(): t.title() for t in tables})

    # Measure runtime
    before_time = time_query(cursor, sql)

    if APPLY_INDEXES and mapped_indexes:
        for table, column in mapped_indexes:
            idx_name = f"{TMP_INDEX_PREFIX}{table}_{column}"
            idx_name = re.sub(r'[^a-z0-9_]', '_', idx_name.lower())[:64]
            try:
                cursor.execute(f"CREATE INDEX {idx_name} ON {table}({column})")
            except Exception as e:
                print(f"Failed to create index {idx_name}: {e}")
        time.sleep(0.5)

    after_time = time_query(cursor, sql)

    if APPLY_INDEXES and mapped_indexes:
        for table, column in mapped_indexes:
            idx_name = f"{TMP_INDEX_PREFIX}{table}_{column}"
            try:
                cursor.execute(f"DROP INDEX {idx_name} ON {table}")
            except:
                pass

    results.append({
        "query": query_text,
        "predicted_labels": predicted_labels,
        "mapped": mapped_indexes,
        "time_before": before_time,
        "time_after": after_time,
        "improvement_ratio": (before_time / after_time) if after_time else None
    })

    print(f"before={before_time:.4f}s after={after_time:.4f}s improvement={(before_time/after_time):.2f}x")
    print(f"mapped_indexes: {mapped_indexes}")

cursor.close()
conn.close()
print("\nDone.")
