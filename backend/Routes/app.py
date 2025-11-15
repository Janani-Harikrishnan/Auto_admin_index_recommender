# app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import joblib
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import mysql.connector
import threading


# ========================= FastAPI Setup =========================
app = FastAPI(title="Custom Index Recommendation API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ========================= Global Variables =========================
MODEL_PATH = "rf_index_model.pkl"
MLB_PATH = "mlb_encoder.pkl"
CSV_PATH = r"C:\Users\Janani\Desktop\5th Sem\DBMS\PROJECT 1\merged_queries_with_indexes.csv"

multi_rf = None
mlb = None
unique_columns = joblib.load("unique_columns.pkl")
with open("feature_schema.json", "r") as f:
    feature_schema = json.load(f)

NUMERIC_FEATURES = feature_schema["numeric"]
VOCAB = feature_schema["columns_vocab"]

# Batch logging variables
BATCH_SIZE = 5
query_batch = []

# ========================= DB Connection =========================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ping#2123",
    database="olist_db"
)

# ========================= Pydantic Schema =========================
class QueryRequest(BaseModel):
    query: str

class QueryListRequest(BaseModel):
    queries: List[str]

# ========================= Feature Extraction =========================
def clean_column_name(col):
    col = col.lower().strip()
    col = re.sub(r'[^a-z0-9_]', '', col)
    return col

def extract_features(query):
    q = query.lower()
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
    join_cols  = [clean_column_name(c) for c in re.findall(r'join\s+\w+\s+on\s+([\w\.]+)', q)]
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

def encode_columns(cols, vocab):
    s = set(cols)
    return [1 if v in s else 0 for v in vocab]

def build_feature_row(features):
    numeric_df = pd.DataFrame([{
        k: features[k] for k in NUMERIC_FEATURES
    }])[NUMERIC_FEATURES]
    col_df = pd.DataFrame(
        [encode_columns(features["columns"], VOCAB)],
        columns=[f"col_{c}" for c in VOCAB]
    )
    return pd.concat([numeric_df, col_df], axis=1)

# ========================= Multi-output predict_proba =========================
def predict_proba_multioutput(model, X_new):
    probs = []
    for i, est in enumerate(model.estimators_):
        classes = getattr(est, "classes_", None)
        proba = est.predict_proba(X_new)
        if proba.shape[1] == 1:
            seen_class = classes[0]
            p = np.ones(X_new.shape[0]) if seen_class == 1 else np.zeros(X_new.shape[0])
        else:
            try:
                class_index = list(classes).index(1)
            except ValueError:
                class_index = 1
            p = proba[:, class_index]
        probs.append(p[0])
    return np.array(probs)

# ========================= Recommendation Function =========================
def recommend_indexes_single(query, threshold=0.25, top_k_fallback=3):
    features = extract_features(query)
    X_new = build_feature_row(features)
    try:
        probs = predict_proba_multioutput(multi_rf, X_new)
        idx_sorted = np.argsort(-probs)
        top_pairs = [(mlb.classes_[i], float(probs[i])) for i in idx_sorted[:10]]
        picks = [mlb.classes_[i] for i in range(len(probs)) if probs[i] >= threshold]
        if not picks:
            picks = [lbl for lbl, _ in top_pairs[:top_k_fallback]]
        return [list(sorted(set(picks)))]
    except Exception as e:
        print(f"Error processing query: {query}", e)
        return [[]]

# ========================= Batch Logging =========================
def log_batch_to_csv():
    global query_batch
    if not query_batch:
        return
    try:
        df_batch = pd.DataFrame(query_batch)
        df_batch.to_csv(CSV_PATH, mode='a', header=False, index=False)
        print(f"Logged {len(query_batch)} queries to CSV")
    except Exception as e:
        print("Error logging batch:", e)
    query_batch = []

# ========================= Background Retraining =========================
def retrain_model_periodically(interval_sec=3600):
    global multi_rf, mlb
    while True:
        time.sleep(interval_sec)
        try:
            print("Retraining model from CSV...")
            df = pd.read_csv(CSV_PATH)
            # Assuming df has columns X features + target indexes in multi-label format
            X_train = df[NUMERIC_FEATURES + [f"col_{c}" for c in VOCAB]]
            y_train = df["indexes"]  # adjust if your CSV column for labels is named differently
            multi_rf.fit(X_train, y_train)
            joblib.dump(multi_rf, MODEL_PATH)
            print("Model retrained successfully!")
        except Exception as e:
            print("Error retraining model:", e)

# ========================= Startup Event =========================
@app.on_event("startup")
def load_model():
    global multi_rf, mlb
    print("Loading ML model into memory...")
    multi_rf = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    print("Model loaded!")
    # Start background retraining thread
    threading.Thread(target=retrain_model_periodically, daemon=True).start()

# ========================= API Endpoints =========================
@app.post("/execute-query/")
def execute_query(request: QueryRequest):
    query = request.query
    start_time = time.time()
    recs = recommend_indexes_single(query)
    query_results = None
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        query_results = {"columns": columns, "rows": rows}
    except Exception as e:
        query_results = {"error": str(e)}
    # Add to batch
    query_batch.append({"query": query, "indexes": recs})
    if len(query_batch) >= BATCH_SIZE:
        log_batch_to_csv()
    end_time = time.time()
    return {
        "query": query,
        "execution_time": round(end_time - start_time, 4),
        "recommendation": recs,
        "result": query_results
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html = Path("templates/index.html").read_text()
    return HTMLResponse(content=html)


# ========================= Debug Endpoints =========================
@app.get("/pending-batch")
def get_pending_batch():
    # Show queries waiting to be written to CSV
    return {"pending_queries": query_batch}


@app.post("/flush-batch")
def flush_batch():
    global query_batch
    if query_batch:
        try:
            df = pd.DataFrame(query_batch)
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
            query_batch = []
            return {"status": "Batch flushed to CSV!"}
        except Exception as e:
            return {"error": str(e)}
    return {"status": "No queries to flush."}
