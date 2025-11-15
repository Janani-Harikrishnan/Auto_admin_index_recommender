import pandas as pd
import re
import json
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score
import joblib

# ----------------------- Helpers -----------------------
def clean_column_name(col: str) -> str:
    col = col.lower().strip()
    col = re.sub(r'[^a-z0-9_]', '', col)
    return col

def extract_features(query: str):
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

# ----------------------- Load data -----------------------
df = pd.read_csv("merged_queries_with_indexes.csv")
df["target_indexes"] = df["target_indexes"].fillna("")

mask_nonempty = df["target_indexes"].str.strip() != ""
df = df[mask_nonempty].copy()
print(f"Kept {len(df)} queries with non-empty target_indexes.")

y_list = df["target_indexes"].apply(
    lambda x: [clean_column_name(t.strip()) for t in str(x).split(",") if t.strip()]
)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_list)

label_counts = Counter([t for row in y_list for t in row])
print("\nTop 20 labels by frequency:")
for lbl, cnt in label_counts.most_common(20):
    print(f"  {lbl}: {cnt}")
print(f"\nTotal unique labels: {len(mlb.classes_)}")

# ----------------------- Build features -----------------------
features = [extract_features(q) for q in df["query"]]
X_basic = pd.DataFrame(features)

all_cols = [c for row in X_basic["columns"] for c in row]
unique_columns = sorted(list(set(all_cols)))

def encode_columns(cols, vocab):
    s = set(cols)
    return [1 if v in s else 0 for v in vocab]

X_cols = pd.DataFrame(
    [encode_columns(cols, unique_columns) for cols in X_basic["columns"]],
    columns=[f"col_{c}" for c in unique_columns],
)

X_numeric = X_basic.drop(columns=["columns"])
X = pd.concat([X_numeric.reset_index(drop=True), X_cols.reset_index(drop=True)], axis=1)

print(f"\nFeature matrix shape: {X.shape}")

# ----------------------- Train / Evaluate -----------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,              # limit tree depth
    min_samples_split=4,       # require at least 4 samples to split
    min_samples_leaf=2,        # leaf must have at least 2 samples
    max_features='sqrt',       # random subset of features per split
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(rf)
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# Evaluation
train_acc = accuracy_score(Y_train, Y_pred_train)
test_acc = accuracy_score(Y_test, Y_pred_test)
f1 = f1_score(Y_test, Y_pred_test, average="micro")
prec = precision_score(Y_test, Y_pred_test, average="micro")
rec = recall_score(Y_test, Y_pred_test, average="micro")
ham_loss = hamming_loss(Y_test, Y_pred_test)

print("\n---------------- Evaluation Summary ----------------")
print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")
print(f"F1 Score (micro)  : {f1:.4f}")
print(f"Precision (micro) : {prec:.4f}")
print(f"Recall (micro)    : {rec:.4f}")
print(f"Hamming Loss (error rate): {ham_loss:.4f}")
print("---------------------------------------------------")

# ----------------------- Persist -----------------------
joblib.dump(model, "rf_index_model.pkl")
joblib.dump(mlb, "mlb_encoder.pkl")
joblib.dump(unique_columns, "unique_columns.pkl")

feature_schema = {
    "numeric": list(X_numeric.columns),
    "columns_vocab": unique_columns,
}
with open("feature_schema.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

print("\n✅ Saved: rf_index_model.pkl, mlb_encoder.pkl, unique_columns.pkl, feature_schema.json")
