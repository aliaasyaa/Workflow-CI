"""
modelling.py
============
Kriteria Basic  - Melatih model dengan MLflow + argparse (untuk MLflow Project)
Dataset         : stroke_preprocessed.csv
Model           : Random Forest Classifier
Logging         : MLflow autolog + manual logging
Target          : stroke  (0 = tidak stroke, 1 = stroke)
"""

import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────
# 1. ARGUMENT PARSER
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Stroke Classification - Random Forest")
parser.add_argument("--n_estimators",  type=int,   default=100,        help="Jumlah pohon")
parser.add_argument("--max_depth",     type=int,   default=0,          help="Kedalaman pohon (0=None)")
parser.add_argument("--random_state",  type=int,   default=42,         help="Random state")
parser.add_argument("--class_weight",  type=str,   default="balanced", help="class_weight parameter")
args = parser.parse_args()

max_depth = None if args.max_depth == 0 else args.max_depth

print("=" * 55)
print("  STROKE CLASSIFICATION — RANDOM FOREST")
print("=" * 55)
print(f"  n_estimators  : {args.n_estimators}")
print(f"  max_depth     : {max_depth}")
print(f"  random_state  : {args.random_state}")
print(f"  class_weight  : {args.class_weight}")
print("=" * 55)

# ─────────────────────────────────────────────
# 2. KONFIGURASI MLflow
#
#    PENTING: Ketika dijalankan via `mlflow run`,
#    env var MLFLOW_RUN_ID sudah di-set otomatis.
#    Jangan panggil set_experiment() karena akan
#    menyebabkan mismatch experiment ID.
#    Panggil set_experiment() HANYA saat lokal.
# ─────────────────────────────────────────────
EXPERIMENT_NAME  = "Stroke_Classification_Basic"
TRACKING_URI     = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
IS_MLFLOW_RUN    = os.environ.get("MLFLOW_RUN_ID") is not None   # True jika via mlflow run

mlflow.set_tracking_uri(TRACKING_URI)
print(f"\n[INFO] Tracking URI : {TRACKING_URI}")
print(f"[INFO] Mode         : {'mlflow run (CI)' if IS_MLFLOW_RUN else 'direct (lokal)'}")

# set_experiment hanya kalau dijalankan langsung (bukan via mlflow run)
if not IS_MLFLOW_RUN:
    mlflow.set_experiment(EXPERIMENT_NAME)

# ─────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "stroke_preprocessed.csv")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    first_line = f.readline()
sep = ';' if first_line.count(';') >= first_line.count(',') else ','
print(f"[INFO] Separator    : '{sep}'")

df = pd.read_csv(DATA_PATH, sep=sep)
print(f"[INFO] Dataset      : {df.shape[0]} baris x {df.shape[1]} kolom")

# ─────────────────────────────────────────────
# 4. BERSIHKAN FORMAT ANGKA
# ─────────────────────────────────────────────
def clean_numeric(series):
    s = series.astype(str).str.strip()
    def parse_val(x):
        x = x.replace(' ', '')
        dot_count   = x.count('.')
        comma_count = x.count(',')
        if dot_count == 0 and comma_count == 0:
            return x
        elif dot_count == 1 and comma_count == 0:
            return x
        elif dot_count == 0 and comma_count == 1:
            return x.replace(',', '.')
        elif dot_count >= 1 and comma_count == 1:
            return x.replace('.', '').replace(',', '.')
        elif dot_count >= 2 and comma_count == 0:
            parts = x.split('.')
            return f"{''.join(parts[:-1])}.{parts[-1]}"
        return x
    s = s.apply(parse_val)
    return pd.to_numeric(s, errors='coerce')

for col in df.columns:
    df[col] = clean_numeric(df[col])

before = len(df)
df.dropna(inplace=True)
if before != len(df):
    print(f"[INFO] Baris dihapus (NaN) : {before - len(df)}")

# ─────────────────────────────────────────────
# 5. FITUR & TARGET + SCALING
# ─────────────────────────────────────────────
X = df.drop(columns=['stroke'])
y = df['stroke'].astype(int)

print(f"[INFO] Jumlah fitur : {X.shape[1]}")
print(f"[INFO] Distribusi target :\n{y.value_counts().to_string()}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X        = pd.DataFrame(X_scaled, columns=X.columns)

# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=args.random_state, stratify=y
)
print(f"\n[INFO] Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 7. TRAINING + MLflow LOGGING
#
#    FIX UTAMA:
#    - Jangan pass run_id ke start_run()
#    - MLflow otomatis baca MLFLOW_RUN_ID dari env
#    - Kalau tidak ada MLFLOW_RUN_ID, buat run baru
# ─────────────────────────────────────────────
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False,
)

# TIDAK pass run_id — MLflow auto-deteksi MLFLOW_RUN_ID dari env var
run_name = None if IS_MLFLOW_RUN else "RandomForest_Stroke_Basic"

with mlflow.start_run(run_name=run_name) as run:

    RUN_ID = run.info.run_id
    print(f"\n[MLflow] Run ID     : {RUN_ID}")
    print(f"[MLflow] Experiment : {EXPERIMENT_NAME}")
    print("[MLflow] Training dimulai...\n")

    model = RandomForestClassifier(
        n_estimators = args.n_estimators,
        max_depth    = max_depth,
        random_state = args.random_state,
        n_jobs       = -1,
        class_weight = args.class_weight,
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("test_roc_auc",   auc)
    mlflow.log_metric("test_f1_score",  f1)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall",    recall)

    print("=" * 55)
    print("  HASIL EVALUASI MODEL")
    print("=" * 55)
    print(f"  Test Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC Score   : {auc:.4f}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"\n  Confusion Matrix :\n  {confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report :")
    print(classification_report(y_test, y_pred,
          target_names=["Tidak Stroke", "Stroke"]))

    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("  Feature Importances (Top 5) :")
    print(importances.sort_values(ascending=False).head(5).to_string())
    print("=" * 55)

# ─────────────────────────────────────────────
# 8. SIMPAN RUN_ID KE FILE
# ─────────────────────────────────────────────
run_id_path = os.path.join(BASE_DIR, "run_id.txt")
with open(run_id_path, "w") as f:
    f.write(RUN_ID)
print(f"\n[INFO] Run ID disimpan → {run_id_path}")
print("[INFO] Training selesai.")
