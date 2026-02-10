"""
Sequential Model Approach for Learner Type Prediction (TensorFlow/Keras)
- Uses true sequences (no per-sequence aggregates)
- Pads variable-length sequences + Masking
- Reduces high-dim embeddings with IncrementalPCA (fit on TRAIN rows only)
- Scales snapshot-level features with StandardScaler (fit on TRAIN rows only)
- Splits train/test by student (user_id) to avoid leakage
Outputs:
- best_sequence_model.keras
- sequence_preprocessing.pkl
- sequence_test_predictions.csv
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 67
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "thinned_snapshots_60s_with_embeddings.parquet"
LABELS_CSV   = "final_labels.csv"

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "thinned_snapshots_60s_with_embeddings.parquet"
LABELS_CSV   = "final_labels.csv"
ALL_FEATURES_CSV = "all_features_df.csv"   # <--- NEW

# ---- Feature selection knobs ----
# Put features into "blocks" so you can enable/disable groups easily.
FEATURE_BLOCKS = {
    # Original snapshot-level numeric features
    "snapshot_numeric": [
        "seconds_from_start", "chars_in_snapshot", "saves_before_snapshot",
        "keystrokes_before_snapshot", "ai_feedback_before_snapshot",
        "submits_before_snapshot", "executes_before_snapshot",
        "snapshot_sequence_num", "total_snapshots_in_session",
        "thinned_snapshots_in_session", "min_interval_seconds",
        "task_code_cosine", "task_code_distance",
    ],

    # all_features_df.csv numeric columns will be auto-added here after load
    # (you can still disable the whole block with ENABLED_BLOCKS)
    "all_features_numeric": [],

    # If you know specific categorical cols to use, list them here.
    # Otherwise we'll auto-detect object/category columns after merge.
    "categorical": []
}

# Enable/disable blocks here (easy toggle)
ENABLED_BLOCKS = [
    "snapshot_numeric",
    "all_features_numeric",
    # "categorical",  # enable if you want to include categorical features
]

EMB_COLS = ["code_embedding", "task_embedding"]  # if present

# Dimensionality reduction
EMB_N_COMPONENTS = 32          # per embedding column (code + task)
IPCA_BATCH_SIZE  = 1024        # adjust if memory issues

# Sequence padding
MAXLEN_PERCENTILE = 95         # use 95th percentile of TRAIN sequence lengths
MIN_MAXLEN        = 2

# Model/training
BATCH_SIZE   = 16
EPOCHS       = 100
LEARNING_RATE = 2e-4
PATIENCE     = 50

# -----------------------------
# Helpers
# -----------------------------
def infer_embedding_dim(series: pd.Series) -> int:
    """Find first non-empty embedding and return its length; else 0."""
    for v in series:
        if isinstance(v, (list, np.ndarray)) and len(v) > 0:
            return len(v)
    return 0

def batch_embedding_matrix(series: pd.Series, dim: int, idx: np.ndarray) -> np.ndarray:
    """Dense matrix from list/ndarray embeddings; fixes NaN/inf by converting to 0."""
    mat = np.zeros((len(idx), dim), dtype=np.float64)

    for i, row_i in enumerate(idx):
        v = series.iloc[row_i]

        if isinstance(v, list):
            v = np.asarray(v, dtype=np.float64)
        elif isinstance(v, np.ndarray):
            v = v.astype(np.float64, copy=False)
        else:
            continue

        if v.size == 0:
            continue

        # Fix weird shapes (pad/truncate)
        if v.shape[0] != dim:
            vv = np.zeros(dim, dtype=np.float64)
            m = min(dim, v.shape[0])
            vv[:m] = v[:m]
            v = vv

        # sanitize NaN/inf
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        mat[i] = v

    return mat


def fit_ipca_on_train(series: pd.Series, dim: int, train_row_idx: np.ndarray,
                      n_components: int, batch_size: int) -> IncrementalPCA | None:
    """Fit IncrementalPCA on training rows only."""
    if dim <= 0:
        return None
    n_train = len(train_row_idx)
    if n_train < 3:
        return None

    n_components_eff = int(min(n_components, dim, n_train - 1))
    if n_components_eff < 1:
        return None

    ipca = IncrementalPCA(n_components=n_components_eff)

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        batch_idx = train_row_idx[start:end]
        Xb = batch_embedding_matrix(series, dim, batch_idx)
        ipca.partial_fit(Xb)

    return ipca

def transform_ipca(series: pd.Series, dim: int, ipca: IncrementalPCA | None,
                   all_row_idx: np.ndarray, batch_size: int) -> np.ndarray:
    """Transform all rows with fitted IPCA. Returns (n_rows, n_components) float32 matrix."""
    n = len(all_row_idx)
    if ipca is None or dim <= 0:
        return np.zeros((n, 0), dtype=np.float32)

    out = np.zeros((n, ipca.n_components_), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = all_row_idx[start:end]
        Xb = batch_embedding_matrix(series, dim, batch_idx)
        Zb = ipca.transform(Xb).astype(np.float32, copy=False)
        out[start:end] = Zb

    return out

# -----------------------------
# Load + merge labels
# -----------------------------
print("Loading data...")
df = pd.read_parquet(PARQUET_PATH)

print("Loading labels...")
labels_df = pd.read_csv(LABELS_CSV)

print("Merging labels...")
df = df.merge(labels_df, on=["user_id", "exercise_id"], how="left")

print("Loading extra features...")
extra_df = pd.read_csv(ALL_FEATURES_CSV)

# Drop obvious index columns if present
extra_df = extra_df.drop(columns=[c for c in ["Unnamed: 0", "index"] if c in extra_df.columns], errors="ignore")

# Ensure keys exist
required_keys = {"user_id", "exercise_id"}
missing_keys = required_keys - set(extra_df.columns)
if missing_keys:
    raise ValueError(f"{ALL_FEATURES_CSV} missing required keys: {missing_keys}")

# If extra_df has duplicates per (user_id, exercise_id), keep first (or aggregate if you prefer)
extra_df = extra_df.drop_duplicates(subset=["user_id", "exercise_id"])

print("Merging extra features (all_features.csv)...")
# validate='m:1' is very helpful: many snapshots -> one row of extra features per user+exercise
df = df.merge(extra_df, on=["user_id", "exercise_id"], how="left", validate="m:1")


if df["label"].isna().any():
    print(f"Warning: {df['label'].isna().sum()} rows have missing labels. Dropping them.")
    df = df.dropna(subset=["label"]).copy()

print(f"Data shape: {df.shape}")
print(f"Unique students: {df['user_id'].nunique()}")

# -----------------------------
# Prepare sequence_id + sorting
# -----------------------------
df["sequence_id"] = df["user_id"].astype(str) + "_" + df["exercise_id"].astype(str)

sort_cols = ["sequence_id"]
if "snapshot_sequence_num" in df.columns:
    sort_cols.append("snapshot_sequence_num")
elif "seconds_from_start" in df.columns:
    sort_cols.append("seconds_from_start")

df = df.sort_values(sort_cols).reset_index(drop=True)


# -----------------------------
# Build numeric feature list dynamically (easy select/deselect)
# -----------------------------
# Auto-detect numeric columns contributed by all_features.csv
all_features_numeric_cols = []
if "extra_df" in locals():
    all_features_numeric_cols = [
        c for c in extra_df.columns
        if c not in ["user_id", "exercise_id"]
        and pd.api.types.is_numeric_dtype(extra_df[c])
    ]
FEATURE_BLOCKS["all_features_numeric"] = all_features_numeric_cols

# If you did not explicitly list categorical features, auto-detect them after merge.
# (Exclude known ID/target columns.)
AUTO_CATEGORICAL_EXCLUDE = {"user_id", "exercise_id", "sequence_id", "label"}
auto_cat = [
    c for c in df.columns
    if c not in AUTO_CATEGORICAL_EXCLUDE
    and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
]
if not FEATURE_BLOCKS["categorical"]:
    FEATURE_BLOCKS["categorical"] = auto_cat

# Finally, this is the master list you toggle via ENABLED_BLOCKS
numeric_features = []
for blk in ENABLED_BLOCKS:
    if blk in ("categorical",):
        continue
    numeric_features.extend(FEATURE_BLOCKS.get(blk, []))

categorical_features = FEATURE_BLOCKS.get("categorical", []) if "categorical" in ENABLED_BLOCKS else []
print(f"Candidate numeric features: {len(numeric_features)}")
print(f"Candidate categorical features: {len(categorical_features)}")


# -----------------------------
# Choose available snapshot features
# -----------------------------
available_numeric_features = [f for f in numeric_features if f in df.columns]
print(f"Available numeric snapshot features: {len(available_numeric_features)}")

# Coerce numeric
for f in available_numeric_features:
    df[f] = pd.to_numeric(df[f], errors="coerce")

# Fill NaNs in numeric columns with global mean (will re-scale later)
for f in available_numeric_features:
    m = df[f].mean()
    df[f] = df[f].fillna(m)

# -----------------------------
# Build sequence metadata (label/user per sequence)
# -----------------------------
seq_meta = (
    df.groupby("sequence_id", sort=False)
      .agg(user_id=("user_id", "first"),
           label=("label", "first"),
           seq_len=("sequence_id", "size"))
      .reset_index()
)

print(f"Total sequences: {len(seq_meta)}")

# -----------------------------
# Train/test split by student
# -----------------------------
df["user_id"] = df["user_id"].astype("object")
seq_meta["user_id"] = seq_meta["user_id"].astype("object")

unique_students = seq_meta["user_id"].to_numpy(dtype=object)  # plain numpy object array
unique_students = np.unique(unique_students)                  # ensure unique

train_students, test_students = train_test_split(
    unique_students, test_size=0.2, random_state=SEED
)
train_seq_ids = seq_meta.loc[seq_meta["user_id"].isin(train_students), "sequence_id"].values
test_seq_ids  = seq_meta.loc[seq_meta["user_id"].isin(test_students),  "sequence_id"].values

print(f"Train students: {len(train_students)} | Test students: {len(test_students)}")
print(f"Train sequences: {len(train_seq_ids)} | Test sequences: {len(test_seq_ids)}")


# Row indices for fitting snapshot-level preprocessors (PCA/scaler) on TRAIN rows only
train_row_mask = df["user_id"].isin(train_students).to_numpy()
train_row_idx = np.where(train_row_mask)[0]
all_row_idx = np.arange(len(df))




# -----------------------------
# Dimensionality reduction for embeddings (fit on TRAIN rows only)
# -----------------------------
reduced_parts = []
pca_models = {}

for col in EMB_COLS:
    if col in df.columns:
        print(f"\nEmbedding column found: {col}")
        dim = infer_embedding_dim(df[col])
        print(f"  inferred dim = {dim}")

        if dim <= 0:
            print(f"  No valid vectors found in {col}. Skipping.")
            reduced = np.zeros((len(df), 0), dtype=np.float32)
            pca_models[col] = None
        else:
            print(f"  Fitting IncrementalPCA on TRAIN rows (n={len(train_row_idx)})...")
            ipca = fit_ipca_on_train(
                df[col], dim, train_row_idx,
                n_components=EMB_N_COMPONENTS,
                batch_size=IPCA_BATCH_SIZE
            )
            if ipca is None:
                print(f"  Could not fit PCA for {col}. Skipping.")
                reduced = np.zeros((len(df), 0), dtype=np.float32)
                pca_models[col] = None
            else:
                print(f"  PCA components kept: {ipca.n_components_}")
                reduced = transform_ipca(
                    df[col], dim, ipca, all_row_idx, IPCA_BATCH_SIZE
                )
                pca_models[col] = ipca

        reduced_parts.append(reduced)
    else:
        print(f"\nEmbedding column missing: {col} (skipping)")
        pca_models[col] = None


# -----------------------------
# Categorical encoding (fit on TRAIN rows only)
# -----------------------------
ohe = None
X_cat = np.zeros((len(df), 0), dtype=np.float32)
cat_feature_names = []

if categorical_features:
    # Fill missing categories
    for c in categorical_features:
        df[c] = df[c].fillna("UNK").astype(str)

    # Fit on TRAIN rows only to avoid leakage
    X_cat_train = df.loc[train_row_mask, categorical_features]

    # OneHotEncoder API differs across sklearn versions; handle both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    ohe.fit(X_cat_train)

    X_cat = ohe.transform(df[categorical_features]).astype(np.float32, copy=False)

    # Feature names (nice for debugging / saving)
    try:
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_feature_names = [f"cat_{i}" for i in range(X_cat.shape[1])]

    print(f"One-hot categorical dim: {X_cat.shape[1]}")
else:
    print("No categorical features enabled/found.")



# -----------------------------
# Build full per-snapshot feature matrix
# -----------------------------
X_num = df[available_numeric_features].to_numpy(dtype=np.float32, copy=False)
X_emb = np.concatenate(reduced_parts, axis=1) if len(reduced_parts) else np.zeros((len(df), 0), dtype=np.float32)

X_all = np.concatenate([X_num, X_emb, X_cat], axis=1).astype(np.float32, copy=False)
print(f"\nPer-snapshot feature dim: {X_all.shape[1]}")

# Scale snapshot features (fit scaler on TRAIN rows only)
print("Scaling snapshot features (fit on TRAIN rows only)...")
scaler = StandardScaler()
scaler.fit(X_all[train_row_idx].astype(np.float64, copy=False))  # scaler is float64 internally
X_all_scaled = scaler.transform(X_all.astype(np.float64, copy=False)).astype(np.float32, copy=False)

# -----------------------------
# Convert snapshots -> padded sequences
# -----------------------------
print("\nBuilding padded sequences...")

# Build mapping: sequence_id -> (start/end indices) in sorted df
# Since df is sorted by sequence_id + time, groups are contiguous.
seq_indices = df.groupby("sequence_id", sort=False).indices  # dict: seq_id -> np.array of row indices

# Determine maxlen from TRAIN sequence lengths
train_lengths = [len(seq_indices[sid]) for sid in train_seq_ids if sid in seq_indices]
maxlen = int(np.percentile(train_lengths, MAXLEN_PERCENTILE)) if len(train_lengths) else MIN_MAXLEN
maxlen = max(maxlen, MIN_MAXLEN)
print(f"Using maxlen = {maxlen} (P{MAXLEN_PERCENTILE} of TRAIN lengths)")

def build_Xy_for_seq_ids(seq_id_list):
    X_list = []
    y_list = []
    sid_out = []

    for sid in seq_id_list:
        idx = seq_indices.get(sid, None)
        if idx is None or len(idx) == 0:
            continue

        # idx already ordered due to df sorting
        seq_X = X_all_scaled[idx]  # (T, D)
        # truncate if too long
        if seq_X.shape[0] > maxlen:
            seq_X = seq_X[:maxlen]  # keep earliest (post truncation)

        # pad to maxlen with zeros
        pad_len = maxlen - seq_X.shape[0]
        if pad_len > 0:
            seq_X = np.pad(seq_X, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)

        # sequence label
        lab = df.loc[idx[0], "label"]
        X_list.append(seq_X)
        y_list.append(lab)
        sid_out.append(sid)

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False) if X_list else np.zeros((0, maxlen, X_all.shape[1]), dtype=np.float32)
    y = np.array(y_list)
    return X, y, np.array(sid_out)

X_train_seq, y_train, train_sids = build_Xy_for_seq_ids(train_seq_ids)
X_test_seq,  y_test,  test_sids  = build_Xy_for_seq_ids(test_seq_ids)

print(f"Train sequences tensor: {X_train_seq.shape}")
print(f"Test  sequences tensor: {X_test_seq.shape}")

# -----------------------------
# Label encoding
# -----------------------------
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc  = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
print(f"Classes ({num_classes}): {list(label_encoder.classes_)}")

# Class weights (helps imbalance)
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train_enc
)
class_weight = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights:", class_weight)

# -----------------------------
# Build sequential model (Masking + BiGRU)
# -----------------------------
feature_dim = X_train_seq.shape[-1]

inputs = keras.Input(shape=(maxlen, feature_dim), dtype=tf.float32)
x = layers.Masking(mask_value=0.0)(inputs)

x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
x = layers.Dropout(0.50)(x)

x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
x = layers.Dropout(0.50)(x)

x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
x = layers.Dropout(0.50)(x)

x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
x = layers.Dropout(0.50)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.50)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, PATIENCE // 2), min_lr=1e-6),
]

history = model.fit(
    X_train_seq, y_train_enc,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
y_pred_proba = model.predict(X_test_seq, batch_size=BATCH_SIZE)
y_pred_enc = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test_enc, y_pred_enc)
print(f"\nTest accuracy: {acc:.4f}\n")

y_pred = label_encoder.inverse_transform(y_pred_enc)
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=label_encoder.classes_))

# -----------------------------
# Save predictions
# -----------------------------
pred_df = pd.DataFrame({
    "sequence_id": test_sids,
    "true_label": y_test,
    "pred_label": y_pred,
})
# add per-class probabilities (optional)
for i, cls in enumerate(label_encoder.classes_):
    pred_df[f"proba_{cls}"] = y_pred_proba[:, i]

pred_df.to_csv("sequence_test_predictions.csv", index=False)
print("\nSaved: sequence_test_predictions.csv")

# -----------------------------
# Save model + preprocessing
# -----------------------------
model.save("best_sequence_model.keras")
print("Saved: best_sequence_model.keras")

preproc = {
    "seed": SEED,
    "available_numeric_features": available_numeric_features,
    "embedding_cols": [c for c in EMB_COLS if c in df.columns],
    "emb_n_components": EMB_N_COMPONENTS,
    "maxlen": maxlen,
    "scaler": scaler,
    "pca_models": pca_models,          # dict: col -> IncrementalPCA or None
    "label_encoder": label_encoder,
    "feature_dim": feature_dim,
    "numeric_features_requested": numeric_features,
    "categorical_features": categorical_features,
    "ohe": ohe,
    "categorical_feature_names": cat_feature_names,
}
joblib.dump(preproc, "sequence_preprocessing.pkl")
print("Saved: sequence_preprocessing.pkl")

print("\nDONE.")
