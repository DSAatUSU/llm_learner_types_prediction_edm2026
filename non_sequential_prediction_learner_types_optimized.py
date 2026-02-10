"""
Aggregated Model for Learner Type Prediction
- Aggregates snapshot-level features per sequence (min/max/mean/etc.)
- Merges sequence-level features from all_features_df.csv
- Filters features by train-only missingness and variance
- Runs group-aware randomized search over strong tree models (ExtraTrees, RF, HGB, LightGBM if available)
- Selects best model by CV macro-F1 and evaluates on a student-held-out test set
Outputs:
  - best_model_<name>.pkl
  - model_predictions.csv
  - model_search_results.csv
  - confusion_matrix.png
  - model_comparison.png
  - model_analysis_summary.txt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from collections import Counter
from functools import partial

from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

# Try to import LightGBM (optional)
try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    USE_LGBM = False

# -----------------------------
# Config
# -----------------------------
SEED = 42
PARQUET_PATH = "thinned_snapshots_60s_with_embeddings.parquet"
LABELS_CSV = "final_labels.csv"
EXTRA_FEATURES_CSV = "all_features_df.csv"

TEST_SIZE = 0.2
CV_SPLITS = 5

MISSINGNESS_THRESH = 0.25
MIN_VARIANCE = 1e-5

N_ITER_ET = 25
N_ITER_RF = 20
N_ITER_HGB = 20
N_ITER_LGBM = 25

np.random.seed(SEED)

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
df = pd.read_parquet(PARQUET_PATH)

print("Loading labels...")
labels_df = pd.read_csv(LABELS_CSV)

print("Merging labels...")
df = df.merge(labels_df, on=["user_id", "exercise_id"], how="left")
if df["label"].isna().any():
    print(f"Warning: {df['label'].isna().sum()} rows have missing labels. Dropping them.")
    df = df.dropna(subset=["label"])

print(f"Data shape: {df.shape}")
print(f"Number of unique students: {df['user_id'].nunique()}")
print(f"Number of unique sequences: {df.groupby(['user_id', 'exercise_id']).ngroups}")

# -----------------------------
# Snapshot-level features to aggregate
# -----------------------------
print("\nPreparing snapshot features...")
base_snapshot_features = [
    "seconds_from_start", "chars_in_snapshot", "saves_before_snapshot",
    "keystrokes_before_snapshot", "ai_feedback_before_snapshot",
    "submits_before_snapshot", "executes_before_snapshot",
    "snapshot_sequence_num", "total_snapshots_in_session",
    "thinned_snapshots_in_session", "min_interval_seconds",
    "task_code_cosine", "task_code_distance"
]

available_snapshot_features = [f for f in base_snapshot_features if f in df.columns]
print(f"Available snapshot features: {available_snapshot_features}")

for f in available_snapshot_features:
    df[f] = pd.to_numeric(df[f], errors="coerce")

# Create sequence_id
df["sequence_id"] = df["user_id"].astype(str) + "_" + df["exercise_id"].astype(str)

# -----------------------------
# Aggregation
# -----------------------------
print("\nCreating sequence-level aggregated features...")
unique_sequences = df["sequence_id"].unique()
print(f"Number of sequences: {len(unique_sequences)}")

def create_aggregated_features(seq_df, feature_list):
    features = {}
    features["sequence_length"] = len(seq_df)

    for feature in feature_list:
        if feature not in seq_df.columns:
            continue
        values = seq_df[feature].astype(np.float64).values
        if len(values) == 0 or np.all(np.isnan(values)):
            continue
        values = np.nan_to_num(values, nan=np.nanmean(values))

        features[f"{feature}_mean"] = np.mean(values)
        features[f"{feature}_std"] = np.std(values) if len(values) > 1 else 0.0
        features[f"{feature}_min"] = np.min(values)
        features[f"{feature}_max"] = np.max(values)
        features[f"{feature}_median"] = np.median(values)
        if len(values) > 1:
            features[f"{feature}_q1"] = np.percentile(values, 25)
            features[f"{feature}_q3"] = np.percentile(values, 75)
        else:
            features[f"{feature}_q1"] = values[0]
            features[f"{feature}_q3"] = values[0]

        if len(values) > 1:
            diffs = np.diff(values)
            features[f"{feature}_diff_mean"] = np.mean(diffs)
            features[f"{feature}_diff_std"] = np.std(diffs) if len(diffs) > 1 else 0.0
            features[f"{feature}_diff_max"] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0.0
            try:
                x = np.arange(len(values)).astype(np.float64)
                mask = ~np.isnan(values)
                if np.sum(mask) >= 2:
                    slope = np.polyfit(x[mask], values[mask], 1)[0]
                    features[f"{feature}_trend"] = slope
                else:
                    features[f"{feature}_trend"] = 0.0
            except Exception:
                features[f"{feature}_trend"] = 0.0

    # Time-based features
    if "seconds_from_start" in seq_df.columns:
        times = seq_df["seconds_from_start"].astype(np.float64).values
        if len(times) > 1:
            features["total_duration"] = times[-1] - times[0]
            features["avg_time_between"] = features["total_duration"] / (len(times) - 1)
            if len(times) > 2:
                time_diffs = np.diff(times)
                features["time_std"] = np.std(time_diffs) if len(time_diffs) > 1 else 0.0
            else:
                features["time_std"] = 0.0
        else:
            features["total_duration"] = 0.0
            features["avg_time_between"] = 0.0
            features["time_std"] = 0.0

    # Snapshot sequence features
    if "snapshot_sequence_num" in seq_df.columns:
        seq_nums = seq_df["snapshot_sequence_num"].astype(np.float64).values
        features["final_snapshot_num"] = seq_nums[-1] if len(seq_nums) > 0 else 0.0

    # Additional derived
    features["avg_chars_per_snapshot"] = features.get("chars_in_snapshot_mean", 0.0)
    features["total_keystrokes"] = features.get("keystrokes_before_snapshot_max", 0.0)
    features["total_saves"] = features.get("saves_before_snapshot_max", 0.0)
    return features

sequence_data = []
success_count = 0
for seq_id in unique_sequences:
    seq_df = df[df["sequence_id"] == seq_id].copy()
    if "snapshot_sequence_num" in seq_df.columns:
        seq_df = seq_df.sort_values("snapshot_sequence_num")
    elif "seconds_from_start" in seq_df.columns:
        seq_df = seq_df.sort_values("seconds_from_start")

    label = seq_df["label"].iloc[0]
    user_id = seq_df["user_id"].iloc[0]
    exercise_id = seq_df["exercise_id"].iloc[0]

    try:
        agg = create_aggregated_features(seq_df, available_snapshot_features)
        agg["sequence_id"] = seq_id
        agg["user_id"] = user_id
        agg["exercise_id"] = exercise_id
        agg["label"] = label
        sequence_data.append(agg)
        success_count += 1
    except Exception as e:
        print(f"Error processing sequence {seq_id}: {e}")
        continue

print(f"Successfully processed {success_count} out of {len(unique_sequences)} sequences")
sequence_df = pd.DataFrame(sequence_data)
print(f"Aggregated sequence dataframe shape: {sequence_df.shape}")

if len(sequence_df) == 0:
    print("ERROR: No sequences were successfully processed!")
    raise SystemExit(1)

# -----------------------------
# Merge all_features_df.csv (sequence-level features)
# -----------------------------
print("\nLoading extra features (all_features_df.csv)...")
extra_df = pd.read_csv(EXTRA_FEATURES_CSV)
extra_df = extra_df.drop(columns=[c for c in ["Unnamed: 0", "index"] if c in extra_df.columns], errors="ignore")
required_keys = {"user_id", "exercise_id"}
missing_keys = required_keys - set(extra_df.columns)
if missing_keys:
    raise ValueError(f"{EXTRA_FEATURES_CSV} missing required keys: {missing_keys}")
extra_df = extra_df.drop_duplicates(subset=["user_id", "exercise_id"])

sequence_df = sequence_df.merge(extra_df, on=["user_id", "exercise_id"], how="left", validate="1:1")
print(f"Sequence dataframe after merge: {sequence_df.shape}")

# -----------------------------
# Train/test split by student
# -----------------------------
print("\nSplitting data by student...")
unique_students = sequence_df["user_id"].to_numpy(dtype=object)
unique_students = np.unique(unique_students)
train_students, test_students = train_test_split(
    unique_students, test_size=TEST_SIZE, random_state=SEED
)

train_sequences = sequence_df[sequence_df["user_id"].isin(train_students)]
test_sequences = sequence_df[sequence_df["user_id"].isin(test_students)]

print(f"Train students: {len(train_students)} | Test students: {len(test_students)}")
print(f"Train sequences: {len(train_sequences)} | Test sequences: {len(test_sequences)}")

if len(train_sequences) == 0 or len(test_sequences) == 0:
    print("ERROR: One of the sets (train or test) is empty.")
    raise SystemExit(1)

# -----------------------------
# Feature selection (train-only stats)
# -----------------------------
exclude_cols = {"sequence_id", "user_id", "exercise_id", "label"}
numeric_cols = [
    c for c in sequence_df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(sequence_df[c])
]
print(f"Total numeric features before filtering: {len(numeric_cols)}")

missing_rate = train_sequences[numeric_cols].isna().mean()
keep_missing = missing_rate[missing_rate <= MISSINGNESS_THRESH].index
dropped_missing = list(missing_rate[missing_rate > MISSINGNESS_THRESH].index)

train_med = train_sequences[keep_missing].median()
train_imputed = train_sequences[keep_missing].fillna(train_med)
variances = train_imputed.var()
keep_var = variances[variances > MIN_VARIANCE].index
dropped_lowvar = list(variances[variances <= MIN_VARIANCE].index)

model_features = list(keep_var)
print(f"Features kept after filtering: {len(model_features)}")
print(f"Dropped for missingness (> {MISSINGNESS_THRESH:.2f}): {len(dropped_missing)}")
print(f"Dropped for low variance (<= {MIN_VARIANCE:g}): {len(dropped_lowvar)}")

if len(model_features) == 0:
    print("ERROR: No features left after filtering.")
    raise SystemExit(1)

X_train = train_sequences[model_features]
X_test = test_sequences[model_features]
y_train = train_sequences["label"].values
y_test = test_sequences["label"].values
groups_train = train_sequences["user_id"].values

label_encoder = LabelEncoder()
label_encoder.fit(sequence_df["label"].values)
y_train_enc = label_encoder.transform(y_train)
y_test_enc = label_encoder.transform(y_test)

classes = np.unique(y_train_enc)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_enc)
class_weight_map = {c: w for c, w in zip(classes, class_weights)}
sample_weight = np.array([class_weight_map[y] for y in y_train_enc])

print(f"Label distribution in train: {dict(Counter(y_train))}")
print(f"Label distribution in test: {dict(Counter(y_test))}")

# -----------------------------
# Model search
# -----------------------------
cv_splits = min(CV_SPLITS, len(np.unique(groups_train)))
cv = GroupKFold(n_splits=cv_splits)
print(f"\nUsing GroupKFold with {cv_splits} splits (by student).")

n_features = len(model_features)
k_candidates = sorted({k for k in [20, 40, 60, 80, 120, n_features] if 2 <= k <= n_features})
if not k_candidates:
    k_candidates = [n_features]

mi_fn = partial(mutual_info_classif, random_state=SEED)

def make_pipeline(model, default_k):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(mi_fn, k=default_k)),
        ("model", model),
    ])

results = []

def run_search(name, estimator, param_distributions, n_iter, fit_params=None):
    default_k = min(80, n_features)
    pipe = make_pipeline(estimator, default_k=default_k)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    fit_params = fit_params or {}
    search.fit(X_train, y_train_enc, groups=groups_train, **fit_params)
    print(f"{name} best CV macro-F1: {search.best_score_:.4f}")
    results.append({
        "name": name,
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "search": search
    })

# ExtraTrees
run_search(
    "ExtraTrees",
    ExtraTreesClassifier(
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"
    ),
    param_distributions={
        "select__k": k_candidates,
        "model__n_estimators": [300, 600, 1000],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__bootstrap": [False, True],
    },
    n_iter=N_ITER_ET
)

# RandomForest
run_search(
    "RandomForest",
    RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"
    ),
    param_distributions={
        "select__k": k_candidates,
        "model__n_estimators": [300, 600, 1000],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__bootstrap": [False, True],
    },
    n_iter=N_ITER_RF
)

# HistGradientBoosting
run_search(
    "HistGradientBoosting",
    HistGradientBoostingClassifier(random_state=SEED),
    param_distributions={
        "select__k": k_candidates,
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 5, 8, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [5, 10, 20],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_iter": [200, 400, 800],
    },
    n_iter=N_ITER_HGB,
    fit_params={"model__sample_weight": sample_weight}
)

# LightGBM (optional)
if USE_LGBM:
    run_search(
        "LightGBM",
        lgb.LGBMClassifier(
            random_state=SEED,
            class_weight="balanced",
            objective="multiclass",
            n_jobs=-1,
            verbose=-1
        ),
        param_distributions={
            "select__k": k_candidates,
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 5, 8, 12],
            "model__min_child_samples": [5, 10, 20],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0],
        },
        n_iter=N_ITER_LGBM
    )

if not results:
    print("ERROR: No model results produced.")
    raise SystemExit(1)

# -----------------------------
# Select best model and evaluate
# -----------------------------
results_sorted = sorted(results, key=lambda x: x["best_score"], reverse=True)
best = results_sorted[0]
best_name = best["name"]
best_search = best["search"]

print(f"\nBest model by CV macro-F1: {best_name} ({best['best_score']:.4f})")
print(f"Best params: {best['best_params']}")

y_pred_enc = best_search.predict(X_test)
acc = accuracy_score(y_test_enc, y_pred_enc)
f1 = f1_score(y_test_enc, y_pred_enc, average="macro")

y_pred = label_encoder.inverse_transform(y_pred_enc)

print(f"\nTest accuracy: {acc:.4f}")
print(f"Test macro-F1: {f1:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
print("Confusion matrix:")
print(cm)

# -----------------------------
# Save outputs
# -----------------------------
pred_df = test_sequences[["sequence_id", "user_id", "label"]].copy()
pred_df["pred_label"] = y_pred
pred_df.to_csv("model_predictions.csv", index=False)
print("Saved: model_predictions.csv")

# Save best model
joblib.dump(best_search.best_estimator_, f"best_model_{best_name}.pkl")
print(f"Saved: best_model_{best_name}.pkl")

# Save preprocessing objects
preprocessing_objects = {
    "label_encoder": label_encoder,
    "model_features": model_features,
    "dropped_missing": dropped_missing,
    "dropped_lowvar": dropped_lowvar,
}
joblib.dump(preprocessing_objects, "model_preprocessing.pkl")
print("Saved: model_preprocessing.pkl")

# Save search results
results_df = pd.DataFrame([
    {"model": r["name"], "best_cv_f1": r["best_score"], "best_params": r["best_params"]}
    for r in results_sorted
])
results_df.to_csv("model_search_results.csv", index=False)
print("Saved: model_search_results.csv")

# Save sequence features (for inspection)
sequence_df.to_csv("sequence_features.csv", index=False)
print("Saved: sequence_features.csv")

# Model comparison chart
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(results_df["model"], results_df["best_cv_f1"], color="steelblue")
ax.set_xlabel("Best CV Macro-F1")
ax.set_title("Model Comparison (Group CV)")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
print("Saved: model_comparison.png")

# Confusion matrix plot for best model
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    title=f"Confusion Matrix - {best_name}",
    ylabel="True label",
    xlabel="Predicted label",
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
thresh = cm.max() / 2.0 if cm.size else 0.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
print("Saved: confusion_matrix.png")

# Summary report
with open("model_analysis_summary.txt", "w") as f:
    f.write("MODEL ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total sequences processed: {len(sequence_df)}\n")
    f.write(f"Total unique students: {len(unique_students)}\n")
    f.write(f"Train students: {len(train_students)}\n")
    f.write(f"Test students: {len(test_students)}\n")
    f.write(f"Train sequences: {len(train_sequences)}\n")
    f.write(f"Test sequences: {len(test_sequences)}\n")
    f.write(f"Numeric features before filter: {len(numeric_cols)}\n")
    f.write(f"Features kept after filtering: {len(model_features)}\n")
    f.write(f"Dropped for missingness: {len(dropped_missing)}\n")
    f.write(f"Dropped for low variance: {len(dropped_lowvar)}\n\n")

    f.write("LABEL DISTRIBUTION\n")
    f.write("-" * 40 + "\n")
    train_counts = dict(Counter(y_train))
    test_counts = dict(Counter(y_test))
    f.write("Training set:\n")
    for label, count in train_counts.items():
        pct = (count / len(y_train)) * 100
        f.write(f"  {label}: {count} ({pct:.1f}%)\n")
    f.write("\nTest set:\n")
    for label, count in test_counts.items():
        pct = (count / len(y_test)) * 100
        f.write(f"  {label}: {count} ({pct:.1f}%)\n")

    f.write("\nMODEL PERFORMANCE (CV)\n")
    f.write("-" * 40 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['model']}: {row['best_cv_f1']:.4f}\n")

    f.write(f"\nBest model: {best_name}\n")
    f.write(f"Best CV macro-F1: {best['best_score']:.4f}\n")
    f.write(f"Test accuracy: {acc:.4f}\n")
    f.write(f"Test macro-F1: {f1:.4f}\n")

print("\nANALYSIS COMPLETE!")
