"""
Sequential Model Approach for Learner Type Prediction using LightGBM
This script uses gradient boosting with sequence aggregation for variable-length sequences.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM (usually easier to install)
try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    print("LightGBM not found. Will use scikit-learn models instead.")
    USE_LGBM = False

import matplotlib.pyplot as plt
import joblib
from collections import Counter

# Load the data
print("Loading data...")
df = pd.read_parquet("thinned_snapshots_60s_with_embeddings.parquet")

# Load labels
print("Loading labels...")
labels_df = pd.read_csv("final_labels.csv")

# Merge labels with the main dataframe
print("\nMerging labels...")
df = df.merge(labels_df, on=['user_id', 'exercise_id'], how='left')

# Check for missing labels
if df['label'].isna().any():
    print(f"Warning: {df['label'].isna().sum()} rows have missing labels")
    df = df.dropna(subset=['label'])

print(f"Data shape: {df.shape}")
print(f"Number of unique students: {df['user_id'].nunique()}")
print(f"Number of unique sequences: {df.groupby(['user_id', 'exercise_id']).ngroups}")

# Prepare features for modeling
print("\nPreparing features...")

# Select features for each snapshot - using only the features that actually exist
numeric_features = [
    'seconds_from_start', 'chars_in_snapshot', 'saves_before_snapshot',
    'keystrokes_before_snapshot', 'ai_feedback_before_snapshot',
    'submits_before_snapshot', 'executes_before_snapshot', 
    'snapshot_sequence_num', 'total_snapshots_in_session',
    'thinned_snapshots_in_session', 'min_interval_seconds',
    'task_code_cosine', 'task_code_distance'
]

# Check which features actually exist in the dataframe
available_features = [f for f in numeric_features if f in df.columns]
print(f"Available features: {available_features}")
print(f"Number of available features: {len(available_features)}")

# Fill NaN values for available features
for feature in available_features:
    if df[feature].isna().any():
        df[feature] = df[feature].fillna(df[feature].mean())

# Create sequence-level aggregated features
print("\nCreating sequence-level aggregated features...")

# Create a unique sequence identifier
df['sequence_id'] = df['user_id'].astype(str) + '_' + df['exercise_id'].astype(str)

# Get all unique sequences
unique_sequences = df['sequence_id'].unique()
print(f"Number of sequences: {len(unique_sequences)}")

# Function to create aggregated features from a sequence
def create_aggregated_features(seq_df):
    """Create aggregated features from a sequence of snapshots"""
    features = {}
    
    # Basic sequence info
    features['sequence_length'] = len(seq_df)
    
    # For each numeric feature, create statistical aggregates
    for feature in available_features:
        if feature in seq_df.columns:
            # Convert to float64 to avoid numpy compatibility issues
            values = seq_df[feature].astype(np.float64).values
            
            # Skip if all values are NaN
            if len(values) == 0 or np.all(np.isnan(values)):
                continue
                
            # Replace any remaining NaN with mean
            values = np.nan_to_num(values, nan=np.nanmean(values))
            
            # Basic statistics
            features[f'{feature}_mean'] = np.mean(values)
            features[f'{feature}_std'] = np.std(values) if len(values) > 1 else 0
            features[f'{feature}_min'] = np.min(values)
            features[f'{feature}_max'] = np.max(values)
            features[f'{feature}_median'] = np.median(values)
            
            if len(values) > 1:
                features[f'{feature}_q1'] = np.percentile(values, 25)
                features[f'{feature}_q3'] = np.percentile(values, 75)
            else:
                features[f'{feature}_q1'] = values[0]
                features[f'{feature}_q3'] = values[0]
            
            # Rate of change features (if sequence has at least 2 points)
            if len(values) > 1:
                diffs = np.diff(values)
                features[f'{feature}_diff_mean'] = np.mean(diffs)
                features[f'{feature}_diff_std'] = np.std(diffs) if len(diffs) > 1 else 0
                features[f'{feature}_diff_max'] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0
                
                # Trend (linear regression slope) - handle with try-except
                try:
                    x = np.arange(len(values)).astype(np.float64)
                    # Ensure no NaN in x or y
                    mask = ~np.isnan(values)
                    if np.sum(mask) >= 2:  # Need at least 2 points for linear fit
                        slope = np.polyfit(x[mask], values[mask], 1)[0]
                        features[f'{feature}_trend'] = slope
                    else:
                        features[f'{feature}_trend'] = 0
                except:
                    features[f'{feature}_trend'] = 0
    
    # Time-based features
    if 'seconds_from_start' in seq_df.columns:
        times = seq_df['seconds_from_start'].astype(np.float64).values
        if len(times) > 1:
            features['total_duration'] = times[-1] - times[0]
            features['avg_time_between'] = features['total_duration'] / (len(times) - 1) if len(times) > 1 else 0
            if len(times) > 2:
                time_diffs = np.diff(times)
                features['time_std'] = np.std(time_diffs) if len(time_diffs) > 1 else 0
            else:
                features['time_std'] = 0
        else:
            features['total_duration'] = 0
            features['avg_time_between'] = 0
            features['time_std'] = 0
    
    # Snapshot sequence features
    if 'snapshot_sequence_num' in seq_df.columns:
        seq_nums = seq_df['snapshot_sequence_num'].astype(np.float64).values
        if len(seq_nums) > 0:
            features['final_snapshot_num'] = seq_nums[-1]
        else:
            features['final_snapshot_num'] = 0
    
    # Additional aggregated features
    features['avg_chars_per_snapshot'] = features.get('chars_in_snapshot_mean', 0)
    features['total_keystrokes'] = features.get('keystrokes_before_snapshot_max', 0)
    features['total_saves'] = features.get('saves_before_snapshot_max', 0)
    
    return features

# Create aggregated features for each sequence
sequence_data = []
sequence_labels = []
sequence_users = []
sequence_ids_list = []

success_count = 0
for seq_id in unique_sequences:
    seq_df = df[df['sequence_id'] == seq_id].copy()
    
    # Sort by timestamp or sequence number
    if 'snapshot_sequence_num' in seq_df.columns:
        seq_df = seq_df.sort_values('snapshot_sequence_num')
    elif 'seconds_from_start' in seq_df.columns:
        seq_df = seq_df.sort_values('seconds_from_start')
    
    # Get label and user
    label = seq_df['label'].iloc[0]
    user_id = seq_df['user_id'].iloc[0]
    
    try:
        # Create aggregated features
        agg_features = create_aggregated_features(seq_df)
        agg_features['sequence_id'] = seq_id
        agg_features['user_id'] = user_id
        agg_features['label'] = label
        
        sequence_data.append(agg_features)
        sequence_labels.append(label)
        sequence_users.append(user_id)
        sequence_ids_list.append(seq_id)
        success_count += 1
    except Exception as e:
        print(f"Error processing sequence {seq_id}: {e}")
        continue

print(f"Successfully processed {success_count} out of {len(unique_sequences)} sequences")

# Convert to dataframe
sequence_df = pd.DataFrame(sequence_data)
print(f"Aggregated sequence dataframe shape: {sequence_df.shape}")

# Remove the identifier columns for modeling
model_features = [col for col in sequence_df.columns 
                 if col not in ['sequence_id', 'user_id', 'label']]
print(f"Number of aggregated features: {len(model_features)}")

# Check if we have any data
if len(sequence_df) == 0:
    print("ERROR: No sequences were successfully processed!")
    exit()

# Fill any NaN values in the aggregated features
for col in model_features:
    if col in sequence_df.columns:
        sequence_df[col] = sequence_df[col].fillna(sequence_df[col].mean())

# Split data by student (80% train, 20% test)
print("\nSplitting data by student...")

# Get unique students
unique_students = list(set(sequence_users))
print(f"Total unique students: {len(unique_students)}")

# Split students into train and test
train_students, test_students = train_test_split(
    unique_students, test_size=0.2, random_state=42
)

print(f"Train students ({len(train_students)}): {train_students}")
print(f"Test students ({len(test_students)}): {test_students}")

# Split sequences based on student membership
train_mask = sequence_df['user_id'].isin(train_students)
test_mask = sequence_df['user_id'].isin(test_students)

train_sequences = sequence_df[train_mask]
test_sequences = sequence_df[test_mask]

print(f"Train sequences: {len(train_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# Check if we have data in both sets
if len(train_sequences) == 0 or len(test_sequences) == 0:
    print("ERROR: One of the sets (train or test) is empty!")
    print("This might happen if all sequences from a student failed to process.")
    exit()

# Prepare features and labels
X_train = train_sequences[model_features].values
y_train = train_sequences['label'].values
X_test = test_sequences[model_features].values
y_test = test_sequences['label'].values

# Fill any remaining NaN values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

print(f"\nFeature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}")

# Encode labels if they're strings
if isinstance(y_train[0], str):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    print(f"Classes: {label_encoder.classes_}")
else:
    y_train_encoded = y_train
    y_test_encoded = y_test
    # Create a simple label encoder for consistency
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_test]))

print(f"Label distribution in train: {dict(Counter(y_train))}")
print(f"Label distribution in test: {dict(Counter(y_test))}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=min(5, len(X_train_scaled))),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Add LightGBM if available
if USE_LGBM:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        
        # Check if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred_proba = None
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Decode predictions if needed
        if isinstance(y_train[0], str):
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
        else:
            y_pred_decoded = y_pred
            y_test_decoded = y_test_encoded
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred_decoded,
            'true_labels': y_test_decoded,
            'probabilities': y_pred_proba
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))
        
    except Exception as e:
        print(f"Error training {name}: {e}")

# Compare model performance
if results:
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, result in sorted_results:
        print(f"{name}: Accuracy = {result['accuracy']:.3f}")
    
    # Get best model
    best_model_name, best_result = sorted_results[0]
    print(f"\nBest model: {best_model_name} (Accuracy: {best_result['accuracy']:.3f})")

# Feature importance for tree-based models
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

for name, result in results.items():
    model = result['model']
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        print(f"\n{name} - Top 10 most important features:")
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print top 10
        for i in range(min(10, len(model_features))):
            feature_idx = indices[i]
            if feature_idx < len(model_features):
                print(f"  {i+1}. {model_features[feature_idx]}: {importances[feature_idx]:.4f}")

# Save predictions
print("\nSaving predictions...")

predictions_df = test_sequences[['sequence_id', 'user_id', 'label']].copy()
predictions_df['true_label'] = predictions_df['label']

# Add predictions from all models
for name, result in results.items():
    predictions_df[f'predicted_{name}'] = result['predictions']

predictions_df.to_csv('model_predictions.csv', index=False)
print("Predictions saved to 'model_predictions.csv'")

# Save the full sequence data with features
sequence_df.to_csv('sequence_features.csv', index=False)
print("Sequence features saved to 'sequence_features.csv'")

# Save models and preprocessing objects
print("\nSaving models...")

# Save the best model
if results:
    best_model_name = sorted_results[0][0]
    best_model = results[best_model_name]['model']
    
    # Save model (using joblib for sklearn models)
    joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
    print(f"Best model ({best_model_name}) saved to 'best_model_{best_model_name}.pkl'")

# Save preprocessing objects
preprocessing_objects = {
    'scaler': scaler,
    'label_encoder': label_encoder if 'label_encoder' in locals() else None,
    'model_features': model_features,
    'available_features': available_features
}

joblib.dump(preprocessing_objects, 'model_preprocessing.pkl')
print("Preprocessing objects saved to 'model_preprocessing.pkl'")

# Create visualizations
print("\nCreating visualizations...")

# 1. Model comparison bar chart
if results:
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [name for name, _ in sorted_results]
    accuracies = [result['accuracy'] for _, result in sorted_results]

    bars = ax.barh(model_names, accuracies, color='skyblue')
    ax.set_xlabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xlim([0, 1.1])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    print("Model comparison chart saved to 'model_comparison.png'")

# 2. Confusion matrix for best model
if results:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique labels
    unique_labels = sorted(set(best_result['true_labels']))
    
    # Create confusion matrix
    cm = confusion_matrix(best_result['true_labels'], best_result['predictions'], 
                          labels=unique_labels)
    
    # Plot
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=unique_labels,
           yticklabels=unique_labels,
           title=f'Confusion Matrix - {best_model_name}',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    print("Confusion matrix saved to 'confusion_matrix.png'")

# 3. Feature importance plot for tree-based models
if results:
    tree_models = [name for name in results.keys() 
                  if hasattr(results[name]['model'], 'feature_importances_')]
    
    if tree_models:
        for model_name in tree_models[:2]:  # Plot for first 2 tree models
            model = results[model_name]['model']
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(indices)), importances[indices], color='b', align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([model_features[i] for i in indices])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top 15 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png', dpi=100, bbox_inches='tight')
            print(f"Feature importance for {model_name} saved to 'feature_importance_{model_name}.png'")

# Create a summary report
with open('model_analysis_summary.txt', 'w') as f:
    f.write("MODEL ANALYSIS SUMMARY\n")
    f.write("="*50 + "\n")
    f.write(f"Total sequences processed: {len(sequence_df)}\n")
    f.write(f"Total unique students: {len(unique_students)}\n")
    f.write(f"Train students: {len(train_students)}\n")
    f.write(f"Test students: {len(test_students)}\n")
    f.write(f"Train sequences: {len(train_sequences)}\n")
    f.write(f"Test sequences: {len(test_sequences)}\n")
    f.write(f"Number of aggregated features: {len(model_features)}\n")
    f.write(f"Original features used: {available_features}\n\n")
    
    f.write("LABEL DISTRIBUTION\n")
    f.write("-"*40 + "\n")
    train_counts = dict(Counter(y_train))
    test_counts = dict(Counter(y_test))
    
    f.write("Training set:\n")
    for label, count in train_counts.items():
        percentage = (count / len(y_train)) * 100
        f.write(f"  {label}: {count} ({percentage:.1f}%)\n")
    
    f.write("\nTest set:\n")
    for label, count in test_counts.items():
        percentage = (count / len(y_test)) * 100
        f.write(f"  {label}: {count} ({percentage:.1f}%)\n")
    
    if results:
        f.write("\nMODEL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        for name, result in sorted_results:
            f.write(f"{name}: {result['accuracy']:.3f}\n")
        
        f.write(f"\nBest model: {best_model_name}\n")
        f.write(f"Best accuracy: {best_result['accuracy']:.3f}\n\n")
        
        f.write("PREDICTION DISTRIBUTION (Best Model)\n")
        f.write("-"*40 + "\n")
        pred_counts = dict(Counter(best_result['predictions']))
        true_counts = dict(Counter(best_result['true_labels']))
        
        f.write("\nTrue labels:\n")
        for label, count in true_counts.items():
            percentage = (count / len(best_result['true_labels'])) * 100
            f.write(f"  {label}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nPredicted labels:\n")
        for label, count in pred_counts.items():
            percentage = (count / len(best_result['predictions'])) * 100
            f.write(f"  {label}: {count} ({percentage:.1f}%)\n")
    
    f.write("\n\nSEQUENCE STATISTICS\n")
    f.write("-"*40 + "\n")
    f.write(f"Average sequence length: {sequence_df['sequence_length'].mean():.1f}\n")
    f.write(f"Min sequence length: {sequence_df['sequence_length'].min()}\n")
    f.write(f"Max sequence length: {sequence_df['sequence_length'].max()}\n")
    
    if 'total_duration' in sequence_df.columns:
        f.write(f"\nAverage total duration: {sequence_df['total_duration'].mean():.1f} seconds\n")
        f.write(f"Min duration: {sequence_df['total_duration'].min():.1f} seconds\n")
        f.write(f"Max duration: {sequence_df['total_duration'].max():.1f} seconds\n")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nFiles saved:")
print("1. model_predictions.csv - All model predictions")
print("2. sequence_features.csv - Aggregated sequence features")
print("3. best_model_<name>.pkl - Best trained model")
print("4. model_preprocessing.pkl - Preprocessing objects")
print("5. model_comparison.png - Model performance comparison")
print("6. confusion_matrix.png - Confusion matrix for best model")
print("7. model_analysis_summary.txt - Summary report")

print("\nNote: If you want to try LightGBM for potentially better performance:")
print("pip install lightgbm")
print("\nThen run this script again.")
