# ================================================================
# PHASE 4 - Proper Ensemble Model Development
# [GitHub Ready Version - FULL]
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, make_scorer
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
from datetime import datetime
import json
import joblib

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    'random_state': 42,
    'base_path': '.', # [GitHub Ready] Relative path
    'phase3_dir': './Phase3_Class_Imbalance',
    'phase4_dir': './Phase4_Model_Development',
    'target_col': 'Triaj',
    'n_jobs': -1,
    'cv_folds': 5,
    'calibration_split': 0.5, # Split validation into val + calibration
    'realistic_gap_target': 0.07,
    'figsize': (12, 8),
    'dpi': 300
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 4: PROPER ENSEMBLE MODELING")
print("="*70)

# [GitHub Ready] Ensure directories exist
os.makedirs(CONFIG['phase4_dir'], exist_ok=True)
for subdir in ['models', 'figures', 'reports']:
    os.makedirs(os.path.join(CONFIG['phase4_dir'], subdir), exist_ok=True)
current_phase_dir = CONFIG['phase4_dir']

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading data from Phase 3...")

try:
    train_df = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'train_resampled.csv'))
    val_df_original = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'validation.csv'))
    test_df = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'test.csv'))
    
    # Load class weights and label encoder
    class_weights = joblib.load(os.path.join(CONFIG['phase3_dir'], 'models', 'class_weights.pkl'))
    le = joblib.load(os.path.join(CONFIG['phase3_dir'], 'models', 'label_encoder.pkl'))

    print(f"Training (resampled): {len(train_df):,}")
    print(f"Validation (original): {len(val_df_original):,}")
    print(f"Test:       {len(test_df):,}")
    print(f"Class weights loaded.")
    print(f"Label encoder loaded. Classes: {le.classes_}")

except Exception as e:
    print(f"Error loading data or models from Phase 3: {e}")
    raise SystemExit("Cannot proceed without Phase 3 data.")


# ================================================================
# CRITICAL FIX: Split validation into val + calibration
# ================================================================

print("\nSplitting validation set for proper calibration...")

val_df, calib_df = train_test_split(
    val_df_original,
    test_size=CONFIG['calibration_split'],
    stratify=val_df_original[CONFIG['target_col']],
    random_state=CONFIG['random_state']
)

print(f"Validation (for selection): {len(val_df):,}")
print(f"Calibration (for isotonic): {len(calib_df):,}")

# ================================================================
# PREPARE DATA
# ================================================================

def prepare_data(df, target_col):
    """Prepare features and target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Use only numeric features (as done in Phase 3)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]

    return X_numeric, y, numeric_cols

X_train, y_train, feature_cols = prepare_data(train_df, CONFIG['target_col'])
X_val, y_val, _ = prepare_data(val_df, CONFIG['target_col'])
X_calib, y_calib, _ = prepare_data(calib_df, CONFIG['target_col'])
X_test, y_test, _ = prepare_data(test_df, CONFIG['target_col'])

print(f"\nFeature count: {X_train.shape[1]}")

# Encode targets
y_train_encoded = le.transform(y_train)
y_val_encoded = le.transform(y_val)
y_calib_encoded = le.transform(y_calib)
y_test_encoded = le.transform(y_test)

# Create sample weights for training
sample_weights_train = np.array([class_weights[label] for label in y_train])

# Create class weight dict for sklearn
class_weight_dict = {le.transform([k])[0]: v for k, v in class_weights.items()}

print(f"Sample weights created for all models")
print(f"Class encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ================================================================
# STEP 1: TRAIN INDIVIDUAL MODELS WITH PROPER CV
# ================================================================

print("\n" + "="*70)
print("STEP 1: TRAINING INDIVIDUAL MODELS")
print("="*70)

# Define scoring for CV
balanced_acc_scorer = make_scorer(balanced_accuracy_score)

# ----------------------------------------------------------------
# XGBoost with moderate regularization
# ----------------------------------------------------------------
print("\n[1/3] Training XGBoost with GridSearchCV...")

param_grid_xgb = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [150, 200, 250],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'reg_alpha': [0.5, 1.0],  # L1 regularization
    'reg_lambda': [2.0, 3.0]  # L2 regularization
}

xgb_base = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    random_state=CONFIG['random_state'],
    n_jobs=CONFIG['n_jobs'],
    eval_metric='mlogloss'
)

cv_strategy = StratifiedKFold(
    n_splits=CONFIG['cv_folds'],
    shuffle=True,
    random_state=CONFIG['random_state']
)

grid_xgb = GridSearchCV(
    xgb_base,
    param_grid_xgb,
    cv=cv_strategy,
    scoring=balanced_acc_scorer,
    n_jobs=CONFIG['n_jobs'],
    verbose=1,
    return_train_score=True
)

# Fit with sample weights
grid_xgb.fit(X_train, y_train_encoded, sample_weight=sample_weights_train)

best_xgb = grid_xgb.best_estimator_

print(f"Best XGBoost params: {grid_xgb.best_params_}")
print(f"Best CV score: {grid_xgb.best_score_:.4f}")
print(f"CV std: {grid_xgb.cv_results_['std_test_score'][grid_xgb.best_index_]:.4f}")

# ----------------------------------------------------------------
# Random Forest with proper sample weights
# ----------------------------------------------------------------
print("\n[2/3] Training Random Forest with GridSearchCV...")

param_grid_rf = {
    'n_estimators': [150, 200, 250],
    'max_depth': [15, 18, 20, None],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [5, 8, 10],
    'max_features': ['sqrt', 'log2'],
    'max_samples': [0.7, 0.8, 0.9]
}

rf_base = RandomForestClassifier(
    class_weight=class_weight_dict,  # Built-in class weighting
    random_state=CONFIG['random_state'],
    n_jobs=CONFIG['n_jobs'],
    oob_score=True,
    bootstrap=True
)

grid_rf = GridSearchCV(
    rf_base,
    param_grid_rf,
    cv=cv_strategy,
    scoring=balanced_acc_scorer,
    n_jobs=CONFIG['n_jobs'],
    verbose=1,
    return_train_score=True
)

# Pass sample_weight to RF as well
grid_rf.fit(X_train, y_train_encoded, sample_weight=sample_weights_train)

best_rf = grid_rf.best_estimator_

print(f"Best RF params: {grid_rf.best_params_}")
print(f"Best CV score: {grid_rf.best_score_:.4f}")
print(f"CV std: {grid_rf.cv_results_['std_test_score'][grid_rf.best_index_]:.4f}")
print(f"OOB Score: {best_rf.oob_score_:.4f}")

# ----------------------------------------------------------------
# Logistic Regression with L2 regularization
# ----------------------------------------------------------------
print("\n[3/3] Training Logistic Regression...")

param_grid_lr = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}

lr_base = LogisticRegression(
    class_weight=class_weight_dict,
    random_state=CONFIG['random_state'],
    n_jobs=CONFIG['n_jobs']
)

grid_lr = GridSearchCV(
    lr_base,
    param_grid_lr,
    cv=cv_strategy,
    scoring=balanced_acc_scorer,
    n_jobs=CONFIG['n_jobs'],
    verbose=1,
    return_train_score=True
)

grid_lr.fit(X_train, y_train_encoded, sample_weight=sample_weights_train)

best_lr = grid_lr.best_estimator_

print(f"Best LR params: {grid_lr.best_params_}")
print(f"Best CV score: {grid_lr.best_score_:.4f}")
print(f"CV std: {grid_lr.cv_results_['std_test_score'][grid_lr.best_index_]:.4f}")

# ================================================================
# STEP 2: EVALUATE INDIVIDUAL MODELS
# ================================================================

print("\n" + "="*70)
print("STEP 2: INDIVIDUAL MODEL EVALUATION")
print("="*70)

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, cv_score=None):
    """Comprehensive model evaluation"""

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_bal = balanced_accuracy_score(y_train, y_train_pred)
    val_bal = balanced_accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    # Gaps
    acc_gap = train_acc - val_acc
    bal_gap = train_bal - val_bal

    # Print results
    print(f"\n{model_name}:")
    if cv_score is not None:
        print(f"   CV Balanced Acc: {cv_score:.4f}")
    print(f"   Train Acc:    {train_acc:.4f} | Balanced: {train_bal:.4f} | F1: {train_f1:.4f}")
    print(f"   Val Acc:      {val_acc:.4f} | Balanced: {val_bal:.4f} | F1: {val_f1:.4f}")
    print(f"   Acc Gap:      {acc_gap:.4f} ({'GOOD' if acc_gap < 0.05 else 'OK' if acc_gap < 0.07 else 'HIGH'})")
    print(f"   Bal Gap:      {bal_gap:.4f}")

    return {
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'train_balanced': float(train_bal),
        'val_balanced': float(val_bal),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'cv_score': float(cv_score) if cv_score else None,
        'acc_gap': float(acc_gap),
        'bal_gap': float(bal_gap)
    }

# Evaluate all models
xgb_metrics = evaluate_model(
    best_xgb, X_train, y_train_encoded, X_val, y_val_encoded,
    "XGBoost", grid_xgb.best_score_
)

rf_metrics = evaluate_model(
    best_rf, X_train, y_train_encoded, X_val, y_val_encoded,
    "Random Forest", grid_rf.best_score_
)

lr_metrics = evaluate_model(
    best_lr, X_train, y_train_encoded, X_val, y_val_encoded,
    "Logistic Regression", grid_lr.best_score_
)

# ================================================================
# STEP 3: ENSEMBLE MODEL (SOFT VOTING)
# ================================================================

print("\n" + "="*70)
print("STEP 3: BUILDING ENSEMBLE MODEL")
print("="*70)

print("\nCreating soft voting ensemble...")

# Create ensemble with optimized weights
ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf),
        ('lr', best_lr)
    ],
    voting='soft',
    weights=[2, 2, 1],  # XGBoost and RF weighted more
    n_jobs=CONFIG['n_jobs']
)

# Fit ensemble with sample weights
ensemble.fit(X_train, y_train_encoded, sample_weight=sample_weights_train)

ensemble_metrics = evaluate_model(
    ensemble, X_train, y_train_encoded, X_val, y_val_encoded,
    "Ensemble (Voting)"
)

# ================================================================
# STEP 4: PROBABILITY CALIBRATION (PROPER WAY)
# ================================================================

print("\n" + "="*70)
print("STEP 4: PROBABILITY CALIBRATION")
print("="*70)

print("\nCalibrating ensemble using separate calibration set...")
print(f"Calibration samples: {len(X_calib):,}")

# Use prefit ensemble and separate calibration set
calibrated_ensemble = CalibratedClassifierCV(
    ensemble,
    method='isotonic',  # Non-parametric, more flexible
    cv='prefit'  # Use prefit model
)

# Fit calibration on the separate calibration set
calibrated_ensemble.fit(X_calib, y_calib_encoded)

print("Calibration complete")

# Evaluate calibrated ensemble
calibrated_metrics = evaluate_model(
    calibrated_ensemble, X_train, y_train_encoded, X_val, y_val_encoded,
    "Calibrated Ensemble"
)

# ================================================================
# STEP 5: MODEL COMPARISON & SELECTION
# ================================================================

print("\n" + "="*70)
print("STEP 5: MODEL COMPARISON & SELECTION")
print("="*70)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest', 'Logistic Reg', 'Ensemble', 'Calibrated Ensemble'],
    'CV_Score': [
        xgb_metrics['cv_score'],
        rf_metrics['cv_score'],
        lr_metrics['cv_score'],
        None,
        None
    ],
    'Val_Balanced': [
        xgb_metrics['val_balanced'],
        rf_metrics['val_balanced'],
        lr_metrics['val_balanced'],
        ensemble_metrics['val_balanced'],
        calibrated_metrics['val_balanced']
    ],
    'Acc_Gap': [
        xgb_metrics['acc_gap'],
        rf_metrics['acc_gap'],
        lr_metrics['acc_gap'],
        ensemble_metrics['acc_gap'],
        calibrated_metrics['acc_gap']
    ],
    'Val_F1': [
        xgb_metrics['val_f1'],
        rf_metrics['val_f1'],
        lr_metrics['val_f1'],
        ensemble_metrics['val_f1'],
        calibrated_metrics['val_f1']
    ]
})

print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# IMPROVED SELECTION: Consider both performance AND gap
comparison_df['Selection_Score'] = (
    comparison_df['Val_Balanced'] * 0.7 +  # 70% weight on performance
    (1 - comparison_df['Acc_Gap'].clip(0, 0.15) / 0.15) * 0.3  # 30% weight on controlling gap
)

best_idx = comparison_df['Selection_Score'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']

print(f"\nBest Model (by balanced criteria): {best_model_name}")
print(f"   Val Balanced Acc: {comparison_df.loc[best_idx, 'Val_Balanced']:.4f}")
print(f"   Gap: {comparison_df.loc[best_idx, 'Acc_Gap']:.4f}")
print(f"   Selection Score: {comparison_df.loc[best_idx, 'Selection_Score']:.4f}")

# Save comparison
comparison_df.to_csv(
    os.path.join(current_phase_dir, 'reports', 'model_comparison.csv'),
    index=False
)

# Select final model
model_map = {
    'XGBoost': (best_xgb, xgb_metrics),
    'Random Forest': (best_rf, rf_metrics),
    'Logistic Reg': (best_lr, lr_metrics),
    'Ensemble': (ensemble, ensemble_metrics),
    'Calibrated Ensemble': (calibrated_ensemble, calibrated_metrics)
}

final_model, final_metrics = model_map[best_model_name]

# ================================================================
# STEP 6: COMPREHENSIVE TEST SET EVALUATION
# ================================================================

print("\n" + "="*70)
print("STEP 6: COMPREHENSIVE TEST SET EVALUATION")
print("="*70)

print(f"\nEvaluating {best_model_name} on test set...")

# Test predictions
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)

# Test metrics
test_acc = accuracy_score(y_test_encoded, y_test_pred)
test_balanced = balanced_accuracy_score(y_test_encoded, y_test_pred)
test_f1_macro = f1_score(y_test_encoded, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test_encoded, y_test_pred, average='weighted')
test_mcc = matthews_corrcoef(y_test_encoded, y_test_pred)

# Multi-class AUC
y_test_bin = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
test_auc = roc_auc_score(y_test_bin, y_test_proba, average='macro', multi_class='ovr')

print(f"\nTest Set Performance:")
print(f"   Accuracy:            {test_acc:.4f}")
print(f"   Balanced Accuracy:   {test_balanced:.4f}")
print(f"   Macro F1:            {test_f1_macro:.4f}")
print(f"   Weighted F1:         {test_f1_weighted:.4f}")
print(f"   Matthews Corr:       {test_mcc:.4f}")
print(f"   ROC AUC (macro):     {test_auc:.4f}")

# Classification report
print(f"\nDetailed Classification Report:")
report_dict = classification_report(
    y_test_encoded, y_test_pred,
    target_names=le.classes_,
    output_dict=True,
    digits=4
)
print(classification_report(
    y_test_encoded, y_test_pred,
    target_names=le.classes_,
    digits=4
))

# Confusion matrix
cm_test = confusion_matrix(y_test_encoded, y_test_pred)

# Gap analysis
train_acc_final = final_metrics['train_acc']
val_acc_final = final_metrics['val_acc']
final_gap = final_metrics['acc_gap']
test_gap = train_acc_final - test_acc

print(f"\nOverfitting Control Analysis:")
print(f"   Train-Val Gap:   {final_gap:.4f} ({'✓ GOOD' if final_gap < CONFIG['realistic_gap_target'] else '⚠ MONITOR'})")
print(f"   Train-Test Gap:  {test_gap:.4f}")
print(f"   Val-Test Diff:   {abs(val_acc_final - test_acc):.4f}")

# ================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ================================================================

print("\n" + "="*70)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Get feature importance from ensemble components
if best_model_name in ['Ensemble', 'Calibrated Ensemble']:
    print("\nExtracting feature importance from ensemble components...")

    # XGBoost importance
    xgb_importance = best_xgb.feature_importances_

    # Random Forest importance
    rf_importance = best_rf.feature_importances_

    # Logistic Regression coefficients (absolute mean across classes)
    lr_coef = np.abs(best_lr.coef_).mean(axis=0)
    lr_coef_normalized = lr_coef / lr_coef.sum()

    # Weighted average (based on ensemble weights)
    ensemble_importance = (
        xgb_importance * 0.4 +  # Weight 2
        rf_importance * 0.4 +  # Weight 2
        lr_coef_normalized * 0.2 # Weight 1
    )

    importance_source = 'ensemble_weighted'

elif best_model_name == 'XGBoost':
    ensemble_importance = best_xgb.feature_importances_
    importance_source = 'xgboost'

elif best_model_name == 'Random Forest':
    ensemble_importance = best_rf.feature_importances_
    importance_source = 'random_forest'

else:  # Logistic Regression
    lr_coef = np.abs(best_lr.coef_).mean(axis=0)
    ensemble_importance = lr_coef / lr_coef.sum()
    importance_source = 'logistic_regression'

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': ensemble_importance
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 Most Important Features ({importance_source}):")
print(importance_df.head(20).to_string(index=False))

# Save importance
importance_df.to_csv(
    os.path.join(current_phase_dir, 'reports', 'feature_importance.csv'),
    index=False
)

# ================================================================
# STEP 8: SAVE ALL MODELS
# ================================================================

print("\n" + "="*70)
print("STEP 8: SAVING MODELS")
print("="*70)

# Save individual models
joblib.dump(best_xgb, os.path.join(current_phase_dir, 'models', 'xgboost.pkl'))
joblib.dump(best_rf, os.path.join(current_phase_dir, 'models', 'random_forest.pkl'))
joblib.dump(best_lr, os.path.join(current_phase_dir, 'models', 'logistic_regression.pkl'))
joblib.dump(ensemble, os.path.join(current_phase_dir, 'models', 'ensemble.pkl'))
joblib.dump(calibrated_ensemble, os.path.join(current_phase_dir, 'models', 'calibrated_ensemble.pkl'))
joblib.dump(final_model, os.path.join(current_phase_dir, 'models', 'final_model.pkl'))
joblib.dump(le, os.path.join(current_phase_dir, 'models', 'label_encoder.pkl'))

# Save grid search results
joblib.dump(grid_xgb, os.path.join(current_phase_dir, 'models', 'grid_search_xgb.pkl'))
joblib.dump(grid_rf, os.path.join(current_phase_dir, 'models', 'grid_search_rf.pkl'))
joblib.dump(grid_lr, os.path.join(current_phase_dir, 'models', 'grid_search_lr.pkl'))

print(f"✓ All models saved")

# ================================================================
# STEP 9: VISUALIZATIONS
# ================================================================

print("\n" + "="*70)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*70)

# 1. Model Comparison Dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

models = comparison_df['Model']
colors_perf = ['steelblue' if m != best_model_name else 'gold' for m in models]

# Panel 1: Validation Balanced Accuracy
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(range(len(models)), comparison_df['Val_Balanced'],
                color=colors_perf, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Balanced Accuracy')
ax1.set_title('Validation Balanced Accuracy', fontweight='bold')
ax1.axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='Target: 0.80')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(comparison_df['Val_Balanced']):
    ax1.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom',
             fontsize=9, fontweight='bold')

# Panel 2: Overfitting Gap
ax2 = fig.add_subplot(gs[0, 1])
gap_colors = ['green' if g < 0.05 else 'orange' if g < 0.07 else 'red'
              for g in comparison_df['Acc_Gap']]
bars2 = ax2.bar(range(len(models)), comparison_df['Acc_Gap'],
                color=gap_colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Train-Val Gap')
ax2.set_title('Overfitting Control', fontweight='bold')
ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Ideal: <0.05')
ax2.axhline(y=0.07, color='orange', linestyle='--', alpha=0.5, label='Target: <0.07')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
for i, gap in enumerate(comparison_df['Acc_Gap']):
    ax2.text(i, gap + 0.002, f'{gap:.3f}', ha='center', va='bottom',
             fontsize=9, fontweight='bold')

# Panel 3: Selection Score
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(range(len(models)), comparison_df['Selection_Score'],
                color=colors_perf, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax3.set_ylabel('Score')
ax3.set_title('Model Selection Score\n(70% perf + 30% gap control)', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i, score in enumerate(comparison_df['Selection_Score']):
    ax3.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom',
             fontsize=9, fontweight='bold')

# Panel 4: Test Set Metrics
ax4 = fig.add_subplot(gs[1, :2])
test_metrics_viz = {
    'Accuracy': test_acc,
    'Balanced\nAcc': test_balanced,
    'Macro F1': test_f1_macro,
    'Weighted\nF1': test_f1_weighted,
    'MCC': test_mcc,
    'ROC AUC': test_auc
}
metric_names = list(test_metrics_viz.keys())
metric_values = list(test_metrics_viz.values())
colors_metric = ['skyblue', 'lightgreen', 'coral', 'gold', 'plum', 'lightpink']

bars4 = ax4.bar(range(len(metric_names)), metric_values,
                color=colors_metric, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(metric_names)))
ax4.set_xticklabels(metric_names, fontsize=10)
ax4.set_ylabel('Score')
ax4.set_title(f'Test Set Performance - {best_model_name}', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(metric_values):
    ax4.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Panel 5: Gap Comparison
ax5 = fig.add_subplot(gs[1, 2])
gap_data = {
    'Train-Val': final_gap,
    'Train-Test': test_gap,
    'Val-Test': abs(val_acc_final - test_acc)
}
gap_names = list(gap_data.keys())
gap_values = list(gap_data.values())
gap_bar_colors = ['green' if v < 0.05 else 'orange' if v < 0.07 else 'red' for v in gap_values]

bars5 = ax5.bar(range(len(gap_names)), gap_values,
                color=gap_bar_colors, alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(gap_names)))
ax5.set_xticklabels(gap_names, fontsize=10)
ax5.set_ylabel('Gap')
ax5.set_title('Generalization Analysis', fontweight='bold')
ax5.axhline(y=CONFIG['realistic_gap_target'], color='orange', linestyle='--',
            alpha=0.5, label=f'Target: {CONFIG["realistic_gap_target"]}')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(gap_values):
    ax5.text(i, val + 0.002, f'{val:.3f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Panel 6: Confusion Matrix (normalized)
ax6 = fig.add_subplot(gs[2, :])
cm_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
im = ax6.imshow(cm_normalized, interpolation='nearest', cmap='Blues', aspect='auto')
cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Count', rotation=270, labelpad=15)

ax6.set_xticks(np.arange(len(le.classes_)))
ax6.set_yticks(np.arange(len(le.classes_)))
ax6.set_xticklabels(le.classes_, fontsize=11)
ax6.set_yticklabels(le.classes_, fontsize=11)
ax6.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax6.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax6.set_title(f'Test Set Confusion Matrix (Normalized)\n{best_model_name}',
              fontsize=12, fontweight='bold')

# Add text annotations
thresh = cm_normalized.max() / 2.
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        text_color = "white" if cm_normalized[i, j] > thresh else "black"
        ax6.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm_test[i, j]})',
                ha="center", va="center", color=text_color,
                fontsize=10, fontweight='bold')

plt.suptitle('Phase 4: Comprehensive Model Evaluation Dashboard',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(current_phase_dir, 'figures', 'comprehensive_evaluation.png'),
            dpi=CONFIG['dpi'], bbox_inches='tight')
plt.show()
plt.close(fig) # Close figure

print("✓ Comprehensive evaluation dashboard saved")

# 2. Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 10))

top_features = importance_df.head(30)
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

ax.barh(range(len(top_features)), top_features['Importance'],
        color=colors_feat, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title(f'Top 30 Feature Importances - {best_model_name}\nSource: {importance_source}',
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(current_phase_dir, 'figures', 'feature_importance.png'),
            dpi=CONFIG['dpi'], bbox_inches='tight')
plt.show()
plt.close(fig) # Close figure

print("✓ Feature importance plot saved")

# ================================================================
# STEP 10: FINAL SUMMARY REPORT
# ================================================================

print("\n" + "="*70)
print("STEP 10: GENERATING FINAL SUMMARY")
print("="*70)

summary = {
    'timestamp': datetime.now().isoformat(),
    'best_model': {
        'name': best_model_name,
        'selection_score': float(comparison_df.loc[best_idx, 'Selection_Score']),
    },
    'test_performance': {
        'accuracy': float(test_acc),
        'balanced_accuracy': float(test_balanced),
        'f1_macro': float(test_f1_macro),
        'mcc': float(test_mcc),
        'auc': float(test_auc),
        'per_class': {
            cls: {
                'precision': float(report_dict[cls]['precision']),
                'recall': float(report_dict[cls]['recall']),
                'f1-score': float(report_dict[cls]['f1-score']),
                'support': int(report_dict[cls]['support'])
            } for cls in le.classes_
        },
        'confusion_matrix': cm_test.tolist()
    },
    'stability_analysis': {
        'train_val_gap': float(final_gap),
        'train_test_gap': float(test_gap),
        'val_test_diff': float(abs(val_acc_final - test_acc)),
    },
    'feature_importance': {
        'source': importance_source,
        'top_10_features': importance_df.head(10)['Feature'].tolist(),
        'top_10_scores': importance_df.head(10)['Importance'].tolist()
    }
}

# Save summary
with open(os.path.join(current_phase_dir, 'reports', 'phase4_final_summary.json'), 'w') as f:
    json.dump(convert_to_serializable(summary), f, indent=2, ensure_ascii=False)

print("✓ Final summary saved")

# ================================================================
# COMPLETION REPORT
# ================================================================

print("\n" + "="*70)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("="*70)
print(f"\nBEST MODEL: {best_model_name}")
print(f"\nTEST SET RESULTS:")
print(f"   Accuracy:          {test_acc:.4f}")
print(f"   Balanced Accuracy: {test_balanced:.4f}")
print(f"   Macro F1:          {test_f1_macro:.4f}")
print(f"   ROC AUC:           {test_auc:.4f}")
print(f"\nREADY FOR: Phase 5 - SHAP Analysis & Clinical Validation")
print("="*70)