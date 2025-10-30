# ================================================================
# PHASE 6: PRE-PUBLICATION CRITICAL ANALYSES
# [GitHub Ready Version]
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime
import json

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    'random_state': 42,
    'base_path': '.', # [GitHub Ready] Relative path
    'phase3_dir': './Phase3_Class_Imbalance',
    'phase4_dir': './Phase4_Model_Development',
    'phase6_dir': './Phase6_Publication_Analyses',
    'target_col': 'Triaj',
    'n_bootstraps': 1000,
    'confidence_level': 0.95,
    'n_jobs': -1,
    'dpi': 300
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 6: PRE-PUBLICATION CRITICAL ANALYSES")
print("="*70)

# [GitHub Ready] Ensure directories exist
os.makedirs(CONFIG['phase6_dir'], exist_ok=True)
for subdir in ['figures', 'reports', 'models']:
    os.makedirs(os.path.join(CONFIG['phase6_dir'], subdir), exist_ok=True)

# ================================================================
# LOAD DATA AND MODELS
# ================================================================
print("\nLoading data and models...")

# [GitHub Ready] Load original (non-resampled) train data for learning curve
train_orig_df = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'train_original.csv'))
val_df = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'validation.csv'))
test_df = pd.read_csv(os.path.join(CONFIG['phase3_dir'], 'data', 'test.csv'))

# Combine train+val for full learning curve
full_train_df = pd.concat([train_orig_df, val_df], ignore_index=True)

final_model = joblib.load(os.path.join(CONFIG['phase4_dir'], 'models', 'final_model.pkl'))
le = joblib.load(os.path.join(CONFIG['phase4_dir'], 'models', 'label_encoder.pkl'))

def prepare_data(df, target_col):
    # ... (function content is unchanged) ...
    pass # Placeholder

X_test, y_test = prepare_data(test_df, CONFIG['target_col'])
y_test_enc = le.transform(y_test)
y_test_pred = final_model.predict(X_test)

# ================================================================
# TEST 1: TEMPORAL VALIDATION (SIMULATED)
# ================================================================
print("\nTEST 1: TEMPORAL VALIDATION (SIMULATED)")
print("WARNING: This analysis MUST be re-run with a real date column for publication.")

# ... (Simulation logic is unchanged, but serves as a placeholder) ...
# ... (Make sure your 'full_df' for this needs to be loaded from original data) ...

# ================================================================
# TEST 2: BOOTSTRAP CONFIDENCE INTERVALS
# ================================================================
print("\nTEST 2: BOOTSTRAP CONFIDENCE INTERVALS")

def bootstrap_metrics(X, y, y_pred, n_bootstraps=1000, random_state=42):
    # ... (function content is unchanged) ...
    pass # Placeholder

bootstrap_results = bootstrap_metrics(X_test, y_test_enc, y_test_pred, n_bootstraps=CONFIG['n_bootstraps'])
# ... (CI calculation and visualization) ...

# ================================================================
# TEST 3: BASELINE MODEL COMPARISON
# ================================================================
print("\nTEST 3: BASELINE MODEL COMPARISON")

# ... (Loading/training dummy, LR, Tree, and Rule-based models) ...
# ... (McNemar test logic) ...

# ================================================================
# TEST 4: SUBGROUP ANALYSIS
# ================================================================
print("\nTEST 4: SUBGROUP PERFORMANCE ANALYSIS")
# ... (Subgroup analysis logic for Age, Ambulance, etc.) ...
# ... (Relies on column names like 'YAŞ', 'ambulans_ile_geldi') ...

# ================================================================
# TEST 5: LEARNING CURVES
# ================================================================
print("\nTEST 5: LEARNING CURVES")

X_train_full, y_train_full = prepare_data(full_train_df, CONFIG['target_col'])
y_train_full_enc = le.transform(y_train_full)

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=15, random_state=CONFIG['random_state']),
    X_train_full, y_train_full_enc,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG['random_state']),
    scoring='balanced_accuracy',
    n_jobs=CONFIG['n_jobs'],
    verbose=0
)
# ... (Plotting logic) ...

# ================================================================
# TEST 6: FEATURE ABLATION STUDY
# ================================================================
print("\nTEST 6: FEATURE ABLATION STUDY")
# ... (Feature group definition and ablation loop) ...
# ... (Relies on column names for groups) ...

# ================================================================
# FINAL PUBLICATION CHECKLIST & SUMMARY
# ================================================================
print("\nFINAL PUBLICATION CHECKLIST")
# ... (Checklist logic and summary saving) ...

print("\nPHASE 6 COMPLETE\n")