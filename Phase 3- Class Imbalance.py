# ================================================================
# PHASE 3 - Balanced Class Imbalance Handling
# [GitHub Ready Version - FULL]
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter
from datetime import datetime
import json
import joblib

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    'random_state': 42,
    'base_path': '.', # [GitHub Ready] Relative path
    'phase2_dir': './Phase2_Feature_Engineering',
    'phase3_dir': './Phase3_Class_Imbalance',
    'test_size': 0.15,
    'validation_size': 0.15,
    'target_col': 'Triaj',

    # RESAMPLING STRATEGY
    'resampling_strategy': 'moderate',  # 'conservative', 'moderate', 'aggressive'
    'target_imbalance_ratio': 2.0,

    # SMOTE PARAMETERS
    'smote_method': 'borderline',  # 'borderline', 'regular', 'combined'
    'k_neighbors': 5,

    # CLASS WEIGHTS
    'use_computed_weights': True,
    'manual_weights': { # Fallback if 'use_computed_weights' is False
        'Kırmızı': 1.0,
        'Sarı': 1.8,
        'Yeşil': 0.6
    },

    'figsize': (12, 8),
    'dpi': 300
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 3: BALANCED CLASS IMBALANCE HANDLING")
print("="*70)

# [GitHub Ready] Ensure directories exist
os.makedirs(CONFIG['phase3_dir'], exist_ok=True)
for subdir in ['data', 'figures', 'reports', 'models']:
    os.makedirs(os.path.join(CONFIG['phase3_dir'], subdir), exist_ok=True)
current_phase_dir = CONFIG['phase3_dir']

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

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

print("\nLoading Phase 2 data...")
data_path = os.path.join(CONFIG['phase2_dir'], 'data', 'data_after_feature_engineering.csv')

try:
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading data from {data_path}: {e}")
    raise SystemExit("Cannot proceed without Phase 2 data")

# Initial distribution analysis
initial_counts = df[CONFIG['target_col']].value_counts().sort_index()
initial_ir = initial_counts.max() / initial_counts.min()

print(f"\nInitial Distribution:")
for cls, count in initial_counts.items():
    print(f"   {cls}: {count:,} ({count/len(df)*100:.1f}%)")
print(f"   Imbalance Ratio: {initial_ir:.2f}")

# ================================================================
# STEP 1: STRATIFIED DATA SPLITTING
# ================================================================

print("\n" + "="*70)
print("STEP 1: STRATIFIED DATA SPLITTING")
print("="*70)
print("Note: Splitting is done at the patient level (1 row per patient).")

# First split: train+val vs test (stratified)
train_val_df, test_df = train_test_split(
    df,
    test_size=CONFIG['test_size'],
    stratify=df[CONFIG['target_col']],
    random_state=CONFIG['random_state']
)

# Second split: train vs val (stratified)
val_size_adjusted = CONFIG['validation_size'] / (1 - CONFIG['test_size'])
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_size_adjusted,
    stratify=train_val_df[CONFIG['target_col']],
    random_state=CONFIG['random_state']
)

print(f"\nData Split Summary:")
print(f"   Training:   {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Validation: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

# Check stratification
print(f"\nClass Distribution by Split:")
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df[CONFIG['target_col']].value_counts(normalize=True).sort_index() * 100
    print(f"   {split_name:8s}: ", end='')
    print(" | ".join([f"{cls}: {pct:.1f}%" for cls, pct in dist.items()]))

# ================================================================
# STEP 2: PREPARE DATA FOR RESAMPLING
# ================================================================

print("\n" + "="*70)
print("STEP 2: DATA PREPARATION")
print("="*70)

def prepare_data(df, target_col):
    """Prepare features and target"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Use only numeric features for SMOTE
    # (SMOTE typically requires numeric input)
    # Categorical features (like Age_Category) should be encoded if used
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]

    print(f"   Features prepared: {len(numeric_cols)} numeric columns")
    return X_numeric, y, numeric_cols

X_train, y_train, feature_cols = prepare_data(train_df, CONFIG['target_col'])

# Encode target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
class_names = le.classes_

print(f"   Target encoded: {len(class_names)} classes")
print(f"   Class mapping: {dict(zip(range(len(class_names)), class_names))}")

# Original distribution
original_dist = Counter(y_train)
print(f"\nOriginal Training Distribution:")
for cls in sorted(original_dist.keys()):
    count = original_dist[cls]
    print(f"   {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")

original_ir = max(original_dist.values()) / min(original_dist.values())
print(f"   Original IR: {original_ir:.2f}")

# ================================================================
# STEP 3: CALCULATE TARGET DISTRIBUTION
# ================================================================

print("\n" + "="*70)
print("STEP 3: RESAMPLING STRATEGY CALCULATION")
print("="*70)

def calculate_target_distribution(original_dist, strategy='moderate', target_ir=2.0):
    """
    Calculate target samples for each class
    """
    majority_class = max(original_dist, key=original_dist.get)
    majority_samples = original_dist[majority_class]
    
    target_samples = {}
    
    if strategy == 'moderate':
        # Oversample minorities to match target_ir relative to majority
        # Undersample nothing
        target_base = int(majority_samples / target_ir)
        
        for cls, count in original_dist.items():
            if count < target_base:
                target_samples[cls] = target_base # Oversample
            else:
                target_samples[cls] = count # Keep as is

    elif strategy == 'conservative':
        # Less aggressive oversampling
        target_base = int(majority_samples / (target_ir + 0.5)) # e.g., target 2.5
        for cls, count in original_dist.items():
            if count < target_base:
                target_samples[cls] = target_base
            else:
                target_samples[cls] = count

    elif strategy == 'aggressive':
        # Nearly equal distribution
        target_base = int(majority_samples / (target_ir - 0.8)) # e.g., target 1.2
        for cls in original_dist.keys():
            if original_dist[cls] < target_base:
                target_samples[cls] = target_base
            else:
                target_samples[cls] = original_dist[cls]

    # Ensure no class is sampled below its original count
    for cls, count in original_dist.items():
        if target_samples[cls] < count:
            target_samples[cls] = count
            
    return target_samples

target_samples = calculate_target_distribution(
    original_dist,
    strategy=CONFIG['resampling_strategy'],
    target_ir=CONFIG['target_imbalance_ratio']
)

print(f"\nTarget Distribution ({CONFIG['resampling_strategy']} strategy):")
total_synthetic = 0
total_target = sum(target_samples.values())

for cls in sorted(target_samples.keys()):
    original = original_dist[cls]
    target = target_samples[cls]
    change = target - original
    
    if change > 0:
        print(f"   {cls}: {target:,} (was {original:,}, +{change:,} synthetic)")
        total_synthetic += change
    else:
        print(f"   {cls}: {target:,} (unchanged)")

predicted_ir = max(target_samples.values()) / min(target_samples.values())
print(f"\nPredicted IR after resampling: {predicted_ir:.2f}")
print(f"Total synthetic samples to add: {total_synthetic:,}")

synthetic_pct = (total_synthetic / total_target) * 100
print(f"Synthetic percentage in new dataset: {synthetic_pct:.1f}%")

if synthetic_pct > 30:
    print(f"⚠️  WARNING: Synthetic samples exceed 30% - risk of overfitting!")
elif synthetic_pct > 20:
    print(f"⚠️  CAUTION: Synthetic samples at {synthetic_pct:.1f}% - monitor for overfitting")
else:
    print(f"✓ Synthetic samples at {synthetic_pct:.1f}% - acceptable range")

# ================================================================
# STEP 4: APPLY RESAMPLING
# ================================================================

print("\n" + "="*70)
print("STEP 4: APPLYING RESAMPLING")
print("="*70)

# Convert to sklearn format
sampling_strategy = {}
for cls, target in target_samples.items():
    encoded_cls = le.transform([cls])[0]
    original = original_dist[cls]

    # Only add to strategy if we need to oversample
    if target > original:
        sampling_strategy[encoded_cls] = target

print(f"\nSMOTE sampling strategy: {len(sampling_strategy)} classes to oversample")
for encoded_cls, target_count in sampling_strategy.items():
    cls_name = le.inverse_transform([encoded_cls])[0]
    original = original_dist[cls_name]
    print(f"   {cls_name}: {original:,} → {target_count:,}")

# Apply SMOTE
method_used = CONFIG['smote_method']
smote_success = False

try:
    if method_used == 'borderline':
        print(f"\nApplying BorderlineSMOTE (k={CONFIG['k_neighbors']})...")
        sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=CONFIG['random_state'],
            k_neighbors=CONFIG['k_neighbors'],
            kind='borderline-1'
        )

    elif method_used == 'combined':
        print(f"\nApplying SMOTETomek (SMOTE + Tomek Links cleaning)...")
        sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=CONFIG['random_state'],
            smote=SMOTE(k_neighbors=CONFIG['k_neighbors'], random_state=CONFIG['random_state'])
        )

    else:  # regular SMOTE
        print(f"\nApplying regular SMOTE (k={CONFIG['k_neighbors']})...")
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=CONFIG['random_state'],
            k_neighbors=CONFIG['k_neighbors']
        )

    X_resampled, y_resampled_encoded = sampler.fit_resample(X_train, y_train_encoded)
    y_resampled = le.inverse_transform(y_resampled_encoded)
    smote_success = True
    print(f"✓ {method_used.upper()} completed successfully")

except Exception as e:
    print(f"✗ {method_used.upper()} failed: {str(e)}")
    print(f"  Using original data without resampling")
    X_resampled = X_train.copy()
    y_resampled = y_train.copy()
    y_resampled_encoded = y_train_encoded.copy()
    method_used = 'None'

# ================================================================
# STEP 5: ANALYZE RESULTS
# ================================================================

print("\n" + "="*70)
print("STEP 5: RESAMPLING RESULTS ANALYSIS")
print("="*70)

resampled_dist = Counter(y_resampled)
achieved_ir = max(resampled_dist.values()) / min(resampled_dist.values())

print(f"\nResampled Distribution:")
for cls in sorted(resampled_dist.keys()):
    resampled_count = resampled_dist[cls]
    original_count = original_dist.get(cls, 0) # Use .get for safety
    change = resampled_count - original_count
    change_pct = (change / original_count * 100) if original_count > 0 else 0

    print(f"   {cls}: {resampled_count:,} (was {original_count:,}, {change:+,}, {change_pct:+.1f}%)")

print(f"\nImbalance Ratio:")
print(f"   Before: {original_ir:.2f}")
print(f"   After:  {achieved_ir:.2f}")
print(f"   Target: {CONFIG['target_imbalance_ratio']:.2f}")

total_samples_added = len(X_resampled) - len(X_train)
final_synthetic_pct = (total_samples_added / len(X_resampled)) * 100

print(f"\nSample Statistics:")
print(f"   Original training samples: {len(X_train):,}")
print(f"   Resampled training samples: {len(X_resampled):,}")
print(f"   Synthetic samples added: {total_samples_added:,}")
print(f"   Synthetic percentage: {final_synthetic_pct:.1f}%")

# ================================================================
# STEP 6: COMPUTE CLASS WEIGHTS
# ================================================================

print("\n" + "="*70)
print("STEP 6: CLASS WEIGHT CALCULATION")
print("="*70)

if CONFIG['use_computed_weights']:
    # Compute balanced weights from resampled data
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_resampled_encoded),
        y=y_resampled_encoded
    )

    class_weights = {}
    for idx, cls_name in enumerate(le.classes_):
        class_weights[cls_name] = float(class_weights_array[idx])

    print(f"\nComputed Class Weights (from resampled data):")
    for cls in sorted(class_weights.keys()):
        print(f"   {cls}: {class_weights[cls]:.3f}")

    # Scale weights to be centered around 1.0 for interpretability
    mean_weight = np.mean(list(class_weights.values()))
    class_weights_scaled = {k: v/mean_weight for k, v in class_weights.items()}

    print(f"\nScaled Class Weights (normalized):")
    for cls in sorted(class_weights_scaled.keys()):
        print(f"   {cls}: {class_weights_scaled[cls]:.3f}")

    class_weights_final = class_weights_scaled
    weights_method = 'computed_balanced'

else:
    class_weights_final = CONFIG['manual_weights']
    weights_method = 'manual'
    print(f"\nUsing Manual Class Weights:")
    for cls in sorted(class_weights_final.keys()):
        print(f"   {cls}: {class_weights_final[cls]:.3f}")

# ================================================================
# STEP 7: SAVE PROCESSED DATA
# ================================================================

print("\n" + "="*70)
print("STEP 7: SAVING PROCESSED DATASETS")
print("="*70)

# Convert to DataFrame
X_train_resampled_df = pd.DataFrame(X_resampled, columns=feature_cols)
train_resampled_df = X_train_resampled_df.copy()
train_resampled_df[CONFIG['target_col']] = y_resampled

# Save all datasets
train_resampled_df.to_csv(
    os.path.join(current_phase_dir, 'data', 'train_resampled.csv'),
    index=False
)
train_df.to_csv(
    os.path.join(current_phase_dir, 'data', 'train_original.csv'),
    index=False
)
val_df.to_csv(
    os.path.join(current_phase_dir, 'data', 'validation.csv'),
    index=False
)
test_df.to_csv(
    os.path.join(current_phase_dir, 'data', 'test.csv'),
    index=False
)

print(f"✓ Training (resampled):  {len(train_resampled_df):,} samples")
print(f"✓ Training (original):   {len(train_df):,} samples")
print(f"✓ Validation:            {len(val_df):,} samples")
print(f"✓ Test:                  {len(test_df):,} samples")

# Save class weights
weight_info = {
    'method': weights_method,
    'weights': class_weights_final,
    'weights_raw': class_weights if CONFIG['use_computed_weights'] else None,
    'resampling_strategy': CONFIG['resampling_strategy'],
    'timestamp': datetime.now().isoformat()
}

with open(os.path.join(current_phase_dir, 'reports', 'class_weights.json'), 'w') as f:
    json.dump(convert_to_serializable(weight_info), f, indent=2, ensure_ascii=False)

joblib.dump(class_weights_final, os.path.join(current_phase_dir, 'models', 'class_weights.pkl'))
joblib.dump(le, os.path.join(current_phase_dir, 'models', 'label_encoder.pkl'))

print(f"✓ Class weights saved")
print(f"✓ Label encoder saved")

# Save comprehensive summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'resampling_strategy': CONFIG['resampling_strategy'],
        'target_ir': CONFIG['target_imbalance_ratio'],
        'smote_method': method_used,
        'k_neighbors': CONFIG['k_neighbors']
    },
    'data_split': {
        'train_original': len(train_df),
        'train_resampled': len(train_resampled_df),
        'validation': len(val_df),
        'test': len(test_df)
    },
    'class_distribution': {
        'original': {str(k): int(v) for k, v in original_dist.items()},
        'resampled': {str(k): int(v) for k, v in resampled_dist.items()},
        'target': {str(k): int(v) for k, v in target_samples.items()}
    },
    'imbalance_ratio': {
        'original': float(original_ir),
        'achieved': float(achieved_ir),
        'target': float(CONFIG['target_imbalance_ratio'])
    },
    'synthetic_samples': {
        'count': int(total_samples_added),
        'percentage': float(final_synthetic_pct)
    },
    'class_weights': {
        'method': weights_method,
        'weights': class_weights_final
    },
    'quality_checks': {
        'synthetic_pct_safe': final_synthetic_pct <= 25,
        'ir_achieved': abs(achieved_ir - CONFIG['target_imbalance_ratio']) <= 0.5,
        'smote_success': smote_success
    }
}

with open(os.path.join(current_phase_dir, 'reports', 'phase3_summary.json'), 'w') as f:
    json.dump(convert_to_serializable(summary), f, indent=2, ensure_ascii=False)

print(f"✓ Phase 3 summary saved")

# ================================================================
# STEP 8: VISUALIZATION
# ================================================================

print("\n" + "="*70)
print("STEP 8: CREATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

classes = sorted(original_dist.keys())
colors = {'Kırmızı': '#d62728', 'Sarı': '#ff7f0e', 'Yeşil': '#2ca02c'}
class_colors = [colors.get(c, 'gray') for c in classes]

# 1. Original Distribution
ax1 = fig.add_subplot(gs[0, 0])
counts_orig = [original_dist[c] for c in classes]
bars1 = ax1.bar(range(len(classes)), counts_orig, color=class_colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(classes)))
ax1.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax1.set_title('BEFORE Resampling', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars1, counts_orig)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({count/sum(counts_orig)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Target Distribution
ax2 = fig.add_subplot(gs[0, 1])
counts_target = [target_samples[c] for c in classes]
bars2 = ax2.bar(range(len(classes)), counts_target, color=class_colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax2.set_title('TARGET Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars2, counts_target)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({count/sum(counts_target)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Resampled Distribution
ax3 = fig.add_subplot(gs[0, 2])
counts_resamp = [resampled_dist[c] for c in classes]
bars3 = ax3.bar(range(len(classes)), counts_resamp, color=class_colors, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(classes)))
ax3.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax3.set_title(f'AFTER Resampling\n(IR: {achieved_ir:.2f})', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count', fontsize=10)
ax3.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars3, counts_resamp)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({count/sum(counts_resamp)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Comparison Bar Chart
ax4 = fig.add_subplot(gs[1, :])
x = np.arange(len(classes))
width = 0.25
ax4.bar(x - width, counts_orig, width, label='Original', color='lightcoral', alpha=0.7, edgecolor='black')
ax4.bar(x, counts_target, width, label='Target', color='lightyellow', alpha=0.7, edgecolor='black')
ax4.bar(x + width, counts_resamp, width, label='Achieved', color='lightgreen', alpha=0.7, edgecolor='black')
ax4.set_xticks(x)
ax4.set_xticklabels(classes, fontsize=11, fontweight='bold')
ax4.set_ylabel('Sample Count', fontsize=11)
ax4.set_title('Distribution Comparison: Original vs Target vs Achieved', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 5. Class Weights
ax5 = fig.add_subplot(gs[2, 0])
weight_values = [class_weights_final[c] for c in classes]
bars5 = ax5.bar(range(len(classes)), weight_values, color=class_colors, alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(classes)))
ax5.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax5.set_title('Class Weights', fontsize=12, fontweight='bold')
ax5.set_ylabel('Weight', fontsize=10)
ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0)')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
for i, (bar, weight) in enumerate(zip(bars5, weight_values)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{weight:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. Synthetic Samples per Class
ax6 = fig.add_subplot(gs[2, 1])
synthetic_per_class = [resampled_dist[c] - original_dist.get(c,0) for c in classes]
bars6 = ax6.bar(range(len(classes)), synthetic_per_class, color=class_colors, alpha=0.7, edgecolor='black')
ax6.set_xticks(range(len(classes)))
ax6.set_xticklabels(classes, fontsize=10, fontweight='bold')
ax6.set_title('Synthetic Samples Added', fontsize=12, fontweight='bold')
ax6.set_ylabel('Count', fontsize=10)
ax6.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars6, synthetic_per_class)):
    if count > 0:
        height = bar.get_height()
        pct = (count / original_dist[classes[i]] * 100) if original_dist[classes[i]] > 0 else 0
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                 f'+{count:,}\n(+{pct:.0f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# 7. Imbalance Ratio Comparison
ax7 = fig.add_subplot(gs[2, 2])
ir_data = ['Original', 'Target', 'Achieved']
ir_values = [original_ir, CONFIG['target_imbalance_ratio'], achieved_ir]
bars7 = ax7.bar(ir_data, ir_values, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
ax7.set_ylabel('Imbalance Ratio', fontsize=10)
ax7.set_title('Imbalance Ratio Reduction', fontsize=12, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)
for bar, value in zip(bars7, ir_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle(f'Phase 3: Class Imbalance Handling - {CONFIG["resampling_strategy"].upper()} Strategy\n'
             f'Method: {method_used} | Synthetic: {final_synthetic_pct:.1f}%',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(current_phase_dir, 'figures', 'phase3_comprehensive_analysis.png'),
            dpi=CONFIG['dpi'], bbox_inches='tight')
plt.show()
plt.close(fig) # Close figure

print(f"✓ Comprehensive visualization saved")

# ================================================================
# STEP 9: QUALITY CHECKS
# ================================================================

print("\n" + "="*70)
print("STEP 9: QUALITY CHECKS")
print("="*70)

quality_checks = []

# Check 1: Synthetic percentage
check1 = final_synthetic_pct <= 25
quality_checks.append({
    'check': 'Synthetic samples <= 25%',
    'status': 'PASS' if check1 else 'FAIL',
    'value': f'{final_synthetic_pct:.1f}%',
    'threshold': '≤25%'
})
print(f"{'✓' if check1 else '✗'} Synthetic samples: {final_synthetic_pct:.1f}% {'(SAFE)' if check1 else '(HIGH - risk of overfitting)'}")

# Check 2: IR achievement
ir_diff = abs(achieved_ir - CONFIG['target_imbalance_ratio'])
check2 = ir_diff <= 0.5
quality_checks.append({
    'check': 'IR within ±0.5 of target',
    'status': 'PASS' if check2 else 'WARN',
    'value': f'{achieved_ir:.2f}',
    'target': f'{CONFIG["target_imbalance_ratio"]:.2f}'
})
print(f"{'✓' if check2 else '⚠'} IR achieved: {achieved_ir:.2f} (target: {CONFIG['target_imbalance_ratio']:.2f}, diff: {ir_diff:.2f})")

# Check 3: Class weights reasonable
max_weight = max(class_weights_final.values())
min_weight = min(class_weights_final.values())
weight_ratio = max_weight / min_weight
check3 = weight_ratio <= 4.0
quality_checks.append({
    'check': 'Class weight ratio <= 4.0',
    'status': 'PASS' if check3 else 'WARN',
    'value': f'{weight_ratio:.2f}',
    'threshold': '≤4.0'
})
print(f"{'✓' if check3 else '⚠'} Class weight ratio: {weight_ratio:.2f} {'(reasonable)' if check3 else '(high - may overemphasize minority)'}")

# Check 4: SMOTE success
check4 = smote_success
quality_checks.append({
    'check': 'SMOTE applied successfully',
    'status': 'PASS' if check4 else 'FAIL',
    'value': method_used
})
print(f"{'✓' if check4 else '✗'} SMOTE status: {method_used} {'applied' if check4 else 'failed'}")

# Save quality checks
with open(os.path.join(current_phase_dir, 'reports', 'quality_checks.json'), 'w') as f:
    json.dump(convert_to_serializable(quality_checks), f, indent=2, ensure_ascii=False)

all_pass = all(c['status'] == 'PASS' for c in quality_checks)
print(f"\nOverall Quality: {'✓ ALL CHECKS PASSED' if all_pass else '⚠ SOME WARNINGS PRESENT'}")

# ================================================================
# COMPLETION REPORT
# ================================================================

print("\n" + "="*70)
print("PHASE 3 COMPLETED SUCCESSFULLY")
print("="*70)

print(f"\nREADY FOR: Phase 4 - Model Development")
print("="*70)