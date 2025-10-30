# ================================================================
# EMERGENCY DEPARTMENT TRIAGE PREDICTION - PHASE 2
# Feature Engineering and Optimization
# [GitHub Ready Version - FULL]
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ML and statistics
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime
import json
import joblib

# ================================================================
# CONFIGURATION
# ================================================================

CONFIG = {
    'random_state': 42,
    'base_path': '.', # [GitHub Ready] Relative path
    'phase1_dir': './Phase1_Data_Preprocessing',
    'phase2_dir': './Phase2_Feature_Engineering',
    'vif_threshold': 10,
    'outlier_method': 'winsorize', # Was 'IQR' in your code, but implementation was winsorize
    'figsize': (12, 8),
    'dpi': 300
}

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 2: FEATURE ENGINEERING AND OPTIMIZATION")
print("="*70)

# [GitHub Ready] Ensure directories exist from config
os.makedirs(CONFIG['phase2_dir'], exist_ok=True)
for subdir in ['figures', 'data', 'reports', 'models']:
    os.makedirs(os.path.join(CONFIG['phase2_dir'], subdir), exist_ok=True)
current_phase_dir = CONFIG['phase2_dir']

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
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
# LOAD PHASE 1 DATA
# ================================================================

print("\nLoading Phase 1 processed data...")
data_path = os.path.join(CONFIG['phase1_dir'], 'data', 'data_after_imputation.csv')

try:
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Create backup
    df_backup = df.copy()

except Exception as e:
    print(f"Error loading data from {data_path}: {e}")
    raise SystemExit("Cannot proceed without Phase 1 data")

# ================================================================
# STEP 1: MULTICOLLINEARITY ANALYSIS AND RESOLUTION
# ================================================================

print("\n" + "="*70)
print("STEP 1: MULTICOLLINEARITY ANALYSIS")
print("="*70)

def calculate_vif(df, target_col='Triaj'):
    """Calculate VIF for all numeric features"""
    print("\nCalculating VIF for numeric features...")

    # Get numeric columns excluding target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Remove columns with zero variance
    variance_cols = []
    for col in numeric_cols:
        if df[col].var() > 1e-6: # Add small threshold for near-zero variance
            variance_cols.append(col)
        else:
            print(f"   Skipping '{col}' (zero or near-zero variance)")
            
    if not variance_cols:
        print("   No numeric features with variance found.")
        return pd.DataFrame(columns=["Feature", "VIF", "VIF_Category"])

    print(f"Analyzing {len(variance_cols)} numeric features...")

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = variance_cols

    X = df[variance_cols].copy()
    # Fill NaNs with median for VIF calculation
    X = X.fillna(X.median()) 
    
    # Check for remaining NaNs or Infs
    if np.any(np.isnan(X.values)) or np.any(np.isinf(X.values)):
        print("   Warning: NaNs or Infs still present after median fill. Filling with 0.")
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

    vif_values = []
    for i in range(len(variance_cols)):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)
        except Exception as e:
            print(f"   Error calculating VIF for {variance_cols[i]}: {e}")
            vif_values.append(np.nan)

    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values('VIF', ascending=False)

    # Categorize VIF levels
    vif_data['VIF_Category'] = pd.cut(vif_data['VIF'],
                                       bins=[0, 5, 10, 100, np.inf],
                                       labels=['Low (<5)', 'Moderate (5-10)',
                                              'High (10-100)', 'Severe (>100)'])

    return vif_data

# Calculate initial VIF
vif_results = calculate_vif(df)

print("\nVIF Summary:")
print(f"   Features with VIF > 10: {len(vif_results[vif_results.VIF > 10])}")
print(f"   Features with VIF > 100: {len(vif_results[vif_results.VIF > 100])}")

print("\nTop 20 features by VIF:")
print(vif_results.head(20)[['Feature', 'VIF', 'VIF_Category']])

# Save VIF results
vif_results.to_csv(os.path.join(current_phase_dir, 'reports', 'vif_analysis_initial.csv'), index=False)

def visualize_vif(vif_data, filename='vif_analysis.png'):
    """Create VIF visualization"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Top 30 VIF values
    top_vif = vif_data.head(30)
    colors = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in top_vif['VIF']]

    ax1.barh(range(len(top_vif)), top_vif['VIF'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_vif)))
    ax1.set_yticklabels(top_vif['Feature'], fontsize=9)
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Critical threshold (VIF=10)')
    ax1.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='Warning threshold (VIF=5)')
    ax1.set_xlabel('VIF Value', fontsize=12)
    ax1.set_title('Top 30 Features by VIF', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis() # Show highest VIF at the top

    # Plot 2: VIF distribution
    vif_categories = vif_data['VIF_Category'].value_counts().reindex(['Low (<5)', 'Moderate (5-10)', 'High (10-100)', 'Severe (>100)']).dropna()
    colors_cat = ['green', 'orange', 'red', 'darkred']
    ax2.pie(vif_categories.values, labels=vif_categories.index, autopct='%1.1f%%',
            colors=colors_cat[:len(vif_categories)], startangle=90)
    ax2.set_title('Distribution of VIF Categories', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(current_phase_dir, 'figures', filename),
                dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    plt.close(fig) # Close figure

    print("VIF visualization saved")

# Visualize initial VIF
if not vif_results.empty:
    visualize_vif(vif_results, 'vif_analysis_initial.png')

def resolve_multicollinearity(df, vif_data, target_col='Triaj', vif_threshold=10):
    """Resolve multicollinearity by removing high VIF features"""
    print(f"\nResolving multicollinearity (VIF threshold: {vif_threshold})...")

    # Clinical importance - features to preserve
    clinical_critical = ['SPO2', 'Nabız', 'KB-S', 'KB-D', 'metin_ates',
                        'YAŞ', 'Ağrı skoru', 'metin_gks']

    removed_features = []
    df_reduced = df.copy()

    # Iteratively remove high VIF features
    iteration = 0
    max_iterations = 50 # Safety break
    
    current_vif_data = vif_data.copy()

    while iteration < max_iterations:
        high_vif = current_vif_data[current_vif_data['VIF'] > vif_threshold].sort_values('VIF', ascending=False)

        if len(high_vif) == 0:
            print(f"\nAll VIF values below threshold after {iteration} iterations")
            break

        # Get feature with highest VIF
        feature_to_remove = high_vif['Feature'].iloc[0]

        # Check if it's clinically critical
        if feature_to_remove in clinical_critical:
            print(f"   Skipping clinically critical feature: {feature_to_remove} (VIF: {high_vif['VIF'].iloc[0]:.2f})")
            # Remove it from consideration in next iteration
            current_vif_data = current_vif_data[current_vif_data['Feature'] != feature_to_remove]
            if len(current_vif_data) == 0:
                print("\n   All remaining high VIF features are clinically critical. Stopping.")
                break
            continue # Try next highest

        # Remove feature
        df_reduced = df_reduced.drop(columns=[feature_to_remove])
        removed_features.append({
            'feature': feature_to_remove,
            'vif': float(high_vif[high_vif['Feature'] == feature_to_remove]['VIF'].iloc[0]),
            'iteration': iteration
        })
        print(f"   Iter {iteration}: Removed '{feature_to_remove}' (VIF: {high_vif['VIF'].iloc[0]:.2f})")

        iteration += 1
        
        # Recalculate VIF for remaining features
        current_vif_data = calculate_vif(df_reduced, target_col)
        
        if current_vif_data.empty:
            print("   No numeric features left to analyze.")
            break

    print(f"\nMulticollinearity resolution complete:")
    print(f"   Features removed: {len(removed_features)}")
    print(f"   Features remaining: {df_reduced.shape[1]}")

    # Save removed features log
    with open(os.path.join(current_phase_dir, 'reports', 'removed_features_multicollinearity.json'), 'w') as f:
        json.dump(convert_to_serializable(removed_features), f, indent=2, ensure_ascii=False)

    return df_reduced, removed_features

# Resolve multicollinearity
df_vif_reduced, removed_vif_features = resolve_multicollinearity(df, vif_results, vif_threshold=CONFIG['vif_threshold'])

# Recalculate VIF after reduction
vif_results_final = calculate_vif(df_vif_reduced)
print("\nFinal VIF Summary:")
if not vif_results_final.empty:
    print(f"   Features with VIF > 10: {len(vif_results_final[vif_results_final.VIF > 10])}")
    print(f"   Mean VIF: {vif_results_final['VIF'].mean():.2f}")
    print(f"   Median VIF: {vif_results_final['VIF'].median():.2f}")
    # Save final VIF
    vif_results_final.to_csv(os.path.join(current_phase_dir, 'reports', 'vif_analysis_final.csv'), index=False)
    visualize_vif(vif_results_final, 'vif_analysis_final.png')
else:
    print("   No features remaining for final VIF analysis.")

# ================================================================
# STEP 2: OUTLIER DETECTION AND TREATMENT
# ================================================================

print("\n" + "="*70)
print("STEP 2: OUTLIER DETECTION AND TREATMENT")
print("="*70)

def detect_outliers_comprehensive(df, target_col='Triaj'):
    """Comprehensive outlier detection"""
    print("\nDetecting outliers using multiple methods...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    outlier_summary = []

    for col in numeric_cols:
        if df[col].var() > 1e-6: # Only check columns with variance
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            # Z-score method
            z_scores = np.abs(zscore(df[col].fillna(df[col].median())))
            zscore_outliers = (z_scores > 3).sum()

            outlier_summary.append({
                'Feature': col,
                'IQR_Outliers': int(iqr_outliers),
                'IQR_Percentage': float((iqr_outliers / len(df)) * 100),
                'Zscore_Outliers': int(zscore_outliers),
                'Zscore_Percentage': float((zscore_outliers / len(df)) * 100),
                'Lower_Bound': float(lower_bound),
                'Upper_Bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3)
            })

    outlier_df = pd.DataFrame(outlier_summary)
    if not outlier_df.empty:
        outlier_df = outlier_df.sort_values('IQR_Percentage', ascending=False)

    return outlier_df

# Detect outliers
outlier_analysis = detect_outliers_comprehensive(df_vif_reduced)

if not outlier_analysis.empty:
    print("\nOutlier Summary:")
    print(f"   Features with >10% outliers (IQR): {len(outlier_analysis[outlier_analysis.IQR_Percentage > 10])}")
    print(f"   Features with >20% outliers (IQR): {len(outlier_analysis[outlier_analysis.IQR_Percentage > 20])}")
    print("\nTop 10 features by outlier percentage:")
    print(outlier_analysis.head(10)[['Feature', 'IQR_Percentage', 'IQR_Outliers']])
    # Save outlier analysis
    outlier_analysis.to_csv(os.path.join(current_phase_dir, 'reports', 'outlier_analysis.csv'), index=False)
else:
    print("\nNo outliers detected or no numeric features to analyze.")

def visualize_outliers(df, outlier_df, top_n=10):
    """Visualize outliers for top features"""
    
    if outlier_df.empty:
        print("No outlier data to visualize.")
        return

    top_outlier_features = outlier_df.head(top_n)['Feature'].tolist()

    if not top_outlier_features:
        print("No outlier features to visualize.")
        return

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, col in enumerate(top_outlier_features):
        if idx < len(axes):
            ax = axes[idx]
            try:
                # Box plot
                df[col].plot(kind='box', ax=ax, vert=False)
                ax.set_title(f'{col}\n({outlier_df[outlier_df.Feature==col]["IQR_Percentage"].iloc[0]:.1f}% outliers)')
                ax.set_xlabel('Value')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Could not plot outlier box for {col}: {e}")
                ax.text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', transform=ax.transAxes, color='red')
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(current_phase_dir, 'figures', 'outlier_boxplots.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    plt.close(fig) # Close figure

    print("Outlier visualization saved")

# Visualize outliers
if not outlier_analysis.empty:
    visualize_outliers(df_vif_reduced, outlier_analysis, top_n=10)

def treat_outliers_clinical(df, outlier_df, method='winsorize', percentile=0.99):
    """Treat outliers using clinical-appropriate methods"""
    print(f"\nTreating outliers using {method} method...")

    df_treated = df.copy()
    treatment_log = []
    
    if outlier_df.empty:
        print("   No outlier analysis data. No outliers treated.")
        return df_treated, treatment_log

    # Features to treat (>10% outliers)
    features_to_treat = outlier_df[outlier_df.IQR_Percentage > 10]['Feature'].tolist()

    for col in features_to_treat:
        if col in df_treated.columns:
            original_outliers = outlier_df[outlier_df.Feature == col]['IQR_Outliers'].iloc[0]

            if method == 'winsorize':
                # Winsorize to 1st and 99th percentile
                lower = df_treated[col].quantile(1 - percentile)
                upper = df_treated[col].quantile(percentile)

                df_treated[col] = df_treated[col].clip(lower=lower, upper=upper)

            elif method == 'log_transform':
                # Log transformation for highly skewed data
                if (df_treated[col] > 0).all():
                    df_treated[col] = np.log1p(df_treated[col])

            treatment_log.append({
                'feature': col,
                'method': method,
                'original_outliers': int(original_outliers),
                'outlier_percentage': float(outlier_df[outlier_df.Feature == col]['IQR_Percentage'].iloc[0])
            })

    print(f"   Treated {len(features_to_treat)} features")

    # Save treatment log
    with open(os.path.join(current_phase_dir, 'reports', 'outlier_treatment_log.json'), 'w') as f:
        json.dump(convert_to_serializable(treatment_log), f, indent=2, ensure_ascii=False)

    return df_treated, treatment_log

# Treat outliers
df_outlier_treated, outlier_treatment_log = treat_outliers_clinical(
    df_vif_reduced,
    outlier_analysis,
    method=CONFIG['outlier_method'],
    percentile=0.99
)

# ================================================================
# STEP 3: FEATURE SCALING AND TRANSFORMATION
# ================================================================

print("\n" + "="*70)
print("STEP 3: FEATURE SCALING AND TRANSFORMATION")
print("="*70)

def scale_features(df, target_col='Triaj', method='standard'):
    """Scale numeric features"""
    print(f"\nScaling features using {method} scaling...")

    df_scaled = df.copy()

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target and binary indicators
    cols_to_scale = [col for col in numeric_cols
                    if col != target_col
                    and not col.endswith('_was_missing')
                    and df[col].nunique() > 2] # Only scale non-binary features

    print(f"   Scaling {len(cols_to_scale)} features")
    
    if not cols_to_scale:
        print("   No features to scale.")
        return df_scaled, None, []

    # Choose scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Fit and transform
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Save scaler
    joblib.dump(scaler, os.path.join(current_phase_dir, 'models', f'{method}_scaler.pkl'))

    # Save scaling info
    scaling_info = {
        'method': method,
        'features_scaled': cols_to_scale,
        'n_features': len(cols_to_scale),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(current_phase_dir, 'reports', 'scaling_info.json'), 'w') as f:
        json.dump(convert_to_serializable(scaling_info), f, indent=2, ensure_ascii=False)

    print(f"   Scaling complete and scaler saved")

    return df_scaled, scaler, cols_to_scale

# Scale features using RobustScaler (better for outliers)
df_scaled, scaler, scaled_features = scale_features(df_outlier_treated, method='robust')

# ================================================================
# STEP 4: CLINICAL DOMAIN FEATURE ENGINEERING
# ================================================================

print("\n" + "="*70)
print("STEP 4: CLINICAL DOMAIN FEATURE ENGINEERING")
print("="*70)

def create_clinical_features(df):
    """Create clinically relevant derived features"""
    print("\nCreating clinical domain features...")

    df_clinical = df.copy()
    new_features = []

    # 1. Modified Early Warning Score (MEWS) components
    if all(col in df.columns for col in ['Nabız', 'KB-S', 'SPO2', 'metin_ates', 'metin_gks']):
        print("   Creating MEWS-based features...")

        # Heart rate score (simplified bins with unique labels)
        conditions_hr = [
            (df_clinical['Nabız'] <= 40),
            (df_clinical['Nabız'] > 40) & (df_clinical['Nabız'] <= 50),
            (df_clinical['Nabız'] > 50) & (df_clinical['Nabız'] <= 100),
            (df_clinical['Nabız'] > 100) & (df_clinical['Nabız'] <= 110),
            (df_clinical['Nabız'] > 110) & (df_clinical['Nabız'] <= 130),
            (df_clinical['Nabız'] > 130)
        ]
        choices_hr = [3, 2, 0, 1, 2, 3]
        df_clinical['HR_Risk_Score'] = np.select(conditions_hr, choices_hr, default=0)
        new_features.append('HR_Risk_Score')

        # Systolic BP score
        conditions_sbp = [
            (df_clinical['KB-S'] <= 70),
            (df_clinical['KB-S'] > 70) & (df_clinical['KB-S'] <= 80),
            (df_clinical['KB-S'] > 80) & (df_clinical['KB-S'] <= 100),
            (df_clinical['KB-S'] > 100) & (df_clinical['KB-S'] <= 199),
            (df_clinical['KB-S'] > 199)
        ]
        choices_sbp = [3, 2, 1, 0, 2]
        df_clinical['SBP_Risk_Score'] = np.select(conditions_sbp, choices_sbp, default=0)
        new_features.append('SBP_Risk_Score')

        # Oxygen saturation score
        conditions_spo2 = [
            (df_clinical['SPO2'] < 85),
            (df_clinical['SPO2'] >= 85) & (df_clinical['SPO2'] < 90),
            (df_clinical['SPO2'] >= 90) & (df_clinical['SPO2'] < 92),
            (df_clinical['SPO2'] >= 92)
        ]
        choices_spo2 = [3, 2, 1, 0]
        df_clinical['SPO2_Risk_Score'] = np.select(conditions_spo2, choices_spo2, default=0)
        new_features.append('SPO2_Risk_Score')

        # Temperature score
        conditions_temp = [
            (df_clinical['metin_ates'] < 35),
            (df_clinical['metin_ates'] >= 35) & (df_clinical['metin_ates'] < 36),
            (df_clinical['metin_ates'] >= 36) & (df_clinical['metin_ates'] <= 38),
            (df_clinical['metin_ates'] > 38) & (df_clinical['metin_ates'] <= 39),
            (df_clinical['metin_ates'] > 39)
        ]
        choices_temp = [2, 1, 0, 1, 2]
        df_clinical['Temp_Risk_Score'] = np.select(conditions_temp, choices_temp, default=0)
        new_features.append('Temp_Risk_Score')

        # Combined MEWS-like score
        df_clinical['Clinical_Risk_Score'] = (
            df_clinical['HR_Risk_Score'] +
            df_clinical['SBP_Risk_Score'] +
            df_clinical['SPO2_Risk_Score'] +
            df_clinical['Temp_Risk_Score']
        )
        new_features.append('Clinical_Risk_Score')

    # 2. Shock Index (HR/SBP)
    if 'Nabız' in df.columns and 'KB-S' in df.columns:
        # Avoid division by zero and infinity
        df_clinical['Shock_Index'] = df_clinical['Nabız'] / df_clinical['KB-S'].replace(0, np.nan)
        # Clip extreme values
        df_clinical['Shock_Index'] = df_clinical['Shock_Index'].clip(upper=5.0)
        # Fill any remaining NaN with median
        df_clinical['Shock_Index'] = df_clinical['Shock_Index'].fillna(df_clinical['Shock_Index'].median())
        df_clinical['Shock_Index_High'] = (df_clinical['Shock_Index'] > 0.9).astype(int)
        new_features.extend(['Shock_Index', 'Shock_Index_High'])
        print("   Created Shock Index features")

    # 3. Mean Arterial Pressure (MAP)
    if all(col in df.columns for col in ['KB-S', 'KB-D']):
        df_clinical['MAP'] = df_clinical['KB-D'] + (df_clinical['KB-S'] - df_clinical['KB-D']) / 3
        # Clip to physiologically reasonable range
        df_clinical['MAP'] = df_clinical['MAP'].clip(lower=30, upper=200)
        df_clinical['MAP_Low'] = (df_clinical['MAP'] < 65).astype(int)
        new_features.extend(['MAP', 'MAP_Low'])
        print("   Created MAP features")

    # 4. Age risk categories
    if 'YAŞ' in df.columns:
        df_clinical['Age_Category'] = pd.cut(df_clinical['YAŞ'],
                                            bins=[0, 18, 40, 65, 100],
                                            labels=['Young', 'Adult', 'Middle', 'Elderly'])
        df_clinical['Elderly_Flag'] = (df_clinical['YAŞ'] >= 65).astype(int)
        new_features.extend(['Elderly_Flag'])
        print("   Created age-based features")

    # 5. Vital signs abnormality count
    abnormal_vitals = []
    if 'Nabız' in df.columns:
        abnormal_vitals.append((df_clinical['Nabız'] < 60) | (df_clinical['Nabız'] > 100))
    if 'KB-S' in df.columns:
        abnormal_vitals.append((df_clinical['KB-S'] < 90) | (df_clinical['KB-S'] > 140))
    if 'SPO2' in df.columns:
        abnormal_vitals.append(df_clinical['SPO2'] < 95)
    if 'metin_ates' in df.columns:
        abnormal_vitals.append((df_clinical['metin_ates'] < 36) | (df_clinical['metin_ates'] > 38))

    if abnormal_vitals:
        df_clinical['Abnormal_Vitals_Count'] = sum(abnormal_vitals).astype(int)
        new_features.append('Abnormal_Vitals_Count')
        print("   Created abnormal vitals count")

    # 6. Pain-based features
    if 'Ağrı skoru' in df.columns:
        df_clinical['Severe_Pain'] = (df_clinical['Ağrı skoru'] >= 7).astype(int)
        df_clinical['No_Pain'] = (df_clinical['Ağrı skoru'] == 0).astype(int)
        new_features.extend(['Severe_Pain', 'No_Pain'])
        print("   Created pain-based features")

    print(f"\n   Total new clinical features created: {len(new_features)}")

    # Save feature engineering log
    feature_log = {
        'new_features': new_features,
        'n_features': len(new_features),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(current_phase_dir, 'reports', 'clinical_features_log.json'), 'w') as f:
        json.dump(convert_to_serializable(feature_log), f, indent=2, ensure_ascii=False)

    return df_clinical, new_features

# Create clinical features
df_with_clinical, clinical_features = create_clinical_features(df_scaled)

# ================================================================
# STEP 5: FEATURE IMPORTANCE AND SELECTION
# ================================================================

print("\n" + "="*70)
print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

def analyze_feature_importance(df, target_col='Triaj'):
    """Analyze feature importance using statistical tests"""
    print("\nAnalyzing feature importance...")

    # Encode target if categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols].copy()
    
    if X.empty:
        print("   No numeric features to analyze for importance.")
        return pd.DataFrame(columns=['Feature', 'F_Score', 'F_PValue', 'MI_Score', 'F_Score_Norm', 'MI_Score_Norm', 'Combined_Score'])

    # Critical: Clean data before analysis
    print("   Cleaning data for analysis...")

    # Replace infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Final check - if still NaN, fill with 0
    X = X.fillna(0)

    # Verify no infinity or NaN
    if np.any(np.isinf(X.values)):
        print("   Warning: Infinity values detected, replacing with large finite values...")
        X = X.replace([np.inf, -np.inf], [1e10, -1e10])

    print(f"   Data cleaned: {X.shape[0]} rows, {X.shape[1]} features")

    # ANOVA F-statistic
    print("   Computing ANOVA F-statistics...")
    selector_f = SelectKBest(f_classif, k='all')
    selector_f.fit(X, y)

    f_scores = pd.DataFrame({
        'Feature': numeric_cols,
        'F_Score': selector_f.scores_,
        'F_PValue': selector_f.pvalues_
    })

    # Mutual Information
    print("   Computing Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=CONFIG['random_state'])

    importance_df = f_scores.copy()
    importance_df['MI_Score'] = mi_scores
    importance_df = importance_df.sort_values('F_Score', ascending=False)

    # Normalize scores for comparison
    importance_df['F_Score_Norm'] = (importance_df['F_Score'] - importance_df['F_Score'].min()) / (importance_df['F_Score'].max() - importance_df['F_Score'].min())
    importance_df['MI_Score_Norm'] = (importance_df['MI_Score'] - importance_df['MI_Score'].min()) / (importance_df['MI_Score'].max() - importance_df['MI_Score'].min())
    importance_df['Combined_Score'] = (importance_df['F_Score_Norm'] + importance_df['MI_Score_Norm']) / 2

    importance_df = importance_df.sort_values('Combined_Score', ascending=False)

    print(f"\n   Top 20 most important features:")
    print(importance_df.head(20)[['Feature', 'F_Score', 'MI_Score', 'Combined_Score']])

    # Save importance analysis
    importance_df.to_csv(os.path.join(current_phase_dir, 'reports', 'feature_importance.csv'), index=False)

    return importance_df

# Analyze feature importance
feature_importance = analyze_feature_importance(df_with_clinical)

def visualize_feature_importance(importance_df, top_n=30):
    """Visualize feature importance"""
    
    if importance_df.empty:
        print("No feature importance data to visualize.")
        return

    top_features = importance_df.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # F-Score
    ax1.barh(range(len(top_features)), top_features['F_Score'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['Feature'], fontsize=9)
    ax1.set_xlabel('F-Score', fontsize=12)
    ax1.set_title(f'Top {top_n} Features by ANOVA F-Score', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Combined Score
    ax2.barh(range(len(top_features)), top_features['Combined_Score'], color='coral', alpha=0.7)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['Feature'], fontsize=9)
    ax2.set_xlabel('Combined Score (Normalized)', fontsize=12)
    ax2.set_title(f'Top {top_n} Features by Combined Score', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(current_phase_dir, 'figures', 'feature_importance.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    plt.close(fig) # Close figure

    print("Feature importance visualization saved")

# Visualize feature importance
visualize_feature_importance(feature_importance, top_n=30)

# ================================================================
# STEP 6: SAVE FINAL ENGINEERED DATASET
# ================================================================

print("\n" + "="*70)
print("STEP 6: SAVING FINAL ENGINEERED DATASET")
print("="*70)

# Save final dataset
print("\nSaving engineered dataset...")

output_path = os.path.join(current_phase_dir, 'data', 'data_after_feature_engineering.csv')
df_with_clinical.to_csv(output_path, index=False)
print(f"CSV saved: {output_path}")

excel_path = os.path.join(current_phase_dir, 'data', 'data_after_feature_engineering.xlsx')
try:
    df_with_clinical.to_excel(excel_path, index=False)
    print(f"Excel saved: {excel_path}")
except Exception as e:
    print(f"Warning: Could not save Excel version: {e}")

# Create comprehensive summary
phase2_summary = {
    'timestamp': datetime.now().isoformat(),
    'input_shape': list(df.shape),
    'output_shape': list(df_with_clinical.shape),
    'multicollinearity': {
        'features_removed': len(removed_vif_features),
        'initial_high_vif_count': int(len(vif_results[vif_results.VIF > 10])) if not vif_results.empty else 0,
        'final_high_vif_count': int(len(vif_results_final[vif_results_final.VIF > 10])) if not vif_results_final.empty else 0,
        'mean_vif_reduction': float(vif_results['VIF'].mean() - vif_results_final['VIF'].mean()) if not vif_results.empty and not vif_results_final.empty else 0
    },
    'outlier_treatment': {
        'features_treated': len(outlier_treatment_log),
        'method': CONFIG['outlier_method']
    },
    'feature_scaling': {
        'method': 'robust',
        'features_scaled': len(scaled_features)
    },
    'clinical_features': {
        'new_features_created': len(clinical_features),
        'feature_list': clinical_features
    },
    'feature_importance': {
        'top_10_features': feature_importance.head(10)['Feature'].tolist() if not feature_importance.empty else []
    },
    'ready_for_next_phase': True,
    'next_phase': 'Phase3_Class_Imbalance'
}

with open(os.path.join(current_phase_dir, 'reports', 'phase2_summary.json'), 'w') as f:
    json.dump(convert_to_serializable(phase2_summary), f, indent=2, ensure_ascii=False)

print("Summary report saved")

# ================================================================
# PHASE 2 COMPLETION REPORT
# ================================================================

print("\n" + "="*70)
print("PHASE 2 COMPLETED SUCCESSFULLY")
print("="*70)

print(f"\nFEATURE ENGINEERING SUMMARY:")
print(f"   Input dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Output dataset: {df_with_clinical.shape[0]:,} rows x {df_with_clinical.shape[1]} columns")
print(f"   Features removed (multicollinearity): {len(removed_vif_features)}")
print(f"   Features treated (outliers): {len(outlier_treatment_log)}")
print(f"   Features scaled: {len(scaled_features)}")
print(f"   New clinical features created: {len(clinical_features)}")
print(f"   Net feature change: {df_with_clinical.shape[1] - df.shape[1]:+d}")

print(f"\nREADY FOR PHASE 3: Class Imbalance Handling")
print("="*70)