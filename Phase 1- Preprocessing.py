# ================================================================
# EMERGENCY DEPARTMENT TRIAGE PREDICTION - PHASE 1
# Data Preprocessing and Missing Data Imputation
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML libraries
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency
import missingno as msno

# Advanced imputation
try:
    from fancyimpute import MICE
    MICE_AVAILABLE = True
except ImportError:
    MICE_AVAILABLE = False
    print("Warning: fancyimpute not available, using sklearn IterativeImputer instead")

from datetime import datetime
import json
import joblib

# ================================================================
# CONFIGURATION AND SETUP
# ================================================================

# Set plotting style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
CONFIG = {
    'random_state': 42,
    'n_jobs': -1,
    'figsize': (12, 8),
    'dpi': 300,
    'base_path': '.',  # [GitHub Ready] Changed to relative path
    'data_file_name': 'raw_data_12335.xlsx' # [GitHub Ready] Generic filename for README
}

# Create phase directories
def setup_directories(base_path):
    """Create directory structure for all phases"""
    phases = [
        'Phase1_Data_Preprocessing',
        'Phase2_Feature_Engineering',
        'Phase3_Class_Imbalance',
        'Phase4_Model_Development',
        'Phase5_Explainability_Validation',
        'Phase6_Publication_Analyses'
    ]
    
    phase_dirs = {}

    for phase in phases:
        phase_dir = os.path.join(base_path, phase)
        os.makedirs(phase_dir, exist_ok=True)
        phase_dirs[phase] = phase_dir
        
        # Create subdirectories
        subdirs = ['figures', 'data', 'reports', 'models']
        for subdir in subdirs:
            os.makedirs(os.path.join(phase_dir, subdir), exist_ok=True)

    print("Directory structure created successfully")
    return phase_dirs

# Setup directories
phase_dirs = setup_directories(CONFIG['base_path'])
current_phase_dir = phase_dirs['Phase1_Data_Preprocessing']

print(f"\nPHASE 1: DATA PREPROCESSING AND IMPUTATION")
print(f"Working directory: {current_phase_dir}")
print("="*70)

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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
# DATA LOADING AND INITIAL ASSESSMENT
# ================================================================

def load_data(data_path):
    """Load data from a specified path."""
    print("Loading dataset...")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(f"Please place your data file (e.g., {CONFIG['data_file_name']}) in the {os.path.dirname(data_path)} directory.")
        return None, None

    try:
        # Check file extension and load accordingly
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path, engine='openpyxl')
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            print(f"Error: Unsupported file format. Please use .xlsx or .csv")
            return None, None
            
        print(f"Data loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns")

        data_info = {
            'file_path': data_path,
            'original_shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
            'load_timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(current_phase_dir, 'reports', 'data_info.json'), 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)

        return df, data_info

    except Exception as e:
        print(f"Error loading data from {data_path}: {str(e)}")
        return None, None

# [GitHub Ready] Load data from 'data' subfolder of Phase 1
data_load_path = os.path.join(current_phase_dir, 'data', CONFIG['data_file_name'])
df, data_info = load_data(data_load_path)

if df is None:
    raise SystemExit("Data loading failed. Cannot proceed with analysis.")

# ================================================================
# DATA TYPE CONVERSION
# ================================================================
print("\nConverting potentially numeric columns to numeric type...")

# Identify columns that should be numeric
numeric_candidate_cols = ['YAŞ', 'Ağrı skoru', 'SPO2', 'Nabız', 'KB-S', 'KB-D', 'metin_ates', 'Vital Bulgu Eksik Sayısı', 'metin_gks', 'Yogunluk_Skoru']

for col in numeric_candidate_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"   Converted '{col}' to numeric (errors coerced to NaN)")
    else:
        print(f"   Column '{col}' not found in DataFrame, skipping conversion.")

print("Data type conversion complete.")


# ================================================================
# MISSING DATA ANALYSIS
# ================================================================

def comprehensive_missing_analysis(df):
    """Comprehensive missing data analysis following TRIPOD guidelines"""
    print("\nCOMPREHENSIVE MISSING DATA ANALYSIS")
    print("="*50)

    # Calculate missing statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes,
        'Unique_Values': [df[col].nunique() for col in df.columns]
    })

    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
    missing_stats['Critical_Missing'] = missing_stats['Missing_Percentage'] > 50

    print(f"Missing Data Summary:")
    print(f"   Total cells: {df.shape[0] * df.shape[1]:,}")
    print(f"   Missing cells: {df.isnull().sum().sum():,}")
    print(f"   Overall missing rate: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    print(f"   Columns with missing data: {len(missing_stats[missing_stats.Missing_Count > 0])}")
    print(f"   Critical missing (>50%): {len(missing_stats[missing_stats.Critical_Missing])}")

    # Save missing statistics
    missing_stats.to_csv(
        os.path.join(current_phase_dir, 'reports', 'missing_data_analysis.csv'),
        index=False
    )

    return missing_stats

def visualize_missing_patterns(df):
    """Create publication-quality missing data visualizations"""
    print("\nCreating missing data visualizations...")

    # 1. Missing data matrix
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Matrix plot
    msno.matrix(df, ax=axes[0,0], color=(0.25, 0.25, 0.75))
    axes[0,0].set_title('Missing Data Pattern Matrix', fontsize=14, fontweight='bold')

    # Bar plot
    msno.bar(df, ax=axes[0,1], color='steelblue')
    axes[0,1].set_title('Missing Data Count by Variable', fontsize=14, fontweight='bold')

    # Heatmap of missing correlations
    msno.heatmap(df, ax=axes[1,0])
    axes[1,0].set_title('Missing Data Correlation Heatmap', fontsize=14, fontweight='bold')

    # Dendrogram
    msno.dendrogram(df, ax=axes[1,1])
    axes[1,1].set_title('Missing Data Clustering Dendrogram', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(
        os.path.join(current_phase_dir, 'figures', 'missing_data_comprehensive.png'),
        dpi=CONFIG['dpi'], bbox_inches='tight'
    )
    plt.show()
    plt.close(fig) # Close figure to free memory

    # 2. Interactive missing data visualization
    missing_counts = df.isnull().sum().sort_values(ascending=True)
    missing_counts = missing_counts[missing_counts > 0]

    if len(missing_counts) > 0:
        fig_interactive = go.Figure(data=[
            go.Bar(
                x=missing_counts.values,
                y=missing_counts.index,
                orientation='h',
                marker_color='indianred',
                text=[f'{count:,} ({count/len(df)*100:.1f}%)' for count in missing_counts.values],
                textposition='outside'
            )
        ])

        fig_interactive.update_layout(
            title='Missing Data Count by Variable (Interactive)',
            xaxis_title='Number of Missing Values',
            yaxis_title='Variables',
            height=800,
            width=1000,
            font=dict(size=12),
            template='plotly_white'
        )

        fig_interactive.write_html(
            os.path.join(current_phase_dir, 'figures', 'missing_data_interactive.html')
        )
        # fig_interactive.show() # [GitHub Ready] Disable auto-show

# Perform missing data analysis
missing_stats = comprehensive_missing_analysis(df)
visualize_missing_patterns(df)

# ================================================================
# MISSING DATA IMPUTATION STRATEGY
# ================================================================

class AdvancedImputer:
    """
    Advanced imputation strategy following medical ML best practices
    Implements MICE + Random Forest hybrid approach
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputers = {}
        self.encoders = {}
        self.imputation_log = []

    def fit_transform(self, df, target_col='Triaj'):
        """Fit and transform with comprehensive imputation strategy"""
        print("\nADVANCED IMPUTATION STRATEGY")
        print("="*45)

        df_imputed = df.copy()

        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Remove target from predictors
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        print(f"Column Analysis:")
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
        print(f"   DateTime columns: {len(datetime_cols)}")

        # 1. Handle critical vital signs (MICE approach)
        vital_signs = ['Nabız', 'SPO2', 'KB-S', 'KB-D', 'metin_ates', 'metin_gks']
        # Ensure vital_signs_available only includes columns present AND numeric
        vital_signs_available = [col for col in vital_signs if col in numeric_cols]

        if vital_signs_available:
            print(f"\nImputing vital signs using MICE approach...")
            df_imputed = self._impute_vital_signs(df_imputed, vital_signs_available)

        # 2. Handle other numeric variables
        other_numeric = [col for col in numeric_cols if col not in vital_signs_available]
        if other_numeric:
            print(f"\nImputing other numeric variables...")
            df_imputed = self._impute_numeric(df_imputed, other_numeric)

        # 3. Handle categorical variables
        if categorical_cols:
            print(f"\nImputing categorical variables...")
            df_imputed = self._impute_categorical(df_imputed, categorical_cols)

        # 4. Create missingness indicators
        print(f"\nCreating missingness indicators...")
        df_imputed = self._create_missingness_indicators(df, df_imputed)

        return df_imputed

    def _impute_vital_signs(self, df, vital_cols):
        """Impute vital signs using MICE (Multiple Imputation by Chained Equations)"""

        # Ensure vital_data is numeric
        vital_data = df[vital_cols].select_dtypes(include=[np.number]).copy()

        if MICE_AVAILABLE:
            try:
                mice_imputer = MICE(random_state=self.random_state)
                vital_imputed = mice_imputer.fit_transform(vital_data)
                vital_imputed_df = pd.DataFrame(vital_imputed, columns=vital_data.columns, index=df.index)

                for col in vital_data.columns:
                    df[col] = vital_imputed_df[col]

                self.imputation_log.append({
                    'method': 'MICE',
                    'columns': vital_data.columns.tolist(),
                    'timestamp': datetime.now().isoformat()
                })

                print(f"   MICE imputation completed for {len(vital_data.columns)} vital signs")

            except Exception as e:
                print(f"   MICE failed, falling back to IterativeImputer: {str(e)}")
                return self._fallback_iterative_imputer(df, vital_data.columns.tolist())
        else:
            return self._fallback_iterative_imputer(df, vital_data.columns.tolist())

        return df

    def _fallback_iterative_imputer(self, df, cols):
        """Fallback iterative imputation"""
        iterative_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=self.random_state),
            random_state=self.random_state,
            max_iter=10
        )
        # Ensure only numeric columns are passed
        numeric_subset = df[cols].select_dtypes(include=np.number)

        if not numeric_subset.empty:
             imputed_data = iterative_imputer.fit_transform(numeric_subset)
             imputed_df = pd.DataFrame(imputed_data, columns=numeric_subset.columns, index=df.index)

             for col in numeric_subset.columns:
                 df[col] = imputed_df[col]

             self.imputers['iterative_vital'] = iterative_imputer
             self.imputation_log.append({
                 'method': 'IterativeImputer',
                 'columns': numeric_subset.columns.tolist(),
                 'timestamp': datetime.now().isoformat()
             })
             print(f"   Iterative imputation completed for {len(numeric_subset.columns)} columns")
        else:
             print("   No numeric columns provided for IterativeImputer fallback.")

        return df

    def _impute_numeric(self, df, numeric_cols):
        """Impute other numeric variables using median or KNN"""

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                missing_pct = df[col].isnull().sum() / len(df)

                if missing_pct < 0.1:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    method_used = f'median({median_val:.2f})'

                else:
                    corr_features = self._select_correlated_features(df, col, numeric_cols)

                    if len(corr_features) >= 2:
                        knn_imputer = KNNImputer(n_neighbors=5)
                        # Ensure only numeric columns are passed to KNNImputer
                        numeric_subset = df[corr_features].select_dtypes(include=np.number)
                        df[numeric_subset.columns] = knn_imputer.fit_transform(numeric_subset)
                        self.imputers[f'knn_{col}'] = knn_imputer
                        method_used = 'KNN'
                    else:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        method_used = f'median({median_val:.2f})'


                self.imputation_log.append({
                    'method': method_used,
                    'column': col,
                    'missing_before': int(df[col].isnull().sum()),
                    'timestamp': datetime.now().isoformat()
                })

        print(f"   Numeric imputation completed for {len(numeric_cols)} columns")
        return df

    def _impute_categorical(self, df, categorical_cols):
        """Impute categorical variables using mode or Random Forest"""

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                missing_pct = df[col].isnull().sum() / len(df)

                if missing_pct < 0.05:
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    method_used = f'mode({mode_val})'

                else:
                    le = LabelEncoder()
                    non_null_mask = df[col].notnull()

                    if non_null_mask.sum() > 10:
                        # Select numeric features for prediction, ensuring they are numeric
                        feature_cols_numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                                              if df[c].notnull().sum() > len(df) * 0.8]
                        # Add categorical features with low missingness
                        feature_cols_categorical = [c for c in df.select_dtypes(include=['object']).columns
                                                  if c != col and df[c].isnull().sum() / len(df) < 0.1]

                        # Combine and prepare features
                        feature_cols = feature_cols_numeric + feature_cols_categorical
                        X_train_features = df.loc[non_null_mask, feature_cols]

                        # One-hot encode categorical features for RF
                        X_train_features = pd.get_dummies(X_train_features, columns=feature_cols_categorical, dummy_na=False)

                        if not X_train_features.empty and len(X_train_features.columns) >= 3:
                            y_train = df.loc[non_null_mask, col]

                            try:
                                le.fit(y_train)
                                y_train_encoded = le.transform(y_train)

                                rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                                rf.fit(X_train_features, y_train_encoded)

                                null_mask = df[col].isnull()
                                if null_mask.sum() > 0:
                                    X_pred_features = df.loc[null_mask, feature_cols]
                                    # Ensure consistent columns after one-hot encoding
                                    X_pred_features = pd.get_dummies(X_pred_features, columns=feature_cols_categorical, dummy_na=False)
                                    # Align columns - crucial for consistent feature sets
                                    X_pred_features = X_pred_features.reindex(columns=X_train_features.columns, fill_value=0)


                                    y_pred_encoded = rf.predict(X_pred_features)
                                    y_pred = le.inverse_transform(y_pred_encoded)
                                    df.loc[null_mask, col] = y_pred

                                self.imputers[f'rf_{col}'] = rf
                                self.encoders[f'le_{col}'] = le
                                method_used = 'RandomForest'
                            except Exception as e:
                                print(f"   Random Forest imputation failed for {col}: {str(e)}")
                                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                                df[col].fillna(mode_val, inplace=True)
                                method_used = f'mode({mode_val}) (fallback)'

                        else:
                            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                            df[col].fillna(mode_val, inplace=True)
                            method_used = f'mode({mode_val}) (fallback - insufficient features/data)'
                    else:
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                        method_used = f'mode({mode_val}) (fallback - insufficient non-null data)'

                self.imputation_log.append({
                    'method': method_used,
                    'column': col,
                    'missing_before': int(df[col].isnull().sum()),
                    'timestamp': datetime.now().isoformat()
                })

        print(f"   Categorical imputation completed for {len(categorical_cols)} columns")
        return df


    def _create_missingness_indicators(self, df_original, df_imputed):
        """Create binary indicators for originally missing values"""

        missing_cols = df_original.columns[df_original.isnull().any()].tolist()

        indicators_created = 0
        for col in missing_cols:
            missing_pct = df_original[col].isnull().sum() / len(df_original)

            # Create indicator only if missingness is substantial
            if missing_pct > 0.05:
                indicator_name = f'{col}_was_missing'
                # Avoid creating indicator if it already exists (e.g., from previous runs)
                if indicator_name not in df_imputed.columns:
                    df_imputed[indicator_name] = df_original[col].isnull().astype(int)
                    indicators_created += 1
                else:
                    print(f"   Indicator for {col} already exists, skipping.")

        print(f"   Created {indicators_created} new missingness indicators")
        # Update log to reflect only *new* indicators
        newly_created_indicators = [
            f'{col}_was_missing' for col in missing_cols
            if df_original[col].isnull().sum() / len(df_original) > 0.05
            and f'{col}_was_missing' in df_imputed.columns # Check if it was actually added
        ]

        self.imputation_log.append({
            'method': 'missingness_indicators',
            'indicators_created': indicators_created,
            'columns': newly_created_indicators,
            'timestamp': datetime.now().isoformat()
        })

        return df_imputed


    def _select_correlated_features(self, df, target_col, available_cols):
        """Select features correlated with target column for imputation"""

        # Ensure we only work with numeric columns for correlation
        numeric_df = df[available_cols].select_dtypes(include=[np.number])

        if target_col not in numeric_df.columns:
            # If target_col is not numeric, just return relevant numeric columns
            return numeric_df.columns.tolist()

        # Drop rows with NaN in the target column for correlation calculation
        numeric_df_clean = numeric_df.dropna(subset=[target_col])

        if len(numeric_df_clean) < 2:
             # Not enough data to calculate meaningful correlation
             return [target_col] if target_col in numeric_df.columns else []


        corr_matrix = numeric_df_clean.corr()

        if target_col in corr_matrix.columns:
            # Select top features based on absolute correlation, including target
            correlations = corr_matrix[target_col].abs().sort_values(ascending=False)
            # Select top N features (including the target itself)
            selected = correlations.head(6).index.tolist()
            return selected
        else:
            # Should not happen if target_col is in numeric_df.columns
            return numeric_df.columns.tolist()


    def get_imputation_summary(self):
        """Get summary of imputation process"""
        return {
            'log': self.imputation_log,
            'imputers_used': list(self.imputers.keys()),
            'encoders_used': list(self.encoders.keys())
        }

# ================================================================
# APPLY IMPUTATION STRATEGY
# ================================================================

print("\nAPPLYING ADVANCED IMPUTATION STRATEGY")
print("="*50)

# Initialize and apply imputer
advanced_imputer = AdvancedImputer(random_state=CONFIG['random_state'])
df_imputed = advanced_imputer.fit_transform(df)

# Verify imputation results
print("\nIMPUTATION RESULTS:")
print("="*25)

# Calculate missing before and after imputation
missing_before = df.isnull().sum().sum()
missing_after = df_imputed.isnull().sum().sum()

print(f"Missing values before: {missing_before:,}")
print(f"Missing values after: {missing_after:,}")

# Avoid division by zero if there were no missing values initially
if missing_before > 0:
    imputation_success_rate = ((missing_before - missing_after) / missing_before) * 100
    print(f"Imputation success rate: {imputation_success_rate:.2f}%")
else:
    imputation_success_rate = 100.0
    print("No missing values to impute.")


# Save imputation log with proper serialization
imputation_summary = advanced_imputer.get_imputation_summary()
imputation_summary_clean = convert_to_serializable(imputation_summary)

try:
    with open(os.path.join(current_phase_dir, 'reports', 'imputation_log.json'), 'w', encoding='utf-8') as f:
        json.dump(imputation_summary_clean, f, indent=2, ensure_ascii=False)
    print("Imputation log saved successfully")
except Exception as e:
    print(f"Warning: Could not save imputation log: {e}")

# ================================================================
# POST-IMPUTATION QUALITY ASSESSMENT
# ================================================================

def assess_imputation_quality(df_original, df_imputed):
    """Assess quality of imputation process"""
    print("\nIMPUTATION QUALITY ASSESSMENT")
    print("="*35)

    quality_metrics = {}

    for col in df_original.columns:
        # Only assess columns that originally had missing values
        if df_original[col].isnull().sum() > 0:

            original_stats = df_original[col].describe()
            imputed_stats = df_imputed[col].describe()

            col_metrics = {
                'missing_count_original': int(df_original[col].isnull().sum()),
                'missing_percentage_original': float(df_original[col].isnull().sum() / len(df_original) * 100),
                'missing_count_after': int(df_imputed[col].isnull().sum()),
                'missing_percentage_after': float(df_imputed[col].isnull().sum() / len(df_imputed) * 100),
            }

            if df_original[col].dtype in ['int64', 'float64']:
                col_metrics['type'] = 'numeric'
                # Handle potential NaN in describe stats for columns with all missing
                original_mean = original_stats.get('mean')
                imputed_mean = imputed_stats.get('mean')
                original_std = original_stats.get('std')
                imputed_std = imputed_stats.get('std')

                col_metrics['mean_original'] = float(original_mean) if not pd.isna(original_mean) else None
                col_metrics['mean_after'] = float(imputed_mean) if not pd.isna(imputed_mean) else None
                col_metrics['std_original'] = float(original_std) if not pd.isna(original_std) else None
                col_metrics['std_after'] = float(imputed_std) if not pd.isna(imputed_std) else None

                if not pd.isna(original_mean) and not pd.isna(imputed_mean):
                     col_metrics['mean_change'] = float(abs(original_mean - imputed_mean))
                else:
                     col_metrics['mean_change'] = None

                if not pd.isna(original_std) and not pd.isna(imputed_std):
                     col_metrics['std_change'] = float(abs(original_std - imputed_std))
                else:
                     col_metrics['std_change'] = None


                # Assess distribution preservation - stricter check
                if col_metrics['mean_change'] is not None and col_metrics['std_original'] is not None:
                    col_metrics['distribution_preserved'] = 'Yes' if col_metrics['mean_change'] < col_metrics['std_original'] * 0.05 else 'No' # Use 5% of std as threshold
                else:
                     col_metrics['distribution_preserved'] = 'N/A'


            else: # Assuming categorical for other types
                col_metrics['type'] = 'categorical'
                original_top = original_stats.get('top')
                imputed_top = imputed_stats.get('top')
                original_unique = original_stats.get('unique')
                imputed_unique = imputed_stats.get('unique')


                col_metrics['mode_original'] = original_top
                col_metrics['mode_after'] = imputed_top
                col_metrics['unique_values_original'] = int(original_unique) if not pd.isna(original_unique) else None
                col_metrics['unique_values_after'] = int(imputed_unique) if not pd.isna(imputed_unique) else None

                if original_top is not None and imputed_top is not None:
                    col_metrics['mode_preserved'] = 'Yes' if original_top == imputed_top else 'No'
                else:
                    col_metrics['mode_preserved'] = 'N/A'

            quality_metrics[col] = col_metrics


    # Save quality assessment with proper serialization
    quality_metrics_clean = convert_to_serializable(quality_metrics)
    try:
        with open(os.path.join(current_phase_dir, 'reports', 'imputation_quality.json'), 'w', encoding='utf-8') as f:
            json.dump(quality_metrics_clean, f, indent=2, ensure_ascii=False)
        print(f"Quality assessment saved for {len(quality_metrics)} variables that had missing data.")
    except Exception as e:
        print(f"Warning: Could not save imputation quality report: {e}")


    return quality_metrics

# Assess imputation quality
if df is not None and df_imputed is not None:
    quality_metrics = assess_imputation_quality(df, df_imputed)
else:
    print("\nCannot assess imputation quality as data loading or imputation failed.")


# ================================================================
# SAVE PROCESSED DATA
# ================================================================

def save_processed_data(df_imputed, df_original, missing_before, missing_after):
    """Save processed data with comprehensive documentation"""
    print("\nSAVING PROCESSED DATA")
    print("="*25)

    # Save main processed dataset
    output_path = os.path.join(current_phase_dir, 'data', 'data_after_imputation.csv')
    df_imputed.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Main dataset saved: {output_path}")

    # Save Excel version for easy viewing
    excel_path = os.path.join(current_phase_dir, 'data', 'data_after_imputation.xlsx')
    try:
        df_imputed.to_excel(excel_path, index=False)
        print(f"Excel version saved: {excel_path}")
    except Exception as e:
        print(f"Warning: Could not save Excel version: {e}")


    # Create summary report with proper type conversion
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'original_shape': list(df_original.shape) if df_original is not None else None,
        'processed_shape': list(df_imputed.shape) if df_imputed is not None else None,
        'columns_added': list(set(df_imputed.columns) - set(df_original.columns)) if df_original is not None else None,
        'missing_data_eliminated': int(missing_before - missing_after),
        'imputation_success_rate': float(((missing_before - missing_after) / missing_before * 100) if missing_before > 0 else 100.0),
        'next_phase': 'Phase2_Feature_Engineering'
    }

    try:
        with open(os.path.join(current_phase_dir, 'reports', 'phase1_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(summary), f, indent=2, ensure_ascii=False)
        print(f"Summary report saved")
    except Exception as e:
        print(f"Warning: Could not save summary report: {e}")


    return summary

# Save processed data
if 'missing_before' in locals() and 'missing_after' in locals() and df is not None and df_imputed is not None:
    phase1_summary = save_processed_data(df_imputed, df, missing_before, missing_after)
else:
    print("\nCannot save processed data as data loading, imputation, or missing value calculation failed.")


# ================================================================
# FINAL VISUALIZATION AND VALIDATION
# ================================================================

def create_final_validation_plots(df_original, df_imputed):
    """Create final validation plots"""
    print("\nCreating final validation visualizations...")

    # 1. Before/After Missing Data Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Before
    missing_before_viz = df_original.isnull().sum().sort_values(ascending=True)
    missing_before_viz = missing_before_viz[missing_before_viz > 0]

    if len(missing_before_viz) > 0:
        ax1.barh(range(len(missing_before_viz)), missing_before_viz.values, color='red', alpha=0.7)
        ax1.set_yticks(range(len(missing_before_viz)))
        ax1.set_yticklabels(missing_before_viz.index, fontsize=10)
        ax1.set_xlabel('Missing Count')
        ax1.set_title('Missing Data BEFORE Imputation', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'NO MISSING DATA\nBEFORE IMPUTATION',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=16, fontweight='bold', color='green')
        ax1.set_title('Missing Data BEFORE Imputation', fontweight='bold', fontsize=14)
        ax1.axis('off') # Hide axes if no data

    # After
    missing_after_viz = df_imputed.isnull().sum().sort_values(ascending=True)
    missing_after_viz = missing_after_viz[missing_after_viz > 0]

    if len(missing_after_viz) > 0:
        ax2.barh(range(len(missing_after_viz)), missing_after_viz.values, color='green', alpha=0.7)
        ax2.set_yticks(range(len(missing_after_viz)))
        ax2.set_yticklabels(missing_after_viz.index, fontsize=10)
        ax2.set_title('Missing Data AFTER Imputation', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Missing Count')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'NO MISSING DATA\nCOMPLETE SUCCESS',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=16, fontweight='bold', color='green')
        ax2.set_title('Missing Data AFTER Imputation', fontweight='bold', fontsize=14)
        ax2.axis('off') # Hide axes if no data


    plt.tight_layout()
    try:
        plt.savefig(
            os.path.join(current_phase_dir, 'figures', 'imputation_before_after.png'),
            dpi=CONFIG['dpi'], bbox_inches='tight'
        )
        print("Before/After Missing Data plot saved.")
    except Exception as e:
        print(f"Warning: Could not save Before/After Missing Data plot: {e}")

    plt.show()
    plt.close(fig) # Close figure

    # 2. Data Quality Improvement Metrics (Requires missing_before and missing_after)
    if 'missing_before' in locals() and 'missing_after' in locals() and df_original is not None and df_imputed is not None:
        improvement_data = {
            'Metric': ['Total Missing Values', 'Columns with Missing Data', 'Complete Cases'],
            'Before': [
                missing_before,
                len(df_original.columns[df_original.isnull().any()]),
                len(df_original.dropna())
            ],
            'After': [
                missing_after,
                len(df_imputed.columns[df_imputed.isnull().any()]),
                len(df_imputed.dropna())
            ]
        }

        improvement_df = pd.DataFrame(improvement_data)
        improvement_df['Improvement'] = improvement_df['Before'] - improvement_df['After']
        # Avoid division by zero if 'Before' is 0
        improvement_df['Improvement_Pct'] = (improvement_df['Improvement'] / improvement_df['Before']) * 100 if improvement_df['Before'].sum() > 0 else 0


        # Save improvement metrics
        try:
            improvement_df.to_csv(
                os.path.join(current_phase_dir, 'reports', 'data_quality_improvement.csv'),
                index=False
            )
            print("Data Quality Improvement metrics saved.")
        except Exception as e:
             print(f"Warning: Could not save Data Quality Improvement metrics: {e}")

    else:
         print("Cannot create Data Quality Improvement metrics as missing value counts are not available.")


# Create final validation plots
if df is not None and df_imputed is not None:
    create_final_validation_plots(df, df_imputed)
else:
    print("\nCannot create final validation plots as data loading or imputation failed.")

# ================================================================
# PHASE 1 COMPLETION REPORT
# ================================================================

print("\n" + "="*70)
print("PHASE 1 COMPLETION REPORT")
print("="*70)

if df is not None and df_imputed is not None:
    print(f"\nPROCESSING SUMMARY:")
    print(f"   Original dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   Processed dataset: {df_imputed.shape[0]:,} rows x {df_imputed.shape[1]} columns")

    if 'missing_before' in locals() and 'missing_after' in locals():
        print(f"   Missing values eliminated: {missing_before:,} to {missing_after:,}")
        if missing_before > 0:
            print(f"   Success rate: {((missing_before - missing_after) / missing_before * 100):.1f}%")
        else:
            print("   Success rate: 100.0% (No missing values initially)")

    if df is not None:
         print(f"   New features added: {len(set(df_imputed.columns) - set(df.columns))}")
    else:
         print("   Could not determine number of new features added.")


    print(f"\nOUTPUT FILES SAVED (if successful):")
    print(f"   - Processed data: data_after_imputation.csv/xlsx")
    print(f"   - Reports: missing_data_analysis.csv, imputation_log.json, etc.")
    print(f"   - Figures: missing_data_comprehensive.png, imputation_before_after.png, etc.")

    print(f"\nREADY FOR PHASE 2: Feature Engineering")
else:
    print("\nPHASE 1 FAILED: Data loading or imputation was not successful. Check logs above.")

print("="*70)