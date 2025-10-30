# ED-Triage-Explainable-ML
This repository contains the complete, reproducible Python code for the study: **"Explainable Machine Learning for Emergency Department Triage: A Multi-Hospital Study with Fairness Analysis."**  

# Explainable Machine Learning for Emergency Department Triage

This repository contains the complete, reproducible Python code for the study: **"Explainable Machine Learning for Emergency Department Triage: A Multi-Hospital Study with Fairness Analysis."**

The project focuses on developing, validating, and explaining a robust machine learning model (XGBoost) to predict three-level triage acuity (Red/Yellow/Green) using data from emergency department visits. The entire pipeline adheres to TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis) guidelines.

---

## Methodological Overview

This project emphasizes methodological rigor, clinical relevance, and transparency. Key methodologies implemented in this codebase include:

* **Advanced Missing Data Imputation:** Utilizes **Multiple Imputation by Chained Equations (MICE)** with Random Forest estimators, a method shown to be superior to simple mean/median imputation.
* **Sophisticated Class Imbalance:** Employs **Borderline-SMOTE** to oversample the minority 'Yellow' triage class, applied *only* to the training set to prevent data leakage.
* **Clinical Feature Engineering:** Creates clinically relevant features such as **Shock Index (SI)**, **Mean Arterial Pressure (MAP)**, and components of the **Modified Early Warning Score (MEWS)**.
* **Robust Validation Strategy:** Implements a strict temporal (out-of-time) data split and uses a separate, dedicated **calibration set** to prevent data leakage during model calibration.
* **Model Explainability (XAI):** Uses **SHAP (SHapley Additive exPlanations)** to ensure the "black box" model is interpretable and that its feature importances are clinically coherent.
* **Comprehensive Publication-Ready Analysis:** Includes a full suite of validation tests required for high-impact publication, including **Bootstrap Confidence Intervals**, **Baseline Model Comparison (McNemar's Test)**, **Subgroup Fairness Analysis**, and **Feature Ablation Studies**.

---

## Repository Structure

The project is organized into six sequential phases. Each phase is a separate script (or notebook) and generates its own outputs (data, figures, models, and reports) in its corresponding directory.

* `Phase1_Data_Preprocessing/`
    * Loads raw data.
    * Performs comprehensive missing data analysis (using `missingno`).
    * Applies the `AdvancedImputer` class (MICE, KNN, RF) to fill missing values.
    * Generates missingness indicators (`_was_missing` features).

* `Phase2_Feature_Engineering/`
    * Conducts multicollinearity analysis (VIF) and removes redundant features.
    * Performs outlier detection and treatment (Winsorization).
    * Engineers clinical domain features (Shock Index, MAP, etc.).
    * Performs statistical feature importance analysis (ANOVA F-test).

* `Phase3_Class_Imbalance/`
    * Splits data into training, validation, and test sets. **Crucially, this is a patient-level split to prevent data leakage.**
    * Applies Borderline-SMOTE *only* to the training set.
    * Calculates balanced class weights for model training.

* `Phase4_Model_Development/`
    * Splits the validation set into `val` (for tuning) and `calib` (for calibration).
    * Tunes multiple models (XGBoost, RandomForest, Logistic Regression) using GridSearchCV.
    * Selects the best model based on a balanced score of performance and overfitting gap.
    * Trains a final calibrated model on the `calib` set to ensure reliable probabilities.

* `Phase5_Explainability_Validation/`
    * Runs **SHAP** analysis on the final model.
    * Generates global and per-class SHAP summary plots.
    * Performs clinical error analysis to identify high-risk misclassifications (e.g., "Red-as-Green").
    * Generates calibration curves and per-class ROC/PR curves.

* `Phase6_Publication_Analyses/`
    * Performs all critical pre-publication checks.
    * **Bootstrap Confidence Intervals (n=1000)** for all key metrics.
    * **Baseline Model Comparison** (vs. Dummy, LR, Simple Rules) with McNemar's test.
    * **Subgroup Fairness Analysis** (e.g., by age, arrival mode).
    * **Learning Curves** to assess if more data is needed.
    * **Feature Ablation Study** to quantify the impact of feature groups (e.g., "Vital Signs").

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
    cd YOUR_REPOSITORY
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage / Execution Order

The code is designed to be run sequentially. You must run the phases in order, as each phase depends on the output data from the previous one.

1.  **Place your data:** Put your (anonymized or synthetic) dataset in the `Phase1_Data_Preprocessing/data/` directory. The scripts are configured to look for a file named `synthetic_dataset.csv` (you can change this in the `CONFIG` dictionary in each script).
2.  **Run Phase 1:** `python Phase1_Data_Preprocessing/phase1_script.py`
3.  **Run Phase 2:** `python Phase2_Feature_Engineering/phase2_script.py`
4.  **Run Phase 3:** `python Phase3_Class_Imbalance/phase3_script.py`
5.  **Run Phase 4:** `python Phase4_Model_Development/phase4_script.py`
6.  **Run Phase 5:** `python Phase5_Explainability_Validation/phase5_script.py`
7.  **Run Phase 6:** `python Phase6_Publication_Analyses/phase6_script.py`

After running all phases, all figures, reports, and models will be populated in their respective phase-specific directories.

---

## Citation

If you use this code or methodology in your research, please cite our paper:

> [Full Paper Citation - e.g., Author, A., & Author, B. (Year). "Title of Paper". *Journal Name*, Vol(Issue), pp. 1-10.]

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.
