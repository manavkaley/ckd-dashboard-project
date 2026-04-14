#!/usr/bin/env python
"""
CKD Advanced Analysis and Dashboard
Generates detailed analysis outputs for research paper and decision support
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = Path('analysis_outputs')
output_dir.mkdir(exist_ok=True)

print("\n" + "="*80)
print("CHRONIC KIDNEY DISEASE RESEARCH FRAMEWORK - COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATORY ANALYSIS
# ============================================================================
print("\n[1/5] LOADING AND EXPLORING DATA...")

df = pd.read_csv('Chronic_Kidney_Dsease_data.csv')

# Summary statistics
summary_stats = df.describe(include='all').T
summary_stats.to_csv(output_dir / '01_summary_statistics.csv')
print(f"✓ Dataset: {df.shape[0]} patients, {df.shape[1]} features")

# Missing value analysis
missing_analysis = df.isna().sum().sort_values(ascending=False)
missing_analysis[missing_analysis > 0].to_csv(output_dir / '02_missing_values_analysis.csv')
print(f"✓ Missing values: {(df.isna().sum().sum() / df.size * 100):.2f}% of data")

# Clinical feature analysis - Focus on key kidney biomarkers
key_features = ['Age', 'SerumCreatinine', 'GFR', 'BUNLevels', 'ProteinInUrine',
                'SystolicBP', 'DiastolicBP', 'FastingBloodSugar', 'BMI']
clinical_analysis = df[key_features].describe(include='all').T
clinical_analysis.to_csv(output_dir / '03_clinical_features_summary.csv')
print(f"✓ Key clinical features analyzed: {len(key_features)} biomarkers")

# ============================================================================
# PART 2: PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================
print("\n[2/5] PREPROCESSING AND FEATURE ENGINEERING...")

X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = df['Diagnosis']

# Create interaction and composite features
X['BP_Age_Interaction'] = X['SystolicBP'] * X['Age'] / 1000
X['Creatinine_Diabetes'] = X['SerumCreatinine'] * (X['FamilyHistoryDiabetes'] == 1).astype(int)
X['GFR_Creatinine_Ratio'] = X['GFR'] / (X['SerumCreatinine'] + 0.1)  # Avoid division by zero
X['HealthRiskScore'] = (
    0.4 * np.clip(X['BMI'] / 35, 0, 2) +
    0.3 * np.clip(X['FastingBloodSugar'] / 200, 0, 2) +
    0.3 * np.clip(X['SerumCreatinine'] / 6, 0, 2)
)

# Bin continuous variables
X['GFR_Category'] = pd.cut(X['GFR'], bins=[-np.inf, 30, 60, 90, np.inf],
                            labels=['Stage 4-5', 'Stage 3', 'Stage 2', 'Stage 1'])
X['Age_Group'] = pd.cut(X['Age'], bins=[0, 40, 60, 80, 150],
                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
X['BP_Category'] = pd.cut(X['SystolicBP'], bins=[0, 120, 140, 180, 300],
                           labels=['Normal', 'Elevated', 'High', 'Crisis'])

feature_engineering_summary = pd.DataFrame({
    'Feature': ['BP_Age_Interaction', 'Creatinine_Diabetes', 'GFR_Creatinine_Ratio',
                'HealthRiskScore', 'GFR_Category', 'Age_Group', 'BP_Category'],
    'Type': ['Numerical', 'Numerical', 'Numerical', 'Numerical', 'Categorical',
             'Categorical', 'Categorical'],
    'Purpose': ['Interaction', 'Interaction', 'Ratio', 'Composite Score', 'Binning',
                'Binning', 'Binning']
})
feature_engineering_summary.to_csv(output_dir / '04_feature_engineering_summary.csv', index=False)
print(f"✓ Total features after engineering: {X.shape[1]}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"✓ Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}")

# ============================================================================
# PART 3: MODEL DEVELOPMENT AND COMPARISON
# ============================================================================
print("\n[3/5] TRAINING MODELS AND COMPARING PERFORMANCE...")

numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

# Define models
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                            class_weight='balanced', max_depth=15))
    ])
}

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                     scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                                     use_label_encoder=False, eval_metric='logloss'))
    ])
except ImportError:
    pass

# Train and evaluate models
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Specificity': confusion_matrix(y_test, y_pred)[0, 0] / (confusion_matrix(y_test, y_pred)[0, 0] + confusion_matrix(y_test, y_pred)[0, 1])
    })

results_df = pd.DataFrame(results).set_index('Model')
results_df.to_csv(output_dir / '05_model_performance_comparison.csv')

# Print comparison
print("\nModel Performance Comparison:")
print(results_df.round(4))

# Select best model
best_model_name = results_df['ROC-AUC'].idxmax()
best_model = trained_models[best_model_name]
print(f"\n✓ Best model selected: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f})")

# ============================================================================
# PART 4: RISK STRATIFICATION
# ============================================================================
print("\n[4/5] RISK STRATIFICATION AND PATIENT ANALYSIS...")

proba_test = best_model.predict_proba(X_test)[:, 1]

# Risk categories based on clinical thresholds
risk_thresholds = [0.0, 0.3, 0.6, 1.0]
risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
risk_categories = pd.cut(proba_test, bins=risk_thresholds, labels=risk_labels, include_lowest=True)

risk_distribution = risk_categories.value_counts().reindex(risk_labels, fill_value=0)
risk_dist_df = pd.DataFrame({
    'Risk Category': risk_labels,
    'Count': [risk_distribution.get(cat, 0) for cat in risk_labels],
    'Percentage': [risk_distribution.get(cat, 0) / len(risk_categories) * 100 for cat in risk_labels]
})
risk_dist_df.to_csv(output_dir / '06_risk_distribution.csv', index=False)
print(f"✓ Risk stratification completed:")
print(risk_dist_df.to_string(index=False))

# High-risk patient identification
high_risk_indices = np.where(proba_test > risk_thresholds[2])[0]
high_risk_patients = X_test.iloc[high_risk_indices].copy()
high_risk_patients['Risk_Score'] = proba_test[high_risk_indices]
high_risk_patients['Actual_Diagnosis'] = y_test.iloc[high_risk_indices].values
high_risk_patients[['Risk_Score', 'Actual_Diagnosis']].to_csv(
    output_dir / '07_high_risk_patients_sample.csv'
)
print(f"✓ High-risk patients identified: {len(high_risk_indices)} patients")

# ============================================================================
# PART 5: FEATURE IMPORTANCE AND INTERPRETATION
# ============================================================================
print("\n[5/5] FEATURE IMPORTANCE AND CLINICAL INTERPRETATION...")

# Extract feature importance from tree-based model
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest'].named_steps['classifier']
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': rf_model.feature_importances_[:len(numeric_features)]
    }).sort_values('Importance', ascending=False).head(15)
    
    feature_importance.to_csv(output_dir / '08_feature_importance.csv', index=False)
    print(f"\nTop 15 Important Features:")
    print(feature_importance.to_string(index=False))

print("\n" + "="*80)
print("✅ COMPREHENSIVE ANALYSIS COMPLETED")
print("="*80)
print(f"\nOutput files saved to: {output_dir.resolve()}")
print("\nGenerated Files:")
print("  1. 01_summary_statistics.csv - Full dataset statistics")
print("  2. 02_missing_values_analysis.csv - Missing data report")
print("  3. 03_clinical_features_summary.csv - Key biomarker analysis")
print("  4. 04_feature_engineering_summary.csv - New features created")
print("  5. 05_model_performance_comparison.csv - Model metrics")
print("  6. 06_risk_distribution.csv - Patient risk stratification")
print("  7. 07_high_risk_patients_sample.csv - Identified high-risk patients")
print("  8. 08_feature_importance.csv - Top predictive features")
print("\n" + "="*80)
print("NEXT STEPS FOR RESEARCH PAPER:")
print("="*80)
print("1. Run the Jupyter notebook for interactive visualizations")
print("2. Generate SHAP explanations for clinical interpretability")
print("3. Create figures for publication (ROC curves, confusion matrices)")
print("4. Write clinical interpretation section using feature importance")
print("5. Prepare results for research publication")
print("="*80 + "\n")
