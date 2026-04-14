#!/usr/bin/env python
"""Test CKD Research Framework - Validate ML Pipeline"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CKD RESEARCH FRAMEWORK - PIPELINE VALIDATION TEST")
print("="*70)

# Load dataset
data_path = r'Chronic_Kidney_Dsease_data.csv'
df = pd.read_csv(data_path)

print(f"\n1. DATA LOADING")
print(f"   Dataset shape: {df.shape}")
print(f"   Diagnosis distribution:\n{df['Diagnosis'].value_counts()}")

# Data preparation
X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis'])
y = df['Diagnosis']

# Feature engineering
X['BP_Age'] = X['SystolicBP'] * X['Age']
X['Creatinine_Diabetes'] = X['SerumCreatinine'] * (X['FamilyHistoryDiabetes'] == 1).astype(int)
X['HealthRiskScore'] = (
    0.4 * (X['BMI'] / 35) +
    0.3 * (X['FastingBloodSugar'] / 200) +
    0.3 * (X['SerumCreatinine'] / 6)
)
X['GFR_Category'] = pd.cut(
    X['GFR'],
    bins=[-np.inf, 30, 60, 90, np.inf],
    labels=['Very Low', 'Low', 'Moderate', 'Normal']
).astype(str)

print(f"\n2. FEATURE ENGINEERING")
print(f"   New features added: BP_Age, Creatinine_Diabetes, HealthRiskScore, GFR_Category")
print(f"   Total features after engineering: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n3. TRAIN-TEST SPLIT")
print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"   Train class balance: {y_train.value_counts(normalize=True).to_dict()}")

# Build preprocessing pipeline
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

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

# Train models
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

print(f"\n4. MODEL TRAINING AND EVALUATION")
print("-" * 70)

results = []
for name, model in models.items():
    print(f"\n   Training: {name}")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'AUC-ROC': auc
    })
    
    print(f"   ✓ Accuracy: {acc:.4f}")
    print(f"   ✓ AUC-ROC: {auc:.4f}")

print("\n" + "="*70)
print("5. RISK STRATIFICATION TEST")
print("="*70)

best_model = models['Random Forest']
proba_test = best_model.predict_proba(X_test)[:, 1]
risk_category = pd.cut(proba_test, bins=[0.0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High'], include_lowest=True)

print(f"\nRisk distribution:")
print(risk_category.value_counts().reindex(['Low', 'Medium', 'High']))

print("\n" + "="*70)
print("✅ FRAMEWORK IMPLEMENTATION: SUCCESS")
print("="*70)
print("\nNext steps:")
print("1. Open the Jupyter notebook: CKD_research_framework.ipynb")
print("2. Run all cells to generate comprehensive analysis and visualizations")
print("3. Review SHAP explanations for clinical interpretability")
print("4. Export results as HTML/PDF for research paper")
print("="*70)
