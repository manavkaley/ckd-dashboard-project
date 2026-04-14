import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

sns.set_style('whitegrid')

DATA_PATH = Path('Chronic_Kidney_Dsease_data.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def preprocess(df):
    data = df.copy()
    data = data.drop(columns=['PatientID', 'DoctorInCharge'], errors='ignore')
    data['BP_Age_Interaction'] = data['SystolicBP'] * data['Age'] / 1000
    data['Creatinine_Diabetes'] = data['SerumCreatinine'] * (data['FamilyHistoryDiabetes'] == 1).astype(int)
    data['GFR_Creatinine_Ratio'] = data['GFR'] / (data['SerumCreatinine'] + 0.1)
    data['HealthRiskScore'] = (
        0.4 * np.clip(data['BMI'] / 35, 0, 2) +
        0.3 * np.clip(data['FastingBloodSugar'] / 200, 0, 2) +
        0.3 * np.clip(data['SerumCreatinine'] / 6, 0, 2)
    )
    data['GFR_Category'] = pd.cut(data['GFR'], bins=[-np.inf, 30, 60, 90, np.inf],
                                  labels=['Stage 4-5', 'Stage 3', 'Stage 2', 'Stage 1'])
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 40, 60, 80, 150],
                               labels=['Young', 'Middle', 'Senior', 'Elderly'])
    data['BP_Category'] = pd.cut(data['SystolicBP'], bins=[0, 120, 140, 180, 300],
                                 labels=['Normal', 'Elevated', 'High', 'Crisis'])
    y = data['Diagnosis'].astype(int)
    X = data.drop(columns=['Diagnosis'])
    return X, y

@st.cache_data
def get_preprocessor(X):
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features

@st.cache_resource
def fit_models(X_train, y_train):
    preprocessor, numeric_features, categorical_features = get_preprocessor(X_train)
    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=15))
        ])
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models, preprocessor, numeric_features, categorical_features

@st.cache_data
def evaluate_models(_models, X_test, y_test):
    results = []
    for name, model in _models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
    results_df = pd.DataFrame(results).set_index('Model')
    return results_df

@st.cache_data
def get_risk_categories(probas, thresholds=(0.4, 0.7)):
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    bins = [0.0, thresholds[0], thresholds[1], 1.0]
    categories = pd.cut(probas, bins=bins, labels=labels, include_lowest=True)
    return categories

@st.cache_data
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            ohe = transformer.named_steps['onehot']
            feature_names.extend(ohe.get_feature_names_out(cols))
    return feature_names

st.set_page_config(page_title='CKD Prediction Dashboard', layout='wide')

st.title('CKD Prediction & Risk Stratification Dashboard')
st.markdown(
    'Interactive healthcare dashboard for Chronic Kidney Disease prediction, risk stratification, and model explainability.'
)

df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
models, preprocessor, numeric_features, categorical_features = fit_models(X_train, y_train)
results_df = evaluate_models(models, X_test, y_test)

best_model_name = results_df['ROC-AUC'].idxmax()
best_model = models[best_model_name]

tabs = st.tabs(['Overview', 'Model Comparison', 'Risk Stratification', 'Patient Prediction', 'Feature Insights'])

with tabs[0]:
    st.header('Dataset Overview')
    st.write('Number of patients:', df.shape[0])
    st.write('Number of features:', df.shape[1])
    st.write('CKD Label Distribution:')
    st.bar_chart(df['Diagnosis'].value_counts())

    st.subheader('Key Clinical Statistics')
    key_stats = df[['Age', 'BMI', 'SerumCreatinine', 'GFR', 'BUNLevels', 'FastingBloodSugar']].describe().T
    st.dataframe(key_stats)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
        ax.set_title('Age Distribution')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df['SerumCreatinine'], bins=20, kde=True, ax=ax)
        ax.set_title('Serum Creatinine Distribution')
        st.pyplot(fig)

with tabs[1]:
    st.header('Model Performance Comparison')
    st.write('Best model selected:', best_model_name)
    st.dataframe(results_df.style.format('{:.3f}'))
    st.write('Model evaluation on the held-out test set.')

    fig, ax = plt.subplots(figsize=(8, 4))
    results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Model Metric Comparison')
    st.pyplot(fig)

with tabs[2]:
    st.header('Risk Stratification')
    proba_test = best_model.predict_proba(X_test)[:, 1]
    categories = get_risk_categories(proba_test)
    dist = categories.value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk']).fillna(0)
    st.bar_chart(dist)
    st.write('Risk distribution on the test set:')
    st.dataframe(dist.to_frame('Count'))
    st.write('Risk thresholds are currently set to:')
    st.markdown('- Low Risk: probability < 0.4')
    st.markdown('- Medium Risk: probability >= 0.4 and < 0.7')
    st.markdown('- High Risk: probability >= 0.7')

with tabs[3]:
    st.header('Patient-Level Prediction')
    st.write('Enter patient attributes to generate a CKD risk score and category.')

    with st.form('patient_form'):
        age = st.number_input('Age', min_value=0, max_value=120, value=45)
        gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=28.4)
        systolic = st.number_input('Systolic BP', min_value=80, max_value=220, value=120)
        diastolic = st.number_input('Diastolic BP', min_value=40, max_value=140, value=80)
        fasting = st.number_input('Fasting Blood Sugar', min_value=50.0, max_value=300.0, value=100.0)
        creatinine = st.number_input('Serum Creatinine', min_value=0.1, max_value=20.0, value=1.0)
        gfr = st.number_input('GFR', min_value=0.0, max_value=200.0, value=90.0)
        bun = st.number_input('BUN Levels', min_value=0.0, max_value=200.0, value=15.0)
        protein = st.number_input('Protein in Urine', min_value=0.0, max_value=500.0, value=100.0)
        family_diabetes = st.selectbox('Family History of Diabetes', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        submit = st.form_submit_button('Predict Risk')

    if submit:
        patient_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Ethnicity': 0,
            'SocioeconomicStatus': 0,
            'EducationLevel': 0,
            'BMI': bmi,
            'Smoking': 0,
            'AlcoholConsumption': 0.0,
            'PhysicalActivity': 0.0,
            'DietQuality': 0.0,
            'SleepQuality': 0.0,
            'FamilyHistoryKidneyDisease': 0,
            'FamilyHistoryHypertension': 0,
            'FamilyHistoryDiabetes': family_diabetes,
            'PreviousAcuteKidneyInjury': 0,
            'UrinaryTractInfections': 0,
            'SystolicBP': systolic,
            'DiastolicBP': diastolic,
            'FastingBloodSugar': fasting,
            'HbA1c': 5.5,
            'SerumCreatinine': creatinine,
            'BUNLevels': bun,
            'GFR': gfr,
            'ProteinInUrine': protein,
            'ACR': 0.0,
            'SerumElectrolytesSodium': 140.0,
            'SerumElectrolytesPotassium': 4.0,
            'SerumElectrolytesCalcium': 9.0,
            'SerumElectrolytesPhosphorus': 3.5,
            'HemoglobinLevels': 13.0,
            'CholesterolTotal': 180.0,
            'CholesterolLDL': 100.0,
            'CholesterolHDL': 50.0,
            'CholesterolTriglycerides': 120.0,
            'ACEInhibitors': 0,
            'Diuretics': 0,
            'NSAIDsUse': 0,
            'Statins': 0,
            'AntidiabeticMedications': 0,
            'Edema': 0,
            'FatigueLevels': 1.0,
            'NauseaVomiting': 0,
            'MuscleCramps': 0,
            'Itching': 0,
            'QualityOfLifeScore': 5.0,
            'HeavyMetalsExposure': 0,
            'OccupationalExposureChemicals': 0,
            'WaterQuality': 1,
            'MedicalCheckupsFrequency': 1.0,
            'MedicationAdherence': 1.0,
            'HealthLiteracy': 1.0
        }])
        patient_data['BP_Age_Interaction'] = patient_data['SystolicBP'] * patient_data['Age'] / 1000
        patient_data['Creatinine_Diabetes'] = patient_data['SerumCreatinine'] * (patient_data['FamilyHistoryDiabetes'] == 1).astype(int)
        patient_data['GFR_Creatinine_Ratio'] = patient_data['GFR'] / (patient_data['SerumCreatinine'] + 0.1)
        patient_data['HealthRiskScore'] = (
            0.4 * np.clip(patient_data['BMI'] / 35, 0, 2) +
            0.3 * np.clip(patient_data['FastingBloodSugar'] / 200, 0, 2) +
            0.3 * np.clip(patient_data['SerumCreatinine'] / 6, 0, 2)
        )
        patient_data['GFR_Category'] = pd.cut(patient_data['GFR'], bins=[-np.inf, 30, 60, 90, np.inf],
                                              labels=['Stage 4-5', 'Stage 3', 'Stage 2', 'Stage 1'])
        patient_data['Age_Group'] = pd.cut(patient_data['Age'], bins=[0, 40, 60, 80, 150],
                                          labels=['Young', 'Middle', 'Senior', 'Elderly'])
        patient_data['BP_Category'] = pd.cut(patient_data['SystolicBP'], bins=[0, 120, 140, 180, 300],
                                            labels=['Normal', 'Elevated', 'High', 'Crisis'])

        score = best_model.predict_proba(patient_data)[0, 1]
        category = get_risk_categories([score])[0]
        st.subheader('Patient Risk Result')
        st.metric('CKD Risk Probability', f'{score:.2f}', delta=None)
        st.write('Risk Category:', category)
        st.write('Key model used:', best_model_name)

with tabs[4]:
    st.header('Feature Insights')
    if 'Random Forest' in models:
        model = models['Random Forest']
        rf = model.named_steps['classifier']
        importances = rf.feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': importances[:len(numeric_features)]
        }).sort_values('Importance', ascending=False).head(15)
        st.write('Top 15 Important Features')
        st.dataframe(feature_importances)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Top Feature Importances')
        st.pyplot(fig)
    else:
        st.write('Random Forest model is not available for feature importance.')

    st.markdown('### Clinical Interpretation')
    st.write(
        'Higher importance for features such as GFR-Creatinine Ratio, ' 
        'Serum Creatinine, and GFR indicates strong relationship with kidney function. ' 
        'These factors are clinically relevant for early CKD detection and risk stratification.'
    )
