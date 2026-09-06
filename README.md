# 🫀 CardioGuard AI: Early Cardiovascular Disease Risk Prediction & Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-74.09%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8085-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Kaggle-CVD_70K_Records-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CardioGuard AI** is an end-to-end clinical machine learning and decision-support platform designed to stratify cardiovascular disease (CVD) risk using non-invasive routine examination biometrics. 

Trained on **70,000 patient records**, this project bridges rigorous algorithmic benchmarking with practical clinical deployment, featuring an interactive **Streamlit web application**, a **full technical Jupyter Notebook**, and an executive **12-slide case study presentation**.

---

## 📑 Project Navigation: Two-Part Architecture

This repository is organized into two primary pillars:

| Pillar | Description | Primary Deliverables |
| :--- | :--- | :--- |
| **1. Technical Project** | Complete end-to-end data science pipeline, cleaning, feature engineering, statistical tests, 5-model benchmarking, hyperparameter tuning, and production pipeline serialization. | • [Full Jupyter Notebook (`cardio_cvd_prediction.ipynb`)](cardio_cvd_prediction.ipynb)<br>• [Training Script (`train_model.py`)](train_model.py)<br>• [Production Web App (`app.py`)](app.py)<br>• [Extracted Chart Assets (`images/`)](images/) |
| **2. Case-Study Presentation** | Executive problem-to-solution story designed for clinical stakeholders and recruiters. Grounded in exact calculated metrics with zero code clutter. | • [12-Slide Case Analysis (`CASE_STUDY.md`)](CASE_STUDY.md)<br>• [Interactive Slide Deck (`case_study_presentation.html`)](case_study_presentation.html) *(Supports 1-click Print to PDF)* |

---

## 🎯 Clinical Problem Formulation

Cardiovascular disease is the leading cause of mortality worldwide, claiming **17.9 million lives each year** (~32% of all global deaths). A substantial fraction of cardiovascular damage develops silently without overt symptoms until an acute ischemic event occurs.

### Core Research Question
> **"Can routine, non-invasive biometrics (age, blood pressure, BMI, serum cholesterol, glucose, and lifestyle factors) reliably predict cardiovascular disease risk before acute clinical onset?"**

### Core Clinical Objectives
1. **Low-Cost Primary Triage:** Screen outpatient cohorts during standard clinic visits without requiring immediate invasive angiograms or advanced imaging.
2. **Prioritizing Confirmatory Diagnostics:** Flag asymptomatic high-risk patients for secondary workups (12-lead ECG, stress echocardiography, carotid Doppler ultrasound).
3. **Preventative Intervention:** Facilitate early lifestyle counseling (AHA diet, exercise prescription, smoking cessation) and preventative pharmacotherapy.

---

## 📊 Dataset Architecture & Preprocessing

The model is trained on the Kaggle Cardiovascular Disease dataset containing **70,000 examination records**:

### Data Dictionary

| Variable | Clinical Feature | Type | Measurement / Categories |
| :--- | :--- | :--- | :--- |
| `age_years` | Chronological Age | Continuous | Derived from days (39.0 – 64.9 years) |
| `gender` | Biological Sex | Categorical | 1: Female, 2: Male |
| `height` | Stature | Continuous | Centimeters (cm) |
| `weight` | Body Mass | Continuous | Kilograms (kg) |
| `ap_hi` | Systolic Blood Pressure | Continuous | mmHg (Cardiac contraction peak) |
| `ap_lo` | Diastolic Blood Pressure | Continuous | mmHg (Resting arterial pressure) |
| `bmi` | Body Mass Index | Continuous | `weight (kg) / (height (m))²` |
| `pulse_pressure` | Arterial Pulse Pressure | Continuous | `ap_hi − ap_lo` (Vascular stiffness indicator) |
| `cholesterol` | Serum Total Cholesterol | Ordinal | 1: Normal, 2: Above Normal, 3: Well Above Normal |
| `gluc` | Fasting Glucose | Ordinal | 1: Normal, 2: Above Normal, 3: Well Above Normal |
| `smoke` | Smoking Inhalation | Binary | 0: Non-smoker, 1: Active smoker |
| `alco` | Alcohol Consumption | Binary | 0: Non-consumer, 1: Consumer |
| `active` | Physical Activity | Binary | 0: Inactive, 1: Regularly active |
| **`cardio`** | **CVD Diagnosis** | **Target** | **0: Healthy Control, 1: Confirmed CVD** |

### Data Preparation Steps
1. **Age Conversion:** Converted raw days to fractional continuous years (`age / 365.0`).
2. **Feature Engineering:** 
   - WHO Body Mass Index (`kg/m²`).
   - Hemodynamic Pulse Pressure (`ap_hi − ap_lo`).
3. **Physiological Filtering:** Filtered out physiological artifacts (`60 ≤ ap_hi ≤ 250`, `40 ≤ ap_lo ≤ 160`, `100 ≤ height ≤ 250`, `30 ≤ weight ≤ 200`), retaining **68,731 high-integrity records**.
4. **Stratified Split:** Split into **54,984 training instances** (80%) and **13,747 test instances** (20%), scaled with `StandardScaler`.

---

## 🔍 Key Exploratory Findings (EDA)

| Age Progression | Blood Pressure Staging |
| :---: | :---: |
| ![Age Distribution](images/02_age_distribution.png) | ![Blood Pressure Distribution](images/03_blood_pressure_distribution.png) |
| *CVD prevalence doubles from **29.26% (<45 yrs)** to **66.80% (60–64 yrs)**.* | *Stage 2 HTN shows **80.04% CVD prevalence** vs **22.14%** in normal BP.* |

| Cholesterol & Glucose Escalation | BMI Adiposity Shift |
| :---: | :---: |
| ![Cholesterol and Glucose](images/06_cholesterol_glucose.png) | ![BMI Distribution](images/05_bmi_distribution.png) |
| *High cholesterol escalates risk from **43.56% (Normal)** to **76.29% (High)**.* | *CVD cohort shows significant upward shift in BMI (**mean 28.48 kg/m²**).* |

---

## 🏆 Model Benchmarking & Evaluation

Five distinct machine learning architectures were trained and evaluated on the identical holdout test cohort (N = 13,747):

| Classification Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Champion)** | **74.09%** | **77.56%** | **67.03%** | **71.91%** | **0.8085** | 🏆 **Best Overall Model** |
| **Decision Tree (d = 6)** | 73.77% | 74.79% | **70.90%** | **72.79%** | 0.8009 | 🥈 Runner-Up (High Recall) |
| **Logistic Regression** | 73.37% | 75.97% | 67.56% | 71.52% | 0.7995 | Linear Baseline |
| **Linear SVM (LinearSVC)** | 73.24% | 76.27% | 66.66% | 71.14% | 0.7993 | Maximum Margin Baseline |
| **K-Nearest Neighbors (k = 11)** | 72.41% | 73.10% | 70.00% | 71.52% | 0.7823 | Non-Parametric Baseline |

![Model Benchmark Comparison](images/10_model_accuracy_comparison.png)

---

## 🎯 Confusion Matrix & Clinical Screening Analysis

The champion **Random Forest** model was evaluated on 13,747 unseen test patients:

![Random Forest Confusion Matrix](images/12_rf_confusion_matrix.png)

### Matrix Breakdown
- **True Negatives (TN): 5,625** (Specificity: **81.01%**) — Healthy individuals correctly identified and spared unnecessary medical anxiety.
- **True Positives (TP): 4,560** (Sensitivity / Recall: **67.03%**) — High-risk patients correctly flagged for early intervention.
- **False Positives (FP): 1,319** (False Positive Rate: **18.99%**) — Healthy patients referred for low-cost secondary checkups (ECG, repeat BP).
- **False Negatives (FN): 2,243** (False Negative Rate: **32.97%**) — Individuals with CVD missed by standard 0.50 decision thresholding.

> ### 💡 Clinical Triage Safety Rationale:
> In health screening, **a False Negative is clinically far more hazardous than a False Positive**. In production settings, CardioGuard AI supports configurable decision threshold calibration: lowering the threshold to **0.38** elevates sensitivity beyond **85%**, ensuring minimal false negatives during initial primary care screening.

---

## 🔑 Feature Importance & Clinical Interpretability

Feature importance rankings extracted from the champion ensemble model:

![Feature Importance](images/11_feature_importance_rf.png)

1. **Systolic Blood Pressure (`ap_hi`):** **38.06%**
2. **Diastolic Blood Pressure (`ap_lo`):** **19.26%**
3. **Pulse Pressure (`ap_hi - ap_lo`):** **14.17%**
4. **Patient Age (`age_years`):** **10.53%**
5. **Serum Cholesterol (`cholesterol`):** **8.66%**
6. **Body Mass Index (`bmi`):** **3.24%**
7. **Patient Weight (`weight`):** **2.92%**

Together, **hemodynamic variables (systolic, diastolic, and pulse pressure) account for 71.49% of total predictive power**, underscoring vascular pressure load as the single greatest physiological biomarker in CVD detection.

---

## 🚀 CardioGuard AI Web Application Features

The interactive clinical decision-support application is built with **Streamlit**:

1. **Interactive Patient Intake & Live Biometrics**:
   - Dynamic real-time calculation of **Body Mass Index (BMI)** with WHO classification badges (Underweight, Normal, Overweight, Obese).
   - Real-time **Pulse Pressure** and **Mean Arterial Pressure (MAP)** computation.
   - American Heart Association (**AHA**) Blood Pressure Staging (Normal, Elevated, Stage 1 & 2 Hypertension, Hypertensive Crisis).
   - 1-click **Quick-Load Clinical Presets** (*Healthy Active Adult*, *Borderline Risk*, *High-Risk Senior*).

2. **Risk Stratification & SVG Gauge Visualizer**:
   - Animated SVG circular gauge indicating continuous risk probability percentage (0% – 100%).
   - Tiered Risk Categorization:
     - 🟢 **Low Risk** (< 30%)
     - 🟡 **Moderate / Borderline Risk** (30% – 60%)
     - 🔴 **High Risk** (> 60%)

3. **Automated Clinical Recommendations & Report Export**:
   - Algorithmic triage guidelines tailored to detected abnormalities (hypertension protocol, lipid profiling, glycemic screening, smoking cessation).
   - 1-click download of a formatted **Clinical Health Assessment Report (`.txt`)**.

4. **Multi-Patient Batch Screening**:
   - Upload patient CSV cohorts (`sample_patients.csv` template included).
   - Automated batch feature engineering and multi-patient risk prediction export.

---

## 💻 Quickstart: Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/ravi8800pandey-ai/cardioguard-ai.git
cd cardioguard-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
Open and run the technical analysis notebook:
```bash
jupyter notebook cardio_cvd_prediction.ipynb
```

### 4. Launch the Streamlit Web App
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### Streamlit Community Cloud (Free & Recommended)
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and link your GitHub account.
3. Select this repository (`cardioguard-ai`), branch `main`, and set main file to `app.py`.
4. Click **Deploy**!

### Docker Container
```bash
docker build -t cardioguard-ai .
docker run -p 8501:8501 cardioguard-ai
```

---

## 📂 Repository File Structure

```
cardioguard-ai/
├── cardio_cvd_prediction.ipynb   # Complete Technical Project Jupyter Notebook
├── CASE_STUDY.md                 # 12-Slide Executive Clinical Case Analysis
├── case_study_presentation.html  # Interactive Presentation Deck (1-Click PDF Print)
├── README.md                     # Executive Documentation & Technical Guide
├── app.py                        # Streamlit Clinical Decision Support Web App
├── train_model.py                # Standalone Scikit-Learn Training & Export Pipeline
├── sample_patients.csv           # Multi-Patient Batch Screening CSV Cohort
├── requirements.txt              # Application Dependencies
├── run_app.bat                   # 1-Click Local Launcher for Windows
├── push_to_github.bat            # 1-Click GitHub Synchronization Script
├── .streamlit/
│   └── config.toml               # Custom Clinical UI Theme Tokens
├── models/
│   ├── best_rf_pipeline.joblib   # Serialized Model Pipeline
│   └── model_metrics.json        # Benchmark Results & Feature Importance Weights
└── images/                       # High-Resolution Visual Assets & Chart Exports
    ├── 01_target_distribution.png
    ├── 02_age_distribution.png
    ├── 03_blood_pressure_distribution.png
    ├── 04_categorical_features.png
    ├── 05_bmi_distribution.png
    ├── 06_cholesterol_glucose.png
    ├── 07_pairplot.png
    ├── 08_correlation_matrix.png
    ├── 09_confusion_matrices_all_models.png
    ├── 10_model_accuracy_comparison.png
    ├── 11_feature_importance_rf.png
    ├── 12_rf_confusion_matrix.png
    ├── 13_roc_curve.png
    └── 14_disease_rate_by_bp_age.png
```

---

## 🛡️ Medical Disclaimer & Clinical Governance

> **Important Notice:** CardioGuard AI is developed solely for educational, research, and clinical decision-support demonstration purposes. It is not an FDA-approved medical diagnostic device and does not substitute for professional medical judgment, clinical examination, or laboratory diagnostic testing. Always seek the advice of a qualified physician with any questions regarding cardiovascular health.

---

## 👤 Author & Acknowledgments

- **Lead Analyst & Developer:** Ravi Pandey
- **GitHub:** [@ravi8800pandey-ai](https://github.com/ravi8800pandey-ai)
- **Dataset Reference:** Kaggle Cardiovascular Disease Dataset (70,000 anonymized examination records).
