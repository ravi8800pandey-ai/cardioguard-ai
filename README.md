# 🫀 CardioGuard AI - Cardiovascular Disease Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-74.09%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8085-success.svg)]()

CardioGuard AI is a clinical-grade decision support and risk stratification web application for Cardiovascular Disease (CVD) prediction. It deploys the top-performing **Random Forest Classifier** trained on 70,000 patient records from the Kaggle Cardiovascular Disease Dataset.

---

## 🌟 Key Features

1. **Interactive Patient Intake & Live Biometrics**:
   - Dynamic real-time calculation of **Body Mass Index (BMI)** with WHO classification badges (Underweight, Normal, Overweight, Obese).
   - Real-time **Pulse Pressure** and **Mean Arterial Pressure (MAP)** calculation.
   - American Heart Association (**AHA**) Blood Pressure Staging (Normal, Elevated, Hypertension Stage 1 & 2, Hypertensive Crisis).
   - 1-click **Quick-Load Demo Presets** ("Healthy Active Adult", "Borderline Risk", "High-Risk Senior").

2. **Risk Stratification & SVG Gauge Visualizer**:
   - Animated SVG circular gauge indicating continuous risk probability percentage ($0\% - 100\%$).
   - Tiered Risk Categorization:
     - 🟢 **Low Risk** ($< 30\%$)
     - 🟡 **Moderate / Borderline Risk** ($30\% - 60\%$)
     - 🔴 **High Risk** ($> 60\%$)

3. **Clinical Recommendations & Report Export**:
   - Automated rule-based clinical recommendations tailored to detected abnormalities (hypertension protocol, lipid profiling, glycemic screening, smoking cessation, physical conditioning).
   - 1-click download of a formatted **Clinical Health Assessment Report (.txt)**.

4. **Model Performance & Explainability Dashboard**:
   - Benchmark leaderboard comparing all 5 algorithms:
     - **Random Forest**: **74.09%** (Best Model 🏆)
     - Decision Tree: 73.77%
     - Logistic Regression: 73.37%
     - SVM (LinearSVC): 73.24%
     - KNN: 72.40%
   - Feature Importance ranking chart (Systolic BP, Diastolic BP, Pulse Pressure, Age, Cholesterol).
   - Full Confusion Matrix analysis for 13,747 test patients.

5. **Multi-Patient Batch Screening (CSV Upload)**:
   - Screen hundreds or thousands of patients in seconds.
   - Automated feature engineering and batch probability prediction.
   - 1-click export of annotated screening results.

---

## 🚀 Quickstart: Running Locally

### 1. Clone or Navigate to the Project
```bash
cd "d:\RAVI DOC\projects\cardio_prediction_app"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### Option A: Streamlit Community Cloud (Recommended & Free)
1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py` (or `cardio_prediction_app/app.py`).
5. Click **"Deploy"**! Streamlit will automatically read `requirements.txt` and host your app live on the web with a public URL.

### Option B: Hugging Face Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Upload the contents of `cardio_prediction_app/` (`app.py`, `requirements.txt`, `models/`, `.streamlit/`).
4. The space will automatically build and launch.

### Option C: Docker Container
Create a `Dockerfile` in the root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t cardioguard-ai .
docker run -p 8501:8501 cardioguard-ai
```

---

## 🔬 Model Technical Specifications

| Parameter | Specification |
|:---|:---|
| **Algorithm** | Random Forest Classifier (Ensemble) |
| **Estimators** | 100 Trees |
| **Max Depth** | 8 |
| **Preprocessing** | StandardScaler Pipeline |
| **Validation Accuracy** | **74.09%** |
| **ROC-AUC Score** | **0.8085** |
| **Precision** | **77.56%** |
| **Recall** | **67.03%** |
| **Training Records** | 54,984 patients |
| **Test Records** | 13,747 patients |

---

## 📂 Project Structure

```
cardio_prediction_app/
├── app.py                     # Streamlit application with modern UI/UX
├── train_model.py             # Model training & pipeline export script
├── sample_patients.csv        # Ready-to-use batch cohort template
├── requirements.txt           # Deployment dependency specifications
├── README.md                  # Comprehensive deployment documentation
├── .streamlit/
│   └── config.toml            # UI theme tokens & styling
└── models/
    ├── best_rf_pipeline.joblib # Serialized scikit-learn Pipeline
    └── model_metrics.json     # Saved evaluation metrics & feature weights
```

---

## 🛡️ Medical Disclaimer
CardioGuard AI is developed as an educational and clinical decision-support reference tool. It does not provide definitive medical diagnoses or replace consultations with licensed healthcare professionals.
