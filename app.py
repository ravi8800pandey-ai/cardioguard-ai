import os
import json
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioGuard AI | CVD Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_THEME = {
    "bg_app": "#0F172A",
    "bg_sidebar": "#1E293B",
    "hero_bg": "linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%)",
    "hero_border": "rgba(56, 189, 248, 0.25)",
    "hero_shadow": "0 10px 30px -10px rgba(0, 0, 0, 0.5)",
    "hero_title": "#F8FAFC",
    "hero_subtitle": "#94A3B8",
    "badge_model_bg": "rgba(2, 132, 199, 0.15)",
    "badge_model_color": "#38BDF8",
    "badge_model_border": "rgba(56, 189, 248, 0.35)",
    "badge_info_bg": "rgba(56, 189, 248, 0.2)",
    "badge_info_color": "#38BDF8",
    "badge_info_border": "rgba(56, 189, 248, 0.4)",
    "card_bg": "rgba(30, 41, 59, 0.7)",
    "card_border": "rgba(148, 163, 184, 0.15)",
    "card_hover_border": "rgba(56, 189, 248, 0.4)",
    "metric_title": "#94A3B8",
    "metric_val": "#F8FAFC",
    "metric_desc": "#64748B",
    "rec_bg": "rgba(30, 41, 59, 0.5)",
    "rec_border": "rgba(148, 163, 184, 0.15)",
    "rec_header": "#F8FAFC",
    "rec_body": "#94A3B8",
    "risk_title_low": "#34D399",
    "risk_title_moderate": "#FBBF24",
    "risk_title_high": "#F87171",
    "gauge_track": "#334155",
    "gauge_text": "#F8FAFC",
    "gauge_subtext": "#94A3B8",
    "gauge_desc": "#E2E8F0",
    "chart_text": "#F8FAFC",
    "chart_bar_bg": "#475569",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8"
}

LIGHT_THEME = {
    "bg_app": "#F8FAFC",
    "bg_sidebar": "#F1F5F9",
    "hero_bg": "linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%)",
    "hero_border": "rgba(2, 132, 199, 0.30)",
    "hero_shadow": "0 10px 25px -10px rgba(0, 0, 0, 0.08)",
    "hero_title": "#0F172A",
    "hero_subtitle": "#475569",
    "badge_model_bg": "rgba(2, 132, 199, 0.10)",
    "badge_model_color": "#0284C7",
    "badge_model_border": "rgba(2, 132, 199, 0.30)",
    "badge_info_bg": "rgba(2, 132, 199, 0.12)",
    "badge_info_color": "#0284C7",
    "badge_info_border": "rgba(2, 132, 199, 0.35)",
    "card_bg": "#FFFFFF",
    "card_border": "rgba(203, 213, 225, 0.85)",
    "card_hover_border": "rgba(2, 132, 199, 0.5)",
    "metric_title": "#64748B",
    "metric_val": "#0F172A",
    "metric_desc": "#64748B",
    "rec_bg": "#FFFFFF",
    "rec_border": "rgba(203, 213, 225, 0.75)",
    "rec_header": "#0F172A",
    "rec_body": "#475569",
    "risk_title_low": "#059669",
    "risk_title_moderate": "#D97706",
    "risk_title_high": "#DC2626",
    "gauge_track": "#E2E8F0",
    "gauge_text": "#0F172A",
    "gauge_subtext": "#64748B",
    "gauge_desc": "#334155",
    "chart_text": "#0F172A",
    "chart_bar_bg": "#CBD5E1",
    "text_primary": "#0F172A",
    "text_secondary": "#475569"
}

# Sidebar Appearance toggle
theme_mode = st.sidebar.radio(
    "🌗 Appearance",
    ["Dark", "Light"],
    index=0,
    horizontal=True,
    help="Toggle visual theme between Dark and Light mode."
)
theme = DARK_THEME if theme_mode == "Dark" else LIGHT_THEME

def get_custom_css(t: dict) -> str:
    return f"""
<style>
/* Global styles & Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp, [data-testid="stAppViewContainer"] {{
    background-color: {t['bg_app']} !important;
    color: {t['text_primary']} !important;
}}

[data-testid="stSidebar"] {{
    background-color: {t['bg_sidebar']} !important;
}}

/* App Header styling */
.hero-card {{
    background: {t['hero_bg']};
    border: 1px solid {t['hero_border']};
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: {t['hero_shadow']};
}}

.hero-title {{
    font-size: 2.2rem;
    font-weight: 800;
    color: {t['hero_title']};
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}}

.hero-subtitle {{
    font-size: 1.05rem;
    color: {t['hero_subtitle']};
    margin-bottom: 12px;
}}

.badge-model {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {t['badge_model_bg']};
    color: {t['badge_model_color']};
    border: 1px solid {t['badge_model_border']};
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}}

/* Metric cards */
.metric-card {{
    background: {t['card_bg']};
    backdrop-filter: blur(10px);
    border: 1px solid {t['card_border']};
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    border-color: {t['card_hover_border']};
}}

.metric-title {{
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {t['metric_title']};
    margin-bottom: 6px;
    font-weight: 600;
}}

.metric-val {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {t['metric_val']};
}}

.metric-desc {{
    font-size: 0.8rem;
    color: {t['metric_desc']};
    margin-top: 4px;
}}

/* Risk Level Banners */
.risk-banner-low {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}}

.risk-banner-moderate {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}}

.risk-banner-high {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}}

.risk-title-low {{ color: {t['risk_title_low']}; font-size: 1.8rem; font-weight: 800; }}
.risk-title-moderate {{ color: {t['risk_title_moderate']}; font-size: 1.8rem; font-weight: 800; }}
.risk-title-high {{ color: {t['risk_title_high']}; font-size: 1.8rem; font-weight: 800; }}

.risk-pct {{
    font-size: 3.2rem;
    font-weight: 800;
    margin: 8px 0;
    letter-spacing: -0.03em;
}}

/* Category Badges */
.badge-pill {{
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
}}
.badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }}
.badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }}
.badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
.badge-info {{ background: {t['badge_info_bg']}; color: {t['badge_info_color']}; border: 1px solid {t['badge_info_border']}; }}

/* Recommendation Cards */
.rec-card {{
    background: {t['rec_bg']};
    border-left: 4px solid #0284C7;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    border-top: 1px solid {t['rec_border']};
    border-right: 1px solid {t['rec_border']};
    border-bottom: 1px solid {t['rec_border']};
}}
.rec-card.rec-alert {{
    border-left-color: #EF4444;
    background: rgba(239, 68, 68, 0.08);
}}
.rec-card.rec-warning {{
    border-left-color: #F59E0B;
    background: rgba(245, 158, 11, 0.08);
}}
.rec-card.rec-healthy {{
    border-left-color: #10B981;
    background: rgba(16, 185, 129, 0.08);
}}

.rec-header {{
    font-weight: 700;
    font-size: 0.95rem;
    color: {t['rec_header']};
    margin-bottom: 4px;
}}
.rec-body {{
    font-size: 0.88rem;
    color: {t['rec_body']};
    line-height: 1.45;
}}

/* Mobile Responsiveness */
@media (max-width: 768px) {{
    .hero-title {{
        font-size: 1.6rem !important;
    }}
    .hero-card {{
        padding: 16px 18px !important;
    }}
    .metric-val {{
        font-size: 1.3rem !important;
    }}
}}
</style>
"""

st.markdown(get_custom_css(theme), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model & Pipeline Loader (with Auto-Trainer Fallback)
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "best_rf_pipeline.joblib")
METRICS_PATH = os.path.join(CURRENT_DIR, "models", "model_metrics.json")
DATASET_PATH = os.path.join(CURRENT_DIR, "..", "cardio_train.csv")

@st.cache_resource(show_spinner="Loading best-performing Random Forest model pipeline...")
def load_trained_model():
    """Load serialized Random Forest pipeline or train if missing."""
    if not os.path.exists(MODEL_PATH):
        st.warning("Pre-trained model artifact not found. Initializing auto-training pipeline...")
        try:
            from train_model import train_and_export
        except ImportError as e:
            st.error(f"Could not import train_model.py: {e}")
            raise
        # Resolve dataset path
        candidates = [
            DATASET_PATH,
            os.path.join(CURRENT_DIR, "cardio_train.csv")
        ]
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break
        if not resolved:
            st.error("Dataset `cardio_train.csv` not found for auto-training.")
            st.stop()
        train_and_export(resolved, os.path.join(CURRENT_DIR, "models"))

    pipeline = joblib.load(MODEL_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return pipeline, metrics

pipeline, model_metrics = load_trained_model()

# Exact test-set metrics evaluated across 13,747 patients for thresholds 0.20 to 0.60 (step 0.01)
# Format: {threshold: (accuracy_pct, precision, recall, specificity, tp, fp, tn, fn)}
THRESHOLD_METRICS = {
    0.20: (56.67, 0.5337, 0.9850, 0.1568, 6701, 5855, 1089, 102),
    0.21: (57.99, 0.5419, 0.9777, 0.1902, 6651, 5623, 1321, 152),
    0.22: (59.31, 0.5503, 0.9721, 0.2219, 6613, 5403, 1541, 190),
    0.23: (60.36, 0.5574, 0.9660, 0.2486, 6572, 5218, 1726, 231),
    0.24: (61.66, 0.5664, 0.9606, 0.2795, 6535, 5003, 1941, 268),
    0.25: (63.13, 0.5778, 0.9469, 0.3220, 6442, 4708, 2236, 361),
    0.26: (65.52, 0.5984, 0.9224, 0.3934, 6275, 4212, 2732, 528),
    0.27: (66.56, 0.6086, 0.9084, 0.4277, 6180, 3974, 2970, 623),
    0.28: (67.34, 0.6172, 0.8950, 0.4562, 6089, 3776, 3168, 714),
    0.29: (67.97, 0.6244, 0.8852, 0.4784, 6022, 3622, 3322, 781),
    0.30: (68.41, 0.6302, 0.8756, 0.4965, 5957, 3496, 3448, 846),
    0.31: (69.11, 0.6381, 0.8681, 0.5176, 5906, 3350, 3594, 897),
    0.32: (69.48, 0.6430, 0.8614, 0.5315, 5860, 3253, 3691, 943),
    0.33: (70.03, 0.6499, 0.8551, 0.5487, 5817, 3134, 3810, 986),
    0.34: (70.43, 0.6555, 0.8484, 0.5631, 5772, 3034, 3910, 1031),
    0.35: (70.83, 0.6613, 0.8415, 0.5778, 5725, 2932, 4012, 1078),
    0.36: (71.23, 0.6671, 0.8358, 0.5913, 5686, 2838, 4106, 1117),
    0.37: (71.39, 0.6712, 0.8271, 0.6030, 5627, 2757, 4187, 1176),
    0.38: (71.78, 0.6787, 0.8161, 0.6215, 5552, 2628, 4316, 1251),
    0.39: (72.52, 0.6933, 0.7976, 0.6544, 5426, 2400, 4544, 1377),
    0.40: (73.40, 0.7118, 0.7772, 0.6917, 5287, 2141, 4803, 1516),
    0.41: (73.65, 0.7186, 0.7685, 0.7052, 5228, 2047, 4897, 1575),
    0.42: (73.85, 0.7252, 0.7592, 0.7182, 5165, 1957, 4987, 1638),
    0.43: (73.97, 0.7315, 0.7486, 0.7308, 5093, 1869, 5075, 1710),
    0.44: (74.12, 0.7396, 0.7361, 0.7461, 5008, 1763, 5181, 1795),
    0.45: (74.15, 0.7468, 0.7226, 0.7599, 4916, 1667, 5277, 1887),
    0.46: (74.05, 0.7520, 0.7097, 0.7707, 4828, 1592, 5352, 1975),
    0.47: (74.18, 0.7604, 0.6984, 0.7844, 4751, 1497, 5447, 2052),
    0.48: (74.22, 0.7663, 0.6893, 0.7941, 4689, 1430, 5514, 2114),
    0.49: (74.22, 0.7716, 0.6804, 0.8027, 4629, 1370, 5574, 2174),
    0.50: (74.09, 0.7756, 0.6703, 0.8101, 4560, 1319, 5625, 2243),
    0.51: (74.02, 0.7805, 0.6607, 0.8180, 4495, 1264, 5680, 2308),
    0.52: (73.99, 0.7863, 0.6516, 0.8265, 4433, 1205, 5739, 2370),
    0.53: (73.94, 0.7913, 0.6428, 0.8340, 4373, 1153, 5791, 2430),
    0.54: (73.94, 0.7969, 0.6352, 0.8414, 4321, 1101, 5843, 2482),
    0.55: (73.74, 0.7993, 0.6268, 0.8458, 4264, 1071, 5873, 2539),
    0.56: (73.68, 0.8029, 0.6205, 0.8508, 4221, 1036, 5908, 2582),
    0.57: (73.65, 0.8070, 0.6146, 0.8560, 4181, 1000, 5944, 2622),
    0.58: (73.54, 0.8118, 0.6056, 0.8625, 4120, 955, 5989, 2683),
    0.59: (73.22, 0.8156, 0.5928, 0.8687, 4033, 912, 6032, 2770),
    0.60: (73.06, 0.8185, 0.5855, 0.8728, 3983, 883, 6061, 2820),
}

# -----------------------------------------------------------------------------
# 3. Clinical Helper Functions & Classifications
# -----------------------------------------------------------------------------
def classify_blood_pressure(ap_hi: int, ap_lo: int) -> tuple[str, str, str]:
    """Classify Blood Pressure according to American Heart Association (AHA) guidelines."""
    if ap_hi >= 180 or ap_lo >= 120:
        return "Hypertensive Crisis", "badge-danger", "Emergency medical care required."
    elif ap_hi >= 140 or ap_lo >= 90:
        return "Hypertension Stage 2", "badge-danger", "Sustained high BP. Clinical treatment advised."
    elif (130 <= ap_hi <= 139) or (80 <= ap_lo <= 89):
        return "Hypertension Stage 1", "badge-warning", "Lifestyle changes + clinical monitoring recommended."
    elif (120 <= ap_hi <= 129) and (ap_lo < 80):
        return "Elevated", "badge-warning", "Elevated systolic pressure. Lifestyle intervention needed."
    elif ap_hi < 120 and ap_lo < 80:
        return "Optimal / Normal", "badge-success", "Healthy blood pressure range."
    else:
        return "Non-standard BP", "badge-info", "Review reading."

def classify_bmi(bmi: float) -> tuple[str, str]:
    """Classify Body Mass Index (BMI) by WHO criteria."""
    if bmi < 18.5:
        return "Underweight", "badge-info"
    elif 18.5 <= bmi <= 24.9:
        return "Normal Weight", "badge-success"
    elif 25.0 <= bmi <= 29.9:
        return "Overweight", "badge-warning"
    else:
        return "Obese", "badge-danger"

def calculate_map(ap_hi: int, ap_lo: int) -> float:
    """Calculate Mean Arterial Pressure (MAP)."""
    return round((ap_hi + 2 * ap_lo) / 3.0, 1)

# -----------------------------------------------------------------------------
# 4. Sidebar: Patient Intake & Preset Profiles
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🫀 Patient Intake Form")
st.sidebar.caption("Enter patient diagnostic metrics or choose a demonstration profile.")

# Demo presets
PRESETS = {
    "Custom Patient Input": None,
    "Healthy Active Young Adult (Low Risk)": {
        "age": 28, "gender": "Female", "height": 168, "weight": 58.0,
        "ap_hi": 115, "ap_lo": 75, "chol": "Normal", "gluc": "Normal",
        "smoke": "No", "alco": "No", "active": "Yes"
    },
    "Borderline Metabolic Risk Adult": {
        "age": 52, "gender": "Male", "height": 175, "weight": 84.0,
        "ap_hi": 135, "ap_lo": 88, "chol": "Above Normal", "gluc": "Normal",
        "smoke": "No", "alco": "Yes", "active": "No"
    },
    "High Risk Hypertensive Senior": {
        "age": 63, "gender": "Male", "height": 170, "weight": 96.0,
        "ap_hi": 165, "ap_lo": 105, "chol": "Well Above Normal", "gluc": "Well Above Normal",
        "smoke": "Yes", "alco": "Yes", "active": "No"
    }
}

preset_choice = st.sidebar.selectbox("⚡ Quick Demo Presets", list(PRESETS.keys()))
preset_data = PRESETS[preset_choice]

# Demographic values
default_age = int(preset_data["age"]) if preset_data else 50
default_gender = str(preset_data["gender"]) if preset_data else "Male"
default_height = int(preset_data["height"]) if preset_data else 170
default_weight = float(preset_data["weight"]) if preset_data else 75.0
default_ap_hi = int(preset_data["ap_hi"]) if preset_data else 130
default_ap_lo = int(preset_data["ap_lo"]) if preset_data else 85
default_chol = str(preset_data["chol"]) if preset_data else "Normal"
default_gluc = str(preset_data["gluc"]) if preset_data else "Normal"
default_smoke = str(preset_data["smoke"]) if preset_data else "No"
default_alco = str(preset_data["alco"]) if preset_data else "No"
default_active = str(preset_data["active"]) if preset_data else "Yes"

with st.sidebar.expander("👤 1. Demographics", expanded=True):
    age_input = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=default_age, step=1)
    gender_input = st.radio("Biological Sex", ["Female", "Male"], index=0 if default_gender == "Female" else 1, horizontal=True)

with st.sidebar.expander("🩺 2. Blood Pressure (mmHg)", expanded=True):
    col_sbp, col_dbp = st.columns(2)
    with col_sbp:
        ap_hi = st.number_input("Systolic (ap_hi)", min_value=70, max_value=240, value=default_ap_hi, step=1, help="Systolic Blood Pressure (upper number)")
    with col_dbp:
        ap_lo = st.number_input("Diastolic (ap_lo)", min_value=40, max_value=150, value=default_ap_lo, step=1, help="Diastolic Blood Pressure (lower number)")

    bp_status, bp_badge_class, bp_hint = classify_blood_pressure(ap_hi, ap_lo)
    st.markdown(f"AHA Status: <span class='badge-pill {bp_badge_class}'>{bp_status}</span>", unsafe_allow_html=True)
    pulse_pressure = ap_hi - ap_lo
    st.caption(f"Pulse Pressure: **{pulse_pressure} mmHg** | MAP: **{calculate_map(ap_hi, ap_lo)} mmHg**")

with st.sidebar.expander("⚖️ 3. Body Dimensions", expanded=True):
    col_h, col_w = st.columns(2)
    with col_h:
        height_cm = st.number_input("Height (cm)", min_value=110, max_value=230, value=default_height, step=1)
    with col_w:
        weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=default_weight, step=0.5)

    calculated_bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 2)
    bmi_status, bmi_badge_class = classify_bmi(calculated_bmi)
    st.markdown(f"BMI: **{calculated_bmi} kg/m²** <span class='badge-pill {bmi_badge_class}'>{bmi_status}</span>", unsafe_allow_html=True)

with st.sidebar.expander("🧪 4. Laboratory Findings", expanded=True):
    chol_options = ["Normal", "Above Normal", "Well Above Normal"]
    gluc_options = ["Normal", "Above Normal", "Well Above Normal"]
    chol_input = st.selectbox("Serum Cholesterol", chol_options, index=chol_options.index(default_chol))
    gluc_input = st.selectbox("Fasting Glucose", gluc_options, index=gluc_options.index(default_gluc))

with st.sidebar.expander("🏃 5. Lifestyle & Habits", expanded=True):
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        smoke_input = st.radio("Tobacco Smoking", ["No", "Yes"], index=0 if default_smoke == "No" else 1, horizontal=True)
        alco_input = st.radio("Alcohol Intake", ["No", "Yes"], index=0 if default_alco == "No" else 1, horizontal=True)
    with col_l2:
        active_input = st.radio("Physical Activity", ["Active", "Sedentary"], index=0 if default_active == "Yes" else 1)

with st.sidebar.expander("⚙️ 6. Model Sensitivity", expanded=True):
    decision_threshold = st.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.60,
        value=0.50,
        step=0.01,
        help="Lower threshold = more sensitive (fewer missed CVD cases, more false alarms). Higher threshold = more specific (fewer false alarms, more missed cases)."
    )
    if decision_threshold < 0.50:
        st.caption(f"🚨 **High-Sensitivity Mode ({decision_threshold:.2f})**: Lower threshold catches early/borderline CVD cases to minimize dangerous false negatives.")
    elif decision_threshold > 0.50:
        st.caption(f"🎯 **High-Specificity Mode ({decision_threshold:.2f})**: Demands higher certainty before flagging cases to prevent unnecessary clinical workups.")
    else:
        st.caption("Standard Balanced Threshold (0.50).")

# Format payload for Model Pipeline
gender_code = 1 if gender_input == "Female" else 2
chol_code = 1 if chol_input == "Normal" else (2 if chol_input == "Above Normal" else 3)
gluc_code = 1 if gluc_input == "Normal" else (2 if gluc_input == "Above Normal" else 3)
smoke_code = 1 if smoke_input == "Yes" else 0
alco_code = 1 if alco_input == "Yes" else 0
active_code = 1 if active_input == "Active" else 0

input_df = pd.DataFrame([{
    'gender': gender_code,
    'height': height_cm,
    'weight': weight_kg,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': chol_code,
    'gluc': gluc_code,
    'smoke': smoke_code,
    'alco': alco_code,
    'active': active_code,
    'age_years': float(age_input),
    'bmi': calculated_bmi,
    'pulse_pressure': pulse_pressure
}])

# Pipeline Inference
prediction_proba = float(pipeline.predict_proba(input_df)[0][1])
prediction_class = int(prediction_proba >= decision_threshold)
risk_percentage = round(prediction_proba * 100, 1)

# -----------------------------------------------------------------------------
# 5. Main Hero Section
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">
        <span>🫀</span> CardioGuard AI
        <span class="badge-model">● Model Live: Random Forest Pipeline</span>
    </div>
    <div class="hero-subtitle">
        Clinical Decision Support & Risk Stratification for Cardiovascular Disease (CVD)
    </div>
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px;">
        <span class="badge-pill badge-info">Validation Accuracy: {model_metrics.get('accuracy', 74.09)}%</span>
        <span class="badge-pill badge-info">ROC-AUC: {model_metrics.get('roc_auc', 0.8085)}</span>
        <span class="badge-pill badge-info">Dataset: 70,000 Verified Records</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Multi-Tab Navigation
# -----------------------------------------------------------------------------
tab_assessment, tab_explain, tab_batch, tab_guidelines = st.tabs([
    "🩺 Patient Risk Assessment",
    "📊 Model Performance & Explainability",
    "📁 Batch Screening (CSV)",
    "📖 Clinical Guidelines & Reference"
])

# =============================================================================
# TAB 1: Patient Assessment & Risk Stratification
# =============================================================================
with tab_assessment:
    col_left, col_right = st.columns([1.1, 1.4], gap="large")

    with col_left:
        st.markdown("#### 🎯 Stratified Risk Score")

        # Determine Risk Tier
        if risk_percentage < 30.0:
            tier_class = "risk-banner-low"
            tier_title_class = "risk-title-low"
            tier_label = "LOW RISK"
            tier_desc = "Patient exhibits low probability of cardiovascular disease based on clinical indicators."
            needle_color = "#10B981"
        elif risk_percentage < 60.0:
            tier_class = "risk-banner-moderate"
            tier_title_class = "risk-title-moderate"
            tier_label = "MODERATE / BORDERLINE RISK"
            tier_desc = "Moderate CVD risk detected. Lifestyle intervention and preventive monitoring advised."
            needle_color = "#F59E0B"
        else:
            tier_class = "risk-banner-high"
            tier_title_class = "risk-title-high"
            tier_label = "HIGH RISK"
            tier_desc = "Significant CVD probability detected. Immediate comprehensive cardiology evaluation recommended."
            needle_color = "#EF4444"

        # SVG Animated-Style Circular Gauge
        circumference = 2 * 3.14159 * 75
        dashoffset = circumference * (1 - (risk_percentage / 100.0))

        svg_gauge = f"""
        <div class="{tier_class}">
            <div class="{tier_title_class}">{tier_label}</div>
            <div style="display: flex; justify-content: center; margin: 15px 0;">
                <svg width="200" height="200" viewBox="0 0 200 200">
                    <circle cx="100" cy="100" r="75" stroke="{theme['gauge_track']}" stroke-width="16" fill="transparent" />
                    <circle cx="100" cy="100" r="75" stroke="{needle_color}" stroke-width="16" fill="transparent"
                            stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}"
                            stroke-linecap="round" transform="rotate(-90 100 100)" style="transition: stroke-dashoffset 1s ease;" />
                    <text x="100" y="95" text-anchor="middle" fill="{theme['gauge_text']}" font-size="34" font-weight="800" font-family="Inter, sans-serif">{risk_percentage}%</text>
                    <text x="100" y="125" text-anchor="middle" fill="{theme['gauge_subtext']}" font-size="14" font-weight="600" font-family="Inter, sans-serif">CVD PROBABILITY</text>
                </svg>
            </div>
            <div style="font-size: 0.95rem; color: {theme['gauge_desc']}; line-height: 1.5;">{tier_desc}</div>
            <div style="margin-top: 14px;">
                <span class="badge-pill {'badge-danger' if prediction_class==1 else 'badge-success'}">
                    Model Binary Decision: {'Cardiovascular Disease Likely (1)' if prediction_class==1 else 'No Disease Indicated (0)'} (Threshold: {decision_threshold:.2f})
                </span>
            </div>
        </div>
        """
        st.markdown(svg_gauge, unsafe_allow_html=True)
        st.caption("⚠️ Educational tool only — not a medical diagnosis. Consult a physician.")

        st.markdown("<br>", unsafe_allow_html=True)
        # Quick Clinical Metrics Tiles
        st.markdown("#### 📋 Diagnostic Summary")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Blood Pressure</div>
                <div class="metric-val">{ap_hi}/{ap_lo}</div>
                <div class="metric-desc"><span class='badge-pill {bp_badge_class}'>{bp_status}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-title">Pulse Pressure</div>
                <div class="metric-val">{pulse_pressure} <span style="font-size: 1rem; font-weight: 400;">mmHg</span></div>
                <div class="metric-desc">Optimal: 30-50 mmHg</div>
            </div>
            """, unsafe_allow_html=True)

        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Body Mass Index</div>
                <div class="metric-val">{calculated_bmi} <span style="font-size: 1rem; font-weight: 400;">kg/m²</span></div>
                <div class="metric-desc"><span class='badge-pill {bmi_badge_class}'>{bmi_status}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-title">Mean Arterial (MAP)</div>
                <div class="metric-val">{calculate_map(ap_hi, ap_lo)} <span style="font-size: 1rem; font-weight: 400;">mmHg</span></div>
                <div class="metric-desc">Normal range: 70-100 mmHg</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🔬 Biomarker Risk Factor Breakdown")

        # Visual deviation from optimal benchmarks
        risk_factors = [
            {"Factor": "Systolic Blood Pressure", "Patient Value": f"{ap_hi} mmHg", "Optimal Baseline": "< 120 mmHg", "Status": "Elevated" if ap_hi >= 130 else ("Borderline" if ap_hi >= 120 else "Normal"), "Flag": ap_hi >= 130},
            {"Factor": "Diastolic Blood Pressure", "Patient Value": f"{ap_lo} mmHg", "Optimal Baseline": "< 80 mmHg", "Status": "Elevated" if ap_lo >= 85 else ("Borderline" if ap_lo >= 80 else "Normal"), "Flag": ap_lo >= 85},
            {"Factor": "Pulse Pressure", "Patient Value": f"{pulse_pressure} mmHg", "Optimal Baseline": "40 mmHg", "Status": "High" if pulse_pressure >= 60 else "Normal", "Flag": pulse_pressure >= 60},
            {"Factor": "Body Mass Index (BMI)", "Patient Value": f"{calculated_bmi} kg/m²", "Optimal Baseline": "18.5 - 24.9 kg/m²", "Status": bmi_status, "Flag": calculated_bmi >= 25.0},
            {"Factor": "Serum Cholesterol", "Patient Value": chol_input, "Optimal Baseline": "Normal", "Status": chol_input, "Flag": chol_input != "Normal"},
            {"Factor": "Fasting Glucose", "Patient Value": gluc_input, "Optimal Baseline": "Normal", "Status": gluc_input, "Flag": gluc_input != "Normal"},
            {"Factor": "Tobacco Smoking", "Patient Value": smoke_input, "Optimal Baseline": "No", "Status": "Smoker" if smoke_input=="Yes" else "Non-Smoker", "Flag": smoke_input == "Yes"},
            {"Factor": "Physical Activity", "Patient Value": active_input, "Optimal Baseline": "Active", "Status": active_input, "Flag": active_input == "Sedentary"}
        ]

        rf_df = pd.DataFrame(risk_factors)

        def style_status(val):
            if val in ["Elevated", "High", "Obese", "Well Above Normal", "Smoker", "Sedentary"]:
                return "color: #F87171; font-weight: bold;"
            elif val in ["Borderline", "Overweight", "Above Normal"]:
                return "color: #FBBF24; font-weight: bold;"
            return "color: #34D399; font-weight: 500;"

        st.dataframe(
            rf_df[["Factor", "Patient Value", "Optimal Baseline", "Status"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💡 Tailored Clinical Interventions")

        recommendations = []
        if ap_hi >= 130 or ap_lo >= 85:
            recommendations.append({
                "type": "rec-alert",
                "icon": "🩸",
                "title": "Hypertension Protocol",
                "body": f"Patient's BP is currently {ap_hi}/{ap_lo} mmHg ({bp_status}). Initiate ambulatory blood pressure monitoring (ABPM), recommend sodium restriction (<1,500 mg/day), and evaluate for antihypertensive pharmacological therapy per AHA/ACC guidelines."
            })

        if calculated_bmi >= 25.0:
            recommendations.append({
                "type": "rec-warning",
                "icon": "⚖️",
                "title": "Weight & Metabolic Optimization",
                "body": f"Current BMI is {calculated_bmi} kg/m² ({bmi_status}). Structured dietary modification focusing on caloric deficit, whole foods, and Mediterranean dietary pattern to achieve a 5-10% body weight reduction."
            })

        if chol_input != "Normal":
            recommendations.append({
                "type": "rec-alert",
                "icon": "🧪",
                "title": "Lipid Profile Assessment",
                "body": f"Serum cholesterol is flagged as '{chol_input}'. Order a comprehensive fasting lipid panel (LDL-C, HDL-C, Triglycerides, ApoB) and evaluate statin eligibility based on ASCVD 10-year risk estimator."
            })

        if gluc_input != "Normal":
            recommendations.append({
                "type": "rec-warning",
                "icon": "🍯",
                "title": "Glycemic Screening",
                "body": f"Fasting glucose is '{gluc_input}'. Schedule HbA1c testing and oral glucose tolerance test (OGTT) to rule out impaired fasting glucose, prediabetes, or type 2 diabetes mellitus."
            })

        if smoke_input == "Yes":
            recommendations.append({
                "type": "rec-alert",
                "icon": "🚭",
                "title": "Tobacco Cessation Support",
                "body": "Active smoking is one of the highest preventable drivers of arterial stiffness and cardiovascular events. Offer nicotine replacement therapy, varenicline, and counseling resources."
            })

        if active_input == "Sedentary":
            recommendations.append({
                "type": "rec-warning",
                "icon": "🏃",
                "title": "Cardiovascular Conditioning",
                "body": "Prescribe at least 150 minutes of moderate-intensity aerobic exercise or 75 minutes of vigorous exercise weekly, paired with twice-weekly resistance training."
            })

        if not recommendations:
            recommendations.append({
                "type": "rec-healthy",
                "icon": "🌟",
                "title": "Maintenance of Cardiovascular Wellness",
                "body": "All monitored biomarkers fall within standard physiological benchmarks. Encourage sustained adherence to regular physical activity, balanced nutrition, and annual health checkups."
            })

        for rec in recommendations:
            st.markdown(f"""
            <div class="rec-card {rec['type']}">
                <div class="rec-header">{rec['icon']} {rec['title']}</div>
                <div class="rec-body">{rec['body']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Downloadable Clinical Summary Report
        report_text = f"""========================================================================
CARDIOGUARD AI - CARDIOVASCULAR HEALTH ASSESSMENT REPORT
========================================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Patient Age: {age_input} years  |  Sex: {gender_input}

DIAGNOSTIC TELEMETRY:
- Blood Pressure: {ap_hi}/{ap_lo} mmHg ({bp_status})
- Pulse Pressure: {pulse_pressure} mmHg
- Mean Arterial Pressure (MAP): {calculate_map(ap_hi, ap_lo)} mmHg
- Height: {height_cm} cm  |  Weight: {weight_kg} kg
- Body Mass Index (BMI): {calculated_bmi} kg/m² ({bmi_status})
- Serum Cholesterol: {chol_input}
- Fasting Glucose: {gluc_input}
- Tobacco Smoking: {smoke_input}
- Alcohol Intake: {alco_input}
- Physical Activity: {active_input}

MODEL INFERENCE:
- Model Architecture: Random Forest Classifier (Depth 8, 100 Trees)
- Overall Model Accuracy: {model_metrics.get('accuracy', 74.09)}%
- Predicted CVD Probability: {risk_percentage}%
- Risk Stratification: {tier_label}
- Binary Decision: {'Cardiovascular Disease Indicated (1)' if prediction_class==1 else 'No Disease Indicated (0)'}

CLINICAL RECOMMENDATIONS:
"""
        for r in recommendations:
            report_text += f"- [{r['title']}]: {r['body']}\n"

        report_text += """
========================================================================
DISCLAIMER: This report is generated by an AI machine learning model for
clinical decision support only. It does not replace professional diagnosis.
========================================================================
"""
        st.download_button(
            label="📥 Download Clinical Assessment Report (TXT)",
            data=report_text,
            file_name=f"CardioGuard_Assessment_Age{age_input}_{gender_input}.txt",
            mime="text/plain",
            use_container_width=True
        )

# =============================================================================
# TAB 2: Model Performance & Explainability
# =============================================================================
with tab_explain:
    st.markdown("### 🏆 Model Architecture & Benchmark Comparisons")
    st.markdown("Detailed breakdown of model selection, evaluation metrics, and feature importance analysis based on the training on 70,000 patient records.")

    # High-level baseline metrics row (at default 0.50 threshold)
    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    with bcol1:
        st.metric("Model Architecture", "Random Forest", "Best Performing")
    with bcol2:
        st.metric("Test Accuracy (0.50)", f"{model_metrics.get('accuracy', 74.09)}%", "+0.32% vs DT")
    with bcol3:
        st.metric("ROC-AUC Score", f"{model_metrics.get('roc_auc', 0.8085)}", "Excellent AUC")
    with bcol4:
        st.metric("Precision (0.50)", f"{round(model_metrics.get('precision', 0.7756)*100, 2)}%", "CVD Class")
    with bcol5:
        st.metric("Recall (0.50)", f"{round(model_metrics.get('recall', 0.6703)*100, 2)}%", "CVD Class")

    # Live-updating threshold calibration stat block
    th_round = round(decision_threshold, 2)
    cal_acc, cal_prec, cal_rec, cal_spec, cal_tp, cal_fp, cal_tn, cal_fn = THRESHOLD_METRICS.get(
        th_round, (74.09, 0.7756, 0.6703, 0.8101, 4560, 1319, 5625, 2243)
    )

    rec_delta = (cal_rec - 0.6703) * 100
    prec_delta = (cal_prec - 0.7756) * 100
    acc_delta = cal_acc - 74.09
    fn_delta = cal_fn - 2243

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### ⚙️ Live Threshold Sensitivity Calibration (`{decision_threshold:.2f}` vs Baseline `0.50`)")
    st.caption("Real-time impact on the 13,747 test patients as you adjust the '⚙️ 6. Model Sensitivity' slider.")

    lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
    with lcol1:
        st.metric(
            label="Calibrated Sensitivity",
            value=f"{cal_rec * 100:.1f}%",
            delta=f"{rec_delta:+.1f}% vs 0.50" if abs(rec_delta) >= 0.05 else "Baseline",
            help="Recall (Sensitivity): Proportion of actual CVD patients correctly detected."
        )
    with lcol2:
        st.metric(
            label="Calibrated Precision",
            value=f"{cal_prec * 100:.1f}%",
            delta=f"{prec_delta:+.1f}% vs 0.50" if abs(prec_delta) >= 0.05 else "Baseline",
            help="Precision: Proportion of flagged patients who truly have cardiovascular disease."
        )
    with lcol3:
        st.metric(
            label="Specificity",
            value=f"{cal_spec * 100:.1f}%",
            delta=f"{(cal_spec - 0.8101)*100:+.1f}% vs 0.50" if abs(cal_spec - 0.8101) >= 0.0005 else "Baseline",
            help="Proportion of healthy patients correctly identified."
        )
    with lcol4:
        st.metric(
            label="Test Accuracy",
            value=f"{cal_acc:.2f}%",
            delta=f"{acc_delta:+.2f}% vs 0.50" if abs(acc_delta) >= 0.01 else "Baseline"
        )
    with lcol5:
        st.metric(
            label="Missed Cases (FN)",
            value=f"{cal_fn:,}",
            delta=f"{fn_delta:+d} cases" if fn_delta != 0 else "Baseline",
            delta_color="inverse",
            help="Clinical False Negatives: CVD patients who were missed at this threshold."
        )

    if decision_threshold <= 0.38:
        st.success(f"🎯 **High-Sensitivity Screening Active (Threshold {decision_threshold:.2f})**: Sensitivity rises to **{cal_rec*100:.1f}%** ({cal_tp:,} detected cases). Missed cases drop from 2,243 to **{cal_fn:,}** ({2243 - cal_fn:,} fewer missed CVD patients!), fulfilling the clinical triage recommendations in the project case study.")
    elif decision_threshold > 0.50:
        st.warning(f"🛡️ **High-Specificity Protocol Active (Threshold {decision_threshold:.2f})**: Precision increases to **{cal_prec*100:.1f}%** with fewer false alarms ({cal_fp:,} FP), but missed CVD cases (FN) rise to **{cal_fn:,}**.")
    else:
        st.info("⚖️ **Standard Decision Threshold (0.50)**: Standard balanced trade-off between diagnostic sensitivity (67.03%) and precision (77.56%).")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1], gap="large")

    with c1:
        st.markdown("#### 🥇 Algorithm Accuracy Comparison")
        benchmarks = model_metrics.get("benchmark_comparison", [
            {"model": "Random Forest", "accuracy": 74.09, "status": "Best Model 🏆"},
            {"model": "Decision Tree", "accuracy": 73.77, "status": "Runner Up"},
            {"model": "Logistic Regression", "accuracy": 73.37, "status": "Baseline"},
            {"model": "SVM (LinearSVC)", "accuracy": 73.24, "status": "Baseline"},
            {"model": "KNN (k=11)", "accuracy": 72.40, "status": "Baseline"}
        ])
        b_df = pd.DataFrame(benchmarks)

        # Altair Horizontal Bar Chart
        chart_bars = alt.Chart(b_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=24).encode(
            x=alt.X('accuracy:Q', title="Validation Accuracy (%)", scale=alt.Scale(domain=[70, 76])),
            y=alt.Y('model:N', title=None, sort='-x'),
            color=alt.condition(
                alt.datum.model == 'Random Forest',
                alt.value('#0284C7'),
                alt.value(theme['chart_bar_bg'])
            ),
            tooltip=['model', 'accuracy', 'status']
        ).properties(height=260)

        chart_text = chart_bars.mark_text(
            align='left',
            baseline='middle',
            dx=6,
            color=theme['chart_text'],
            fontWeight=700
        ).encode(
            text=alt.Text('accuracy:Q', format='.2f')
        )

        st.altair_chart(chart_bars + chart_text, use_container_width=True)

        st.info("💡 **Clinical Observation**: The ensemble mechanism of Random Forest successfully reduces variance across individual noisy biometric measurements (e.g. episodic blood pressure spikes), yielding the highest generalizability.")

    with c2:
        st.markdown(f"#### 🎯 Live Confusion Matrix (Threshold {decision_threshold:.2f})")
        cm_df = pd.DataFrame(
            [[cal_tn, cal_fp], [cal_fn, cal_tp]],
            columns=["Predicted No CVD", "Predicted CVD"],
            index=["Actual No CVD", "Actual CVD"]
        )

        st.dataframe(cm_df, use_container_width=True)

        st.markdown(f"""
        - **True Negatives (TN)**: **{cal_tn:,}** (Healthy correctly identified)
        - **True Positives (TP)**: **{cal_tp:,}** (CVD correctly detected)
        - **False Positives (FP)**: **{cal_fp:,}** (Screening false alarms)
        - **False Negatives (FN)**: **{cal_fn:,}** (Missed cases — clinical hazard)
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔑 Feature Importance Breakdown")
    st.caption("Gini-importance scores computed by the trained Random Forest ensemble showing which features drive predictions most.")

    feat_importances = model_metrics.get("feature_importances", {
        "ap_hi": 0.3806,
        "ap_lo": 0.1926,
        "pulse_pressure": 0.1417,
        "age_years": 0.1053,
        "cholesterol": 0.0866,
        "bmi": 0.0410,
        "weight": 0.0165,
        "height": 0.0125,
        "gluc": 0.0095,
        "active": 0.0049,
        "gender": 0.0041,
        "smoke": 0.0035,
        "alco": 0.0030
    })

    feat_names_readable = {
        "ap_hi": "Systolic BP (ap_hi)",
        "ap_lo": "Diastolic BP (ap_lo)",
        "pulse_pressure": "Pulse Pressure (Engineered)",
        "age_years": "Patient Age (Years)",
        "cholesterol": "Serum Cholesterol",
        "bmi": "Body Mass Index (BMI)",
        "weight": "Body Weight (kg)",
        "height": "Height (cm)",
        "gluc": "Fasting Glucose",
        "active": "Physical Activity",
        "gender": "Biological Sex",
        "smoke": "Tobacco Smoking",
        "alco": "Alcohol Intake"
    }

    fi_data = [
        {"Feature": feat_names_readable.get(k, k), "Code": k, "Importance": round(v * 100, 2)}
        for k, v in feat_importances.items()
    ]
    fi_df = pd.DataFrame(fi_data).sort_values("Importance", ascending=False)

    fi_chart = alt.Chart(fi_df).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, size=18).encode(
        x=alt.X('Importance:Q', title="Feature Importance (%)"),
        y=alt.Y('Feature:N', sort='-x', title=None),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Feature', 'Importance']
    ).properties(height=380)

    st.altair_chart(fi_chart, use_container_width=True)

# =============================================================================
# TAB 3: Batch Screening Mode
# =============================================================================
with tab_batch:
    st.markdown("### 📁 Multi-Patient Cohort Batch Screening")
    st.markdown("Upload a CSV cohort file to process cardiovascular screening for dozens or thousands of patients simultaneously.")

    # Sample CSV Download Template
    sample_patients = pd.DataFrame([
        {"age_years": 54, "gender": 1, "height": 162, "weight": 68.0, "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age_years": 62, "gender": 2, "height": 178, "weight": 92.0, "ap_hi": 160, "ap_lo": 100, "cholesterol": 3, "gluc": 2, "smoke": 1, "alco": 1, "active": 0},
        {"age_years": 45, "gender": 1, "height": 165, "weight": 55.0, "ap_hi": 110, "ap_lo": 70, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1},
        {"age_years": 58, "gender": 2, "height": 172, "weight": 85.0, "ap_hi": 145, "ap_lo": 90, "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 0},
        {"age_years": 38, "gender": 2, "height": 180, "weight": 76.0, "ap_hi": 125, "ap_lo": 82, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
    ])

    sample_csv = sample_patients.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Sample Batch CSV Template",
        data=sample_csv,
        file_name="cardio_cohort_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded cohort with **{len(batch_df)} patients**.")

            # Check / adapt columns
            processed_df = batch_df.copy()
            if 'age' in processed_df.columns and 'age_years' not in processed_df.columns:
                processed_df['age_years'] = (processed_df['age'] / 365.0).round(1)

            if 'bmi' not in processed_df.columns:
                processed_df['bmi'] = (processed_df['weight'] / ((processed_df['height'] / 100.0) ** 2)).round(2)

            if 'pulse_pressure' not in processed_df.columns:
                processed_df['pulse_pressure'] = processed_df['ap_hi'] - processed_df['ap_lo']

            # Required feature order
            feature_cols = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi', 'pulse_pressure']
            missing_cols = [c for c in feature_cols if c not in processed_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                X_batch = processed_df[feature_cols]
                probs = pipeline.predict_proba(X_batch)[:, 1]
                preds = (probs >= decision_threshold).astype(int)

                processed_df['CVD_Probability_%'] = (probs * 100).round(1)
                processed_df['Predicted_CVD'] = preds
                processed_df['Risk_Stratification'] = np.where(
                    probs < 0.30, "Low Risk",
                    np.where(probs < 0.60, "Moderate Risk", "High Risk")
                )

                # Cohort Distribution
                scol1, scol2, scol3 = st.columns(3)
                low_cnt = (processed_df['Risk_Stratification'] == "Low Risk").sum()
                mod_cnt = (processed_df['Risk_Stratification'] == "Moderate Risk").sum()
                high_cnt = (processed_df['Risk_Stratification'] == "High Risk").sum()

                with scol1:
                    st.metric("Low Risk Patients (<30%)", f"{low_cnt}", f"{round(low_cnt/len(processed_df)*100, 1)}%")
                with scol2:
                    st.metric("Moderate Risk (30-60%)", f"{mod_cnt}", f"{round(mod_cnt/len(processed_df)*100, 1)}%")
                with scol3:
                    st.metric("High Risk Patients (>60%)", f"{high_cnt}", f"{round(high_cnt/len(processed_df)*100, 1)}%")

                st.markdown("#### Cohort Results Preview")
                st.dataframe(processed_df, use_container_width=True)

                out_csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Cohort Results (CSV)",
                    data=out_csv,
                    file_name="cardio_cohort_screened_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {str(e)}")

# =============================================================================
# TAB 4: Clinical Guidelines & Reference
# =============================================================================
with tab_guidelines:
    st.markdown("### 📖 Clinical Frameworks & Diagnostic Thresholds")

    gcol1, gcol2 = st.columns(2, gap="large")

    with gcol1:
        st.markdown("#### 🩸 American Heart Association (AHA) BP Staging")
        bp_guide_data = [
            {"Category": "Normal", "Systolic (mmHg)": "< 120", "Diastolic (mmHg)": "and < 80"},
            {"Category": "Elevated", "Systolic (mmHg)": "120 - 129", "Diastolic (mmHg)": "and < 80"},
            {"Category": "Hypertension Stage 1", "Systolic (mmHg)": "130 - 139", "Diastolic (mmHg)": "or 80 - 89"},
            {"Category": "Hypertension Stage 2", "Systolic (mmHg)": "≥ 140", "Diastolic (mmHg)": "or ≥ 90"},
            {"Category": "Hypertensive Crisis", "Systolic (mmHg)": "> 180", "Diastolic (mmHg)": "and/or > 120"}
        ]
        st.dataframe(pd.DataFrame(bp_guide_data), use_container_width=True, hide_index=True)

        st.markdown("#### ⚡ Pulse Pressure Clinical Importance")
        st.markdown("""
        **Pulse Pressure** = $\\text{Systolic BP} - \\text{Diastolic BP}$.
        - A sustained pulse pressure greater than **60 mmHg** in older adults is a recognized clinical marker of large artery stiffness and has been shown to be an independent predictor of adverse cardiovascular events.
        """)

    with gcol2:
        st.markdown("#### ⚖️ World Health Organization (WHO) BMI Classifications")
        bmi_guide_data = [
            {"Classification": "Underweight", "BMI Range (kg/m²)": "< 18.5"},
            {"Classification": "Normal Weight", "BMI Range (kg/m²)": "18.5 - 24.9"},
            {"Classification": "Overweight", "BMI Range (kg/m²)": "25.0 - 29.9"},
            {"Classification": "Obesity Class I", "BMI Range (kg/m²)": "30.0 - 34.9"},
            {"Classification": "Obesity Class II", "BMI Range (kg/m²)": "35.0 - 39.9"},
            {"Classification": "Obesity Class III", "BMI Range (kg/m²)": "≥ 40.0"}
        ]
        st.dataframe(pd.DataFrame(bmi_guide_data), use_container_width=True, hide_index=True)

        st.markdown("#### 🧪 Laboratory Encoding Dictionary")
        st.markdown("""
        - **Cholesterol**: 1: Normal, 2: Above Normal, 3: Well Above Normal
        - **Glucose**: 1: Normal, 2: Above Normal, 3: Well Above Normal
        - **Gender**: 1: Female, 2: Male
        - **Lifestyle**: 0: No / Inactive, 1: Yes / Active
        """)

    st.markdown("---")
    st.caption("🛡️ **Medical Disclaimer**: CardioGuard AI is designed as a clinical decision-support and educational tool. It is not an automated medical diagnostic device. Final medical judgments should always be synthesized with clinical history, laboratory workups, and physician examination.")
