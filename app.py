#Multi-Disease Prediction System 

from __future__ import annotations
import base64
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent


# PREPROCESSING raw data
 
_KIDNEY_BINARY_TARGET_COL = {
    "pc": "pc_abnormal", "pcc": "pcc_present", "ba": "ba_present",
    "htn": "htn_yes", "dm": "dm_yes", "cad": "cad_yes",
    "appet": "appet_good", "pe": "pe_yes", "ane": "ane_yes",
}


def preprocess_liver(raw: dict, artifact: dict) -> pd.DataFrame:
    row = dict(raw)
    ratio = row.get("Albumin_and_Globulin_Ratio")
    if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
        globulin = row["Total_Protiens"] - row["Albumin"]
        row["Albumin_and_Globulin_Ratio"] = row["Albumin"] / globulin if globulin else np.nan

    row["Gender_Male"] = 1 if row.get("Gender") == "Male" else 0

    for col in artifact["skewed_cols"]:
        row[col] = np.log1p(row[col])

    return pd.DataFrame([row])[artifact["feature_columns"]]


def preprocess_kidney(raw: dict, artifact: dict) -> pd.DataFrame:
    row = dict(raw)

    for raw_col, target_col in _KIDNEY_BINARY_TARGET_COL.items():
        mapping = artifact["binary_value_maps"][raw_col]
        val = row.pop(raw_col, None)
        row[target_col] = mapping.get(val, np.nan) if val is not None else np.nan

    bounds = artifact["sod_pot_outlier_bounds"]
    if row.get("sod") is not None and row["sod"] < bounds["sod_min"]:
        row["sod"] = np.nan
    if row.get("pot") is not None and row["pot"] > bounds["pot_max"]:
        row["pot"] = np.nan

    for col in artifact["skewed_cols"]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[col] = np.log1p(val)

    def _is_missing(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    for col in artifact["cont_cols"]:
        if _is_missing(row.get(col)):
            row[col] = artifact["impute_medians"][col]
    for col in artifact["binary_cols"]:
        if _is_missing(row.get(col)):
            row[col] = artifact["impute_modes"][col]

    return pd.DataFrame([row])[artifact["feature_columns"]]


def preprocess_parkinsons(raw: dict, artifact: dict) -> pd.DataFrame:
    return pd.DataFrame([raw])[artifact["feature_columns"]]


_PREPROCESSORS = {
    "liver": preprocess_liver,
    "kidney": preprocess_kidney,
    "parkinsons": preprocess_parkinsons,
}


# MODEL LOADING + PREDICTION

_ARTIFACT_FILES = {
    "liver": "liver_model.joblib",
    "kidney": "kidney_model.joblib",
    "parkinsons": "parkinsons_model.joblib",
}


@st.cache_resource(show_spinner=False)
def load_artifact(disease: str) -> dict:
    path = MODELS_DIR / _ARTIFACT_FILES[disease]
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


_POSITIVE_LABEL = {
    "liver": "Liver disease indicators present",
    "kidney": "Chronic kidney disease indicators present",
    "parkinsons": "Parkinson's indicators present",
}
_NEGATIVE_LABEL = {
    "liver": "No liver disease indicators",
    "kidney": "No chronic kidney disease indicators",
    "parkinsons": "No Parkinson's indicators",
}


@dataclass
class PredictionResult:
    probability: float
    is_positive: bool
    label: str
    risk_tier: str
    model_name: str


def _risk_tier(p: float) -> str:
    if p < 0.35:
        return "Low"
    if p < 0.65:
        return "Moderate"
    return "High"


def predict(disease: str, artifact: dict, raw_inputs: dict) -> PredictionResult:
    X = _PREPROCESSORS[disease](raw_inputs, artifact)
    X_scaled = artifact["scaler"].transform(X)
    proba = float(artifact["model"].predict_proba(X_scaled)[0, 1])
    is_positive = proba >= 0.5
    label = _POSITIVE_LABEL[disease] if is_positive else _NEGATIVE_LABEL[disease]
    return PredictionResult(
        probability=proba,
        is_positive=is_positive,
        label=label,
        risk_tier=_risk_tier(proba),
        model_name=artifact.get("model_name", "model"),
    )



ICONS = {"home": "icons/medical.png",
        "liver": "icons/liver.png", 
         "kidney": "icons/kidney.png", 
         "parkinsons": "icons/parkinsons.png"}


@st.cache_data(show_spinner=False)
def _image_data_uri(path: Path) -> str | None:
    """Reads a local image file into a base64 data URI so it can be embedded
    directly inside a raw HTML <img> tag (a plain relative src="icons/x.png"
    won't resolve in the browser, since Streamlit doesn't serve that folder
    over HTTP). Returns None if the file doesn't exist yet, so the card just
    renders without an icon instead of crashing the page."""
    if not path.exists():
        return None
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{encoded}"


# STYLE — design tokens, sidebar, hover cards, result panel

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --bg: #F4F6F8; --surface: #FFFFFF; --ink: #101820; --ink-soft: #55636F; --line: #E3E8EC;
  --sidebar-bg: #0F1B2D; --sidebar-text: #A9B7C4; --sidebar-text-active: #FFFFFF; --sidebar-hover: #16283F;
  --liver: #B5652D; --kidney: #1E6E6A; --parkinsons: #6B4A85;
  --risk-low: #2F8F5B; --risk-moderate: #B8871E; --risk-high: #B23B3B;
  --accent: #0F1B2D;
}
html, body, [class*="css"]{ font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp{ background: var(--bg); }
h1, h2, h3, .brand{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
#MainMenu, footer, header[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1080px; }

[data-testid="stSidebar"]{ background: var(--sidebar-bg); }
[data-testid="stSidebar"] *{ color: var(--sidebar-text); }
.sidebar-brand{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem;
  color: #fff !important; padding: 0.4rem 0 0.2rem 0; line-height: 1.3; }
.sidebar-brand span{ color: #7C8CA0; font-weight: 500; font-size: 0.78rem; display:block; }
[data-testid="stSidebar"] hr{ border-color: #223349; margin: 0.9rem 0; }
[data-testid="stSidebar"] [data-testid="stButton"] button{
  background: transparent; border: 1px solid transparent; color: var(--sidebar-text);
  text-align: left; justify-content: flex-start; width: 100%; font-weight: 500;
  border-radius: 8px; padding: 0.55rem 0.8rem; transition: background 0.15s ease, color 0.15s ease; }
[data-testid="stSidebar"] [data-testid="stButton"] button:hover{
  background: var(--sidebar-hover); color: var(--sidebar-text-active); border-color: transparent; }
[data-testid="stSidebar"] [data-testid="stButton"] button:focus{ box-shadow: none; }
[data-testid="stSidebar"] [data-testid="stButton"] button p{ font-size: 0.92rem; }
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{
  background: var(--sidebar-hover); color: #fff; border-left: 3px solid var(--accent); font-weight: 600; }

.hero-eyebrow{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--ink-soft);
  text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3rem; }
.hero-title{ font-size: 2.1rem; font-weight: 700; margin: 0 0 0.4rem 0; color: var(--ink); }
.hero-sub{ color: var(--ink-soft); font-size: 1rem; margin: 0 0 1.6rem 0; line-height: 1.5; }

.disease-stack{ display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; }
a.disease-card{ position: relative; display: flex; align-items: center; gap: 1.3rem;
  background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.1rem 1.5rem; text-decoration: none; color: var(--ink);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease; }
a.disease-card:hover{ transform: translateY(-4px); box-shadow: 0 18px 32px -18px rgba(16,24,32,0.28); border-color: var(--card-accent); }
.disease-card-icon{ flex-shrink: 0; width: 72px; height: 72px; border-radius: 14px;
  background: color-mix(in srgb, var(--card-accent) 12%, white);
  display: flex; align-items: center; justify-content: center; overflow: hidden;
  transition: transform 0.18s ease; }
a.disease-card:hover .disease-card-icon{ transform: scale(1.06); }
.disease-card-icon img{ width: 100%; height: 100%; object-fit: contain; padding: 12px; }
.disease-card-body{ flex: 1 1 auto; min-width: 0; }
.disease-card .tag{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
  color: var(--card-accent); background: color-mix(in srgb, var(--card-accent) 12%, white);
  padding: 0.2rem 0.55rem; border-radius: 999px; margin-bottom: 0.5rem; }
.disease-card h3{ margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.disease-card p{ color: var(--ink-soft); font-size: 0.86rem; line-height: 1.4; margin: 0; }
.disease-card .go{ flex-shrink: 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
  font-weight: 600; color: var(--card-accent); white-space: nowrap; }
.card-liver{ --card-accent: var(--liver); } .card-kidney{ --card-accent: var(--kidney); } .card-parkinsons{ --card-accent: var(--parkinsons); }
@media (max-width: 520px){
  a.disease-card{ flex-wrap: wrap; }
  .disease-card .go{ width: 100%; text-align: right; }
}

.page-header{ display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.3rem; }
.page-header .icon-badge{ width: 46px; height: 46px; border-radius: 12px;
  background: color-mix(in srgb, var(--accent) 14%, white); color: var(--accent);
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.page-header .icon-badge svg{ width: 26px; height: 26px; }
.page-header h1{ font-size: 1.5rem; margin: 0; }
.page-sub{ color: var(--ink-soft); font-size: 0.92rem; margin: 0.2rem 0 1.6rem 0; }

.section-label{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--accent); border-bottom: 1px solid var(--line);
  padding-bottom: 0.4rem; margin: 1.3rem 0 0.9rem 0; }

[data-testid="stForm"] [data-testid="stButton"] button[kind="primary"], .stButton button[kind="primary"]{
  background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; border-radius: 9px; }
.stButton button[kind="primary"]:hover{ filter: brightness(1.08); }

.result-card{ background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.5rem 1.6rem; margin-top: 0.6rem; }
.result-top{ display:flex; align-items:center; justify-content:space-between; gap: 1rem; flex-wrap: wrap; }
.result-verdict{ font-size: 1.15rem; font-weight: 700; }
.risk-badge{ font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600;
  padding: 0.28rem 0.7rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.04em; }
.risk-low{ color: var(--risk-low); background: color-mix(in srgb, var(--risk-low) 14%, white); }
.risk-moderate{ color: var(--risk-moderate); background: color-mix(in srgb, var(--risk-moderate) 16%, white); }
.risk-high{ color: var(--risk-high); background: color-mix(in srgb, var(--risk-high) 14%, white); }
.prob-row{ margin-top: 1rem; }
.prob-label{ display:flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--ink-soft); margin-bottom: 0.3rem; }
.prob-track{ height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; }
.prob-fill{ height: 100%; border-radius: 999px; }
.result-foot{ margin-top: 1rem; font-size: 0.78rem; color: var(--ink-soft); border-top: 1px solid var(--line); padding-top: 0.7rem; }
.disclaimer{ font-size: 0.78rem; color: var(--ink-soft); background: color-mix(in srgb, var(--accent) 6%, white);
  border: 1px solid var(--line); border-radius: 10px; padding: 0.7rem 0.9rem; margin-top: 1.4rem; }
"""


def load_css(accent: str | None = None) -> None:
    override = f":root{{ --accent: {accent}; }}" if accent else ""
    st.markdown(f"<style>{BASE_CSS}\n{override}</style>", unsafe_allow_html=True)


# UI 

PAGES = [
    ("home", "Home"),
    ("liver", "Liver Disease Prediction"),
    ("kidney", "Kidney Disease Prediction"),
    ("parkinsons", "Parkinson's Prediction"),
]
ACCENTS = {"liver": "#B5652D", "kidney": "#1E6E6A", "parkinsons": "#6B4A85"}

DISEASE_META = {
    "liver": {
        "title": "Liver Disease Prediction",
        "tag": "Hepatic panel",
        "blurb": "Estimate liver disease likelihood from a standard blood chemistry panel — bilirubin, liver enzymes, and protein levels.",
        "sub": "Enter a patient's liver function panel to estimate the likelihood of liver disease.",
    },
    "kidney": {
        "title": "Kidney Disease Prediction",
        "tag": "Renal panel",
        "blurb": "Estimate chronic kidney disease likelihood from blood, urine, and clinical history fields.",
        "sub": "Enter a patient's blood, urine, and clinical history values to estimate CKD likelihood.",
    },
    "parkinsons": {
        "title": "Parkinson's Prediction",
        "tag": "Acoustic voice analysis",
        "blurb": "Estimate Parkinson's likelihood from acoustic measures of sustained vowel phonation.",
        "sub": "Enter acoustic voice-recording measures to estimate Parkinson's likelihood.",
    },
}


def get_current_page() -> str:
    valid = {key for key, _ in PAGES}
    page = st.query_params.get("page", "home")
    return page if page in valid else "home"


def go_to(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def render_sidebar(current: str) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">Multi-Disease Prediction'
            '<span>Clinical risk screening</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)
        for key, label in PAGES:
            btn_type = "primary" if key == current else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                go_to(key)


def render_page_header(disease: str) -> None:
    meta = DISEASE_META[disease]

    col1, col2 = st.columns([1, 6], vertical_alignment="center")

    with col1:
        icon_path = MODELS_DIR / ICONS[disease]
        if icon_path.exists():
            st.image(str(icon_path), width=100)

    with col2:
        st.markdown(
            f"""
            <h1>{meta['title']}</h1>
            <p class="page-sub">{meta['sub']}</p>
            """,
        unsafe_allow_html=True,
    )


def render_result(disease: str, result: PredictionResult) -> None:
    accent = ACCENTS[disease]
    risk_class = f"risk-{result.risk_tier.lower()}"
    pct = round(result.probability * 100, 1)
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-top">
            <div class="result-verdict">{result.label}</div>
            <span class="risk-badge {risk_class}">{result.risk_tier} risk</span>
          </div>
          <div class="prob-row">
            <div class="prob-label"><span>Predicted probability</span><span>{pct}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:{pct}%; background:{accent};"></div></div>
          </div>
          <div class="result-foot">Model: {result.model_name} · Threshold: 50% probability</div>
        </div>
        <div class="disclaimer">
          This tool estimates statistical likelihood from historical data. It is not a
          medical diagnosis — please consult a qualified clinician for interpretation
          and next steps.
        </div>
        """,
        unsafe_allow_html=True,
    )


# PAGES

def render_home() -> None:
    icon_path = MODELS_DIR / ICONS["home"]
    col1, col2 = st.columns([1, 6], vertical_alignment="center")
    with col1:
        if icon_path.exists():
            st.image(str(icon_path), width=100)
    with col2:
        st.markdown(
            '<h1 class="hero-title">Multi-Disease Prediction System</h1>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p class="hero-sub">
            Three independently trained models — liver, kidney, and Parkinson's — 
            each evaluated across Logistic Regression, Random Forest, and XGBoost,
            with the strongest performer deployed per disease.
        </p>
        """,
        unsafe_allow_html=True,
    )

    cards_html = '<div class="disease-stack">'
    for key in ("liver", "kidney", "parkinsons"):
        meta = DISEASE_META[key]
        icon_uri = _image_data_uri(MODELS_DIR / ICONS[key])
        icon_html = f'<img src="{icon_uri}" alt=""/>' if icon_uri else ""
        cards_html += (
            f'<a class="disease-card card-{key}" href="?page={key}" target="_self">'
            f'<div class="disease-card-icon">{icon_html}</div>'
            f'<div class="disease-card-body">'
            f'<span class="tag">{meta["tag"]}</span>'
            f'<h3>{meta["title"]}</h3>'
            f'<p>{meta["blurb"]}</p>'
            f'</div>'
            f'<span class="go"> &rarr;</span>'
            f'</a>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)



def render_liver() -> None:
    load_css(ACCENTS["liver"])
    render_page_header("liver")
    artifact = load_artifact("liver")

    with st.form("liver_form"):
        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=1, max_value=100, value=45)
        gender = c2.selectbox("Gender", ["Male", "Female"])

        st.markdown('<div class="section-label">Liver Function Panel</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        total_bilirubin = c1.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=80.0, value=1.0, step=0.1)
        direct_bilirubin = c2.number_input("Direct Bilirubin (mg/dL)", min_value=0.1, max_value=20.0, value=0.3, step=0.1)
        alk_phos = c3.number_input("Alkaline Phosphotase (IU/L)", min_value=60, max_value=2200, value=210)
        c1, c2 = st.columns(2)
        alt = c1.number_input("Alamine Aminotransferase — ALT (IU/L)", min_value=5, max_value=2100, value=35)
        ast = c2.number_input("Aspartate Aminotransferase — AST (IU/L)", min_value=5, max_value=5000, value=42)

        st.markdown('<div class="section-label">Protein Panel</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        total_proteins = c1.number_input("Total Proteins (g/dL)", min_value=2.0, max_value=10.0, value=6.6, step=0.1)
        albumin = c2.number_input("Albumin (g/dL)", min_value=0.5, max_value=6.0, value=3.1, step=0.1)
        auto_ratio = c3.checkbox("Auto-calculate A/G ratio", value=True)
        ag_ratio = None
        if not auto_ratio:
            ag_ratio = c3.number_input("Albumin/Globulin Ratio", min_value=0.1, max_value=3.0, value=0.9, step=0.1)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = {
            "Age": age, "Gender": gender, "Total_Bilirubin": total_bilirubin,
            "Direct_Bilirubin": direct_bilirubin, "Alkaline_Phosphotase": alk_phos,
            "Alamine_Aminotransferase": alt, "Aspartate_Aminotransferase": ast,
            "Total_Protiens": total_proteins, "Albumin": albumin,
            "Albumin_and_Globulin_Ratio": ag_ratio,
        }
        render_result("liver", predict("liver", artifact, raw))


def render_kidney() -> None:
    load_css(ACCENTS["kidney"])
    render_page_header("kidney")
    artifact = load_artifact("kidney")
    YES_NO = ["no", "yes"]

    with st.form("kidney_form"):
        st.markdown('<div class="section-label">Vitals & Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=2, max_value=100, value=55)
        bp = c2.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=180, value=80)

        st.markdown('<div class="section-label">Urinalysis</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        sg = c1.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
        al = c2.selectbox("Albumin (urine, 0–5 scale)", [0, 1, 2, 3, 4, 5], index=0)
        su = c3.selectbox("Sugar (urine, 0–5 scale)", [0, 1, 2, 3, 4, 5], index=0)
        c1, c2, c3 = st.columns(3)
        pc = c1.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = c2.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = c3.selectbox("Bacteria", ["notpresent", "present"])

        st.markdown('<div class="section-label">Blood Chemistry</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        bgr = c1.number_input("Blood Glucose Random (mg/dL)", min_value=20, max_value=500, value=121)
        bu = c2.number_input("Blood Urea (mg/dL)", min_value=1.0, max_value=400.0, value=42.0, step=0.5)
        sc = c3.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=80.0, value=1.3, step=0.1)
        c1, c2, c3 = st.columns(3)
        sod = c1.number_input("Sodium (mEq/L)", min_value=50.0, max_value=180.0, value=138.0, step=0.5)
        pot = c2.number_input("Potassium (mEq/L)", min_value=1.5, max_value=15.0, value=4.4, step=0.1)
        hemo = c3.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=18.0, value=12.6, step=0.1)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=9, max_value=54, value=40)

        st.markdown('<div class="section-label">Clinical History</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        htn = c1.selectbox("Hypertension", YES_NO)
        dm = c2.selectbox("Diabetes Mellitus", YES_NO)
        cad = c3.selectbox("Coronary Artery Disease", YES_NO)
        c1, c2, c3 = st.columns(3)
        appet = c1.selectbox("Appetite", ["good", "poor"])
        pe = c2.selectbox("Pedal Edema", YES_NO)
        ane = c3.selectbox("Anemia", YES_NO)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
            "hemo": hemo, "pcv": pcv,
            "pc": pc, "pcc": pcc, "ba": ba, "htn": htn, "dm": dm,
            "cad": cad, "appet": appet, "pe": pe, "ane": ane,
        }
        render_result("kidney", predict("kidney", artifact, raw))


def _num(col, label, key, min_v, max_v, default, step, fmt=None):
    kwargs = dict(min_value=min_v, max_value=max_v, value=default, step=step)
    if fmt:
        kwargs["format"] = fmt
    return key, col.number_input(label, **kwargs)


def render_parkinsons() -> None:
    load_css(ACCENTS["parkinsons"])
    render_page_header("parkinsons")
    st.caption(
        "Values come from acoustic analysis of a sustained vowel recording "
        "(e.g. via Praat or MDVP software) rather than a manual exam."
    )
    artifact = load_artifact("parkinsons")
    raw = {}

    with st.form("parkinsons_form"):
        st.markdown('<div class="section-label">Fundamental Frequency</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "MDVP:Fo (Hz) — average", "MDVP:Fo(Hz)", 80.0, 270.0, 148.79, 0.1),
            _num(c2, "MDVP:Fhi (Hz) — max", "MDVP:Fhi(Hz)", 100.0, 600.0, 175.83, 0.1),
            _num(c3, "MDVP:Flo (Hz) — min", "MDVP:Flo(Hz)", 60.0, 245.0, 104.32, 0.1),
        ]))

        st.markdown('<div class="section-label">Jitter (frequency perturbation)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "MDVP:Jitter(%)", "MDVP:Jitter(%)", 0.0, 0.035, 0.00494, 0.0001, "%.5f"),
            _num(c2, "MDVP:Jitter(Abs)", "MDVP:Jitter(Abs)", 0.0, 0.0003, 0.00003, 0.00001, "%.6f"),
            _num(c3, "MDVP:RAP", "MDVP:RAP", 0.0, 0.022, 0.00250, 0.0001, "%.5f"),
        ]))
        c1, c2 = st.columns(2)
        raw.update(dict([
            _num(c1, "MDVP:PPQ", "MDVP:PPQ", 0.0, 0.020, 0.00269, 0.0001, "%.5f"),
            _num(c2, "Jitter:DDP", "Jitter:DDP", 0.0, 0.065, 0.00749, 0.0001, "%.5f"),
        ]))

        st.markdown('<div class="section-label">Shimmer (amplitude perturbation)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "MDVP:Shimmer", "MDVP:Shimmer", 0.0, 0.12, 0.02297, 0.001, "%.5f"),
            _num(c2, "MDVP:Shimmer(dB)", "MDVP:Shimmer(dB)", 0.0, 1.4, 0.221, 0.01, "%.3f"),
            _num(c3, "Shimmer:APQ3", "Shimmer:APQ3", 0.0, 0.06, 0.01279, 0.001, "%.5f"),
        ]))
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "Shimmer:APQ5", "Shimmer:APQ5", 0.0, 0.08, 0.01347, 0.001, "%.5f"),
            _num(c2, "MDVP:APQ", "MDVP:APQ", 0.0, 0.14, 0.01826, 0.001, "%.5f"),
            _num(c3, "Shimmer:DDA", "Shimmer:DDA", 0.0, 0.17, 0.03836, 0.001, "%.5f"),
        ]))

        st.markdown('<div class="section-label">Noise & Nonlinear Dynamics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "NHR", "NHR", 0.0, 0.32, 0.01166, 0.001, "%.5f"),
            _num(c2, "HNR", "HNR", 5.0, 35.0, 22.09, 0.1),
            _num(c3, "RPDE", "RPDE", 0.2, 0.7, 0.4960, 0.001, "%.4f"),
        ]))
        c1, c2, c3 = st.columns(3)
        raw.update(dict([
            _num(c1, "DFA", "DFA", 0.5, 0.85, 0.7223, 0.001, "%.4f"),
            _num(c2, "spread1", "spread1", -8.0, -2.0, -5.72, 0.01),
            _num(c3, "spread2", "spread2", 0.0, 0.5, 0.2190, 0.001, "%.4f"),
        ]))
        c1, c2 = st.columns(2)
        raw.update(dict([
            _num(c1, "D2", "D2", 1.0, 4.0, 2.362, 0.01),
            _num(c2, "PPE", "PPE", 0.0, 0.55, 0.1940, 0.001, "%.4f"),
        ]))

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        render_result("parkinsons", predict("parkinsons", artifact, raw))


PAGE_RENDERERS = {
    "home": render_home,
    "liver": render_liver,
    "kidney": render_kidney,
    "parkinsons": render_parkinsons,
}


# ENTRY POINT

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_page = get_current_page()
load_css()  
render_sidebar(current_page)
PAGE_RENDERERS[current_page]()