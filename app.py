import os
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Type Predictor",
    page_icon="🌦️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark atmospheric background */
.stApp {
    background: linear-gradient(160deg, #0f0c29, #1a1a3e, #24243e);
    color: #e0e0e0;
}

/* Section card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* Hero title */
.hero {
    text-align: center;
    padding: 30px 0 10px 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a8edea, #fed6e3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero p {
    color: #9ca3af;
    font-size: 1rem;
}

/* Section header */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a8edea;
    margin-bottom: 16px;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 24px 0;
    animation: fadeIn 0.5s ease;
}
.result-card h2 {
    font-size: 2rem;
    font-weight: 700;
    margin: 12px 0 4px 0;
}
.result-card p {
    font-size: 1rem;
    opacity: 0.85;
}
.result-icon {
    font-size: 4rem;
    line-height: 1;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Predict button */
div.stButton > button {
    width: 100%;
    padding: 14px;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #a8edea, #fed6e3);
    color: #1a1a3e;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(168,237,234,0.35);
}

/* Input labels */
label {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
}

/* Selectbox and number input */
.stSelectbox > div, .stNumberInput > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model & Preprocessors ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all():
    m  = load_model(os.path.join(BASE_DIR, "weather_model.keras"))
    s  = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    o  = joblib.load(os.path.join(BASE_DIR, "ohe.pkl"))
    l  = joblib.load(os.path.join(BASE_DIR, "le.pkl"))
    return m, s, o, l

model, scaler, ohe, le = load_all()

# Weather visual config
WEATHER_CONFIG = {
    "Sunny":  {"icon": "☀️",  "gradient": "linear-gradient(135deg, #f6d365, #fda085)", "text": "#7c3000"},
    "Rainy":  {"icon": "🌧️",  "gradient": "linear-gradient(135deg, #4facfe, #00f2fe)", "text": "#00264d"},
    "Cloudy": {"icon": "☁️",  "gradient": "linear-gradient(135deg, #b0bec5, #78909c)", "text": "#1c2c35"},
    "Snowy":  {"icon": "❄️",  "gradient": "linear-gradient(135deg, #e0f7fa, #b2ebf2)", "text": "#003c47"},
}

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌦️ Weather Type Predictor</h1>
    <p>Enter atmospheric conditions to classify the weather using a trained ANN model</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Numeric Inputs ────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌡️ Atmospheric Measurements</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    temperature          = st.number_input("Temperature (°C)",          value=25.0, step=0.5)
    humidity             = st.number_input("Humidity (%)",              value=60.0, step=1.0, min_value=0.0, max_value=100.0)
    wind_speed           = st.number_input("Wind Speed (km/h)",         value=10.0, step=0.5, min_value=0.0)
    precipitation        = st.number_input("Precipitation (%)",         value=5.0,  step=0.5, min_value=0.0)
with col2:
    atmospheric_pressure = st.number_input("Atmospheric Pressure (hPa)",value=1013.0, step=1.0)
    uv_index             = st.number_input("UV Index",                  value=5.0,  step=0.5, min_value=0.0, max_value=11.0)
    visibility           = st.number_input("Visibility (km)",           value=10.0, step=0.5, min_value=0.0)

st.markdown('</div>', unsafe_allow_html=True)

# ── Categorical Inputs ────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🗂️ Conditions & Location</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    cloud_cover = st.selectbox("Cloud Cover",  ["clear", "partly cloudy", "overcast", "cloudy"])
with col4:
    season      = st.selectbox("Season",       ["Spring", "Summer", "Autumn", "Winter"])
with col5:
    location    = st.selectbox("Location",     ["inland", "mountain", "coastal"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Weather Type")

if predict_clicked:
    # 1. Scale numeric features
    numeric_input   = np.array([[temperature, humidity, wind_speed,
                                  precipitation, atmospheric_pressure,
                                  uv_index, visibility]])
    numeric_scaled  = scaler.transform(numeric_input)

    # 2. Encode categorical features
    categorical_input   = np.array([[cloud_cover, season, location]])
    categorical_encoded = ohe.transform(categorical_input)

    # 3. Combine
    final_input = np.concatenate([numeric_scaled, categorical_encoded], axis=1)

    # 4. Predict
    proba           = model.predict(final_input)[0]
    predicted_idx   = np.argmax(proba)
    predicted_label = le.inverse_transform([predicted_idx])[0]
    confidence      = proba[predicted_idx] * 100

    cfg = WEATHER_CONFIG.get(predicted_label, {
        "icon": "🌡️",
        "gradient": "linear-gradient(135deg, #667eea, #764ba2)",
        "text": "#ffffff"
    })

    # ── Result Card ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card" style="background:{cfg['gradient']}; color:{cfg['text']};">
        <div class="result-icon">{cfg['icon']}</div>
        <h2>{predicted_label}</h2>
        <p>Model confidence: <strong>{confidence:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence Gauge ──────────────────────────────────────────────────────
    col_g, col_b = st.columns(2)

    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = confidence,
            title = {"text": "Confidence %", "font": {"color": "#e0e0e0", "size": 14}},
            number= {"suffix": "%", "font": {"color": "#a8edea", "size": 28}},
            gauge = {
                "axis" : {"range": [0, 100], "tickcolor": "#555"},
                "bar"  : {"color": "#a8edea"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,80,80,0.2)"},
                    {"range": [40, 70], "color": "rgba(255,200,0,0.2)"},
                    {"range": [70,100], "color": "rgba(80,255,160,0.2)"},
                ],
                "threshold": {
                    "line" : {"color": "#fed6e3", "width": 3},
                    "thickness": 0.75,
                    "value": confidence
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   ="#e0e0e0",
            height=240,
            margin=dict(t=30, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Probability Bar Chart ─────────────────────────────────────────────────
    with col_b:
        classes = list(le.classes_)
        colors  = ["#a8edea" if c == predicted_label else "rgba(255,255,255,0.15)"
                   for c in classes]

        fig_bar = go.Figure(go.Bar(
            x         = [f"{c} {WEATHER_CONFIG.get(c,{}).get('icon','')}" for c in classes],
            y         = proba * 100,
            marker_color = colors,
            text      = [f"{v*100:.1f}%" for v in proba],
            textposition = "outside",
            textfont  = {"color": "#e0e0e0", "size": 12}
        ))
        fig_bar.update_layout(
            title     = {"text": "All Class Probabilities", "font": {"color": "#e0e0e0", "size": 13}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            yaxis     = {"range": [0, 115], "gridcolor": "rgba(255,255,255,0.06)", "title": "%"},
            xaxis     = {"title": ""},
            height    = 240,
            margin    = dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Input Summary ─────────────────────────────────────────────────────────
    with st.expander("📋 View Input Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Temperature",   f"{temperature} °C")
            st.metric("Humidity",      f"{humidity} %")
            st.metric("Wind Speed",    f"{wind_speed} km/h")
            st.metric("Precipitation", f"{precipitation} %")
        with summary_col2:
            st.metric("Pressure",      f"{atmospheric_pressure} hPa")
            st.metric("UV Index",      f"{uv_index}")
            st.metric("Visibility",    f"{visibility} km")
            st.metric("Cloud Cover",   cloud_cover)
        st.write(f"**Season:** {season} &nbsp;&nbsp; **Location:** {location}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#4b5563; font-size:0.8rem;'>
    Built with TensorFlow · Streamlit · Plotly &nbsp;|&nbsp; ANN Weather Classifier
</p>
""", unsafe_allow_html=True)