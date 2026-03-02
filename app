import streamlit as st
import cv2
import tempfile
import pandas as pd
from PIL import Image

# Import your custom engines
import computer_vision as cv_engine
import math_engine as risk_engine

# --- UI SETUP ---
st.set_page_config(page_title="B-SVaR Urban Risk Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ B-SVaR: Multimodal Urban Risk Engine")
st.markdown("Real-time Bayesian fusion of visual telemetry and spatial-temporal fundamentals.")
st.divider()

# --- INITIALIZE SESSION STATE FOR CHARTS ---
if 'history_posterior' not in st.session_state:
    st.session_state.history_posterior = []
if 'history_var' not in st.session_state:
    st.session_state.history_var = []

# --- SIDEBAR: CONTEXTUAL INPUTS ---
st.sidebar.header("📍 Spatial Fundamentals")
# Strings must MATCH math_engine.py EXACTLY
precinct = st.sidebar.selectbox("Precinct Risk Profile", ["Low", "Medium", "High"], index=1)
active_event = st.sidebar.selectbox("Active City Events", ["None", "Concert/Sports (Crowds)", "Protest/Riot"], index=0)
osint = st.sidebar.toggle("🚨 OSINT Sentiment Spike", value=False)

st.sidebar.header("🌦️ Macro Indicators")
time_of_day = st.sidebar.selectbox("Time of Day", ["Day (6 AM - 5 PM)", "Evening (6 PM - 1 AM)", "Night (2 AM - 5 AM)"],
                                   index=0)
weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Fog/Low Visibility", "Heavy Rain/Storm"], index=0)

# --- RUN MATH ENGINE ---
# Generates the Prior and VaR based on sidebar context
prior_pt, var_threshold = risk_engine.monte_carlo_sims(
    time=time_of_day,
    weather=weather,
    precinct_risks=precinct,
    active_events=active_event,
    osint_spike=osint
)

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1.2])

with col2:
    st.subheader("📊 Quantitative Risk Metrics")

    # Display baseline metrics
    m1, m2 = st.columns(2)
    m1.metric(label="Contextual Prior $P(T)$", value=f"{prior_pt:.4f}")
    m2.metric(label="95% VaR Threshold", value=f"{var_threshold:.4f}")

    st.markdown("### Live Telemetry")
    # Placeholders for dynamic updates
    status_placeholder = st.empty()
    likelihood_placeholder = st.empty()
    chart_placeholder = st.empty()

with col1:
    st.subheader("📹 AI Vision Sensor Feed")
    uploaded_file = st.file_uploader("Deploy Video Stream", type=["mp4", "mov", "avi"])

    stop_button = st.button("🛑 Stop Stream")

    if uploaded_file is not None and not stop_button:
        # 1. Load CLIP Model
        with st.spinner("Initializing CLIP Vision Transformer..."):
            cv_engine.load_clip_engine()

        # 2. Process Video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()
        frame_count = 0

        # Reset history on new video upload
        st.session_state.history_posterior = []
        st.session_state.history_var = []

        # --- THE CONTINUOUS EXECUTION LOOP ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame_count += 1

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True, channels="RGB")

            # RUN AI SENSOR (Sample every 30 frames to simulate 1 FPS processing)
            if frame_count % 30 == 0:
                pil_img = Image.fromarray(frame_rgb)

                # Extract visual likelihood
                cv_results = cv_engine.calc_clip_risk(pil_img)
                likelihood_pvt = cv_results["p_vgt"]

                # Run Bayesian Fusion
                posterior_ptv = risk_engine.calculate_bayesian_posterior(prior_pt, likelihood_pvt)

                # Update History for charts
                st.session_state.history_posterior.append(posterior_ptv)
                st.session_state.history_var.append(var_threshold)

                # --- UPDATE UI DYNAMICALLY ---
                likelihood_placeholder.info(f"👁️ **Vision Likelihood $P(V|T)$:** `{likelihood_pvt:.4f}`")

                if posterior_ptv > var_threshold:
                    status_placeholder.error(
                        f"🚨 **BREACH DETECTED** \n\nPosterior Risk ({posterior_ptv:.4f}) exceeded VaR.")
                else:
                    status_placeholder.success(
                        f"✅ **NOMINAL** \n\nPosterior Risk ({posterior_ptv:.4f}) absorbed by context.")

                # Draw the live time-series chart
                chart_data = pd.DataFrame({
                    "Live Posterior Risk": st.session_state.history_posterior,
                    "VaR Threshold": st.session_state.history_var
                })
                chart_placeholder.line_chart(chart_data, color=["#FF4B4B", "#1f77b4"])

        cap.release()
