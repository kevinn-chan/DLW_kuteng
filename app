import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from PIL import Image

# Import your custom engines
import computer_vision as cv_engine
import math_engine as risk_engine

# --- UI SETUP ---
st.set_page_config(page_title="B-SVaR Urban Risk Monitor", layout="wide")
st.title("🛡️ B-SVaR: Multimodal Urban Risk Engine")
st.markdown("---")

# --- SIDEBAR: CONTEXTUAL INPUTS ---
st.sidebar.header("📍 Contextual Microservices")
precinct = st.sidebar.selectbox("Precinct Risk Profile", ["Low", "Medium", "High"], index=1)
active_event = st.sidebar.selectbox("Active City Events", ["None", "Concert/Sports (Crowds)", "Protest/Riot"])
osint = st.sidebar.toggle("OSINT Sentiment Spike Detected", value=False)

st.sidebar.header("🌦️ Environmental Macro")
time_of_day = st.sidebar.selectbox("Time of Day", ["Day (6AM - 5PM)", "Evening (6PM - 1AM)", "Night (2AM - 5AM)"],
                                   index=0)
weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Fog/Low Visibility", "Heavy Rain/Storm/Snow"], index=0)

# --- INITIALIZE MATH ENGINE ---
# This runs the Monte Carlo simulation immediately based on sidebar inputs
prior_pt, var_threshold = risk_engine.monte_carlo_sims(
    time=time_of_day,
    weather=weather,
    precinct_risks=precinct,
    active_events=active_event,
    osint_spike=osint
)

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Risk Metrics")
    st.metric("Contextual Prior P(T)", f"{prior_pt:.4f}")
    st.metric("VaR Threshold (95%)", f"{var_threshold:.4f}")

    # Placeholder for live results
    result_placeholder = st.empty()
    chart_placeholder = st.empty()

with col1:
    st.subheader("📹 Live Surveillance Feed")
    uploaded_file = st.file_uploader("Upload Video Feed...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Load the CLIP model once
        with st.spinner("Initializing CLIP Vision Transformer..."):
            cv_engine.load_clip_engine()

        # Save uploaded video to a temp file for OpenCV to read
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()

        frame_count = 0

        # --- THE CONTINUOUS EXECUTION LOOP ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 1. DISPLAY THE VIDEO (Normal Speed)
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True)

            # 2. RUN AI SENSOR (Sampling every 30 frames / 1 second)
            if frame_count % 30 == 0:
                # Get likelihood from computer_vision.py
                # We pass the PIL image directly to avoid re-encoding
                pil_img = Image.fromarray(frame_rgb)
                cv_results = cv_engine.calc_clip_risk(pil_img)
                likelihood_pvt = cv_results["p_vgt"]

                # 3. RUN BAYESIAN FUSION
                posterior_ptv = risk_engine.calculate_bayesian_posterior(prior_pt, likelihood_pvt)

                # 4. UPDATE UI
                with result_placeholder.container():
                    st.write(f"**Vision Likelihood P(V|T):** `{likelihood_pvt:.4f}`")

                    if posterior_ptv > var_threshold:
                        st.error(f"🚨 ALERT: POSTERIOR RISK BREACHED! ({posterior_ptv:.4f})")
                    else:
                        st.success(f"✅ STATUS: NORMAL ({posterior_ptv:.4f})")

                # Dynamic Bar Chart for visual punch
                chart_placeholder.bar_chart({
                    "Prior": prior_pt,
                    "VaR Threshold": var_threshold,
                    "Likelihood": likelihood_pvt,
                    "Final Posterior": posterior_ptv
                })

        cap.release()
