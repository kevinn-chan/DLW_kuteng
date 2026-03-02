# 🚦 B-SVaR: Bayesian-Simulated Value-at-Risk
**Autonomous Traffic Incident Engine for Context-Aware Public Safety**

### **1. The Vision**
Traditional safety AI is binary—it detects objects but ignores context. This creates an "Interpretation Gap" where a rainy night triggers the same alerts as a clear afternoon, leading to fatal alarm fatigue. 

**B-SVaR** closes this gap by porting **Quantitative Financial Risk Modeling** into public safety. We move from *detecting objects* to *quantifying environmental chaos*.

### **2. Key Innovations**
* **The Stochastic Noise Floor (Monte Carlo)**: We establish a dynamic **Value-at-Risk (VaR)** threshold using 10,000 Monte Carlo simulations based on road architecture, traffic flow, and weather. This represents the environmental "noise floor."
* **Semantic Intent Engine (Zero-Shot CLIP)**: Unlike standard object detection, we use a **CLIP Transformer** to understand the *intent* of a scene using 22 engineered semantic anchors (e.g., "crumpled metal" vs. "normal traffic").
* **Actionable Certainty**: Our **Persistence Gate** requires consecutive breaches of the VaR threshold before confirming an incident, ensuring emergency responders only act on verified data.

### **3. System Architecture**
* `math_engine.py`: The analytical core. Runs the Monte Carlo simulations and Odds-form Bayesian fusion.
* `computer_vision.py`: The semantic sensor. Performs zero-shot inference using OpenAI's CLIP, featuring Out-of-Domain (OOD) suppression.
* `app.py`: The orchestration layer. A real-time Streamlit dashboard for telemetry and environmental toggle testing.
* `testbench/`: Automated headless scripts to verify mathematical stability across extreme weather and traffic scenarios.

---

### **4. Setup Instructions**

**Step 1: Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/B-SVaR.git](https://github.com/YOUR_USERNAME/B-SVaR.git)
cd B-SVaR
