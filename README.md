# 🚦 B-SVaR: Bayesian-Simulated Value-at-Risk for Urban Traffic

**B-SVaR** is an autonomous traffic incident engine that applies quantitative financial risk methodologies (Monte Carlo Simulations, Value-at-Risk) to spatial urban infrastructure. 

By fusing zero-shot semantic computer vision (OpenAI CLIP) with a stochastic environmental risk matrix, B-SVaR dynamically adjusts its sensitivity based on real-world conditions (weather, traffic flow, road type). This mathematically eliminates the false-positive rate commonly associated with 2D CCTV cameras in complex urban environments.

## 🧠 Core Architecture
1. **The Vision Sensor (`computer_vision.py`):** Utilizes a heavily optimized, custom linear algebra implementation of OpenAI's CLIP model. It extracts the semantic likelihood $P(V|T)$ of a traffic incident and applies a 5-frame temporal smoothing buffer to eliminate visual jitter.
2. **The Stochastic Engine (`math_engine.py`):** Calculates a contextual Prior $P(T)$ and a 95% Value-at-Risk (VaR) threshold using a Monte Carlo simulation across a parameter-shifted Beta distribution.
3. **The Bayesian Fusion (`app.py`):** A real-time Streamlit command center that updates the posterior risk $P(T|V)$. An alert is only triggered if the visual evidence statistically outweighs the environmental noise floor.

## ⚙️ Quickstart Installation
Ensure you have Python 3.10+ installed.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/B-SVaR.git](https://github.com/YOUR_USERNAME/B-SVaR.git)
cd B-SVaR

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit Dashboard
streamlit run app.py
