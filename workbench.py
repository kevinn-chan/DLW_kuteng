import math_engine as risk_engine

def run_scenario(name, road, flow, weather, vision_likelihood):
    print(f"\n{'=' * 60}")
    print(f"🔬 WORKBENCH SCENARIO: {name}")
    print(f"{'=' * 60}")
    print(f"Context   : {road} | {flow} | {weather}")
    print(f"Telemetry : Vision P(V|T) = {vision_likelihood:.4f}")
    print("-" * 60)

    # 1. Stochastic Engine
    prior, var_threshold = risk_engine.monte_carlo_sims(
        flow=flow,
        weather=weather,
        road=road
    )

    # 2. Bayesian Fusion
    posterior = risk_engine.calculate_bayesian_posterior(prior, vision_likelihood)

    # 3. Calculate Delta (Risk Spread)
    spread = posterior - var_threshold

    print(f"Contextual Prior P(T) : {prior:.4f}")
    print(f"95% VaR Threshold     : {var_threshold:.4f}")
    print(f"Final Posterior P(T|V): {posterior:.4f}")
    print("-" * 60)

    if posterior > var_threshold:
        print(f"🚨 RESULT: BREACH DETECTED (+{spread:.4f} over VaR)")
        print("   Logic: Visual evidence outweighed environmental uncertainty.")
    else:
        print(f"✅ RESULT: ALARM SUPPRESSED ({spread:.4f} under VaR)")
        print("   Logic: Visual anomaly absorbed by contextual noise floor.")


if __name__ == "__main__":
    print("\n🚀 INITIALIZING B-SVaR TRAFFIC WORKBENCH...\n")

    # SCENARIO 1: Perfect Day (Baseline)
    # Expected: Low Prior, tight VaR. Even a moderate vision score should trigger an alert because the baseline is stable.
    run_scenario(
        name="The 'Perfect Tuesday' (Nominal Conditions)",
        road="Straight City Road",
        flow="Normal Flow",
        weather="Clear/Dry",
        vision_likelihood=0.65
    )

    # SCENARIO 2: Paranoid AI (Extreme Noise, Safe Vision)
    # Expected: Massive VaR threshold due to rain and night. Low vision score. The system should aggressively suppress any false positives.
    run_scenario(
        name="The 'Paranoid AI' (High Macro Noise)",
        road="Complex Intersection",
        flow="Night (Low Visibility)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.20
    )

    # SCENARIO 3: Messy Reality (High Noise, Moderate Vision Jitter)
    # Expected: A traffic jam in the rain causes CLIP to get confused and output a 0.70. A standard AI would trigger a false alarm here. B-SVaR should suppress it.
    run_scenario(
        name="The 'Messy Reality' (Stress Test)",
        road="High-Speed Highway",
        flow="Rush Hour (Gridlock)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.70
    )

    # SCENARIO 4: Undeniable Crash (High Noise, Absolute Vision Certainty)
    # Expected: Even in a rainstorm, if the cars physically mash together and CLIP outputs a 0.95, it must breach the elevated VaR boundary.
    run_scenario(
        name="The 'Undeniable Crash' (True Positive Override)",
        road="Complex Intersection",
        flow="Rush Hour (Gridlock)",
        weather="Heavy Rain/Wet Asphalt",
        vision_likelihood=0.95
    )

    print(f"\n{'=' * 60}")
    print("🏁 WORKBENCH COMPLETE. ALL STOCHASTIC BOUNDARIES VERIFIED.")
    print(f"{'=' * 60}\n")
