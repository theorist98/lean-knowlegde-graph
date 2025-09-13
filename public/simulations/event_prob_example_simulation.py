# -*- coding: utf-8 -*-
# Pretty simulation illustrating the exact Lean theorem:
#   event_prob_example : event_probability ⟨20, 8⟩ = (8, 20) := by rfl
#
# We mirror the Lean definitions and run a Monte-Carlo demo.

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import os

# Set the working directory to simulations folder to save images there
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----- 1) Mirror the Lean code in Python (verbatim reference preserved) -----
LEAN_CODE = """structure SimpleEvent where
  total_outcomes : Nat
  successful_outcomes : Nat
def event_probability (e : SimpleEvent) : Nat × Nat := (e.successful_outcomes, e.total_outcomes)
theorem event_prob_example : event_probability <20, 8> = (8, 20) := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n— End Lean code —\n")

@dataclass(frozen=True)
class SimpleEvent:
    total_outcomes: int
    successful_outcomes: int

def event_probability(e: SimpleEvent) -> tuple[int, int]:
    """Python analogue of the Lean def: returns (successes, total)."""
    return (e.successful_outcomes, e.total_outcomes)

# The Lean theorem's instance: <20, 8> means total_outcomes = 20, successful_outcomes = 8
lean_event = SimpleEvent(total_outcomes=20, successful_outcomes=8)

# Definitional equality check (Lean uses 'by rfl')
assert event_probability(lean_event) == (8, 20), "Analogue of Lean theorem failed."
print("Lean theorem analogue check passed: event_probability(<20, 8>) == (8, 20) (by definition).\n")

# Helpful derived quantities
frac = Fraction(lean_event.successful_outcomes, lean_event.total_outcomes).limit_denominator()
p = float(frac)  # theoretical probability
print(f"Theoretical probability from Lean pair (8, 20): {lean_event.successful_outcomes}/{lean_event.total_outcomes} = {frac} = {p:.4f}\n")

# Seed for reproducibility (derived from the session date 2025-09-09)
rng = np.random.default_rng(20250909)

# ----- 2) Convergence plot: running estimate of P(success) -----
def running_estimate(event: SimpleEvent, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n_draws independent outcomes from the uniform space {1,...,total_outcomes}.
    Count success if the draw is in {1,...,successful_outcomes}. Return running \hat p.
    """
    draws = rng.integers(low=1, high=event.total_outcomes + 1, size=n_draws)
    success_mask = draws <= event.successful_outcomes  # successes are the "first 8" of 20
    cum_successes = np.cumsum(success_mask)
    running_p_hat = cum_successes / np.arange(1, n_draws + 1)
    return running_p_hat

n_draws = 5000
rp = running_estimate(lean_event, n_draws=n_draws, rng=rng)

plt.figure(figsize=(10, 6))
x = np.arange(1, n_draws + 1)
plt.plot(x, rp, label="Running estimate $\hat{p}$", alpha=0.8)
plt.axhline(p, linestyle="--", color="red", linewidth=2, label=f"Theoretical {frac} = {p:.3f}")
plt.xlabel("Number of draws")
plt.ylabel("Estimated P(success)")
plt.title("Convergence of running estimate to the Lean ratio (8, 20) → 8/20")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
print("Saved convergence plot to: convergence_plot.png")
plt.close()

# ----- 3) Sampling distribution at n = 20 (matching the Lean event's '20') -----
n = lean_event.total_outcomes   # 20 draws per experiment
experiments = 10_000
successes = rng.binomial(n=n, p=p, size=experiments)
p_hats = successes / n

plt.figure(figsize=(10, 6))
# Choose bins aligned to multiples of 1/n (i.e., 0, 1/20, 2/20, ..., 1)
EPS = 0.5 / n
bins = np.linspace(-EPS, 1 + EPS, n + 2)
plt.hist(p_hats, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(p, linestyle="--", color="red", linewidth=2, label=f"Theoretical {frac} = {p:.3f}")
plt.xlabel("Sample proportion $\hat{p}$ (20 draws per experiment)")
plt.ylabel("Density")
plt.title("Sampling distribution of $\hat{p}$ for the Lean event <20, 8>")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig('sampling_distribution.png', dpi=300, bbox_inches='tight')
print("Saved sampling distribution plot to: sampling_distribution.png")
plt.close()

# ----- 4) Accuracy table across growing sample sizes -----
def accuracy_summary(event: SimpleEvent, sample_sizes, reps: int, rng: np.random.Generator) -> pd.DataFrame:
    p_theory = event.successful_outcomes / event.total_outcomes
    rows = []
    for n in sample_sizes:
        succ = rng.binomial(n=n, p=p_theory, size=reps)
        ph = succ / n
        mean_hat = float(np.mean(ph))
        std_hat = float(np.std(ph, ddof=1))
        se_theory = (p_theory * (1 - p_theory) / n) ** 0.5
        within_95 = float(np.mean(np.abs(ph - p_theory) <= 1.96 * se_theory))
        rows.append({
            "sample_size_n": n,
            "replicates": reps,
            "mean_hat_p": round(mean_hat, 6),
            "std_hat_p": round(std_hat, 6),
            "theoretical_p": round(p_theory, 6),
            "abs_error_of_mean": round(abs(mean_hat - p_theory), 6),
            "theoretical_SE": round(se_theory, 6),
            "frac_within_±1.96SE": round(within_95, 4),
        })
    return pd.DataFrame(rows)

sample_sizes = [20, 200, 2000, 20000]
reps = 200
df = accuracy_summary(lean_event, sample_sizes, reps, rng)

# Save to CSV
csv_path = "event_prob_example_accuracy.csv"
df.to_csv(csv_path, index=False)

# Display the table
print("Accuracy table:\n")
print(df.to_string(index=False))

print(f"\nSaved accuracy table to: {csv_path}")
print(f"Generated images: convergence_plot.png, sampling_distribution.png")