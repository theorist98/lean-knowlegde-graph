# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a basic stochastic‑process theorem.

Exact Lean code referenced (verbatim):

structure RadioactiveDecay where
  halfLife : Nat
  initialAmount : Nat
def decayAmount (rd : RadioactiveDecay) (time : Nat) : Nat := rd.initialAmount / (2 ^ (time / rd.halfLife))
theorem decay_at_zero : decayAmount ⟨10, 100⟩ 0 = 100 := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean structure and function exactly over integers (Nat semantics):
     decayAmount(rd, t) = rd.initialAmount // 2^(t // rd.halfLife)
   Note the use of integer division (//) to match Lean's Nat division.
2) Verifies the Lean theorem analogue:
     decayAmount(<10,100>, 0) == 100    (holds by definition, i.e., Lean's `rfl`).
3) Provides a *stochastic* model consistent with the half‑life interpretation:
   - Time is continuous but *decay events* only occur at checkpoints t = k·halfLife.
   - At each checkpoint, every surviving atom independently survives with prob 1/2.
   - Between checkpoints, the count stays constant.
   This yields E[N(t)] = initialAmount * (1/2)^floor(t/halfLife), while the Lean function
   returns floor(initialAmount / 2^floor(t/halfLife)) due to Nat arithmetic.
4) Visuals (each on its own figure, matplotlib defaults, no custom colors):
   - Step curve of the Lean `decayAmount` vs Monte‑Carlo mean with 5–95% band.
   - Histogram of survivors at t = 3*halfLife with markers for Lean value (floor)
     and the real expectation.
   - LLN demo: running mean at t = 5*halfLife approaching the real expectation,
     with the Lean (floored) value shown for comparison.
5) Saves a CSV summary of the trajectories.

Dependencies: numpy, matplotlib, pandas.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

# ----- Mirror the Lean definitions exactly over integers (Nat semantics) -----

LEAN_CODE = """structure RadioactiveDecay where
  halfLife : Nat
  initialAmount : Nat
def decayAmount (rd : RadioactiveDecay) (time : Nat) : Nat := rd.initialAmount / (2 ^ (time / rd.halfLife))
theorem decay_at_zero : decayAmount <10, 100> 0 = 100 := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n— End Lean code —\n")

@dataclass(frozen=True)
class RadioactiveDecay:
    halfLife: int
    initialAmount: int

def decay_amount(rd: RadioactiveDecay, time: int) -> int:
    """
    Python analogue of the Lean definition with Nat division:
      decayAmount (rd) (time) = rd.initialAmount / (2 ^ (time / rd.halfLife))
    where '/' and '^' are on Nats. In Python we use:
      // for integer division, and ** for exponentiation.
    """
    if rd.halfLife <= 0:
        raise ValueError("halfLife must be positive to mirror Lean Nat semantics.")
    k = time // rd.halfLife
    return rd.initialAmount // (2 ** k)

# ----- Verify the Lean theorem instance (by definitional equality) -----
assert decay_amount(RadioactiveDecay(10, 100), 0) == 100, "Lean theorem analogue failed."
print("Lean theorem analogue check passed: decayAmount <10,100> at t=0 equals 100 (by definition).\n")

# ----- Stochastic model consistent with half-life checkpoints -----

def simulate_decay_trial(rd: RadioactiveDecay, t_max: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate one trial:
      - For times t, compute k = floor(t / halfLife).
      - Each time k increases by 1, apply Binomial(n, 1/2) to survivors.
      - Between checkpoints, state is constant.
    Returns an array of survivors for times 0..t_max (inclusive).
    """
    if rd.halfLife <= 0:
        raise ValueError("halfLife must be positive.")
    survivors = rd.initialAmount
    out = np.empty(t_max + 1, dtype=int)
    prev_k = 0
    for t in range(t_max + 1):
        k = t // rd.halfLife
        if k > prev_k:
            # Apply as many 1/2 survival rounds as checkpoints crossed since last t
            for _ in range(k - prev_k):
                survivors = rng.binomial(survivors, 0.5)
            prev_k = k
        out[t] = survivors
    return out

def simulate_decay_ensemble(rd: RadioactiveDecay, t_max: int, n_trials: int, seed: int = 202509122325) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trials = np.empty((n_trials, t_max + 1), dtype=int)
    for i in range(n_trials):
        trials[i] = simulate_decay_trial(rd, t_max, rng)
    return np.arange(0, t_max + 1), trials

def lean_curve(rd: RadioactiveDecay, times: np.ndarray) -> np.ndarray:
    return np.array([decay_amount(rd, int(t)) for t in times], dtype=int)

def real_expectation(rd: RadioactiveDecay, times: np.ndarray) -> np.ndarray:
    k = (times // rd.halfLife).astype(int)
    return rd.initialAmount * (0.5 ** k)

# ----- Parameters tied to the Lean theorem instance -----
rd = RadioactiveDecay(halfLife=10, initialAmount=100)
t_max = 100
n_trials = 5000

# ----- Run ensemble simulation -----
times, trials = simulate_decay_ensemble(rd, t_max=t_max, n_trials=n_trials)
lean_vals = lean_curve(rd, times)
mean_vals = trials.mean(axis=0)
q05 = np.quantile(trials, 0.05, axis=0)
q95 = np.quantile(trials, 0.95, axis=0)
exp_vals = real_expectation(rd, times)

# ----- Save summary CSV -----
df = pd.DataFrame({
    "time": times,
    "lean_decayAmount_nat": lean_vals,
    "mc_mean": mean_vals,
    "q05": q05,
    "q95": q95,
    "real_expectation": exp_vals
})
csv_path = "./radioactive_decay_summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved time-series summary to: {csv_path}\n")

# ----- Visualization 1: Lean step curve vs Monte-Carlo mean with band -----
plt.figure(figsize=(8, 5))
plt.step(times, lean_vals, where="post", label="Lean decayAmount (Nat arithmetic)")
plt.plot(times, mean_vals, label="Monte-Carlo mean (checkpoint halving)")
plt.plot(times, exp_vals, linestyle="--", label="Real expectation $E[N(t)]$")
plt.fill_between(times, q05, q95, alpha=0.2, label="5-95% Monte-Carlo band")
for m in range(0, t_max + 1, rd.halfLife):
    plt.axvline(m, linestyle=":", alpha=0.5)
plt.xlabel("time")
plt.ylabel("amount remaining")
plt.title("Radioactive decay: Lean step function vs stochastic ensemble")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('./nuclear_decay_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Distribution at t = 3*halfLife -----
t_star = 3 * rd.halfLife
idx = int(t_star)
vals_star = trials[:, idx]
plt.figure(figsize=(8, 5))
# integer-aligned bins
bins = np.arange(vals_star.min() - 0.5, vals_star.max() + 1.5, 1.0)
plt.hist(vals_star, bins=bins, density=True)
plt.axvline(lean_vals[idx], linestyle="--", label=f"Lean decayAmount={lean_vals[idx]}")
plt.axvline(exp_vals[idx], linestyle="-.", label=f"Real expectation={exp_vals[idx]:.2f}")
plt.xlabel(f"survivors at t={t_star}")
plt.ylabel("density")
plt.title(f"Distribution of survivors at t = {t_star} (k = {t_star // rd.halfLife} checkpoints)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('./nuclear_decay_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: LLN - running mean at t = 5*halfLife -----
t_star2 = 5 * rd.halfLife
vals_star2 = trials[:, t_star2]
running_mean = np.cumsum(vals_star2) / np.arange(1, n_trials + 1)
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, n_trials + 1), running_mean, label="Running mean")
plt.axhline(exp_vals[t_star2], linestyle="--", label=f"Real expectation={exp_vals[t_star2]:.2f}")
plt.axhline(lean_vals[t_star2], linestyle=":", label=f"Lean decayAmount={lean_vals[t_star2]}")
plt.xlabel("number of trials")
plt.ylabel(f"mean survivors at t={t_star2}")
plt.title("Running mean converges to the real expectation (LLN)\n(Lean value shown for Nat rounding)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('./nuclear_decay_convergence.png', dpi=150, bbox_inches='tight')
plt.show()