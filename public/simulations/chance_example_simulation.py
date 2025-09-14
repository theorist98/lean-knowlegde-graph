# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a measure theory chance theorem.

Exact Lean code referenced (verbatim):

structure BasicEvent where
  totalPossibilities : Nat
  favorableCount : Nat
def eventChance (e : BasicEvent) : Nat × Nat := (e.favorableCount, e.totalPossibilities)
theorem chance_example : eventChance <20, 5> = (5, 20) := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean BasicEvent structure and eventChance function exactly:
   - BasicEvent contains total possibilities and favorable count
   - eventChance returns the favorable count and total as a tuple
2) Verifies the Lean theorem analogue:
   - eventChance(<20, 5>) == (5, 20) (basic probability structure)
3) Provides stochastic models that demonstrate probability 5/20 = 0.25:
   - Monte Carlo simulation with trials drawing from 20 possibilities
   - Statistical analysis showing convergence to theoretical 0.25 probability
   - Binomial distribution validation with n=20, p=0.25
4) Visuals (three complementary plots):
   - Convergence of running probability estimate to theoretical 5/20 = 0.25
   - Binomial distribution showing success counts in multiple experiments
   - Confidence intervals demonstrating sampling variability
5) Saves CSV summary of simulation trials and statistical analysis

Dependencies: numpy, matplotlib, pandas, scipy
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from typing import Tuple

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure BasicEvent where
  totalPossibilities : Nat
  favorableCount : Nat
def eventChance (e : BasicEvent) : Nat × Nat := (e.favorableCount, e.totalPossibilities)
theorem chance_example : eventChance <20, 5> = (5, 20) := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n— End Lean code —\n")

@dataclass(frozen=True)
class BasicEvent:
    totalPossibilities: int
    favorableCount: int

def event_chance(e: BasicEvent) -> Tuple[int, int]:
    """
    Python analogue of the Lean eventChance function:
    Returns (favorableCount, totalPossibilities) tuple
    """
    return (e.favorableCount, e.totalPossibilities)

# ----- Verify the Lean theorem instance -----
lean_event = BasicEvent(totalPossibilities=20, favorableCount=5)
assert event_chance(lean_event) == (5, 20), "Lean theorem analogue failed."
print("Lean theorem analogue check passed: eventChance(<20, 5>) = (5, 20) (verified).\n")

# Theoretical probability
p_theoretical = 5/20  # 0.25
print(f"Theoretical probability: {lean_event.favorableCount}/{lean_event.totalPossibilities} = {p_theoretical:.4f}\n")

# ----- Monte Carlo simulation functions -----

def simulate_single_trial(event: BasicEvent, rng: np.random.Generator) -> bool:
    """
    Simulate a single trial: randomly select from totalPossibilities,
    return True if selection is among favorableCount
    """
    # Draw random number from 1 to totalPossibilities
    draw = rng.integers(1, event.totalPossibilities + 1)
    # Success if draw <= favorableCount
    return draw <= event.favorableCount

def simulate_convergence_trials(event: BasicEvent, n_trials: int, seed: int = 20250914) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run n_trials and track running probability estimate convergence
    """
    rng = np.random.default_rng(seed)
    successes = 0
    running_probabilities = np.empty(n_trials)

    for i in range(n_trials):
        if simulate_single_trial(event, rng):
            successes += 1
        running_probabilities[i] = successes / (i + 1)

    trial_numbers = np.arange(1, n_trials + 1)
    return trial_numbers, running_probabilities

def simulate_binomial_experiments(event: BasicEvent, n_experiments: int, trials_per_experiment: int, seed: int = 20250914) -> np.ndarray:
    """
    Run n_experiments, each with trials_per_experiment trials
    Return array of success counts per experiment
    """
    rng = np.random.default_rng(seed + 1000)  # Different seed for independence
    success_counts = np.empty(n_experiments)

    for exp in range(n_experiments):
        successes = 0
        for _ in range(trials_per_experiment):
            if simulate_single_trial(event, rng):
                successes += 1
        success_counts[exp] = successes

    return success_counts

# ----- Run simulations -----
n_convergence_trials = 10000
n_experiments = 1000
trials_per_experiment = 20  # Match the totalPossibilities

print("Running simulations...")
trial_numbers, running_probs = simulate_convergence_trials(lean_event, n_convergence_trials)
success_counts = simulate_binomial_experiments(lean_event, n_experiments, trials_per_experiment)

# ----- Statistical analysis -----
final_estimate = running_probs[-1]
theoretical_successes = trials_per_experiment * p_theoretical
empirical_mean_successes = np.mean(success_counts)
empirical_std_successes = np.std(success_counts)
theoretical_std_successes = np.sqrt(trials_per_experiment * p_theoretical * (1 - p_theoretical))

stats_summary = {
    "theoretical_probability": p_theoretical,
    "final_convergence_estimate": final_estimate,
    "convergence_error": abs(final_estimate - p_theoretical),
    "theoretical_mean_successes": theoretical_successes,
    "empirical_mean_successes": empirical_mean_successes,
    "theoretical_std_successes": theoretical_std_successes,
    "empirical_std_successes": empirical_std_successes,
    "trials_for_convergence": n_convergence_trials,
    "binomial_experiments": n_experiments,
    "trials_per_experiment": trials_per_experiment
}

# ----- Save CSV summaries -----
# Convergence data (sample every 100 trials to reduce file size)
convergence_sample = np.arange(99, n_convergence_trials, 100)  # Every 100th trial
convergence_df = pd.DataFrame({
    "trial_number": trial_numbers[convergence_sample],
    "running_probability": running_probs[convergence_sample],
    "theoretical_probability": p_theoretical,
    "absolute_error": np.abs(running_probs[convergence_sample] - p_theoretical)
})
convergence_df.to_csv("./chance_example_convergence.csv", index=False)

# Binomial experiments data
binomial_df = pd.DataFrame({
    "experiment": range(1, n_experiments + 1),
    "success_count": success_counts,
    "success_rate": success_counts / trials_per_experiment,
    "theoretical_rate": p_theoretical
})
binomial_df.to_csv("./chance_example_binomial.csv", index=False)

# Statistics summary
stats_df = pd.DataFrame([stats_summary])
stats_df.to_csv("./chance_example_statistics.csv", index=False)

print("Saved simulation data to CSV files.")

# ----- Visualization 1: Convergence to theoretical probability -----
plt.figure(figsize=(10, 6))

plt.plot(trial_numbers, running_probs, 'b-', alpha=0.7, linewidth=1, label="Running probability estimate")
plt.axhline(p_theoretical, color='red', linestyle='--', linewidth=2, label=f"Theoretical probability = {p_theoretical:.4f}")

# Add confidence bounds (approximate)
n_vals = trial_numbers
lower_bound = p_theoretical - 1.96 * np.sqrt(p_theoretical * (1 - p_theoretical) / n_vals)
upper_bound = p_theoretical + 1.96 * np.sqrt(p_theoretical * (1 - p_theoretical) / n_vals)
plt.fill_between(trial_numbers, lower_bound, upper_bound, alpha=0.2, color='gray', label="95% confidence bounds")

plt.xlabel("Number of trials")
plt.ylabel("Probability estimate")
plt.title(f"Convergence to Lean theorem: eventChance(<20, 5>) → 5/20 = {p_theoretical:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(1, n_convergence_trials)
plt.ylim(0.15, 0.35)
plt.tight_layout()
plt.savefig('./chance_example_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Binomial distribution of success counts -----
plt.figure(figsize=(10, 6))

# Plot histogram of success counts
bins = np.arange(-0.5, max(success_counts) + 1.5, 1.0)
plt.hist(success_counts, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Empirical distribution')

# Overlay theoretical binomial distribution
x_theory = np.arange(0, int(max(success_counts)) + 1)
y_theory = stats.binom.pmf(x_theory, trials_per_experiment, p_theoretical)
plt.plot(x_theory, y_theory, 'ro-', markersize=6, label=f'Theoretical Binomial(n={trials_per_experiment}, p={p_theoretical:.3f})')

# Mark theoretical mean
plt.axvline(theoretical_successes, color='red', linestyle='--', alpha=0.8,
           label=f'Theoretical mean = {theoretical_successes:.1f}')
plt.axvline(empirical_mean_successes, color='orange', linestyle=':', alpha=0.8,
           label=f'Empirical mean = {empirical_mean_successes:.2f}')

plt.xlabel("Number of successes (out of 20 trials)")
plt.ylabel("Probability density")
plt.title(f"Distribution of Success Counts: {n_experiments} experiments of 20 trials each")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./chance_example_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Probability estimates from batches -----
batch_size = 50
n_batches = len(success_counts) // batch_size
batch_probabilities = []
batch_numbers = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_mean = np.mean(success_counts[start_idx:end_idx]) / trials_per_experiment
    batch_probabilities.append(batch_mean)
    batch_numbers.append(i + 1)

batch_probabilities = np.array(batch_probabilities)
batch_numbers = np.array(batch_numbers)

plt.figure(figsize=(10, 6))
plt.plot(batch_numbers, batch_probabilities, 'bo-', alpha=0.7, markersize=4, label=f"Batch estimates (n={batch_size} experiments each)")
plt.axhline(p_theoretical, color='red', linestyle='--', linewidth=2, label=f"Theoretical probability = {p_theoretical:.4f}")

# Add error bars showing standard error
batch_std_error = theoretical_std_successes / (trials_per_experiment * np.sqrt(batch_size))
plt.fill_between(batch_numbers,
                p_theoretical - 1.96 * batch_std_error,
                p_theoretical + 1.96 * batch_std_error,
                alpha=0.2, color='gray', label="95% theoretical bounds")

plt.xlabel("Batch number")
plt.ylabel("Probability estimate")
plt.title(f"Stability of Probability Estimates: Batches of {batch_size} experiments")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.15, 0.35)
plt.tight_layout()
plt.savefig('./chance_example_stability.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Chance Example Simulation Results ===")
print(f"Lean theorem: eventChance(<20, 5>) = (5, 20)")
print(f"Theoretical probability: {stats_summary['theoretical_probability']:.6f}")
print(f"Final convergence estimate: {stats_summary['final_convergence_estimate']:.6f}")
print(f"Convergence error: {stats_summary['convergence_error']:.6f}")
print(f"Theoretical mean successes (per 20 trials): {stats_summary['theoretical_mean_successes']:.2f}")
print(f"Empirical mean successes: {stats_summary['empirical_mean_successes']:.2f}")
print(f"Theoretical std successes: {stats_summary['theoretical_std_successes']:.3f}")
print(f"Empirical std successes: {stats_summary['empirical_std_successes']:.3f}")
print(f"\nConvergence trials: {stats_summary['trials_for_convergence']:,}")
print(f"Binomial experiments: {stats_summary['binomial_experiments']:,}")
print(f"Total simulated trials: {stats_summary['trials_for_convergence'] + stats_summary['binomial_experiments'] * stats_summary['trials_per_experiment']:,}")
print("\nTheorem verification: eventChance(<20, 5>) = (5, 20) (verified)")