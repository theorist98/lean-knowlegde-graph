# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a basic stochastic process theorem.

Exact Lean code referenced (verbatim):

structure RandomWalk where
  steps : List Int
  probability : Nat -> Nat
def walkStep (rw : RandomWalk) (n : Nat) : Int := rw.steps.get? n |>.getD 0
theorem walk_starts_zero : walkStep <[0, 1, -1, 2], fun _ => 1> 0 = 0 := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean RandomWalk structure and walkStep function exactly:
   - RandomWalk contains a list of steps and a probability function
   - walkStep gets the nth step from the list, defaulting to 0 if index is out of bounds
2) Verifies the Lean theorem analogue:
   - walkStep(<[0, 1, -1, 2], fun _ => 1>, 0) == 0 (the walk starts at zero)
3) Provides stochastic models that demonstrate random walk properties:
   - Simple random walks with +1/-1 steps
   - Cumulative sum trajectories showing walk evolution
   - Statistical analysis of walk properties (mean, variance, distribution)
4) Visuals (three complementary plots):
   - Multiple random walk trajectories starting from zero
   - Distribution of final positions after N steps
   - Analysis of step-by-step position statistics
5) Saves a CSV summary of walk trajectories and statistics

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Callable, Tuple

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure RandomWalk where
  steps : List Int
  probability : Nat -> Nat
def walkStep (rw : RandomWalk) (n : Nat) : Int := rw.steps.get? n |>.getD 0
theorem walk_starts_zero : walkStep <[0, 1, -1, 2], fun _ => 1> 0 = 0 := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n— End Lean code —\n")

@dataclass(frozen=True)
class RandomWalk:
    steps: List[int]
    probability: Callable[[int], int]

def walk_step(rw: RandomWalk, n: int) -> int:
    """
    Python analogue of the Lean walkStep function:
    Gets the nth step from the steps list, defaulting to 0 if index is out of bounds.
    """
    if 0 <= n < len(rw.steps):
        return rw.steps[n]
    return 0

# ----- Verify the Lean theorem instance -----
lean_walk = RandomWalk(steps=[0, 1, -1, 2], probability=lambda _: 1)
assert walk_step(lean_walk, 0) == 0, "Lean theorem analogue failed."
print("Lean theorem analogue check passed: walkStep(<[0, 1, -1, 2], fun _ => 1>, 0) = 0 (walk starts at zero).\n")

# ----- Stochastic random walk models -----

def simulate_simple_random_walk(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a simple random walk: each step is +1 or -1 with equal probability.
    Returns cumulative sum starting from 0.
    """
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.cumsum(np.concatenate([[0], steps]))  # Start at 0
    return positions

def simulate_random_walk_ensemble(n_walks: int, n_steps: int, seed: int = 20250914) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate multiple random walk trajectories.
    Returns (time_points, walk_matrix) where each row is one walk trajectory.
    """
    rng = np.random.default_rng(seed)
    walks = np.empty((n_walks, n_steps + 1))

    for i in range(n_walks):
        walks[i] = simulate_simple_random_walk(n_steps, rng)

    time_points = np.arange(0, n_steps + 1)
    return time_points, walks

def analyze_walk_statistics(walks: np.ndarray) -> dict:
    """
    Analyze statistical properties of random walk ensemble.
    """
    n_walks, n_steps = walks.shape
    final_positions = walks[:, -1]

    stats = {
        "n_walks": n_walks,
        "n_steps": n_steps - 1,
        "mean_final_position": np.mean(final_positions),
        "std_final_position": np.std(final_positions),
        "min_final_position": np.min(final_positions),
        "max_final_position": np.max(final_positions),
        "theoretical_variance": n_steps - 1,  # For simple random walk: Var[X_n] = n
        "empirical_variance": np.var(final_positions)
    }

    return stats

# ----- Parameters for simulation -----
n_walks = 1000
n_steps = 100
time_points, walks = simulate_random_walk_ensemble(n_walks, n_steps)

# ----- Statistical analysis -----
stats = analyze_walk_statistics(walks)

# ----- Save summary CSV -----
# Create DataFrame with walk trajectories
df_walks = pd.DataFrame(walks.T, columns=[f"walk_{i}" for i in range(n_walks)])
df_walks.insert(0, "step", time_points)

# Add summary statistics
stats_df = pd.DataFrame([stats])

csv_path = "./random_walk_summary.csv"
df_walks.to_csv(csv_path, index=False)
print(f"Saved walk trajectories to: {csv_path}")

stats_csv_path = "./random_walk_statistics.csv"
stats_df.to_csv(stats_csv_path, index=False)
print(f"Saved statistics summary to: {stats_csv_path}\n")

# ----- Visualization 1: Multiple random walk trajectories -----
plt.figure(figsize=(10, 6))

# Plot a subset of walks for clarity
n_display = min(50, n_walks)
for i in range(n_display):
    alpha = 0.3 if i > 10 else 0.7  # Highlight first few walks
    plt.plot(time_points, walks[i], alpha=alpha, linewidth=0.8)

# Highlight the mean trajectory
mean_trajectory = np.mean(walks, axis=0)
plt.plot(time_points, mean_trajectory, 'r-', linewidth=2, label=f'Mean of {n_walks} walks')

# Mark the starting point (theorem verification)
plt.scatter([0], [0], color='black', s=100, zorder=5, label='Start: walk_starts_zero')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel("Step number")
plt.ylabel("Position")
plt.title(f"Random Walk Trajectories: {n_display} of {n_walks} walks shown")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./random_walk_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Distribution of final positions -----
final_positions = walks[:, -1]

plt.figure(figsize=(10, 6))
plt.hist(final_positions, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Overlay theoretical normal distribution
x_theory = np.linspace(final_positions.min(), final_positions.max(), 200)
theoretical_std = np.sqrt(n_steps)  # For simple random walk
y_theory = (1/np.sqrt(2*np.pi*theoretical_std**2)) * np.exp(-x_theory**2/(2*theoretical_std**2))
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label=f'Theoretical N(0, {n_steps})')

plt.axvline(np.mean(final_positions), color='orange', linestyle='--',
            label=f'Empirical mean: {np.mean(final_positions):.2f}')
plt.axvline(0, color='black', linestyle='-', alpha=0.8, label='Theoretical mean: 0')

plt.xlabel("Final position after 100 steps")
plt.ylabel("Density")
plt.title(f"Distribution of Final Positions ({n_walks} random walks)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./random_walk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Variance growth over time -----
step_variances = np.var(walks, axis=0)
theoretical_variances = time_points  # For simple random walk: Var[X_n] = n

plt.figure(figsize=(10, 6))
plt.plot(time_points, step_variances, 'b-', linewidth=2, label='Empirical variance')
plt.plot(time_points, theoretical_variances, 'r--', linewidth=2, label='Theoretical variance = n')

plt.xlabel("Step number")
plt.ylabel("Variance of position")
plt.title("Variance Growth in Random Walks")
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotation about the starting point theorem
plt.annotate('Theorem: walk starts at 0\n(variance = 0)',
            xy=(0, 0), xytext=(20, 50),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
            fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('./random_walk_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("=== Random Walk Simulation Results ===")
print(f"Number of walks simulated: {stats['n_walks']}")
print(f"Number of steps per walk: {stats['n_steps']}")
print(f"Mean final position: {stats['mean_final_position']:.4f}")
print(f"Std final position: {stats['std_final_position']:.4f}")
print(f"Theoretical standard deviation: {np.sqrt(stats['theoretical_variance']):.4f}")
print(f"Position range: [{stats['min_final_position']}, {stats['max_final_position']}]")
print(f"\nTheorem verification: All walks start at position 0 (verified)")
print(f"Empirical variance: {stats['empirical_variance']:.2f}")
print(f"Theoretical variance: {stats['theoretical_variance']}")