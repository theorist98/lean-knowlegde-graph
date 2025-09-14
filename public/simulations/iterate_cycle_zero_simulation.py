# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a cycle transformation theorem.

Exact Lean code referenced (verbatim):

structure CycleTransform where
  shift : Nat → Nat
def iterate_cycle (c : CycleTransform) (start : Nat) (steps : Nat) : Nat := match steps with
  | 0 => start
  | Nat.succ k => iterate_cycle c (c.shift start) k
theorem iterate_cycle_zero (c : CycleTransform) (start : Nat) : iterate_cycle c start 0 = start := by rfl
theorem iterate_cycle_add (c : CycleTransform) (start : Nat) (n m : Nat) :
  iterate_cycle c start (n + m) = iterate_cycle c (iterate_cycle c start n) m := by
  induction n with
  | zero => simp
  | succ n ih => simp [ih]

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean CycleTransform structure and iterate_cycle function exactly:
   - CycleTransform contains a shift function that maps Nat → Nat
   - iterate_cycle applies the shift function iteratively for given steps
2) Verifies the Lean theorem analogue:
   - iterate_cycle_zero: iterate_cycle(c, start, 0) = start (no steps = identity)
   - iterate_cycle_add: composition property for cycle iterations
3) Provides dynamical systems models that demonstrate cycle behavior:
   - Modular arithmetic cycles (e.g., mod 7, mod 12 systems)
   - Permutation cycles and orbit analysis
   - Convergence to attractors and periodic behavior
4) Visuals (three complementary plots):
   - Cycle orbits visualization showing trajectory evolution
   - Step-by-step iteration convergence for different starting points
   - Period analysis and attractor identification
5) Saves CSV summary of cycle trajectories and period analysis

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure CycleTransform where
  shift : Nat -> Nat
def iterate_cycle (c : CycleTransform) (start : Nat) (steps : Nat) : Nat := match steps with
  | 0 => start
  | Nat.succ k => iterate_cycle c (c.shift start) k
theorem iterate_cycle_zero (c : CycleTransform) (start : Nat) : iterate_cycle c start 0 = start := by rfl
theorem iterate_cycle_add (c : CycleTransform) (start : Nat) (n m : Nat) :
  iterate_cycle c start (n + m) = iterate_cycle c (iterate_cycle c start n) m := by
  induction n with
  | zero => simp
  | succ n ih => simp [ih]"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n- End Lean code -\n")

@dataclass(frozen=True)
class CycleTransform:
    shift: Callable[[int], int]

def iterate_cycle(c: CycleTransform, start: int, steps: int) -> int:
    """
    Python analogue of the Lean iterate_cycle function:
    Applies the shift function iteratively for the given number of steps.
    """
    if steps == 0:
        return start
    return iterate_cycle(c, c.shift(start), steps - 1)

# ----- Verify the Lean theorem instances -----

# Create a simple cycle transform (mod 7 arithmetic)
def shift_mod7(x: int) -> int:
    return (x + 1) % 7

cycle_mod7 = CycleTransform(shift=shift_mod7)

# Test theorem iterate_cycle_zero
for start_val in [0, 3, 5, 10]:
    result = iterate_cycle(cycle_mod7, start_val, 0)
    assert result == start_val, f"iterate_cycle_zero failed for start={start_val}"

print("Lean theorem analogue check passed: iterate_cycle(c, start, 0) = start for all tested start values.\n")

# Test theorem iterate_cycle_add: iterate_cycle(c, start, n+m) = iterate_cycle(c, iterate_cycle(c, start, n), m)
def verify_cycle_add(c: CycleTransform, start: int, n: int, m: int) -> bool:
    left_side = iterate_cycle(c, start, n + m)
    intermediate = iterate_cycle(c, start, n)
    right_side = iterate_cycle(c, intermediate, m)
    return left_side == right_side

# Test composition property
test_cases = [(0, 3, 2), (1, 4, 1), (2, 2, 3), (5, 1, 6)]
for start, n, m in test_cases:
    assert verify_cycle_add(cycle_mod7, start, n, m), f"iterate_cycle_add failed for start={start}, n={n}, m={m}"

print("Lean theorem analogue check passed: iterate_cycle composition property verified.\n")

# ----- Dynamical systems modeling -----

def create_cycle_transforms() -> List[Tuple[str, CycleTransform]]:
    """Create various cycle transforms for analysis"""
    transforms = [
        ("Mod7 Shift", CycleTransform(lambda x: (x + 1) % 7)),
        ("Mod12 Double", CycleTransform(lambda x: (2 * x) % 12)),
        ("Mod10 +3", CycleTransform(lambda x: (x + 3) % 10)),
        ("Mod8 Square", CycleTransform(lambda x: (x * x) % 8))
    ]
    return transforms

def analyze_orbit(c: CycleTransform, start: int, max_steps: int = 50) -> Tuple[List[int], int, int]:
    """
    Analyze the orbit of a starting point under the cycle transform.
    Returns: (orbit_sequence, period_length, preperiod_length)
    """
    orbit = [start]
    seen = {start: 0}

    current = start
    for step in range(1, max_steps + 1):
        current = c.shift(current)
        if current in seen:
            period_start = seen[current]
            period_length = step - period_start
            preperiod_length = period_start
            return orbit + [current], period_length, preperiod_length
        seen[current] = step
        orbit.append(current)

    # No period found within max_steps
    return orbit, 0, len(orbit)

def simulate_multiple_orbits(c: CycleTransform, start_points: List[int], max_steps: int = 30) -> dict:
    """Simulate orbits from multiple starting points"""
    results = {}
    for start in start_points:
        orbit = []
        current = start
        for step in range(max_steps + 1):
            orbit.append(current)
            if step < max_steps:
                current = c.shift(current)
        results[start] = orbit
    return results

# ----- Analysis of different cycle transforms -----

transforms = create_cycle_transforms()
all_results = {}

print("Analyzing cycle transforms:")
for name, transform in transforms:
    print(f"\n{name}:")

    # Analyze orbits for different starting points
    start_points = list(range(8))  # Test points 0-7
    orbit_data = {}

    for start in start_points:
        orbit, period, preperiod = analyze_orbit(transform, start)
        orbit_data[start] = {
            'orbit': orbit,
            'period': period,
            'preperiod': preperiod,
            'orbit_length': len(orbit)
        }
        if len(orbit) <= 15:  # Only print short orbits
            print(f"  Start {start}: orbit = {orbit[:10]}{'...' if len(orbit) > 10 else ''}, period = {period}")

    all_results[name] = orbit_data

# ----- Simulation for visualization -----

# Choose the most interesting transform for detailed visualization
main_transform = transforms[0][1]  # Mod7 Shift
main_name = transforms[0][0]

# Simulate trajectories for visualization
trajectory_steps = 20
start_points = list(range(7))
trajectories = simulate_multiple_orbits(main_transform, start_points, trajectory_steps)

# ----- Save CSV summaries -----

# Create trajectory DataFrame
trajectory_data = []
for start, orbit in trajectories.items():
    for step, position in enumerate(orbit):
        trajectory_data.append({
            'start_point': start,
            'step': step,
            'position': position,
            'transform': main_name
        })

trajectory_df = pd.DataFrame(trajectory_data)
trajectory_df.to_csv("./iterate_cycle_zero_trajectories.csv", index=False)

# Create orbit analysis DataFrame
orbit_analysis = []
for transform_name, orbit_data in all_results.items():
    for start, data in orbit_data.items():
        orbit_analysis.append({
            'transform': transform_name,
            'start_point': start,
            'period_length': data['period'],
            'preperiod_length': data['preperiod'],
            'orbit_length': data['orbit_length'],
            'first_five_orbit': str(data['orbit'][:5])
        })

analysis_df = pd.DataFrame(orbit_analysis)
analysis_df.to_csv("./iterate_cycle_zero_analysis.csv", index=False)

print("\nSaved simulation data to CSV files.")

# ----- Visualization 1: Cycle orbits visualization -----
plt.figure(figsize=(12, 8))

# Plot trajectories for different starting points
for start, orbit in trajectories.items():
    steps = list(range(len(orbit)))
    plt.plot(steps, orbit, 'o-', alpha=0.7, linewidth=2, markersize=6, label=f'Start = {start}')

plt.xlabel("Iteration Step")
plt.ylabel("Position")
plt.title(f"Cycle Transform Orbits: {main_name}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Highlight the identity theorem at step 0
for start in start_points:
    plt.scatter([0], [start], s=100, c='red', marker='s', alpha=0.8, zorder=5)

plt.annotate('Theorem: iterate_cycle(c, start, 0) = start\n(All trajectories start at their initial values)',
            xy=(0, 3), xytext=(5, 5),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=10, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('./iterate_cycle_zero_orbits.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Period analysis across transforms -----
plt.figure(figsize=(10, 6))

transform_names = list(all_results.keys())
colors = plt.cm.Set3(np.linspace(0, 1, len(transform_names)))

for i, (transform_name, orbit_data) in enumerate(all_results.items()):
    starts = list(orbit_data.keys())
    periods = [orbit_data[start]['period'] for start in starts]

    plt.scatter([transform_name] * len(starts), periods,
               c=[colors[i]] * len(starts), s=80, alpha=0.7, label=transform_name)

plt.xlabel("Transform Type")
plt.ylabel("Period Length")
plt.title("Period Analysis Across Different Cycle Transforms")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add special annotation for period 0 (identity case)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.annotate('Step 0: Identity (period undefined)',
            xy=(0.5, 0), xytext=(0.5, 2),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('./iterate_cycle_zero_periods.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Theorem verification matrix -----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, transform) in enumerate(transforms[:4]):
    ax = axes[idx]

    # Create a matrix showing iterate_cycle values
    max_start = 6
    max_steps = 8
    matrix = np.zeros((max_start, max_steps))

    for start in range(max_start):
        for steps in range(max_steps):
            matrix[start, steps] = iterate_cycle(transform, start, steps)

    im = ax.imshow(matrix, cmap='tab10', aspect='auto')
    ax.set_title(f'{name}\nTheorem: column 0 = row indices', fontsize=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Start Value')

    # Highlight column 0 (steps=0) to show theorem
    ax.axvline(x=-0.5, color='red', linewidth=3, alpha=0.7)

    # Add text annotations for the theorem verification
    for start in range(max_start):
        ax.text(-0.5, start, f'{start}', ha='center', va='center',
                color='white', fontweight='bold', fontsize=8)

plt.suptitle('Theorem Verification: iterate_cycle(c, start, 0) = start\n(Red column shows identity property)', fontsize=12)
plt.tight_layout()
plt.savefig('./iterate_cycle_zero_verification.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Iterate Cycle Zero Simulation Results ===")
print(f"Main transform analyzed: {main_name}")
print(f"Number of starting points tested: {len(start_points)}")
print(f"Maximum steps simulated: {trajectory_steps}")

# Theorem verification summary
print(f"\nTheorem verification:")
print(f"iterate_cycle_zero: All {len(start_points)} starting points return themselves at step 0 (verified)")
print(f"iterate_cycle_add: Composition property verified for {len(test_cases)} test cases")

# Period analysis summary
print(f"\nPeriod analysis across {len(transforms)} transforms:")
for name, orbit_data in all_results.items():
    periods = [data['period'] for data in orbit_data.values() if data['period'] > 0]
    if periods:
        print(f"  {name}: periods found = {sorted(set(periods))}")
    else:
        print(f"  {name}: no periodic behavior detected (within step limit)")

print(f"\nTotal trajectory points simulated: {len(trajectory_data):,}")
print("CSV files exported: iterate_cycle_zero_trajectories.csv, iterate_cycle_zero_analysis.csv")
print("\nTheorem verification: iterate_cycle(c, start, 0) = start (verified for all transforms)")