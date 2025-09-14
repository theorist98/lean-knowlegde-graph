# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a stateful transform theorem.

Exact Lean code referenced (verbatim):

structure StatefulTransform where
  state : Nat -> Nat
def apply_transform (t : StatefulTransform) (initial : Nat) (iterations : Nat) : Nat :=
  match iterations with
  | 0 => initial
  | Nat.succ k => apply_transform t (t.state initial) k
theorem transform_identity (t : StatefulTransform) (initial : Nat) : apply_transform t initial 0 = initial := by rfl
theorem transform_one_step (t : StatefulTransform) (initial : Nat) : apply_transform t initial 1 = t.state initial := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean StatefulTransform structure and apply_transform function exactly:
   - StatefulTransform contains a state function that maps Nat -> Nat
   - apply_transform recursively applies the state function for given iterations
2) Verifies the Lean theorem analogues:
   - transform_identity: apply_transform(t, initial, 0) = initial (zero iterations = identity)
   - transform_one_step: apply_transform(t, initial, 1) = t.state(initial) (one iteration)
3) Provides dynamical systems models that demonstrate iterative state evolution:
   - Linear transformations (scaling, shifting)
   - Nonlinear maps (logistic map, quadratic maps)
   - Chaotic and convergent behavior analysis
4) Visuals (three complementary plots):
   - State evolution trajectories for different initial conditions
   - Fixed point analysis and convergence behavior
   - Phase portraits and attractor visualization
5) Saves CSV summary of transformation trajectories and convergence analysis

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure StatefulTransform where
  state : Nat -> Nat
def apply_transform (t : StatefulTransform) (initial : Nat) (iterations : Nat) : Nat :=
  match iterations with
  | 0 => initial
  | Nat.succ k => apply_transform t (t.state initial) k
theorem transform_identity (t : StatefulTransform) (initial : Nat) : apply_transform t initial 0 = initial := by rfl
theorem transform_one_step (t : StatefulTransform) (initial : Nat) : apply_transform t initial 1 = t.state initial := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n- End Lean code -\n")

@dataclass(frozen=True)
class StatefulTransform:
    state: Callable[[int], int]

def apply_transform(t: StatefulTransform, initial: int, iterations: int) -> int:
    """
    Python analogue of the Lean apply_transform function:
    Recursively applies the state function for the given number of iterations.
    """
    if iterations == 0:
        return initial
    return apply_transform(t, t.state(initial), iterations - 1)

# ----- Verify the Lean theorem instances -----

# Create sample stateful transforms for testing
def linear_state(x: int) -> int:
    """Linear transformation: multiply by 2 and add 1"""
    return 2 * x + 1

def quadratic_state(x: int) -> int:
    """Quadratic transformation: x^2 mod 100"""
    return (x * x) % 100

def mod_state(x: int) -> int:
    """Modular transformation: (x + 3) mod 10"""
    return (x + 3) % 10

# Test transforms
linear_transform = StatefulTransform(state=linear_state)
quadratic_transform = StatefulTransform(state=quadratic_state)
mod_transform = StatefulTransform(state=mod_state)

# Test theorem transform_identity: apply_transform(t, initial, 0) = initial
test_initials = [0, 5, 10, 25, 50]
test_transforms = [linear_transform, quadratic_transform, mod_transform]

print("Testing theorem transform_identity:")
for i, transform in enumerate(test_transforms):
    transform_name = ["Linear", "Quadratic", "Modular"][i]
    for initial in test_initials:
        result = apply_transform(transform, initial, 0)
        assert result == initial, f"transform_identity failed for {transform_name}, initial={initial}"
    print(f"  {transform_name} transform: identity verified for all initial values")

print("\nLean theorem analogue check passed: apply_transform(t, initial, 0) = initial for all tested cases.\n")

# Test theorem transform_one_step: apply_transform(t, initial, 1) = t.state(initial)
print("Testing theorem transform_one_step:")
for i, transform in enumerate(test_transforms):
    transform_name = ["Linear", "Quadratic", "Modular"][i]
    for initial in test_initials:
        result_one_step = apply_transform(transform, initial, 1)
        expected = transform.state(initial)
        assert result_one_step == expected, f"transform_one_step failed for {transform_name}, initial={initial}"
    print(f"  {transform_name} transform: one-step property verified for all initial values")

print("\nLean theorem analogue check passed: apply_transform(t, initial, 1) = t.state(initial) for all tested cases.\n")

# ----- Advanced dynamical systems modeling -----

def create_test_transforms() -> List[Tuple[str, StatefulTransform]]:
    """Create various stateful transforms for analysis"""

    def collatz_step(n: int) -> int:
        """Collatz conjecture step: n/2 if even, 3n+1 if odd"""
        if n <= 0:
            return 1
        return n // 2 if n % 2 == 0 else 3 * n + 1

    def logistic_discrete(x: int) -> int:
        """Discrete logistic map: r*x*(100-x)/100 with r=3"""
        if x < 0 or x > 100:
            return x % 100
        return (3 * x * (100 - x)) // 100

    def fibonacci_mod(x: int) -> int:
        """Fibonacci-inspired: (x + prev_x) mod 50, simplified as x+1 for demo"""
        return (x + (x + 1)) % 50

    def tent_map(x: int) -> int:
        """Tent map: 2*x if x<50, 2*(100-x) if x>=50"""
        if x < 50:
            return (2 * x) % 100
        else:
            return (2 * (100 - x)) % 100

    transforms = [
        ("Linear 2x+1", StatefulTransform(lambda x: (2 * x + 1) % 100)),
        ("Quadratic x^2", StatefulTransform(lambda x: (x * x) % 100)),
        ("Collatz Step", StatefulTransform(collatz_step)),
        ("Discrete Logistic", StatefulTransform(logistic_discrete)),
        ("Tent Map", StatefulTransform(tent_map))
    ]
    return transforms

def simulate_trajectory(transform: StatefulTransform, initial: int, max_iterations: int) -> List[int]:
    """Simulate the trajectory of a transform starting from initial value"""
    trajectory = [initial]
    current = initial

    for i in range(1, max_iterations + 1):
        current = apply_transform(transform, initial, i)
        trajectory.append(current)

        # Early stopping for cycles or fixed points
        if len(trajectory) > 10 and current in trajectory[-10:-1]:
            break

    return trajectory

def analyze_fixed_points(transform: StatefulTransform, search_range: int = 100) -> List[int]:
    """Find fixed points: points where state(x) = x"""
    fixed_points = []
    for x in range(search_range):
        if transform.state(x) == x:
            fixed_points.append(x)
    return fixed_points

def analyze_convergence(transform: StatefulTransform, initial_values: List[int], max_iterations: int = 50) -> dict:
    """Analyze convergence behavior for different initial values"""
    results = {}

    for initial in initial_values:
        trajectory = simulate_trajectory(transform, initial, max_iterations)

        # Check for fixed point convergence
        converged = False
        fixed_point = None
        if len(trajectory) > 5:
            last_values = trajectory[-5:]
            if len(set(last_values)) == 1:
                converged = True
                fixed_point = last_values[0]

        # Check for cycles
        cycle_length = 0
        if not converged and len(trajectory) > 10:
            for period in range(2, min(10, len(trajectory) // 2)):
                if trajectory[-period:] == trajectory[-2*period:-period]:
                    cycle_length = period
                    break

        results[initial] = {
            'trajectory': trajectory,
            'length': len(trajectory),
            'converged': converged,
            'fixed_point': fixed_point,
            'cycle_length': cycle_length,
            'final_value': trajectory[-1]
        }

    return results

# ----- Run comprehensive analysis -----

transforms = create_test_transforms()
analysis_results = {}

print("Analyzing stateful transforms:")
for name, transform in transforms:
    print(f"\n{name}:")

    # Find fixed points
    fixed_points = analyze_fixed_points(transform)
    print(f"  Fixed points: {fixed_points[:5]}{'...' if len(fixed_points) > 5 else ''} (total: {len(fixed_points)})")

    # Analyze convergence for various initial values
    initial_values = [1, 5, 10, 25, 50, 75, 90]
    convergence_analysis = analyze_convergence(transform, initial_values)

    # Count convergent vs. non-convergent trajectories
    converged_count = sum(1 for result in convergence_analysis.values() if result['converged'])
    cyclic_count = sum(1 for result in convergence_analysis.values() if result['cycle_length'] > 0)

    print(f"  Convergent trajectories: {converged_count}/{len(initial_values)}")
    print(f"  Cyclic trajectories: {cyclic_count}/{len(initial_values)}")

    analysis_results[name] = {
        'fixed_points': fixed_points,
        'convergence_analysis': convergence_analysis
    }

# ----- Simulation for detailed visualization -----

# Choose the most interesting transform for detailed analysis
main_transform_name = "Tent Map"
main_transform = next(transform for name, transform in transforms if name == main_transform_name)

# Simulate multiple trajectories
initial_values = [5, 15, 25, 35, 45, 55, 65, 75]
max_iterations = 30

detailed_trajectories = {}
for initial in initial_values:
    trajectory = simulate_trajectory(main_transform, initial, max_iterations)
    detailed_trajectories[initial] = trajectory

# ----- Save CSV summaries -----

# Create trajectory data
trajectory_data = []
for initial, trajectory in detailed_trajectories.items():
    for iteration, value in enumerate(trajectory):
        trajectory_data.append({
            'initial_value': initial,
            'iteration': iteration,
            'value': value,
            'transform': main_transform_name
        })

trajectory_df = pd.DataFrame(trajectory_data)
trajectory_df.to_csv("./transform_identity_trajectories.csv", index=False)

# Create analysis summary
analysis_data = []
for transform_name, results in analysis_results.items():
    for initial, analysis in results['convergence_analysis'].items():
        analysis_data.append({
            'transform': transform_name,
            'initial_value': initial,
            'trajectory_length': analysis['length'],
            'converged': analysis['converged'],
            'fixed_point': analysis['fixed_point'],
            'cycle_length': analysis['cycle_length'],
            'final_value': analysis['final_value']
        })

analysis_df = pd.DataFrame(analysis_data)
analysis_df.to_csv("./transform_identity_analysis.csv", index=False)

print(f"\nSaved simulation data to CSV files.")

# ----- Visualization 1: Trajectory evolution -----
plt.figure(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, len(detailed_trajectories)))

for i, (initial, trajectory) in enumerate(detailed_trajectories.items()):
    iterations = list(range(len(trajectory)))
    plt.plot(iterations, trajectory, 'o-', color=colors[i], alpha=0.7,
             linewidth=2, markersize=4, label=f'Initial = {initial}')

    # Highlight theorem verification at iteration 0
    plt.scatter([0], [initial], s=120, c='red', marker='s', alpha=0.8, zorder=5)

plt.xlabel("Iteration")
plt.ylabel("State Value")
plt.title(f"Stateful Transform Evolution: {main_transform_name}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Annotate theorem
plt.annotate('Theorem: apply_transform(t, initial, 0) = initial\n(Red squares show identity at iteration 0)',
            xy=(0, 40), xytext=(5, 70),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=10, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('./transform_identity_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Fixed point analysis -----
plt.figure(figsize=(10, 6))

transform_names = [name for name, _ in transforms]
fixed_point_counts = [len(analysis_results[name]['fixed_points']) for name in transform_names]

bars = plt.bar(transform_names, fixed_point_counts, alpha=0.7, color='skyblue', edgecolor='black')

# Annotate bars with counts
for bar, count in zip(bars, fixed_point_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.xlabel("Transform Type")
plt.ylabel("Number of Fixed Points")
plt.title("Fixed Point Analysis: Points where state(x) = x")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add horizontal line for identity reference
plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Single fixed point reference')
plt.legend()

plt.tight_layout()
plt.savefig('./transform_identity_fixed_points.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Convergence behavior matrix -----
fig, ax = plt.subplots(figsize=(12, 8))

# Create a matrix of convergence types
transform_list = list(analysis_results.keys())
initial_list = [1, 5, 10, 25, 50, 75, 90]

convergence_matrix = np.zeros((len(transform_list), len(initial_list)))

for i, transform_name in enumerate(transform_list):
    for j, initial in enumerate(initial_list):
        if initial in analysis_results[transform_name]['convergence_analysis']:
            analysis = analysis_results[transform_name]['convergence_analysis'][initial]
            if analysis['converged']:
                convergence_matrix[i, j] = 1  # Converged
            elif analysis['cycle_length'] > 0:
                convergence_matrix[i, j] = 0.5  # Cyclic
            else:
                convergence_matrix[i, j] = 0.2  # Chaotic/other

im = ax.imshow(convergence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Convergence Type', rotation=270, labelpad=15)
cbar.set_ticks([0.2, 0.5, 1.0])
cbar.set_ticklabels(['Chaotic', 'Cyclic', 'Converged'])

# Set labels
ax.set_xticks(range(len(initial_list)))
ax.set_xticklabels(initial_list)
ax.set_yticks(range(len(transform_list)))
ax.set_yticklabels(transform_list)
ax.set_xlabel('Initial Value')
ax.set_ylabel('Transform Type')
ax.set_title('Convergence Behavior Analysis\n(Theorem: All start at identity for iteration 0)')

# Add text annotations
for i in range(len(transform_list)):
    for j in range(len(initial_list)):
        value = convergence_matrix[i, j]
        if value == 1.0:
            text = 'C'  # Converged
        elif value == 0.5:
            text = 'Y'  # Cyclic
        else:
            text = 'X'  # Other
        ax.text(j, i, text, ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('./transform_identity_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Transform Identity Simulation Results ===")
print(f"Main transform analyzed: {main_transform_name}")
print(f"Number of transforms studied: {len(transforms)}")
print(f"Initial values tested: {initial_values}")

# Theorem verification summary
total_tests = len(test_transforms) * len(test_initials)
print(f"\nTheorem verification:")
print(f"transform_identity: {total_tests} test cases verified (all passed)")
print(f"transform_one_step: {total_tests} test cases verified (all passed)")

# Analysis summary
total_fixed_points = sum(len(results['fixed_points']) for results in analysis_results.values())
print(f"\nFixed point analysis:")
print(f"Total fixed points found across all transforms: {total_fixed_points}")

convergence_summary = {}
for transform_name, results in analysis_results.items():
    convergent = sum(1 for analysis in results['convergence_analysis'].values() if analysis['converged'])
    cyclic = sum(1 for analysis in results['convergence_analysis'].values() if analysis['cycle_length'] > 0)
    convergence_summary[transform_name] = {'converged': convergent, 'cyclic': cyclic}

print(f"Convergence behavior summary:")
for transform_name, summary in convergence_summary.items():
    print(f"  {transform_name}: {summary['converged']} converged, {summary['cyclic']} cyclic")

print(f"\nTotal trajectory points simulated: {len(trajectory_data):,}")
print("CSV files exported: transform_identity_trajectories.csv, transform_identity_analysis.csv")
print("\nTheorem verification: apply_transform(t, initial, 0) = initial (verified for all transforms)")
print("Theorem verification: apply_transform(t, initial, 1) = t.state(initial) (verified for all transforms)")