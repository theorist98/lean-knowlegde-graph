# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a bounded function composition theorem.

Exact Lean code referenced (verbatim):

structure BoundedFunction where
  domain : List Nat
  range : List Nat
  bound : Nat
def boundedCompose (f g : BoundedFunction) : BoundedFunction := ⟨f.domain, g.range, max f.bound g.bound⟩
theorem bounded_compose_bound (f g : BoundedFunction) : boundedCompose f g = ⟨f.domain, g.range, max f.bound g.bound⟩ := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean BoundedFunction structure and boundedCompose function exactly:
   - BoundedFunction contains domain, range (as lists), and bound (maximum value)
   - boundedCompose creates a new function with combined domain/range and max bound
2) Verifies the Lean theorem analogue:
   - bounded_compose_bound: boundedCompose(f, g) preserves max bound property
3) Provides functional analysis models demonstrating composition behavior:
   - Linear function compositions with bound preservation
   - Polynomial function compositions and bound analysis
   - Trigonometric function approximations within bounded domains
4) Visuals (three complementary plots):
   - Function composition visualization showing bound preservation
   - Bound evolution analysis across multiple compositions
   - Domain-range mapping with bound verification
5) Saves CSV summary of composition analysis and bound verification

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Callable
import math

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure BoundedFunction where
  domain : List Nat
  range : List Nat
  bound : Nat
def boundedCompose (f g : BoundedFunction) : BoundedFunction := <f.domain, g.range, max f.bound g.bound>
theorem bounded_compose_bound (f g : BoundedFunction) : boundedCompose f g = <f.domain, g.range, max f.bound g.bound> := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n- End Lean code -\n")

@dataclass(frozen=True)
class BoundedFunction:
    domain: List[int]
    range: List[int]
    bound: int
    func: Callable[[int], int] = None  # Optional function implementation

def bounded_compose(f: BoundedFunction, g: BoundedFunction) -> BoundedFunction:
    """
    Python analogue of the Lean boundedCompose function:
    Creates composition with f.domain, g.range, and max(f.bound, g.bound)
    """
    return BoundedFunction(
        domain=f.domain,
        range=g.range,
        bound=max(f.bound, g.bound)
    )

# ----- Verify the Lean theorem instances -----

# Create sample bounded functions for testing
def create_test_functions():
    """Create various bounded functions for comprehensive testing"""

    # Linear functions
    linear_f1 = BoundedFunction(
        domain=[0, 1, 2, 3, 4, 5],
        range=[0, 2, 4, 6, 8, 10],
        bound=10,
        func=lambda x: 2*x
    )

    linear_f2 = BoundedFunction(
        domain=[0, 2, 4, 6, 8, 10],
        range=[0, 1, 2, 3, 4, 5],
        bound=5,
        func=lambda x: x // 2
    )

    # Quadratic function (bounded)
    quad_f = BoundedFunction(
        domain=[0, 1, 2, 3],
        range=[0, 1, 4, 9],
        bound=9,
        func=lambda x: x*x
    )

    # Modular arithmetic function
    mod_f = BoundedFunction(
        domain=[0, 1, 2, 3, 4, 5, 6, 7],
        range=[0, 1, 2, 3, 0, 1, 2, 3],
        bound=7,
        func=lambda x: x % 4
    )

    # Trigonometric approximation (discrete)
    trig_domain = list(range(0, 13))  # 0 to 12 (representing 0 to 2π in 12 steps)
    trig_range = [int(5 * math.sin(x * math.pi / 6) + 5) for x in trig_domain]  # Scaled to [0,10]
    trig_f = BoundedFunction(
        domain=trig_domain,
        range=trig_range,
        bound=10,
        func=lambda x: int(5 * math.sin(x * math.pi / 6) + 5)
    )

    # Exponential decay (bounded)
    exp_domain = list(range(0, 8))
    exp_range = [int(10 * math.exp(-x/3)) for x in exp_domain]
    exp_f = BoundedFunction(
        domain=exp_domain,
        range=exp_range,
        bound=10,
        func=lambda x: int(10 * math.exp(-x/3))
    )

    return [
        ("Linear 2x", linear_f1),
        ("Linear x/2", linear_f2),
        ("Quadratic x²", quad_f),
        ("Modular x%4", mod_f),
        ("Sine Wave", trig_f),
        ("Exp Decay", exp_f)
    ]

test_functions = create_test_functions()

# Test theorem bounded_compose_bound
print("Testing theorem bounded_compose_bound:")

composition_results = []
for i, (name_f, f) in enumerate(test_functions):
    for j, (name_g, g) in enumerate(test_functions):
        if i != j:  # Don't compose function with itself for variety
            composed = bounded_compose(f, g)
            expected_bound = max(f.bound, g.bound)

            # Verify the theorem
            assert composed.bound == expected_bound, f"bounded_compose_bound failed for {name_f} ∘ {name_g}"
            assert composed.domain == f.domain, f"Domain preservation failed for {name_f} ∘ {name_g}"
            assert composed.range == g.range, f"Range preservation failed for {name_f} ∘ {name_g}"

            composition_results.append({
                'f_name': name_f,
                'g_name': name_g,
                'f_bound': f.bound,
                'g_bound': g.bound,
                'composed_bound': composed.bound,
                'expected_bound': expected_bound,
                'theorem_verified': composed.bound == expected_bound
            })

            if len(composition_results) % 5 == 0:  # Print every 5th result
                print(f"  {name_f} o {name_g}: bounds({f.bound}, {g.bound}) -> max = {composed.bound} (verified)")

total_compositions = len(composition_results)
verified_count = sum(1 for r in composition_results if r['theorem_verified'])
print(f"\nLean theorem analogue check passed: bounded_compose_bound verified for all {verified_count}/{total_compositions} compositions.\n")

# ----- Advanced function composition analysis -----

def analyze_composition_chain(functions: List[BoundedFunction], chain_length: int = 4):
    """Analyze chains of function compositions and bound evolution"""

    if len(functions) < chain_length:
        return []

    chain_results = []

    # Create a chain of compositions
    current = functions[0]
    bounds_evolution = [current.bound]
    names_chain = [f"f0(bound={current.bound})"]

    for i in range(1, chain_length):
        next_func = functions[i % len(functions)]
        current = bounded_compose(current, next_func)
        bounds_evolution.append(current.bound)
        names_chain.append(f"f{i}(bound={next_func.bound})")

    chain_results.append({
        'chain': " o ".join(names_chain),
        'bounds_evolution': bounds_evolution,
        'final_bound': current.bound,
        'domain_size': len(current.domain),
        'range_size': len(current.range)
    })

    return chain_results

def simulate_composition_statistics(functions: List[BoundedFunction], num_trials: int = 100):
    """Generate statistics on random function compositions"""

    np.random.seed(20250914)  # For reproducibility

    composition_stats = {
        'bound_increases': 0,
        'bound_stays_same': 0,
        'max_bound_achieved': 0,
        'bound_distribution': [],
        'composition_pairs': []
    }

    for _ in range(num_trials):
        # Randomly select two functions
        f_idx, g_idx = np.random.choice(len(functions), 2, replace=False)
        f = functions[f_idx][1]  # Get BoundedFunction from (name, function) tuple
        g = functions[g_idx][1]

        composed = bounded_compose(f, g)

        if composed.bound > max(f.bound, g.bound):
            composition_stats['bound_increases'] += 1
        elif composed.bound == max(f.bound, g.bound):
            composition_stats['bound_stays_same'] += 1

        composition_stats['max_bound_achieved'] = max(composition_stats['max_bound_achieved'], composed.bound)
        composition_stats['bound_distribution'].append(composed.bound)
        composition_stats['composition_pairs'].append((f.bound, g.bound, composed.bound))

    return composition_stats

# Run advanced analysis
print("Running advanced composition analysis:")

# Analyze composition chains
chain_analysis = analyze_composition_chain([f for _, f in test_functions], chain_length=4)
for result in chain_analysis:
    print(f"  Composition chain: {result['chain']}")
    print(f"    Bounds evolution: {result['bounds_evolution']}")
    print(f"    Final bound: {result['final_bound']}")

# Generate composition statistics
comp_stats = simulate_composition_statistics(test_functions, num_trials=50)
print(f"\nComposition statistics (50 random trials):")
print(f"  Bound stays at max: {comp_stats['bound_stays_same']}/50 = {comp_stats['bound_stays_same']/50:.1%}")
print(f"  Maximum bound achieved: {comp_stats['max_bound_achieved']}")
print(f"  Average composed bound: {np.mean(comp_stats['bound_distribution']):.2f}")

# ----- Save CSV summaries -----

# Create composition results DataFrame
composition_df = pd.DataFrame(composition_results)
composition_df.to_csv("./bounded_compose_bound_results.csv", index=False)

# Create composition statistics DataFrame
stats_data = []
for f_bound, g_bound, comp_bound in comp_stats['composition_pairs']:
    stats_data.append({
        'f_bound': f_bound,
        'g_bound': g_bound,
        'composed_bound': comp_bound,
        'theoretical_max': max(f_bound, g_bound),
        'theorem_holds': comp_bound == max(f_bound, g_bound)
    })

stats_df = pd.DataFrame(stats_data)
stats_df.to_csv("./bounded_compose_bound_statistics.csv", index=False)

print(f"\nSaved simulation data to CSV files.")

# ----- Visualization 1: Bound preservation in composition -----
plt.figure(figsize=(12, 8))

# Plot composition results showing bound relationships
f_bounds = [r['f_bound'] for r in composition_results[:20]]  # First 20 for clarity
g_bounds = [r['g_bound'] for r in composition_results[:20]]
composed_bounds = [r['composed_bound'] for r in composition_results[:20]]
expected_bounds = [r['expected_bound'] for r in composition_results[:20]]

x_positions = range(len(f_bounds))

plt.plot(x_positions, f_bounds, 'o-', label='f.bound', alpha=0.7, linewidth=2, markersize=6)
plt.plot(x_positions, g_bounds, 's-', label='g.bound', alpha=0.7, linewidth=2, markersize=6)
plt.plot(x_positions, composed_bounds, '^-', label='composed.bound', alpha=0.9, linewidth=3, markersize=8, color='red')
plt.plot(x_positions, expected_bounds, 'x--', label='max(f.bound, g.bound)', alpha=0.8, linewidth=2, markersize=8, color='black')

plt.xlabel("Composition Index")
plt.ylabel("Bound Value")
plt.title("Bounded Function Composition: Theorem Verification\nbounded_compose_bound: composed.bound = max(f.bound, g.bound)")
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight theorem verification
for i, (comp, exp) in enumerate(zip(composed_bounds, expected_bounds)):
    if comp == exp:
        plt.scatter([i], [comp], s=100, c='green', marker='o', alpha=0.8, zorder=5)

plt.tight_layout()
plt.savefig('./bounded_compose_bound_verification.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Bound distribution analysis -----
plt.figure(figsize=(10, 6))

bound_dist = comp_stats['bound_distribution']
unique_bounds = sorted(set(bound_dist))
bound_counts = [bound_dist.count(b) for b in unique_bounds]

bars = plt.bar(unique_bounds, bound_counts, alpha=0.7, color='skyblue', edgecolor='black')

# Annotate bars
for bar, count in zip(bars, bound_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.xlabel("Composed Function Bound")
plt.ylabel("Frequency")
plt.title("Distribution of Bounds in Random Function Compositions\n(50 trials demonstrating max bound preservation)")
plt.grid(True, alpha=0.3, axis='y')

# Add theorem annotation
plt.axvline(x=np.mean(bound_dist), color='red', linestyle='--', alpha=0.7, label=f'Mean bound: {np.mean(bound_dist):.1f}')
plt.legend()

plt.tight_layout()
plt.savefig('./bounded_compose_bound_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Function domain-range mapping -----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, func) in enumerate(test_functions[:6]):
    ax = axes[idx]

    # Plot function mapping
    domain_vals = func.domain[:min(len(func.domain), 10)]  # Limit for clarity
    range_vals = func.range[:len(domain_vals)]

    ax.plot(domain_vals, range_vals, 'o-', linewidth=2, markersize=8, alpha=0.8)

    # Highlight bound
    ax.axhline(y=func.bound, color='red', linestyle='--', alpha=0.7,
               label=f'Bound = {func.bound}')

    # Fill area under bound
    ax.fill_between(domain_vals, 0, func.bound, alpha=0.2, color='yellow')

    ax.set_title(f'{name}\nBound = {func.bound}', fontsize=10)
    ax.set_xlabel('Domain')
    ax.set_ylabel('Range')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Bounded Functions: Domain-Range Mappings\n(Yellow area shows bounded region)', fontsize=14)
plt.tight_layout()
plt.savefig('./bounded_compose_bound_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Bounded Compose Bound Simulation Results ===")
print(f"Functions analyzed: {len(test_functions)}")
print(f"Total compositions tested: {total_compositions}")
print(f"Theorem verifications: {verified_count}/{total_compositions} (100%)")

print(f"\nTheorem verification:")
print(f"bounded_compose_bound: boundedCompose(f, g).bound = max(f.bound, g.bound) (verified for all compositions)")

print(f"\nFunction types analyzed:")
for name, func in test_functions:
    print(f"  {name}: domain size {len(func.domain)}, range size {len(func.range)}, bound {func.bound}")

print(f"\nComposition chain analysis:")
for result in chain_analysis:
    print(f"  Final bound after 4 compositions: {result['final_bound']}")
    print(f"  Domain size preserved: {result['domain_size']}")

print(f"\nStatistical summary:")
print(f"  Mean composed bound: {np.mean(comp_stats['bound_distribution']):.2f}")
print(f"  Max bound achieved: {comp_stats['max_bound_achieved']}")
print(f"  Theorem compliance: {comp_stats['bound_stays_same']}/50 = {comp_stats['bound_stays_same']/50:.1%}")

print(f"\nCSV files exported: bounded_compose_bound_results.csv, bounded_compose_bound_statistics.csv")
print("Theorem verification: boundedCompose(f, g).bound = max(f.bound, g.bound) (verified for all test cases)")