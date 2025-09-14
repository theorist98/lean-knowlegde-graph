# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for a circular transform theorem.

Exact Lean code referenced (verbatim):

structure CircularTransform where
  rotation : Nat -> Nat
def rotate (t : CircularTransform) (x : Nat) (n : Nat) : Nat := match n with | 0 => x | Nat.succ k => t.rotation (rotate t x k)
theorem rotate_zero (t : CircularTransform) (x : Nat) : rotate t x 0 = x := by rfl

What this script does (and how each step ties to the Lean code):
1) Mirrors the Lean CircularTransform structure and rotate function exactly:
   - CircularTransform contains a rotation function that maps Nat -> Nat
   - rotate recursively applies the rotation function n times
2) Verifies the Lean theorem analogue:
   - rotate_zero: rotate(t, x, 0) = x (zero rotations = identity)
3) Provides diverse rotational models demonstrating circular dynamics:
   - Mathematical group rotations and permutations
   - Physical rotation analogies (wheels, gears, planetary motion)
   - Abstract algebraic circular structures
4) Visuals (three complementary plots):
   - Multi-scale rotation visualization across different moduli
   - Convergence and periodicity analysis for rotation sequences
   - Identity preservation demonstration and theorem verification
5) Saves CSV summary of rotation dynamics and theoretical analysis

Dependencies: numpy, matplotlib, pandas
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Tuple
import math

# ----- Mirror the Lean definitions exactly -----

LEAN_CODE = """structure CircularTransform where
  rotation : Nat -> Nat
def rotate (t : CircularTransform) (x : Nat) (n : Nat) : Nat := match n with | 0 => x | Nat.succ k => t.rotation (rotate t x k)
theorem rotate_zero (t : CircularTransform) (x : Nat) : rotate t x 0 = x := by rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n- End Lean code -\n")

@dataclass(frozen=True)
class CircularTransform:
    rotation: Callable[[int], int]

def rotate(t: CircularTransform, x: int, n: int) -> int:
    """
    Python analogue of the Lean rotate function:
    Recursively applies the rotation function n times.
    """
    if n == 0:
        return x
    return t.rotation(rotate(t, x, n - 1))

# ----- Verify the Lean theorem instances -----

# Create diverse circular transforms for comprehensive testing
def create_rotation_transforms() -> List[Tuple[str, CircularTransform, str]]:
    """Create various rotation transforms with descriptions"""

    def gear_rotation_10(x: int) -> int:
        """10-tooth gear: advance by 1 tooth"""
        return (x + 1) % 10

    def planetary_rotation_24(x: int) -> int:
        """24-hour planetary rotation: advance by 1 hour"""
        return (x + 1) % 24

    def fibonacci_rotation(x: int) -> int:
        """Fibonacci-inspired rotation: (x + fib_step) mod 13"""
        fib_step = ((x % 5) + ((x // 5) % 5)) % 8 + 1  # Simplified Fibonacci-like
        return (x + fib_step) % 13

    def binary_rotation(x: int) -> int:
        """Binary rotation: bit shift in 8-bit space"""
        return ((x << 1) | (x >> 7)) & 0xFF

    def prime_rotation(x: int) -> int:
        """Prime-based rotation: advance by next prime-like step"""
        primes = [1, 2, 3, 5, 7, 11]
        step = primes[x % len(primes)]
        return (x + step) % 17

    def harmonic_rotation(x: int) -> int:
        """Harmonic rotation: musical octave (12-tone)"""
        return (x + 7) % 12  # Perfect fifth interval

    def golden_rotation(x: int) -> int:
        """Golden ratio rotation: phi-based step"""
        phi_step = int(x * 1.618) % 7 + 1
        return (x + phi_step) % 21

    transforms = [
        ("10-Tooth Gear", CircularTransform(gear_rotation_10), "Mechanical gear with 10 teeth"),
        ("24-Hour Planet", CircularTransform(planetary_rotation_24), "Planetary 24-hour rotation cycle"),
        ("Fibonacci Spiral", CircularTransform(fibonacci_rotation), "Fibonacci-inspired growth rotation"),
        ("8-Bit Binary", CircularTransform(binary_rotation), "Binary bit rotation in 8-bit space"),
        ("Prime Steps", CircularTransform(prime_rotation), "Prime number step rotation"),
        ("Harmonic 12-Tone", CircularTransform(harmonic_rotation), "Musical harmonic rotation"),
        ("Golden Ratio", CircularTransform(golden_rotation), "Golden ratio based rotation")
    ]
    return transforms

# Create test transforms
rotation_transforms = create_rotation_transforms()

# Test positions across different ranges
test_positions = [0, 1, 3, 7, 12, 21, 42, 100, 255]

print("Testing theorem rotate_zero:")
total_tests = 0
for name, transform, description in rotation_transforms:
    tests_passed = 0
    for x in test_positions:
        result = rotate(transform, x, 0)
        if result == x:
            tests_passed += 1
        total_tests += 1
        assert result == x, f"rotate_zero failed for {name}, x={x}: got {result}, expected {x}"
    print(f"  {name}: identity verified for all {tests_passed} test positions")

print(f"\nLean theorem analogue check passed: rotate(t, x, 0) = x for all {total_tests} tested cases.\n")

# ----- Advanced rotation analysis -----

def analyze_rotation_sequence(transform: CircularTransform, x: int, max_rotations: int = 30) -> Tuple[List[int], dict]:
    """
    Analyze the rotation sequence starting from position x.
    Returns: (sequence, analysis_dict)
    """
    sequence = [x]
    seen_positions = {x: 0}

    current = x
    period_length = 0
    convergence_point = None

    for n in range(1, max_rotations + 1):
        current = rotate(transform, x, n)

        if current in seen_positions:
            period_start = seen_positions[current]
            period_length = n - period_start
            convergence_point = current
            break

        seen_positions[current] = n
        sequence.append(current)

    analysis = {
        'sequence_length': len(sequence),
        'period_length': period_length,
        'convergence_point': convergence_point,
        'unique_positions': len(set(sequence)),
        'position_range': (min(sequence), max(sequence)) if sequence else (0, 0),
        'average_position': np.mean(sequence) if sequence else 0
    }

    return sequence, analysis

def simulate_multi_start_rotations(transform: CircularTransform, start_positions: List[int],
                                 max_rotations: int = 20) -> dict:
    """Simulate rotations from multiple starting positions"""
    results = {}

    for x in start_positions:
        rotation_sequence = []
        for n in range(max_rotations + 1):
            position = rotate(transform, x, n)
            rotation_sequence.append(position)
        results[x] = rotation_sequence

    return results

def calculate_rotation_statistics(sequences: dict) -> dict:
    """Calculate comprehensive statistics for rotation sequences"""
    all_sequences = list(sequences.values())
    all_positions = [pos for seq in all_sequences for pos in seq]

    stats = {
        'total_positions': len(all_positions),
        'unique_positions': len(set(all_positions)),
        'position_density': len(set(all_positions)) / len(all_positions) if all_positions else 0,
        'mean_position': np.mean(all_positions) if all_positions else 0,
        'std_position': np.std(all_positions) if all_positions else 0,
        'min_position': min(all_positions) if all_positions else 0,
        'max_position': max(all_positions) if all_positions else 0,
        'sequence_lengths': [len(seq) for seq in all_sequences]
    }

    return stats

# ----- Run comprehensive rotation analysis -----

print("Analyzing circular transforms:")
transform_analysis = {}

for name, transform, description in rotation_transforms:
    print(f"\n{name} - {description}:")

    # Test multiple starting positions
    start_positions = [0, 1, 5, 10, 25, 50]
    position_analyses = {}

    for x in start_positions:
        sequence, analysis = analyze_rotation_sequence(transform, x)
        position_analyses[x] = {
            'sequence': sequence,
            'analysis': analysis
        }

        if analysis['sequence_length'] <= 12:  # Only print short sequences
            print(f"  Start {x}: {sequence[:8]}{'...' if len(sequence) > 8 else ''}, "
                  f"period={analysis['period_length']}, unique={analysis['unique_positions']}")

    transform_analysis[name] = position_analyses

# ----- Detailed simulation for visualization -----

# Select the most interesting transform for detailed analysis
main_transform_name = "Golden Ratio"
main_transform = next(transform for name, transform, _ in rotation_transforms if name == main_transform_name)

# Simulate comprehensive rotation sequences
start_positions = list(range(8))
max_rotations = 25
detailed_sequences = simulate_multi_start_rotations(main_transform, start_positions, max_rotations)
rotation_stats = calculate_rotation_statistics(detailed_sequences)

# ----- Save CSV summaries -----

# Create detailed sequence data
sequence_data = []
for start_x, sequence in detailed_sequences.items():
    for n, position in enumerate(sequence):
        sequence_data.append({
            'start_position': start_x,
            'rotation_count': n,
            'current_position': position,
            'transform': main_transform_name,
            'is_identity': n == 0 and position == start_x
        })

sequence_df = pd.DataFrame(sequence_data)
sequence_df.to_csv("./rotate_zero_sequences.csv", index=False)

# Create comprehensive analysis data
analysis_data = []
for transform_name, positions in transform_analysis.items():
    for start_x, data in positions.items():
        analysis = data['analysis']
        analysis_data.append({
            'transform': transform_name,
            'start_position': start_x,
            'sequence_length': analysis['sequence_length'],
            'period_length': analysis['period_length'],
            'unique_positions': analysis['unique_positions'],
            'position_range_min': analysis['position_range'][0],
            'position_range_max': analysis['position_range'][1],
            'average_position': analysis['average_position']
        })

analysis_df = pd.DataFrame(analysis_data)
analysis_df.to_csv("./rotate_zero_analysis.csv", index=False)

print(f"\nSaved simulation data to CSV files.")

# ----- Visualization 1: Multi-transform rotation comparison -----
plt.figure(figsize=(15, 10))

# Create subplot grid for different transforms
n_transforms = min(6, len(rotation_transforms))
cols = 3
rows = 2

for i, (name, transform, description) in enumerate(rotation_transforms[:n_transforms]):
    plt.subplot(rows, cols, i + 1)

    # Simulate short sequences for this transform
    test_starts = [0, 2, 5]
    colors = ['red', 'blue', 'green']

    for j, start in enumerate(test_starts):
        sequence = []
        for n in range(12):
            pos = rotate(transform, start, n)
            sequence.append(pos)

        plt.plot(range(len(sequence)), sequence, 'o-', color=colors[j],
                alpha=0.7, linewidth=2, markersize=4, label=f'Start {start}')

        # Highlight theorem verification at rotation 0
        plt.scatter([0], [start], s=100, c='black', marker='s', alpha=0.8, zorder=5)

    plt.title(f'{name}\n{description[:30]}...', fontsize=8)
    plt.xlabel('Rotation Count')
    plt.ylabel('Position')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6)

plt.suptitle('Circular Transform Rotation Analysis\nBlack squares verify rotate_zero theorem: rotation 0 = identity', fontsize=12)
plt.tight_layout()
plt.savefig('./rotate_zero_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 2: Identity preservation and convergence -----
plt.figure(figsize=(12, 8))

# Plot all sequences for the main transform
for start_x, sequence in detailed_sequences.items():
    rotations = list(range(len(sequence)))
    plt.plot(rotations, sequence, 'o-', alpha=0.6, linewidth=1.5,
             markersize=4, label=f'Start = {start_x}')

    # Emphasize the identity point (rotation 0)
    plt.scatter([0], [start_x], s=120, c='red', marker='s', alpha=0.8, zorder=5)

plt.xlabel("Rotation Count")
plt.ylabel("Current Position")
plt.title(f"Rotation Sequences: {main_transform_name}\nRed squares demonstrate rotate_zero theorem: rotate(t, x, 0) = x")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Add theorem annotation
plt.annotate('Theorem: rotate(t, x, 0) = x\n(Red squares at rotation 0)',
            xy=(0, 4), xytext=(5, 15),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            fontsize=10, ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('./rotate_zero_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Visualization 3: Statistical analysis heatmap -----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Prepare data for heatmaps
transform_names = [name for name, _, _ in rotation_transforms]
start_positions = [0, 1, 5, 10, 25, 50]

# Period length heatmap
period_matrix = np.zeros((len(transform_names), len(start_positions)))
sequence_length_matrix = np.zeros((len(transform_names), len(start_positions)))
unique_positions_matrix = np.zeros((len(transform_names), len(start_positions)))
range_span_matrix = np.zeros((len(transform_names), len(start_positions)))

for i, name in enumerate(transform_names):
    if name in transform_analysis:
        for j, start_pos in enumerate(start_positions):
            if start_pos in transform_analysis[name]:
                analysis = transform_analysis[name][start_pos]['analysis']
                period_matrix[i, j] = analysis['period_length']
                sequence_length_matrix[i, j] = analysis['sequence_length']
                unique_positions_matrix[i, j] = analysis['unique_positions']
                pos_range = analysis['position_range']
                range_span_matrix[i, j] = pos_range[1] - pos_range[0]

# Plot heatmaps
heatmap_data = [
    (period_matrix, "Period Length", "viridis"),
    (sequence_length_matrix, "Sequence Length", "plasma"),
    (unique_positions_matrix, "Unique Positions", "cividis"),
    (range_span_matrix, "Position Range Span", "inferno")
]

for idx, (data, title, cmap) in enumerate(heatmap_data):
    ax = axes[idx // 2, idx % 2]
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(start_positions)))
    ax.set_xticklabels(start_positions)
    ax.set_yticks(range(len(transform_names)))
    ax.set_yticklabels([name[:12] for name in transform_names], fontsize=8)
    ax.set_xlabel('Start Position')
    ax.set_ylabel('Transform')
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add value annotations for non-zero entries
    for i in range(len(transform_names)):
        for j in range(len(start_positions)):
            value = int(data[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha='center', va='center',
                       color='white' if value > np.max(data) * 0.5 else 'black',
                       fontsize=8, fontweight='bold')

plt.suptitle('Rotation Transform Statistical Analysis\n(All transforms satisfy rotate_zero theorem)', fontsize=12)
plt.tight_layout()
plt.savefig('./rotate_zero_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

# ----- Summary output -----
print("\n=== Rotate Zero Simulation Results ===")
print(f"Main transform analyzed: {main_transform_name}")
print(f"Number of transforms studied: {len(rotation_transforms)}")
print(f"Total starting positions tested: {len(start_positions)}")

# Theorem verification summary
print(f"\nTheorem verification:")
print(f"rotate_zero: rotate(t, x, 0) = x verified for {total_tests} test cases across {len(rotation_transforms)} transforms")

# Statistical summary
print(f"\nRotation statistics for {main_transform_name}:")
print(f"Total positions simulated: {rotation_stats['total_positions']}")
print(f"Unique positions visited: {rotation_stats['unique_positions']}")
print(f"Position density: {rotation_stats['position_density']:.3f}")
print(f"Average position: {rotation_stats['mean_position']:.2f}")
print(f"Position standard deviation: {rotation_stats['std_position']:.2f}")
print(f"Position range: [{rotation_stats['min_position']}, {rotation_stats['max_position']}]")

# Transform characteristics summary
print(f"\nTransform characteristics:")
for name, positions in transform_analysis.items():
    period_lengths = [data['analysis']['period_length'] for data in positions.values() if data['analysis']['period_length'] > 0]
    unique_counts = [data['analysis']['unique_positions'] for data in positions.values()]

    if period_lengths:
        print(f"  {name}: periods = {sorted(set(period_lengths))}, avg unique positions = {np.mean(unique_counts):.1f}")

print(f"\nTotal sequence data points: {len(sequence_data):,}")
print("CSV files exported: rotate_zero_sequences.csv, rotate_zero_analysis.csv")
print(f"\nTheorem verification: rotate(t, x, 0) = x (verified for all {len(rotation_transforms)} circular transforms)")
print("Identity preservation confirmed: Zero rotations always return the original position")