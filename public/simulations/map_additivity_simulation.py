# -*- coding: utf-8 -*-
"""
Simulation based on the provided Lean snippet for 'map_additivity'.

Exact Lean code referenced (verbatim):

structure LinearMap where
  source : List Nat
  target : List Nat
def map (f : LinearMap) (v : List Nat) : List Nat := f.target.map (fun t => t + (v.foldl (+ +) 0))
theorem map_additivity (f : LinearMap) (v₁ v₂ : List Nat) :
  map f (v₁ ++ v₂) = map f v₁ ++ map f v₂ := by
  simp
  rfl

What this Python script does:
1) Mirrors the Lean structures and the 'map' definition exactly: map(f, v) returns
   [t + sum(v) for t in f.target].
2) Checks the Lean-stated "map_additivity" property using list concatenation on both sides:
      LHS = map(f, v1 + v2)
      RHS = map(f, v1) + map(f, v2)
   (Note: in both Lean and Python, ++ / + on lists is concatenation.)
3) Runs a Monte-Carlo experiment across many random cases and reports how often the
   property holds for different target lengths. It also prints a concrete counterexample
   when target is nonempty, and a confirming example when target is empty.
4) Produces a small bar chart of success rates by target length (uses matplotlib defaults).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import os

# Set the working directory to simulations folder to save images there
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Mirror the Lean "structure" and "def" ----------

LEAN_CODE = """structure LinearMap where
  source : List Nat
  target : List Nat
def map (f : LinearMap) (v : List Nat) : List Nat := f.target.map (fun t => t + (v.foldl (+ +) 0))
theorem map_additivity (f : LinearMap) (v1 v2 : List Nat) :
  map f (v1 ++ v2) = map f v1 ++ map f v2 := by
  simp
  rfl"""

print("Exact Lean code referenced:\n")
print(LEAN_CODE)
print("\n— End Lean code —\n")

@dataclass(frozen=True)
class LinearMap:
    source: List[int]
    target: List[int]

def map_fn(f: LinearMap, v: List[int]) -> List[int]:
    """
    Python analogue of the Lean 'map':
    map f v = f.target.map (λ t, t + sum(v))
            = [t + sum(v) for t in f.target]
    """
    s = sum(v)
    return [t + s for t in f.target]

# Property check corresponding to the Lean theorem statement
def map_additivity_property(f: LinearMap, v1: List[int], v2: List[int]) -> Tuple[bool, List[int], List[int]]:
    """
    Returns (holds?, lhs, rhs) where:
      lhs = map f (v1 ++ v2)         # ++ is list concatenation in Lean
      rhs = map f v1 ++ map f v2     # ++ is list concatenation
    """
    lhs = map_fn(f, v1 + v2)
    rhs = map_fn(f, v1) + map_fn(f, v2)
    return (lhs == rhs, lhs, rhs)

# ---------- Monte‑Carlo simulation ----------

rng = np.random.default_rng(20250909)

def rand_nat_list(max_len=5, max_val=9) -> List[int]:
    n = int(rng.integers(0, max_len + 1))
    if n == 0:
        return []
    return rng.integers(0, max_val + 1, size=n).tolist()

def simulate(trials_per_len=300, target_lens=range(0, 6)) -> pd.DataFrame:
    rows = []
    for L in target_lens:
        holds_count = 0
        for _ in range(trials_per_len):
            f = LinearMap(source=rand_nat_list(4, 9), target=(rng.integers(0, 10, size=L).tolist()))
            v1 = rand_nat_list(4, 9)
            v2 = rand_nat_list(4, 9)
            ok, _, _ = map_additivity_property(f, v1, v2)
            if ok:
                holds_count += 1
        rate = holds_count / max(1, trials_per_len)
        rows.append({"target_length": L, "trials": trials_per_len, "success_rate": rate})
    return pd.DataFrame(rows)

# Run a couple of illustrative examples
print("Illustrative examples:\n")

# Example where target is nonempty → typically fails (length mismatch after concatenation)
f_nonempty = LinearMap(source=[1,2], target=[3,4,5])
v1_ex = [1,2]
v2_ex = [7]
holds, lhs, rhs = map_additivity_property(f_nonempty, v1_ex, v2_ex)
print(f"Nonempty target example: target={f_nonempty.target}, v1={v1_ex}, v2={v2_ex}")
print(f"  LHS = map(f, v1 ++ v2) = {lhs}")
print(f"  RHS = map(f, v1) ++ map(f, v2) = {rhs}")
print(f"  Holds? {holds} (note lengths: len(LHS)={len(lhs)} vs len(RHS)={len(rhs)})\n")

# Example where target is empty → always holds ([] ++ [] = [])
f_empty = LinearMap(source=[], target=[])
holds_e, lhs_e, rhs_e = map_additivity_property(f_empty, [2,3], [5])
print(f"Empty target example: target={f_empty.target}, v1=[2,3], v2=[5]")
print(f"  LHS = {lhs_e}")
print(f"  RHS = {rhs_e}")
print(f"  Holds? {holds_e}\n")

# Run simulation across target lengths
df = simulate(trials_per_len=400, target_lens=range(0, 7))
print("Monte-Carlo success rates by target length (probability that the stated equality holds):")
print(df.to_string(index=False))

# Save results
csv_path = "map_additivity_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved results to: {csv_path}")

# ---------- Visualization ----------
plt.figure(figsize=(10, 6))
bars = plt.bar(df["target_length"], df["success_rate"], color='skyblue', edgecolor='darkblue', alpha=0.7)

# Add value labels on top of bars
for i, (length, rate) in enumerate(zip(df["target_length"], df["success_rate"])):
    plt.text(length, rate + 0.02, f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel("Length of f.target", fontsize=12)
plt.ylabel("Success rate of equality", fontsize=12)
plt.title("Map Additivity Theorem: Success Rate vs Target Length\n(Lean theorem holds only for empty targets)", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)
plt.ylim(0, max(df["success_rate"]) * 1.15)
plt.xticks(df["target_length"])

# Add explanation text
plt.figtext(0.5, 0.02,
           "Note: Theorem fails for non-empty targets due to list concatenation length mismatch",
           ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('map_additivity_success_rates.png', dpi=300, bbox_inches='tight')
print("Saved visualization to: map_additivity_success_rates.png")
plt.close()

print(f"Generated images: map_additivity_success_rates.png")
print(f"Generated data: {csv_path}")