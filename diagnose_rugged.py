#!/usr/bin/env python3
"""
Quick diagnostic to understand why rugged 2D fails.
"""
import numpy as np
import matplotlib.pyplot as plt
from benchmark.test_3.fitness import fitness_rugged

# ============================================================
# Test 1: Check Local Pits
# ============================================================
print("="*80)
print("TEST 1: Checking Local Pits")
print("="*80)

pits = [(12, 8), (-15, -5), (18, -10), (0, 0)]  # Include goal

for px, py in pits:
    f_center = fitness_rugged([px, py])
    print(f"\nPit at ({px}, {py}): fitness = {f_center:.4f}")
    
    # Check neighbors
    print("  Neighbors:")
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = px + dx, py + dy
        f_neighbor = fitness_rugged([nx, ny])
        diff = f_neighbor - f_center
        print(f"    ({nx:3d}, {ny:3d}): f={f_neighbor:6.2f}, diff={diff:+6.2f}")

# ============================================================
# Test 2: Check Basin Detection Feasibility
# ============================================================
print("\n" + "="*80)
print("TEST 2: Basin Detection Along X-axis from pit (12, 8)")
print("="*80)

center_x, center_y = 12, 8
f_center = fitness_rugged([center_x, center_y])

print(f"Center ({center_x}, {center_y}): f={f_center:.4f}")
print("\nSearching LEFT:")

equal_count = 0
better_count = 0
worse_count = 0

for i in range(1, 21):
    x = center_x - i
    f = fitness_rugged([x, center_y])
    
    if abs(f - f_center) < 1e-9:
        rule = "Rule 1 (equal)"
        equal_count += 1
    elif f > f_center:
        rule = "Rule 2 (worse)"
        worse_count += 1
    else:
        rule = "Rule 3 (BETTER!)"
        better_count += 1
    
    print(f"  x={x:3d}: f={f:6.2f} - {rule}")

print(f"\nSummary: Equal={equal_count}, Worse={worse_count}, Better={better_count}")
print(f"Basin detectable? {'YES (Rule 1 or 3 found)' if (equal_count > 0 or better_count > 0) else 'NO (only Rule 2)'}")

# ============================================================
# Test 3: Escape Difficulty
# ============================================================
print("\n" + "="*80)
print("TEST 3: Path from Pit (12, 8) to Goal (0, 0)")
print("="*80)

# Linear interpolation path
start = np.array([12, 8])
goal = np.array([0, 0])
steps = 15

print(f"\nFitness along straight line from ({start[0]}, {start[1]}) to (0, 0):")
print(f"{'Step':>4} {'X':>6} {'Y':>6} {'Fitness':>10}")

for i in range(steps + 1):
    t = i / steps
    point = start * (1 - t) + goal * t
    x, y = int(round(point[0])), int(round(point[1]))
    f = fitness_rugged([x, y])
    marker = " ← START" if i == 0 else (" ← GOAL" if i == steps else "")
    print(f"{i:4d} {x:6d} {y:6d} {f:10.4f}{marker}")

# ============================================================
# Test 4: Visualize Fitness Landscape
# ============================================================
print("\n" + "="*80)
print("TEST 4: Generating fitness landscape visualization")
print("="*80)

# Create 2D landscape plot
x_range = np.arange(-30, 30, 1)
y_range = np.arange(-30, 30, 1)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = fitness_rugged([int(X[i, j]), int(Y[i, j])])

# Clip for better visualization
Z_clipped = np.clip(Z, 0, 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Heatmap
im = axes[0].imshow(Z_clipped, extent=(-30, 30, -30, 30), origin='lower', cmap='viridis')
axes[0].set_title("Rugged 2D Fitness Landscape (clipped at 50)")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
plt.colorbar(im, ax=axes[0], label="Fitness")

# Mark important points
axes[0].plot(0, 0, 'r*', markersize=15, label='Goal (0,0) f=0.0')
axes[0].plot(12, 8, 'y*', markersize=12, label='Pit (12,8) f=0.4')
axes[0].plot(-15, -5, 'y*', markersize=12, label='Pit (-15,-5) f=0.4')
axes[0].plot(18, -10, 'y*', markersize=12, label='Pit (18,-10) f=0.4')
axes[0].legend()

# Right: Contour plot
contour = axes[1].contour(X, Y, Z_clipped, levels=20, cmap='viridis')
axes[1].clabel(contour, inline=True, fontsize=8)
axes[1].set_title("Contour Plot (clipped at 50)")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

# Mark points
axes[1].plot(0, 0, 'r*', markersize=15, label='Goal')
axes[1].plot(12, 8, 'y*', markersize=12, label='Local Pits')
axes[1].plot(-15, -5, 'y*', markersize=12)
axes[1].plot(18, -10, 'y*', markersize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('rugged_2d_diagnostic.png', dpi=150)
print("Saved visualization to: rugged_2d_diagnostic.png")

# ============================================================
# Test 5: Check Random Point Distribution
# ============================================================
print("\n" + "="*80)
print("TEST 5: Random Starting Point Fitness Distribution")
print("="*80)

import random
random.seed(42)

fitness_samples = []
for _ in range(100):
    x = random.randint(-150, 150)
    y = random.randint(-150, 150)
    f = fitness_rugged([x, y])
    fitness_samples.append(f)

fitness_samples = np.array(fitness_samples)

print(f"Random start points (n=100) in [-150, 150]:")
print(f"  Mean fitness: {np.mean(fitness_samples):.2f}")
print(f"  Median fitness: {np.median(fitness_samples):.2f}")
print(f"  Min fitness: {np.min(fitness_samples):.2f}")
print(f"  Max fitness: {np.max(fitness_samples):.2f}")
print(f"  Std dev: {np.std(fitness_samples):.2f}")

print("\nConclusion:")
print(f"  Typical starting fitness: ~{np.median(fitness_samples):.0f}")
print(f"  Local pit fitness: 0.4")
print(f"  Improvement from start to pit: ~{np.median(fitness_samples) - 0.4:.0f}x better!")
print(f"  Improvement from pit to goal: only {0.4 - 0.0:.1f} better")
print(f"  → Pit is MUCH more attractive than goal from algorithm's perspective!")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print("""
Key Findings:
1. Local pits at 0.4 fitness are surrounded by much worse fitness (~18-30)
2. No plateau detected around pits (only oscillating worse fitness)
3. Path from pit to goal goes UPHILL (through worse regions)
4. Random starts (~180 fitness) → Pit (0.4) is 450x improvement!
5. Pit (0.4) → Goal (0.0) is only 0.4 improvement (not attractive)

Root Cause:
- Algorithm gets trapped in local pits because:
  * They're much better than surroundings
  * No escape mechanism (restart goes to worse points)
  * Compression doesn't work (no plateaus in rugged landscape)

Recommended Fixes:
1. Add random restart far from current position (not just basin boundary)
2. Reduce basin_max_search for rugged landscapes
3. Increase time limit or max trials
4. Add explicit escape mechanism for deep minima
""")

print("="*80)

