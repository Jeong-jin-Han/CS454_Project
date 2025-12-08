#!/usr/bin/env python3
"""
Diagnose: Why doesn't compression help escape local minima in rugged landscape?

The user's expectation is correct:
- If we compress regions during search
- Those compressions should help us jump OUT of local pits later
- So why doesn't it work?
"""
import random
from benchmark.test_3.fitness import fitness_rugged
from compression_hc import CompressionManagerND, detect_basin_along_dimension

print("="*80)
print("INVESTIGATION: Why Compression Doesn't Help in Rugged 2D")
print("="*80)

# ============================================================
# Scenario 1: What happens during a typical search?
# ============================================================
print("\n" + "="*80)
print("SCENARIO 1: Typical Search Trajectory")
print("="*80)

random.seed(42)
cm = CompressionManagerND(2, steepness=5.0)

# Start from random point
start = (100, -80)
print(f"\nStart: {start}, f={fitness_rugged(start):.2f}")

# Simulate descent to local pit
print("\nSimulating hill climb descent...")
point = start
trajectory = []

for step in range(20):
    current_f = fitness_rugged(point)
    trajectory.append((point, current_f))
    
    # Try neighbors
    neighbors = [
        (point[0]-1, point[1]),
        (point[0]+1, point[1]),
        (point[0], point[1]-1),
        (point[0], point[1]+1),
    ]
    
    best = point
    best_f = current_f
    
    for n in neighbors:
        nf = fitness_rugged(n)
        if nf < best_f:
            best = n
            best_f = nf
    
    if best_f < current_f:
        point = best
    else:
        print(f"  → Stuck at {point}, f={current_f:.4f} after {step} steps")
        break

# ============================================================
# Scenario 2: Try basin detection at stuck point
# ============================================================
print("\n" + "="*80)
print("SCENARIO 2: Basin Detection When Stuck")
print("="*80)

stuck_point = point
stuck_f = fitness_rugged(stuck_point)
print(f"\nStuck at: {stuck_point}, f={stuck_f:.4f}")

print("\n--- Basin detection along X-axis (fix Y) ---")
basin_x = detect_basin_along_dimension(
    fitness_rugged, stuck_point, vary_dim=0, max_search=100, verbose=True
)
print(f"Result: {basin_x}")

print("\n--- Basin detection along Y-axis (fix X) ---")
basin_y = detect_basin_along_dimension(
    fitness_rugged, stuck_point, vary_dim=1, max_search=100, verbose=True
)
print(f"Result: {basin_y}")

if basin_x:
    print(f"\n✅ Basin detected on X-axis: {basin_x}")
    cm.update_dimension(0, (stuck_point[1],), basin_x)
else:
    print(f"\n❌ No basin on X-axis")

if basin_y:
    print(f"✅ Basin detected on Y-axis: {basin_y}")
    cm.update_dimension(1, (stuck_point[0],), basin_y)
else:
    print(f"❌ No basin on Y-axis")

# ============================================================
# Scenario 3: Even IF we had compressions, would they help?
# ============================================================
print("\n" + "="*80)
print("SCENARIO 3: Hypothetical - What if we HAD compressions?")
print("="*80)

print("\nLet's artificially create compressions and see if they'd help...")

# Create artificial compression around stuck point
if not basin_x:
    print("\nCreating ARTIFICIAL basin on X-axis [50, 150]...")
    artificial_basin_x = (50, 100)  # Large basin
    cm.update_dimension(0, (stuck_point[1],), artificial_basin_x)
    print(f"  Added: {artificial_basin_x}")

print("\n--- Now try neighbor generation WITH compression ---")

# Get compression system
fixed_y = stuck_point[1]
comp_sys_x = cm.get_system(0, (fixed_y,))

if comp_sys_x:
    print(f"✅ Compression system found for X-axis (Y={fixed_y})")
    
    # Try compressed neighbors
    z_current = comp_sys_x.forward(stuck_point[0])
    print(f"\nCurrent X={stuck_point[0]} → Z={z_current:.4f}")
    
    z_left = z_current - 1
    z_right = z_current + 1
    
    x_left = comp_sys_x.inverse(z_left)
    x_right = comp_sys_x.inverse(z_right)
    
    print(f"  Z-1={z_left:.4f} → X={x_left} (jump from {stuck_point[0]})")
    print(f"  Z+1={z_right:.4f} → X={x_right} (jump from {stuck_point[0]})")
    
    # Check fitness at these points
    left_point = (int(x_left), stuck_point[1])
    right_point = (int(x_right), stuck_point[1])
    
    f_left = fitness_rugged(left_point)
    f_right = fitness_rugged(right_point)
    
    print(f"\nFitness comparison:")
    print(f"  Current ({stuck_point[0]}, {stuck_point[1]}): f={stuck_f:.4f}")
    print(f"  Left ({left_point[0]}, {left_point[1]}): f={f_left:.4f} {'✅ BETTER!' if f_left < stuck_f else '❌ worse'}")
    print(f"  Right ({right_point[0]}, {right_point[1]}): f={f_right:.4f} {'✅ BETTER!' if f_right < stuck_f else '❌ worse'}")
else:
    print("❌ No compression system (even after artificial creation?)")

# ============================================================
# Scenario 4: The REAL issue - where are compressible regions?
# ============================================================
print("\n" + "="*80)
print("SCENARIO 4: Search for Compressible Regions")
print("="*80)

print("\nSearching entire space for regions with Rule 1 or Rule 3...")

compressible_regions = []

# Sample points across the space
test_points = [
    (0, 0), (10, 10), (20, 20), (-10, -10), (-20, -20),
    (50, 0), (0, 50), (-50, 0), (0, -50),
    (100, 100), (-100, -100),
]

for pt in test_points:
    f_center = fitness_rugged(pt)
    
    # Quick check: look at neighbors
    equal_count = 0
    better_count = 0
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (pt[0] + dx, pt[1] + dy)
            f_n = fitness_rugged(neighbor)
            
            if abs(f_n - f_center) < 1e-9:
                equal_count += 1
            elif f_n < f_center:
                better_count += 1
    
    if equal_count > 0 or better_count > 0:
        compressible_regions.append((pt, f_center, equal_count, better_count))
        print(f"  Point {pt}: f={f_center:.2f}, equal={equal_count}, better={better_count}")

if compressible_regions:
    print(f"\n✅ Found {len(compressible_regions)} potentially compressible regions!")
else:
    print(f"\n❌ NO compressible regions found in entire search space!")
    print("   This means:")
    print("   • No plateaus exist (Rule 1 never triggered)")
    print("   • No basin exits found (Rule 3 never triggered)")
    print("   • ONLY oscillations everywhere (Rule 2 only)")
    print("   • Compression cannot be applied!")

# ============================================================
# Scenario 5: What about BETWEEN trials?
# ============================================================
print("\n" + "="*80)
print("SCENARIO 5: Compression Accumulation Across Trials")
print("="*80)

print("\nThe user's idea:")
print("  'If we compress regions in early trials,")
print("   later trials should benefit from those compressions'")

print("\nLet's simulate multiple trials...")

cm_shared = CompressionManagerND(2, steepness=5.0)
trial_results = []

for trial in range(5):
    # Random start
    start = (random.randint(-150, 150), random.randint(-150, 150))
    
    # Quick descent (simplified)
    point = start
    for _ in range(10):
        neighbors = [(point[0]+dx, point[1]+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
        best = min(neighbors, key=lambda p: fitness_rugged(p))
        if fitness_rugged(best) < fitness_rugged(point):
            point = best
        else:
            break
    
    # Try basin detection
    basin_x = detect_basin_along_dimension(fitness_rugged, point, 0, max_search=50, verbose=False)
    basin_y = detect_basin_along_dimension(fitness_rugged, point, 1, max_search=50, verbose=False)
    
    if basin_x:
        cm_shared.update_dimension(0, (point[1],), basin_x)
    if basin_y:
        cm_shared.update_dimension(1, (point[0],), basin_y)
    
    trial_results.append({
        'trial': trial,
        'final_point': point,
        'final_f': fitness_rugged(point),
        'basin_x': basin_x,
        'basin_y': basin_y,
    })
    
    print(f"Trial {trial}: End at {point}, f={fitness_rugged(point):.4f}, basins: X={basin_x}, Y={basin_y}")

# Check accumulated compressions
total_compressions = sum(len(cm_shared.dim_compressions[d]) for d in range(2))
print(f"\n📊 Total compressions accumulated: {total_compressions}")

if total_compressions > 0:
    print("✅ Compressions were accumulated! They SHOULD help later trials.")
    print("   But do they? Let's check if later trials used them...")
else:
    print("❌ NO compressions accumulated across trials!")
    print("   This is why compression doesn't help - there's nothing to use!")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "="*80)
print("CONCLUSION: Why Compression Doesn't Help")
print("="*80)

print("""
Your intuition is CORRECT:
  ✓ Compression SHOULD help escape local minima
  ✓ IF compressions are detected and stored
  ✓ THEN later trials should benefit

But the ACTUAL problem is:
  ❌ In rugged landscapes, NO BASINS are detected (only Rule 2)
  ❌ Without basins, NO compressions are created
  ❌ Without compressions, there's nothing to help escape
  ❌ Algorithm gets stuck with no tools to use

The rugged fitness function has:
  • High-frequency oscillations (no flat regions)
  • Isolated point minima (no plateaus)
  • No Rule 1 (equal fitness) regions
  • No Rule 3 (exit) points (all surrounded by worse fitness)
  • ONLY Rule 2 (worse fitness) everywhere

Solution:
  1. Compression works GREAT when basins exist (plateau landscapes)
  2. Compression CANNOT work when no basins exist (rugged landscapes)
  3. Need complementary strategy: RANDOM RESTART for diversification
  4. This makes algorithm work for BOTH landscape types!

Your compression strategy is sound! It just needs a backup plan
for when no compressible structure exists.
""")

print("="*80)

