#!/usr/bin/env python3
"""Check the fitness landscape around the stuck pit at (19, ±22)"""

from benchmark.test_3.fitness import fitness_rugged

# The stuck position
stuck_positions = [(19, -22), (19, 22), (19, -23), (19, 23)]

print("=" * 80)
print("Checking fitness landscape around stuck pit")
print("=" * 80)

for pos in stuck_positions:
    print(f"\n📍 Position {pos}, f={fitness_rugged(pos):.4f}")
    print("   Neighbors:")
    
    # Axis-aligned neighbors
    x, y = pos
    neighbors = {
        f"  X+1: ({x+1}, {y})": fitness_rugged((x+1, y)),
        f"  X-1: ({x-1}, {y})": fitness_rugged((x-1, y)),
        f"  Y+1: ({x}, {y+1})": fitness_rugged((x, y+1)),
        f"  Y-1: ({x}, {y-1})": fitness_rugged((x, y-1)),
    }
    
    # Diagonal neighbors
    diag_neighbors = {
        f"  Diag++: ({x+1}, {y+1})": fitness_rugged((x+1, y+1)),
        f"  Diag+-: ({x+1}, {y-1})": fitness_rugged((x+1, y-1)),
        f"  Diag-+: ({x-1}, {y+1})": fitness_rugged((x-1, y+1)),
        f"  Diag--: ({x-1}, {y-1})": fitness_rugged((x-1, y-1)),
    }
    
    all_neighbors = {**neighbors, **diag_neighbors}
    
    # Sort by fitness
    sorted_neighbors = sorted(all_neighbors.items(), key=lambda x: x[1])
    
    for name, fit in sorted_neighbors:
        better = "✅" if fit < fitness_rugged(pos) else "❌"
        print(f"{name}: f={fit:.4f} {better}")

# Check path to goal
print("\n" + "=" * 80)
print("Path from stuck position to goal (0, 0)")
print("=" * 80)

start = (19, -22)
goal = (0, 0)

print(f"Start: {start}, f={fitness_rugged(start):.4f}")
print(f"Goal:  {goal}, f={fitness_rugged(goal):.4f}")
print(f"\nDirect path fitness values:")

# Check points along straight line
steps = 20
for i in range(steps + 1):
    t = i / steps
    x = int(start[0] * (1 - t) + goal[0] * t)
    y = int(start[1] * (1 - t) + goal[1] * t)
    pt = (x, y)
    f = fitness_rugged(pt)
    print(f"  Step {i:2d}: {pt}, f={f:.4f}")

