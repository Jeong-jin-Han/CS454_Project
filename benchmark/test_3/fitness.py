import math
import numpy as np
import matplotlib.pyplot as plt


def fitness_needle(args: list[int]) -> float:
    """Multi-basin Needle-in-Haystack landscape."""
    x = args[0]
    y = args[1] if len(args) > 1 else 0

    # Global optimum
    if x == 0 and y == 0:
        return 0.0

    # Repeated fake needles every 30 units (ND)
    if (x % 30 == 0) and (y % 30 == 0):
        return 1.0

    # Additional specific fake needles
    fake_needles = [(5, 5), (-7, -7), (10, -10)]
    for fx, fy in fake_needles:
        if x == fx and y == fy:
            return 1.0

    # Base landscape: flat but slightly bumpy
    bump = abs(math.sin(x * 0.25)) + abs(math.cos(y * 0.3))
    return 50.0 + bump * 5


def fitness_plateau(args: list[int]) -> float:
    """
    True wide plateau landscape:
    - Very flat central region
    - Slight ripples to create small misleading basins
    - Global minimum located on the right side
    - Works for 1D/2D/ND
    """
    x = args[0]
    y = args[1] if len(args) > 1 else 0

    # A huge plateau around x ∈ [-100, 40], y ∈ [-100, 40]
    if -100 <= x <= 40 and -100 <= y <= 40:
        # Small gentle ripples to make fake basins
        ripple = (
            2.0 * math.sin(0.05 * x)
            + 2.0 * math.cos(0.05 * y)
            + 1.0 * math.sin(0.1 * (x + y))
        )
        return 150.0 + ripple  # nearly flat

    # Left deep bowl (much worse)
    if x < -100:
        return 0.005 * (x + 150) ** 2 + 300.0  # always higher than plateau

    # Right bowl with global optimum
    # Target location: (60, 0)
    gx, gy = 60, 0
    if x == gx and y == gy:
        return 0.0  # global minimum

    return 0.01 * ((x - gx) ** 2 + (y - gy) ** 2)


def fitness_rugged(args: list[int]) -> float:
    """Highly rugged multi-basin landscape."""
    x = args[0]
    y = args[1] if len(args) > 1 else 0

    # Multi-scale ruggedness
    freqs = [0.4, 1.2, 2.5, 5.0]
    amps = [8, 5, 3, 1.5]
    rugged = sum(
        amps[i] * abs(math.sin(freqs[i] * x) + math.cos(freqs[i] * y))
        for i in range(len(freqs))
    )

    # Large periodic basins (super compressible)
    basin = (abs(x % 40 - 20) + abs(y % 40 - 20)) * 1.5

    # Deep local pits
    pits = [(12, 8), (-15, -5), (18, -10)]
    for px, py in pits:
        if x == px and y == py:
            return 0.4

    # Global minimum at (0,0)
    if x == 0 and y == 0:
        return 0.0

    return rugged + basin + 0.01 * (x * x + y * y)


def fitness_combined(args: list[int]) -> float:
    x = args[0]
    y = args[1] if len(args) > 1 else 0

    # Global optimum
    if x == 0 and y == 0:
        return 0.0

    # Plateau region near the center
    if abs(x) < 15 and abs(y) < 15:
        base = 8.0
    else:
        base = (abs(x) + abs(y)) * 0.4

    # Multi-scale ruggedness
    rugged = (
        abs(math.sin(x * 0.7)) * 3
        + abs(math.cos(y * 1.3)) * 4
        + abs(math.sin((x + y) * 0.25)) * 5
    )

    # Repeating fake basins every 50 units
    basin = (abs(x % 50 - 25) + abs(y % 50 - 25)) * 1.2

    # Several fake local minima
    fake_local = [(6, 6), (-5, 8), (-10, -10), (12, -15)]
    for fx, fy in fake_local:
        if x == fx and y == fy:
            return 1.5

    return base + rugged + basin


def plot_1d(fitness_fn, fn_name, xmin=-20, xmax=20):
    xs = np.arange(xmin, xmax + 1)
    ys = np.array([fitness_fn([x]) for x in xs])

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("Fitness")
    plt.grid(True)

    fname = f"landscape_{fn_name.lower()}_1d.png"
    plt.savefig(fname, dpi=200)
    plt.savefig(f"landscape_{fn_name.lower()}_1d.pdf", dpi=200)
    plt.close()


def plot_3d(fitness_fn, fn_name, xmin=-20, xmax=20, ymin=-20, ymax=20):
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_fn((int(X[i, j]), int(Y[i, j])))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Fitness")

    fname = f"landscape_{fn_name.lower()}_2d.png"
    plt.savefig(fname, dpi=200)
    plt.savefig(f"landscape_{fn_name.lower()}_2d.pdf", dpi=200)
    plt.close()


if __name__ == "__main__":
    # 1D plots
    plot_1d(fitness_needle, "Needle-in-the-Haystack", xmin=-150, xmax=150)
    plot_1d(fitness_plateau, "Plateau", xmin=-200, xmax=200)
    plot_1d(fitness_rugged, "Rugged")
    plot_1d(fitness_combined, "Combined")

    # 3D plots
    plot_3d(fitness_needle, "Needle-in-the-Haystack")
    plot_3d(fitness_plateau, "Plateau", xmin=-200, xmax=200)
    plot_3d(fitness_rugged, "Rugged")
    plot_3d(fitness_combined, "Combined")
