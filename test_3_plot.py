import os


def plot_fitness_landscape(
    fitness_fn, history, fitness_name, num_args, value_range, save_path
):
    """
    Plot GA/HCC trajectory on top of the fitness landscape.
    - 1D → int-grid curve + trajectory dots (later steps = darker)
    - 2D → 3D surface + trajectory dots (later steps = darker)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    lo, hi = value_range
    # 0.2 → 1.0 : 앞은 연하고, 뒤로 갈수록 진해짐
    alphas = np.linspace(0.2, 1.0, len(history))

    # ============================================================
    # 1D LANDSCAPE
    # ============================================================
    if num_args == 1:

        # ---- integer grid ----
        X = np.arange(lo, hi + 1)
        Y = np.array([fitness_fn((int(x),)) for x in X])

        plt.figure(figsize=(8, 5))

        # Draw landscape curve
        plt.plot(X, Y, color="lightgray", linewidth=2, label="Landscape")

        # ---- Trajectory (re-evaluated on SAME int grid) ----
        traj_x = [pt[0] for (_, pt, _) in history]
        traj_y = [fitness_fn((int(pt[0]),)) for (_, pt, _) in history]

        # 점 하나씩 찍되, 뒤로 갈수록 alpha ↑
        for i in range(len(history)):
            plt.scatter(
                traj_x[i],
                traj_y[i],
                c=[(0, 0, 0, alphas[i])],  # RGBA: black + varying alpha
                s=40,
            )

        plt.xlabel("x")
        plt.ylabel("Fitness")
        plt.title(f"{fitness_name} Landscape with Trajectory ({num_args}D)")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved 1D plot: {save_path}")
        return

    # ============================================================
    # 2D LANDSCAPE
    # ============================================================
    elif num_args == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # ---- integer grid ----
        X = np.arange(lo, hi + 1)
        Y = np.arange(lo, hi + 1)
        XX, YY = np.meshgrid(X, Y)

        Z = np.zeros_like(XX, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = fitness_fn((int(XX[i, j]), int(YY[i, j])))

        # ---- Draw surface ----
        ax.plot_surface(XX, YY, Z, cmap="viridis", alpha=0.70, edgecolor="none")

        # ---- Trajectory (on same landscape) ----
        traj_x = [pt[0] for (_, pt, _) in history]
        traj_y = [pt[1] for (_, pt, _) in history]
        traj_z = [fitness_fn((int(pt[0]), int(pt[1]))) for (_, pt, _) in history]

        # RGBA 리스트로 alpha 점점 진하게
        traj_colors = [(0, 0, 0, a) for a in alphas]

        ax.scatter(traj_x, traj_y, traj_z, c=traj_colors, s=35)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("fitness")
        ax.set_title(f"{fitness_name} Landscape with Trajectory ({num_args}D)")

        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved 2D→3D plot: {save_path}")
        plot_fitness_landscape_projected(
            fitness_fn, history, fitness_name, num_args, value_range, save_path
        )
        return

    else:
        raise ValueError(
            "plot_fitness_landscape only supports 1D or 2D fitness functions."
        )


def plot_fitness_landscape_projected(
    fitness_fn, history, fitness_name, num_args, value_range, save_path
):
    """
    2D landscape도 x만 찍어서 1D로 투영하여 가독성을 높인 버전.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    lo, hi = value_range
    alphas = np.linspace(0.2, 1.0, len(history))

    # === 1D 또는 2D 모두 x만 사용 ===
    X = np.arange(lo, hi + 1)
    Y = (
        np.array([fitness_fn((int(x), 0)) for x in X])
        if num_args == 2
        else np.array([fitness_fn((int(x),)) for x in X])
    )

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, color="lightgray", linewidth=2, label="Landscape")

    # trajectory points
    traj_x = [pt[0] for (_, pt, _) in history]
    traj_y = [
        fitness_fn((pt[0], pt[1])) if num_args == 2 else fitness_fn((pt[0],))
        for (_, pt, _) in history
    ]

    for i in range(len(history)):
        plt.scatter(traj_x[i], traj_y[i], c=[(0, 0, 0, alphas[i])], s=40)

    plt.xlabel("x (only first dimension used)")
    plt.ylabel("Fitness")
    plt.title(f"{fitness_name} Trajectory (Projected {num_args}D → 1D)")
    root, ext = os.path.splitext(save_path)
    projected_path = f"{root}_projected{ext}"

    plt.savefig(projected_path, dpi=200)
    plt.close()

    print(f"Saved projected plot: {projected_path}")
