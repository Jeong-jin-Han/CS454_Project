import os


def plot_fitness_landscape(
    fitness_fn,
    history,
    fitness_name,
    num_args,
    value_range,
    save_path,
    is_hcc=False,
    is_hc=False,
):
    """
    Plot GA or HCC trajectory on top of the fitness landscape.
    HISTORY FORMAT:
        HCC → (trial_id, gen_idx, pt, fitness_value)
        GA  → (gen_idx, pt, fitness_value)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    lo, hi = value_range
    alphas = np.linspace(0.2, 1.0, len(history))

    # --- Normalize history into uniform format ---
    if is_hcc or is_hc:
        # HCC: (trial_id, gen, pt, f)
        parsed = [(tid, gen, pt, f) for tid, gen, pt, f in history]
    else:
        # GA or non-HCC:
        # Allow both:
        #   (gen, pt, f)  OR  (tid, gen, pt, f)
        parsed = []
        for entry in history:
            if len(entry) == 4:
                tid, gen, pt, f = entry
            else:
                # no trial_id → treat as single trial (id=0)
                gen, pt, f = entry
                tid = 0
            parsed.append((tid, gen, pt, f))

    # Split fields
    trial_ids = [t for (t, _, _, _) in parsed]
    pts = [pt for (_, _, pt, _) in parsed]
    fits = [f for (_, _, _, f) in parsed]

    # ============================================================
    # 1D LANDSCAPE
    # ============================================================
    if num_args == 1:

        X = np.arange(lo, hi + 1)
        Y = np.array([fitness_fn((int(x),)) for x in X])

        plt.figure(figsize=(8, 5))
        plt.plot(X, Y, color="lightgray", linewidth=2)

        # trial segmentation
        segments = {}
        for tid, gen, pt, f in parsed:
            segments.setdefault(tid, [[], []])
            segments[tid][0].append(pt[0])
            segments[tid][1].append(f)

        # Draw each trial as connected line
        for tid, (xs, ys) in segments.items():
            plt.plot(xs, ys, linewidth=2, alpha=0.7)
            # scatter with full-gradient
            for i, (xv, fv) in enumerate(zip(xs, ys)):
                plt.scatter(xv, fv, c=[(0, 0, 0, alphas[i])], s=35)

        plt.xlabel("x")
        plt.ylabel("Fitness")
        # plt.title(f"{fitness_name} Landscape with Trial-aware Trajectory (1D)")
        plt.savefig(save_path, dpi=200)
        # save pdf
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved 1D plot: {save_path}")
        return

    # ============================================================
    # 2D LANDSCAPE
    # ============================================================
    elif num_args == 2:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        X = np.arange(lo, hi + 1)
        Y = np.arange(lo, hi + 1)
        XX, YY = np.meshgrid(X, Y)

        Z = np.zeros_like(XX, dtype=float)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = fitness_fn((int(XX[i, j]), int(YY[i, j])))

        ax.plot_surface(XX, YY, Z, cmap="viridis", alpha=0.70, edgecolor="none")

        # trial-aware segmentation
        segments = {}
        for tid, gen, pt, f in parsed:
            segments.setdefault(tid, [[], [], []])  # xs, ys, zs
            segments[tid][0].append(pt[0])
            segments[tid][1].append(pt[1])
            segments[tid][2].append(f)

        # draw each trial separately
        for tid, (xs, ys, zs) in segments.items():
            ax.plot(xs, ys, zs, color="red", linewidth=2, alpha=0.8)
            for i, (xv, yv, zv) in enumerate(zip(xs, ys, zs)):
                ax.scatter(xv, yv, zv, c=[(0, 0, 0, alphas[i])], s=35)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("fitness")
        # ax.set_title(f"{fitness_name} Landscape Trajectory (2D)")

        plt.savefig(save_path, dpi=200)
        # save pdf
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved 2D→3D plot: {save_path}")

        # contour version
        plot_fitness_landscape_projected(
            fitness_fn,
            parsed,
            fitness_name,
            value_range,
            save_path,
        )
        if is_hcc or is_hc:
            plot_fitness_landscape_projected_last_trial(
                fitness_fn,
                parsed,
                fitness_name,
                value_range,
                save_path,
            )
        return

    else:
        raise ValueError("plot_fitness_landscape only supports 1D or 2D.")


def plot_fitness_landscape_projected(
    fitness_fn,
    history,
    fitness_name,
    value_range,
    save_path,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    lo, hi = value_range

    # history already normalized: (tid, gen, pt, f)
    parsed = history
    # =====================================================
    # 2D CASE
    # =====================================================
    grid_res = 200
    xs = np.linspace(lo, hi, grid_res)
    ys = np.linspace(lo, hi, grid_res)
    Xg, Yg = np.meshgrid(xs, ys)

    Z = np.zeros_like(Xg)
    for i in range(grid_res):
        for j in range(grid_res):
            Z[i, j] = fitness_fn((int(Xg[i, j]), int(Yg[i, j])))

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(Xg, Yg, Z, levels=60, cmap="viridis", alpha=0.85)
    plt.colorbar(contour)

    # trial segmentation
    segments = {}
    for tid, gen, pt, f in parsed:
        segments.setdefault(tid, [[], []])
        segments[tid][0].append(pt[0])
        segments[tid][1].append(pt[1])

    for tid, (xs, ys) in segments.items():
        plt.plot(xs, ys, linewidth=2)
        plt.scatter(xs, ys, s=25, edgecolors="black")

    # plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(f"{fitness_name} - 2D Trial Segmented Contour")

    root, ext = os.path.splitext(save_path)
    contour_path = f"{root}_contour{ext}"
    plt.savefig(contour_path, dpi=200)
    # save pdf
    pdf_path = os.path.splitext(contour_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved 2D segmented contour plot: {contour_path}")
    return


def plot_fitness_landscape_projected_last_trial(
    fitness_fn,
    history,
    fitness_name,
    value_range,
    save_path,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    lo, hi = value_range

    # history already normalized: (tid, gen, pt, f)
    parsed = history
    # get the last trial only
    last_tid = max(tid for tid, _, _, _ in parsed)
    parsed = [entry for entry in parsed if entry[0] == last_tid]
    # =====================================================
    # 2D CASE
    # =====================================================
    grid_res = 200
    xs = np.linspace(lo, hi, grid_res)
    ys = np.linspace(lo, hi, grid_res)
    Xg, Yg = np.meshgrid(xs, ys)

    Z = np.zeros_like(Xg)
    for i in range(grid_res):
        for j in range(grid_res):
            Z[i, j] = fitness_fn((int(Xg[i, j]), int(Yg[i, j])))

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(Xg, Yg, Z, levels=60, cmap="viridis", alpha=0.85)
    plt.colorbar(contour)

    # trial segmentation
    segments = {}
    for tid, gen, pt, f in parsed:
        segments.setdefault(tid, [[], []])
        segments[tid][0].append(pt[0])
        segments[tid][1].append(pt[1])
    for tid, (xs, ys) in segments.items():
        plt.plot(xs, ys, linewidth=2, label=f"{last_tid+1}th trial")
        plt.scatter(xs, ys, s=25, edgecolors="black")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(f"{fitness_name} - 2D Trial Segmented Contour")

    root, ext = os.path.splitext(save_path)
    contour_path = f"{root}_contour_last_trial{ext}"
    plt.savefig(contour_path, dpi=200)
    # save pdf
    pdf_path = os.path.splitext(contour_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved 2D segmented contour plot: {contour_path}")
    return