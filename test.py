import numpy as np
import matplotlib.pyplot as plt
from hill_climb_multiD import hill_climb_with_compression_nd, hill_climb_simple_nd


# ===============================
# 1) 2D test function 정의
# ===============================
def f2d_complex(x, y):
    """
    f(x,y) = 1 - exp(-(x^2 + y^2)) + (sin(0.5*x) * cos^3(y - x))^2
    - 항상 >= 0
    - (0,0)이 global minimum (값 0)
    - 주변에 굴곡/로컬 미니마가 많은 landscape
    """
    x = float(x)
    y = float(y)

    r2 = x * x + y * y
    base = 1.0 - np.exp(-r2)
    osc  = np.sin(0.5 * x) * (np.cos(y - x) ** 3)
    return base + osc ** 2

# hill_climb_with_compression_nd가 요구하는 N-D 형태 래퍼
def f2d_complex_nd(point):
    x, y = point
    return f2d_complex(x, y)

# 시각화 / 탐색 도메인
X_MIN, X_MAX = -50, 50
Y_MIN, Y_MAX = -50, 50

# 시작점
START_2D = (35, -10)

print("Test function ready. Global min is at (0,0) with f=0.")
print(f"Start: {START_2D}, f={f2d_complex(*START_2D):.4f}")

# ===============================
# 2) Hill climb + basin compression 실행 (WITH compression)
# ===============================
traj2d, cm2d = hill_climb_with_compression_nd(
    fitness_func_nd=f2d_complex_nd,
    start_point=START_2D,
    dim=2,
    max_iterations=20,      # compression iteration 횟수
    basin_max_search=60,
    global_min_threshold=1e-6,
)

# traj2d 원소: (point, fitness, used_compression)
final_point, final_f, _ = traj2d[-1]
print(f"\n===== WITH COMPRESSION =====")
print(f"End:   {final_point}, f={final_f:.6g}")
print(f"Trajectory length: {len(traj2d)}")

# trajectory 분해 (N-D 형식에 맞게 언팩)
tx   = np.array([step[0][0] for step in traj2d], dtype=float)   # point[0] = x
ty   = np.array([step[0][1] for step in traj2d], dtype=float)   # point[1] = y
tf   = np.array([step[1]      for step in traj2d], dtype=float) # fitness
tcomp = np.array([step[2]     for step in traj2d], dtype=bool)  # True면 압축 사용 후 단계
idx_after  = np.where(tcomp)[0]   # compression 사용 후 단계 인덱스

# ===============================
# 2') Baseline hill climb 실행 (WITHOUT compression)
# ===============================
traj2d_base = hill_climb_simple_nd(
    fitness_func_nd=f2d_complex_nd,
    start_point=START_2D,
    dim=2,
    max_steps=2000
)

base_final_point, base_final_f = traj2d_base[-1]
print(f"\n===== BASELINE (NO COMPRESSION) =====")
print(f"End:   {base_final_point}, f={base_final_f:.6g}")
print(f"Trajectory length: {len(traj2d_base)}")

bx = np.array([p[0] for (p, f) in traj2d_base], dtype=float)
by = np.array([p[1] for (p, f) in traj2d_base], dtype=float)

# ===============================
# 3) Surface + path 시각화
# ===============================
x_grid = np.arange(X_MIN, X_MAX + 1, 1)
y_grid = np.arange(Y_MIN, Y_MAX + 1, 1)
X_vis, Y_vis = np.meshgrid(x_grid, y_grid)
Z_vis = np.zeros_like(X_vis, dtype=float)

for i in range(X_vis.shape[0]):
    for j in range(X_vis.shape[1]):
        Z_vis[i, j] = f2d_complex(X_vis[i, j], Y_vis[i, j])

fig = plt.figure(figsize=(18, 7))

# --- (1) 3D surface + path (compression 전체 경로는 그대로 빨강) ---
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.plot_surface(
    X_vis, Y_vis, Z_vis,
    cmap='viridis',
    alpha=0.7,
    linewidth=0,
    antialiased=True,
)
ax3d.plot(tx, ty, tf, 'r-o', linewidth=3, markersize=5, label='Search path (with compression)')

ax3d.scatter(tx[0], ty[0], tf[0], c='white', s=80, marker='o', label='Start')
ax3d.scatter(tx[-1], ty[-1], tf[-1], c='yellow', s=120, marker='*', label='End (compression)')

ax3d.set_title(
    "2D Hill Climb + Basin Compression (N-D framework)\n"
    r"$f(x,y)=1-e^{-(x^2+y^2)}+(\sin(0.5x)\cos^3(y-x))^2$",
    fontsize=13
)
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("fitness")
ax3d.legend(loc='upper left')

# --- (2) Contour + path (after compression + baseline) ---
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(X_vis, Y_vis, Z_vis, levels=40, cmap='viridis')
fig.colorbar(cs, ax=ax2, shrink=0.8)

# compression 사용 후 단계만 초록색
if len(idx_after) > 0:
    ax2.plot(
        tx[idx_after], ty[idx_after],
        'g-s',
        label='After compression',
        linewidth=3,
    )

# baseline: 흰색 점선
ax2.plot(
    bx, by,
    'w--',
    linewidth=2,
    label='Baseline (no compression)',
)

# start / end 마커
ax2.scatter(tx[0],  ty[0],
            c='white',  edgecolors='k', s=80,  marker='o', label='Start')
ax2.scatter(tx[-1], ty[-1],
            c='yellow', edgecolors='k', s=120, marker='*', label='End (compression)')
ax2.scatter(bx[-1], by[-1],
            c='cyan',   edgecolors='k', s=120, marker='X', label='End (baseline)')

ax2.set_title("Top view with trajectories", fontsize=14)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("hillclimb_result.png", dpi=300)
