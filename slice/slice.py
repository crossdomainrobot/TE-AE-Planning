# @File : slice .py
# author: Hou Chenfei
# Time：2026-1-23
import numpy as np
import matplotlib.pyplot as plt

# ======================== 地图数据 ========================
rows, cols = 10, 10

data = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
]

height_map = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 2, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 3, 0, 0, 2, 1, 3, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [2, 0, 3, 0, 0, 0, 0, 2, 2, 3],
    [3, 2, 0, 1, 0, 0, 0, 2, 1, 1],
    [2, 0, 2, 1, 0, 0, 1, 0, 1, 2],
    [3, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 2, 0, 0, 0, 0, 1, 2, 0],
    [0, 1, 2, 2, 0, 0, 0, 0, 0, 0],
], dtype=float)


def draw_plane(ax, height_map, target_h, title, draw_cell_grid=False,
               obstacle_color="black", free_color="white",
               grid_color="black", grid_lw=1.0):
    """
    画一个“高度阈值平面”：
    - target_h: 0/1/2/3
      规则：把 height_map > target_h 视为“障碍物”(黑色)，否则为空白
    - draw_cell_grid: 是否画每个栅格的黑色边框
    """
    # 生成二值图：障碍物=1（黑），空白=0（白）
    obstacle = (height_map > target_h).astype(int)

    # 用灰度 colormap：0->白, 1->黑
    ax.imshow(
        obstacle,
        cmap="gray_r",          # 0白1黑（gray_r 是反转灰度）
        vmin=0, vmax=1,
        interpolation="nearest",
        origin="upper"
    )

    ax.set_title(title, fontsize=16)

    # 不需要横纵坐标值
    ax.set_xticks([])
    ax.set_yticks([])

    # 栅格边框（每格一条线）
    if draw_cell_grid:
        # 网格线放在像素边界：-0.5, 0.5, 1.5, ...
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color=grid_color, linewidth=grid_lw)
        ax.tick_params(which="minor", bottom=False, left=False)

    # 外边框也设成黑色
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)


def main():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # 1) h=0 平面：height > 0 都当障碍物（黑）
    draw_plane(
        axes[0, 0], height_map, target_h=0,
        title="Plane h=0",
        draw_cell_grid=True
    )

    # 2) h=1 平面：height > 1 当障碍物（黑）
    draw_plane(
        axes[0, 1], height_map, target_h=1,
        title="Plane h=1",
        draw_cell_grid=True
    )

    # 3) h=2 平面：height > 2 当障碍物（黑）
    draw_plane(
        axes[1, 0], height_map, target_h=2,
        title="Plane h=2",
        draw_cell_grid=True
    )

    # 3) h=3 平面：按你的要求，每格黑色边框；（height>3 才算障碍物，通常全白）
    draw_plane(
        axes[1, 1], height_map, target_h=3,
        title="Plane h=3",
        draw_cell_grid=True,
        grid_lw=1.0
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
