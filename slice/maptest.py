import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.colors as mcolors

# ======================== 地图数据 ========================
rows, cols = 10, 10

# 柱子“面”的透明度（0~1）
BAR_FACE_ALPHA = 0.45
# 柱子“边框”的透明度
BAR_EDGE_ALPHA = 0.5
# ⭐ 柱子高度缩放系数（<1 会让柱子显得更矮）
BAR_HEIGHT_SCALE = 0.4

# 侧面(XZ / YZ)网格线样式，只在两个空白面上用
SIDE_GRID_ALPHA = 0.4         # 透明度
SIDE_GRID_LINEWIDTH = 1       # 粗细（越大越粗）
# 虚线间距：(线段长度, 空白长度)
SIDE_GRID_DASHES = (4, 8)

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


color_map = {
    1: "#0473AF",  # h=1
    2: "#55288F",  # h=2
    3: "#B2355D",  # h=3
}


def make_plane_facecolors(height_map, color_map):
    rows, cols = height_map.shape
    fc = np.zeros((rows, cols, 4))
    for i in range(rows):
        for j in range(cols):
            h = int(height_map[i, j])
            if h == 0:
                rgba = mcolors.to_rgba("white")
            else:
                rgba = mcolors.to_rgba(color_map.get(h, "#cccccc"))
            fc[i, j, :] = rgba
    return fc


def draw_ground_border(ax):
    border_width = 2.0  # ⭐ 调整地面边框粗细
    ax.plot([0, cols], [0, 0], [0, 0], color="black", linewidth=border_width)
    ax.plot([0, cols], [rows, rows], [0, 0], color="black", linewidth=border_width)
    ax.plot([0, 0], [0, rows], [0, 0], color="black", linewidth=border_width)
    ax.plot([cols, cols], [0, rows], [0, 0], color="black", linewidth=border_width)


def draw_side_grids(ax):
    """
    只在两个空白侧面上画 Z 方向的虚线：
    - 后面的 XZ 平面：y = rows
    - 右侧的 YZ 平面：x = cols
    """
    z_min, z_max = 0, 4  # 和 set_zlim 一致

    # ========= 后面 XZ 平面：y = rows =========
    # 只画竖线：沿 z 方向
    for x in range(cols + 1):
        ax.plot(
            [x, x], [rows, rows], [z_min, z_max],
            color="black",
            alpha=SIDE_GRID_ALPHA,
            linewidth=SIDE_GRID_LINEWIDTH,
            dashes=SIDE_GRID_DASHES,
        )

    # ========= 右侧 YZ 平面：x = cols =========
    # 只画竖线：沿 z 方向
    for y in range(rows + 1):
        ax.plot(
            [cols, cols], [y, y], [z_min, z_max],
            color="black",
            alpha=SIDE_GRID_ALPHA,
            linewidth=SIDE_GRID_LINEWIDTH,
            dashes=SIDE_GRID_DASHES,
        )


def draw_map_with_plane(ax):
    # ========= 3D 柱子 =========
    X, Y = np.meshgrid(np.arange(0, cols), np.arange(0, rows))
    Xr, Yr = X.ravel(), Y.ravel()
    Zr = np.zeros_like(Xr)

    DZ_raw = height_map.ravel()  # 原始高度（0~3）
    mask = DZ_raw > 0           # 只画有障碍物的格子

    # ⭐ 实际绘制的高度 = 原始高度 * 缩放系数
    DZ_scaled = DZ_raw * BAR_HEIGHT_SCALE

    dx = dy = 1.0

    # 为每个高度生成带“面透明度”的 RGBA 颜色（h>0 的柱子）
    face_colors = []
    for h in DZ_raw:
        if h > 0:
            base_rgba = mcolors.to_rgba(color_map.get(int(h), "#cccccc"))
            face_colors.append((base_rgba[0], base_rgba[1], base_rgba[2], BAR_FACE_ALPHA))

    ax.bar3d(
        Xr[mask], Yr[mask], Zr[mask],
        dx, dy, DZ_scaled[mask],      # ⭐ 使用缩放后的高度
        color=face_colors,
        edgecolor=(0, 0, 0, BAR_EDGE_ALPHA),
        linewidth=0.75
    )

    # ========= 上层 Z=4 的 2D 视图 =========
    Xp, Yp = np.meshgrid(np.arange(0, cols + 1), np.arange(0, rows + 1))
    Zp_top = np.full_like(Xp, 4)
    facecolors = make_plane_facecolors(height_map, color_map)

    ax.plot_surface(
        Xp, Yp, Zp_top,
        facecolors=facecolors,
        edgecolor="black",
        linewidth=0.5,
        shade=False,
        alpha=1
    )

    # ⭐ 为 Z=4 的平面添加边框线
    top_border_width = 2.5
    ax.plot([0, cols], [0, 0], [4, 4], color="black", linewidth=top_border_width)
    ax.plot([0, cols], [rows, rows], [4, 4], color="black", linewidth=top_border_width)
    ax.plot([0, 0], [0, rows], [4, 4], color="black", linewidth=top_border_width)
    ax.plot([cols, cols], [0, rows], [4, 4], color="black", linewidth=top_border_width)

    # 只保留地面边框
    draw_ground_border(ax)

    # ⭐ 两个空白侧面的虚线网格
    draw_side_grids(ax)

    # ⭐ 在 X=20, Y=0 加一条竖线，把 Z=0 到 Z=4 连起来
    ax.plot([20, 20], [0, 0], [0, 4], color="black", linewidth=1.5)

    # 轴范围
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_zlim(0, 4)

    # ========= 轴刻度 =========
    ax.set_xticks(range(0, cols + 1, 2))
    ax.set_yticks(range(0, rows + 1, 2))

    # ⭐ Z 轴刻度位置用“缩放后高度”，标签仍然显示原始高度 1,2,3
    logical_heights = np.array([1, 2, 3])
    scaled_ticks = logical_heights * BAR_HEIGHT_SCALE
    ax.set_zticks(scaled_ticks)
    ax.set_zticklabels([str(h) for h in logical_heights])

    # 把 X、Y 反向（保持你原有习惯）
    ax.invert_xaxis()
    ax.invert_yaxis()

    # ⭐ 这里加 labelpad，让 XYZ 和刻度数字拉开一点
    ax.set_xlabel("X", labelpad=8)
    ax.set_ylabel("Y", labelpad=8)
    ax.set_zlabel("Height", labelpad=8)


def main():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32  # 比如 12、14、16，数字越大越大

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # 修改 Z 轴的线条样式
    ax.zaxis.line.set_color("black")
    ax.zaxis.line.set_linewidth(1.7)

    # 关掉 3D 默认网格线
    ax.grid(False)

    # 坐标平面背景设为透明，只保留边框和刻度
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("black")

    draw_map_with_plane(ax)

    # 调整刻度线长度和粗细
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['tick']['inward_factor'] = 0.35
        axis._axinfo['tick']['outward_factor'] = 0.0
        axis._axinfo['tick']['linewidth'] = {True: 1, False: 1.5}

    # 刻度字体大小
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.tick_params(axis='z', labelsize=28)

    ax.view_init(elev=30, azim=60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
