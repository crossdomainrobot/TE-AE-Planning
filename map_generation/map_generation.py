# @File : map_generation.py
# author: Hou Chenfei (modified per request)
# Time：2026-01-17
# -*- coding: UTF-8 -*-

import random
import math

# ======================== 与你原来一致的高度分布函数 ========================

def generate_heights(n: int):
    """
    生成 n 个障碍的高度列表（高度取 1/2/3），满足原代码的比例约束：
    - 高度1至少占 50%
    - 高度2至少占 25%
    - 高度3在 10%~20% 之间
    """
    min1 = math.ceil(0.5 * n)
    min2 = math.ceil(0.25 * n)
    min3 = math.ceil(0.1 * n)
    max3 = math.floor(0.2 * n)

    candidates = []
    for n3 in range(min3, max3 + 1):
        rest = n - n3
        for n1 in range(min1, rest - min2 + 1):
            n2 = rest - n1
            if n2 >= min2:
                candidates.append((n1, n2, n3))

    if not candidates:
        raise ValueError(f"No valid combination for n={n}")

    n1, n2, n3 = random.choice(candidates)
    heights = [1] * n1 + [2] * n2 + [3] * n3
    random.shuffle(heights)
    return heights


# ======================== 与你原来一致的 Map2D ========================

class Map2D:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.data = None    # 0/1，是否是障碍
        self.height = None  # 障碍高度（3D）

    def map_init(self):
        self.data = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.height = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def map_obstacle(self, num: int):
        """
        与原代码一致：随机放置障碍，并给出高度（1,2,3）。
        四个角固定留空。
        """
        desired_obstacles = num
        corners = {(0, 0), (self.rows - 1, 0), (0, self.cols - 1), (self.rows - 1, self.cols - 1)}

        all_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in corners
        ]

        num = min(desired_obstacles, len(all_cells))

        obstacle_cells = random.sample(all_cells, num)
        heights_list = generate_heights(num)

        self.map_init()
        for (r, c), h in zip(obstacle_cells, heights_list):
            self.data[r][c] = 1
            self.height[r][c] = h

        # corners 固定留空
        for (r, c) in corners:
            self.data[r][c] = 0
            self.height[r][c] = 0


# ======================== 导出工具 ========================

def export_map_as_python(map2d: Map2D, name: str):
    """
    把一张地图打印成可复制的 Python 结构。
    name: 该地图在输出中的名字，比如 "map0"
    """
    rows, cols = map2d.rows, map2d.cols

    print(f"# ========== {name} ==========")
    print(f"{name} = {{")
    print(f"    'rows': {rows},")
    print(f"    'cols': {cols},")
    print("    'data': [")
    for r in range(rows):
        print("        " + str(map2d.data[r]) + ",")
    print("    ],")
    print("    'height': [")
    for r in range(rows):
        print("        " + str(map2d.height[r]) + ",")
    print("    ]")
    print("}")
    print()  # 空行分隔


def generate_map_bank(num_maps: int = 100, rows: int = 20, cols: int = 20, obstacles: int = 120):
    """
    一次性生成若干张地图，并以 Python 形式打印出来。
    - num_maps  : 要生成多少张地图（按你的要求默认 100）
    - rows,cols : 地图尺寸
    - obstacles : 障碍数量
    注意：不使用随机种子，每次运行都会随机产生不同的地图。
    """
    for i in range(num_maps):
        m = Map2D(rows, cols)
        m.map_obstacle(obstacles)

        name = f"map{i}"
        export_map_as_python(m, name)


if __name__ == "__main__":
    """
    运行本文件，会在终端打印出 map0, map1, ... map99 的 Python 字典结构
    你可以直接复制这些结果到路径规划算法代码中使用。
    """
    generate_map_bank(
        num_maps=100,  # 每次生成 100 张
        rows=20,
        cols=20,
        obstacles=120
    )
