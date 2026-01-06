import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ================== 1. 原始数据 ==================
# 年份和对应的数值
years = np.array([2015, 2016, 2017, 2018, 2019,
                  2020, 2021, 2022, 2023, 2024])
values = np.array([2, 5, 4, 6, 6,
                   4, 14, 13, 25, 38], dtype=float)

# ================== 2. 归一化自变量（为了数值稳定） ==================
# 把年份映射到 [0, 1] 区间
x = (years - years.min()) / (years.max() - years.min())

# ================== 3. 定义单调递增的 Logistic 函数 ==================
def logistic_raw(x, L, k_raw, x0):
    """
    L: 上限（最大值）
    k_raw: 斜率参数（通过 abs 保证单调递增）
    x0: 拐点位置
    """
    k = np.abs(k_raw)  # 保证递增
    return L / (1.0 + np.exp(-k * (x - x0)))

# 初始参数猜测
L0 = values.max()
k0 = 5.0
x0_0 = 0.5
p0 = [L0, k0, x0_0]

# ================== 4. 拟合参数 ==================
params, _ = curve_fit(logistic_raw, x, values, p0=p0, maxfev=20000)
L_fit, k_fit_raw, x0_fit = params
k_fit = abs(k_fit_raw)

print("拟合得到的参数：")
print("L =", L_fit, "k =", k_fit, "x0 =", x0_fit)

# 每一年的拟合值（可以对比看看）
values_fit = logistic_raw(x, L_fit, k_fit, x0_fit)
print("\n逐年拟合结果：")
for year, v_raw, v_fit in zip(years, values, values_fit):
    print(year, "原始:", v_raw, "  拟合:", round(v_fit, 2))

# ================== 5. 生成 1000 个平滑采样点 ==================
num_points = 1000
x_dense = np.linspace(x.min(), x.max(), num_points)
y_dense = logistic_raw(x_dense, L_fit, k_fit, x0_fit)

# 把归一化的 x_dense 映射回真实年份坐标
years_dense = x_dense * (years.max() - years.min()) + years.min()

# ================== 6. 保存到当前代码目录下的 CSV ==================
df = pd.DataFrame({
    "year": years_dense,
    "value": y_dense
})
output_path = "fitted_curve.csv"   # 保存在代码所在目录
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n已将 {num_points} 个拟合点保存到: {output_path}")

# ================== 7. 画个图检查一下（可选） ==================
plt.figure(figsize=(7, 4))
plt.scatter(years, values, label="原始数据点")
plt.plot(years_dense, y_dense, label="平滑单调增长拟合曲线")
plt.xlabel("年份")
plt.ylabel("数值")
plt.title("Logistic 平滑单调增长拟合")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
