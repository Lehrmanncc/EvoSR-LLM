# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据定义
# alphas = [0.1, 1, 5, 10, 50]
# osc_id_alpha = [1.05e-6, 1.28e-6, 5.39e-8, 1.53e-6, 2.58e-5]
# osc_ood_alpha = [0.0287, 0.0038, 1.86e-4, 0.0067, 0.0309]
#
# ts = [40, 80, 120, 160, 200]
# osc_id_t = [5.04e-7, 6.54e-8, 5.39e-8, 9.89e-6, 1.12e-6]
# osc_ood_t = [0.0311, 1.79e-4, 1.86e-4, 0.0371, 0.2292]
#
# # 设置风格
# plt.rcParams.update({
#         'font.family': 'serif',
#         'font.serif': 'Times New Roman',
#         'axes.titlesize': 18,
#         'axes.labelsize': 16,
#         # 'axes.labelweight': 'bold',
#         'xtick.labelsize': 14,
#         'ytick.labelsize': 14,
#         'legend.fontsize': 12,
#         'lines.linewidth': 2,  # 线条宽度
#         'grid.alpha': 1,  # 网格线透明度
#     })
#
# # ---------------------- 图 1: Penalty Weight α ----------------------
# plt.figure(figsize=(10, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(alphas, osc_id_alpha, marker='o', label='Oscillation 1 (ID)', color='blue')
# plt.plot(alphas, osc_ood_alpha, marker='s', label='Oscillation 1 (OOD)', color='orange')
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Penalty Weight α')
# plt.ylabel('Oscillation Value')
# plt.title('Effect of Penalty Weight α')
# plt.legend()
# # 标出最小值
# min_id_idx = np.argmin(osc_id_alpha)
# min_ood_idx = np.argmin(osc_ood_alpha)
# plt.scatter(alphas[min_id_idx], osc_id_alpha[min_id_idx], color='red', zorder=5)
# plt.scatter(alphas[min_ood_idx], osc_ood_alpha[min_ood_idx], color='red', zorder=5)
#
# # ---------------------- 图 2: Sampling Interval t ----------------------
# plt.subplot(1, 2, 2)
# plt.plot(ts, osc_id_t, marker='o', label='Oscillation 1 (ID)', color='blue')
# plt.plot(ts, osc_ood_t, marker='s', label='Oscillation 1 (OOD)', color='orange')
# plt.yscale('log')
# plt.xlabel('Sampling Interval t')
# plt.ylabel('Oscillation Value')
# plt.title('Effect of Interval t')
# plt.legend()
# # 标出最小值
# min_id_idx = np.argmin(osc_id_t)
# min_ood_idx = np.argmin(osc_ood_t)
# plt.scatter(ts[min_id_idx], osc_id_t[min_id_idx], color='red', zorder=5)
# plt.scatter(ts[min_ood_idx], osc_ood_t[min_ood_idx], color='red', zorder=5)
#
# plt.tight_layout()
# plt.savefig("./results_final/sensitivity.pdf")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 启用 LaTeX 字体
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif': 'Times New Roman',
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
})


# 数据
alphas = [0.1, 1, 5, 10, 50]
osc_id_alpha = [1.05e-6, 1.28e-6, 5.39e-8, 1.53e-6, 2.58e-5]
osc_ood_alpha = [0.0287, 0.0038, 1.86e-4, 0.0067, 0.0309]

ts = [40, 80, 120, 160, 200]
osc_id_t = [5.04e-7, 6.54e-8, 5.39e-8, 9.89e-6, 1.12e-6]
osc_ood_t = [0.0311, 1.79e-4, 1.86e-4, 0.0371, 0.2292]

# 图形设置
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))

# -------- 图1: α参数 --------

axs[0].plot(alphas, osc_id_alpha, 'k-', marker='o', markersize=6, color='black', label='ID', linewidth=1.5, markerfacecolor='skyblue')
axs[0].plot(alphas, osc_ood_alpha, marker='D', markersize=6,  color='black', label='OOD', linewidth=1.5, markerfacecolor='coral')

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'Penalty scaling factor $\alpha$')
axs[0].set_ylabel('Normalized MSE')
axs[0].set_title(r'(a)')
axs[0].legend(loc='center left')
axs[0].grid(True)

# 标注最优点
# idx1 = np.argmin(osc_id_alpha)
# idx2 = np.argmin(osc_ood_alpha)
# axs[0].scatter(alphas[idx1], osc_id_alpha[idx1], s=60, color='red', zorder=5)
# axs[0].scatter(alphas[idx2], osc_ood_alpha[idx2], s=60, color='red', zorder=5)

# -------- 图2: t参数 --------
axs[1].plot(ts, osc_id_t, 'k-', marker='o', markersize=6, color='black', label='ID', linewidth=1.5, markerfacecolor='skyblue')
axs[1].plot(ts, osc_ood_t, 'k-', marker='D', markersize=6,  color='black', label='OOD', linewidth=1.5, markerfacecolor='coral')

axs[1].set_yscale('log')
axs[1].set_xlabel(r'Interval $t$')
axs[1].set_ylabel('Normalized MSE')
axs[1].set_title(r'(b)')
axs[1].legend(loc='center right')
axs[1].grid(True)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 标注最优点
# idx1 = np.argmin(osc_id_t)
# idx2 = np.argmin(osc_ood_t)
# axs[1].scatter(ts[idx1], osc_id_t[idx1], s=60, facecolors='none', edgecolors='red', linewidths=1.5)
# axs[1].scatter(ts[idx2], osc_ood_t[idx2], s=60, facecolors='none', edgecolors='red', linewidths=1.5)

# 总体布局
plt.tight_layout()
plt.savefig("./results_final/sensitivity.pdf")
plt.show()

