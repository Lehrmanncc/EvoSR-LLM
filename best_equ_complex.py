import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# 数据
datasets = ["Oscillator 1", "Oscillator 2", "E. coli growth", "Stress-Strain"]
evosr = [24, 46, 61, 16]
llmsr = [42, 42, 80, 46]
drsr = [53, 158, 96, 261]
funsearch = [43, 45, 45, 43]
DSR = [68, 131, 214, 94]
uDSR = [50, 126, 234, 63]
PySR = [26, 32, 27, 33]


# 位置参数
x = np.arange(len(datasets))
# 调整宽度以适应更多柱状图
width = 0.11  # 从0.15调整为0.11以容纳7个算法

plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.titlesize': 32,      # 与multi_convergence_plot.py保持一致 (18 -> 32)
        'axes.labelsize': 28,      # 与multi_convergence_plot.py保持一致 (16 -> 28)
        'xtick.labelsize': 24,     # 与multi_convergence_plot.py保持一致 (14 -> 24)
        'ytick.labelsize': 24,     # 与multi_convergence_plot.py保持一致 (14 -> 24)
        'legend.fontsize': 26,     # 与multi_convergence_plot.py保持一致 (16 -> 26)
        'lines.linewidth': 3.0,    # 与multi_convergence_plot.py保持一致 (2 -> 3.0)
        'grid.alpha': 0.8,         # 与multi_convergence_plot.py保持一致 (1 -> 0.8)
        'grid.linewidth': 1.0,     # 增加网格线宽度
        'axes.linewidth': 1.5,     # 增加坐标轴边框线宽
    })

# 绘制直方图
fig, ax = plt.subplots(figsize=(12, 6))  # 增加图像尺寸以适应更大字体 (10, 5) -> (12, 6)
# 绘制每组柱状图（包含所有7个算法）
rects1 = ax.bar(x - 3*width, evosr, width, label='EvoSR-LLM', color='#E15759')
rects2 = ax.bar(x - 2*width, llmsr, width, label='LLM-SR', color='#F28E2B')
rects3 = ax.bar(x - width, drsr, width, label='DrSR', color='#AF7AA1')
rects4 = ax.bar(x, funsearch, width, label='FunSearch', color='#FF9D9A')
rects5 = ax.bar(x + width, DSR, width, label='DSR', color='#76B7B2')
rects6 = ax.bar(x + 2*width, uDSR, width, label='uDSR', color='#59A14F')
rects7 = ax.bar(x + 3*width, PySR, width, label='PySR', color='#4E79A7')

# rects1 = ax.bar(x - width/2, evosr, width, label='EvoSR-LLM', color='royalblue', alpha=0.8)
# rects2 = ax.bar(x + width/2, llmsr, width, label='LLM-SR', color='orange', alpha=0.8)
# rects3 = ax.bar(x - width/2, PySR, width, label='EvoSR-LLM', color='royalblue', alpha=0.8)
# rects4 = ax.bar(x + width/2, uDSR, width, label='LLM-SR', color='orange', alpha=0.8)
# rects5 = ax.bar(x - width/2, DSR, width, label='EvoSR-LLM', color='royalblue', alpha=0.8)


# 标注
ax.set_xlabel('Problems', fontsize=28)  # 与rcParams保持一致 (16 -> 28)
ax.set_ylabel('Expression Tree Complexity', fontsize=28)  # 与rcParams保持一致 (16 -> 28)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=24)  # 与rcParams保持一致 (14 -> 24)
ax.legend(fontsize=20)  # 适当减小图例字体 (26 -> 20)
ax.grid(axis='y', linestyle='--', alpha=0.8, linewidth=1.0)  # 与multi_convergence_plot.py保持一致

# 显示数值
for rects in [rects1, rects2, rects3, rects4, rects5, rects6, rects7]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height+1, f'{height}', ha='center', va='bottom', fontsize=15)  # 减小数值标注字体 (18 -> 12)

plt.tight_layout()
plt.savefig("./figures/exp/Complexity.pdf")
plt.show()


