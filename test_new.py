import matplotlib.pyplot as plt

# 数据
model_capacity = [18, 34, 50]  # ResNet18, ResNet34, ResNet50
accuracy_tsat_resnet = [25, 32, 37]  # TSAT-ResNet
accuracy_tsat_wideresnet = [25, 30, 35]  # TSAT-WideResNet
accuracy_avmixup_resnet = [27, 32, 35]  # Avmixup-ResNet

# 绘制图像
plt.figure(figsize=(8, 6))

# TSAT-ResNet
plt.plot(model_capacity, accuracy_tsat_resnet, 'k-', marker='D', markersize=8, label='TSAT-ResNet', color='skyblue')
for x, y, label in zip(model_capacity, accuracy_tsat_resnet, ['Res18', 'Res34', 'Res50']):
    plt.text(x, y, label, fontsize=10, ha='right')

# TSAT-WideResNet
plt.plot(model_capacity, accuracy_tsat_wideresnet, 'k-', marker='s', markersize=8, label='TSAT-WideResNet', color='coral')
for x, y, label in zip(model_capacity, accuracy_tsat_wideresnet, ['WRN16', 'WRN28', 'WRN34']):
    plt.text(x, y, label, fontsize=10, ha='left')

# Avmixup-ResNet
plt.plot(model_capacity, accuracy_avmixup_resnet, 'k-', marker='o', markersize=8, label='Avmixup-ResNet', color='limegreen')
for x, y, label in zip(model_capacity, accuracy_avmixup_resnet, ['Res18', 'Res34', 'Res50']):
    plt.text(x, y, label, fontsize=10, ha='left')

# 设置轴标签
plt.xlabel('Model Capacity', fontsize=14)
plt.ylabel('Robust accuracy', fontsize=14)

# 设置刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 添加图例
plt.legend(fontsize=12, loc='best', frameon=True)

# 设置边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图像
plt.tight_layout()
plt.savefig("./results_final/sensitivity_1.pdf")
plt.show()