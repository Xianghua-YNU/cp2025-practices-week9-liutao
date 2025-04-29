"""
项目5: 盒计数法估算分形维数
实现盒计数算法计算分形图像的盒维数

任务说明：
1. 实现盒计数算法计算分形图像的盒维数
2. 完成以下函数实现
3. 在main函数中测试你的实现
"""

"""
项目5: 盒计数法估算分形维数
实现盒计数算法计算分形图像的盒维数
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_binarize_image(image_path, threshold=128):
    """
    加载图像并转换为二值数组
    """
    img = Image.open(image_path).convert('L')  # 转为灰度图像
    np_image = np.array(img)
    binary = (np_image >= threshold).astype(int)  # 二值化处理（假设分形为白色）
    return binary

def box_count(binary_image, box_sizes):
    """
    盒计数算法实现
    """
    h, w = binary_image.shape
    counts = {}
    for s in box_sizes:
        if s <= 0:
            continue
        rows = (h + s - 1) // s  # 向上取整
        cols = (w + s - 1) // s
        count = 0
        for i in range(rows):
            y_start = i * s
            y_end = min(y_start + s, h)
            for j in range(cols):
                x_start = j * s
                x_end = min(x_start + s, w)
                box = binary_image[y_start:y_end, x_start:x_end]
                if np.any(box == 1):
                    count += 1
        counts[s] = count
    return counts

def calculate_fractal_dimension(binary_image, min_box_size=1, max_box_size=None, num_sizes=10):
    """
    计算分形维数
    """
    h, w = binary_image.shape
    min_dim = min(h, w)
    if max_box_size is None:
        max_box_size = min_dim // 2
    max_box_size = max(max_box_size, min_box_size + 1)  # 确保max > min
    
    # 生成等比数列的盒子尺寸（对数空间）
    log_max = np.log(max_box_size)
    log_min = np.log(min_box_size)
    log_sizes = np.linspace(log_max, log_min, num=num_sizes)
    box_sizes = np.exp(log_sizes).astype(int)
    box_sizes = np.unique(box_sizes)
    box_sizes = box_sizes[box_sizes >= min_box_size]
    box_sizes = sorted(box_sizes, reverse=True)
    
    counts = box_count(binary_image, box_sizes)
    epsilons = np.array(list(counts.keys()))
    N_epsilons = np.array(list(counts.values()))
    
    # 过滤无效数据（N=0）
    valid = N_epsilons > 0
    epsilons = epsilons[valid]
    N_epsilons = N_epsilons[valid]
    if len(epsilons) < 2:
        raise ValueError("数据点不足，无法进行线性回归。")
    
    # 线性回归
    log_eps = np.log(epsilons)
    log_N = np.log(N_epsilons)
    slope, intercept = np.polyfit(log_eps, log_N, 1)
    D = -slope
    return D, (epsilons, N_epsilons, slope, intercept)

def plot_log_log(epsilons, N_epsilons, slope, intercept, save_path=None):
    """
    绘制log-log图
    """
    plt.figure()
    log_eps = np.log(epsilons)
    log_N = np.log(N_epsilons)
    plt.scatter(log_eps, log_N, label='Data Points')
    regression_line = slope * log_eps + intercept
    plt.plot(log_eps, regression_line, 'r', label=f'Fit: D = {-slope:.3f}')
    plt.xlabel('log(ε)')
    plt.ylabel('log(N(ε))')
    plt.legend()
    plt.title('Log-Log Plot of Box Counting')
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    IMAGE_PATH = "../../images/barnsley_fern.png"  # 测试图像路径
    
    # 1. 加载并二值化图像
    binary_img = load_and_binarize_image(IMAGE_PATH, threshold=128)
    
    # 2. 计算分形维数
    D, results = calculate_fractal_dimension(binary_img)
    epsilons, N_epsilons, slope, intercept = results
    
    # 3. 输出结果
    print(f"估算的盒维数 D = {D:.5f}")
    print("盒子尺寸 (ε):", epsilons)
    print("盒子计数 (N(ε)):", N_epsilons)
    
    # 4. 绘制log-log图
    plot_log_log(epsilons, N_epsilons, slope, intercept, "log_log_plot.png")
