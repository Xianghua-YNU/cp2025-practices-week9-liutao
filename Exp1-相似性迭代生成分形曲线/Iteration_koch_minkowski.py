import numpy as np
import matplotlib.pyplot as plt

def koch_generator(u, level):
    """
    递归/迭代生成科赫曲线的点序列。

    参数:
        u: 初始线段的端点数组（复数表示）
        level: 迭代层数

    返回:
        numpy.ndarray: 生成的所有点（复数数组）
    """
    if level == 0:
        return u
    
    # 将每条线段替换为4段（5个点）
    new_points = []
    for i in range(len(u)-1):
        start = u[i]
        end = u[i+1]
        
        # 计算4个等分点
        segment = (end - start)
        p1 = start
        p2 = start + segment / 3
        p3 = p2 + (segment / 3) * np.exp(1j * np.pi/3)  # 旋转60度
        p4 = start + 2 * segment / 3
        p5 = end
        
        new_points.extend([p1, p2, p3, p4])
    
    new_points.append(u[-1])  # 添加最后一个点
    
    return koch_generator(np.array(new_points), level-1)

def minkowski_generator(u, level):
    """
    递归/迭代生成闵可夫斯基香肠曲线的点序列。

    参数:
        u: 初始线段的端点数组（复数表示）
        level: 迭代层数

    返回:
        numpy.ndarray: 生成的所有点（复数数组）
    """
    if level == 0:
        return u
    
    # 将每条线段替换为8段（9个点）
    new_points = []
    for i in range(len(u)-1):
        start = u[i]
        end = u[i+1]
        
        segment = (end - start) / 4
        # 生成闵可夫斯基香肠的8个线段（9个点）
        p1 = start
        p2 = p1 + segment
        p3 = p2 + segment * np.exp(1j * np.pi/2)  # 向上90度
        p4 = p3 + segment
        p5 = p4 + segment * np.exp(-1j * np.pi/2)  # 向下90度
        p6 = p5 + segment * np.exp(-1j * np.pi/2)  # 继续向下90度
        p7 = p6 + segment
        p8 = p7 + segment * np.exp(1j * np.pi/2)   # 向上90度
        p9 = end
        
        new_points.extend([p1, p2, p3, p4, p5, p6, p7, p8])
    
    new_points.append(u[-1])  # 添加最后一个点
    
    return minkowski_generator(np.array(new_points), level-1)

if __name__ == "__main__":
    # 初始线段（转换为复数表示）
    init_u = np.array([0 + 0j, 1 + 0j])

    # 绘制不同层级的科赫曲线
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        koch_points = koch_generator(init_u, i+1)
        axs[i//2, i%2].plot(
            np.real(koch_points), np.imag(koch_points), 'k-', lw=1
        )
        axs[i//2, i%2].set_title(f"Koch Curve Level {i+1}")
        axs[i//2, i%2].axis('equal')
        axs[i//2, i%2].axis('off')
    plt.tight_layout()
    plt.savefig('koch_curves.png')
    plt.show()

    # 绘制不同层级的闵可夫斯基香肠曲线
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        minkowski_points = minkowski_generator(init_u, i+1)
        axs[i//2, i%2].plot(
            np.real(minkowski_points), np.imag(minkowski_points), 'k-', lw=1
        )
        axs[i//2, i%2].set_title(f"Minkowski Sausage Level {i+1}")
        axs[i//2, i%2].axis('equal')
        axs[i//2, i%2].axis('off')
    plt.tight_layout()
    plt.savefig('minkowski_sausages.png')
    plt.show()
