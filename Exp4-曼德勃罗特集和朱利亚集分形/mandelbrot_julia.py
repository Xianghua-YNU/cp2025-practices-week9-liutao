
import numpy as np
import matplotlib.pyplot as plt


def generate_mandelbrot(width=800, height=800, max_iter=100):
    """
    生成Mandelbrot集数据(向量化版本)
    :param width: 图像宽度
    :param height: 图像高度
    :param max_iter: 最大迭代次数
    :return: Mandelbrot集数据
    """
    x = np.linspace(-2.0, 1.0, width)
    y = np.linspace(-1.5, 1.5, height)
    Re, Im = np.meshgrid(x, y)
    C = Re + 1j * Im

    B = np.zeros(C.shape)  # 记录迭代次数
    Z = np.zeros(C.shape, dtype=np.complex128)  # Z的初值为0

    for j in range(max_iter):
        mask = np.abs(Z) <= 2
        B += mask
        Z[mask] = Z[mask]**2 + C[mask]

    return B.T  # 转置以保持与原始代码一致的维度


def generate_julia(c, width=800, height=800, max_iter=100):
    """
    生成Julia集数据(向量化版本)
    :param c: Julia集参数
    :param width: 图像宽度
    :param height: 图像高度
    :param max_iter: 最大迭代次数
    :return: Julia集数据
    """
    x = np.linspace(-2.0, 2.0, width)
    y = np.linspace(-2.0, 2.0, height)
    Re, Im = np.meshgrid(x, y)
    Z0 = Re + 1j * Im

    B = np.zeros(Z0.shape)  # 记录迭代次数
    Z = Z0.copy()  # 初始值为网格点

    for j in range(max_iter):
        mask = np.abs(Z) <= 2
        B += mask
        Z[mask] = Z[mask]**2 + c

    return B.T  # 转置以保持与原始代码一致的维度


def plot_fractal(data, title, filename=None, cmap='magma'):
    """
    绘制分形图像
    :param data: 分形数据
    :param title: 图像标题
    :param filename: 保存文件名
    :param cmap: 颜色映射
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(data.T, cmap=cmap, origin='lower')
    plt.title(title)
    plt.axis('off')

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.show()


if __name__ == "__main__":
    # 生成并绘制Mandelbrot集
    mandelbrot = generate_mandelbrot(max_iter=100)
    plot_fractal(mandelbrot, "Mandelbrot Set", "mandelbrot.png")

    # 生成并绘制不同c值的Julia集
    julia_c_values = [
        -0.8 + 0.156j,  # 经典Julia集
        -0.4 + 0.6j,    # 树枝状Julia集
        0.285 + 0.01j   # 复杂结构Julia集
    ]

    for i, c in enumerate(julia_c_values):
        julia = generate_julia(c, max_iter=100)
        plot_fractal(julia, f"Julia Set (c = {c:.3f})", f"julia_{i+1}.png")
