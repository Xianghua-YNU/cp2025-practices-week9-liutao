import numpy as np
import matplotlib.pyplot as plt

def get_fern_params():
    """
    返回巴恩斯利蕨的IFS参数
    """
    return [
        [0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01],   # 茎干
        [0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85],  # 小叶
        [0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07], # 左大叶
        [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07]  # 右大叶
    ]

def get_tree_params():
    """
    返回概率树的IFS参数
    """
    return [
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1],          # 树干
        [0.42, -0.42, 0.42, 0.42, 0.0, 0.2, 0.45],    # 左分支
        [0.42, 0.42, -0.42, 0.42, 0.0, 0.2, 0.45]     # 右分支
    ]


def apply_transform(point, params):
    """
    应用单个变换到点
    """
    x, y = point
    a, b, c, d, e, f = params[:6]
    x_new = a * x + b * y + e
    y_new = c * x + d * y + f
    return (x_new, y_new)

def run_ifs(ifs_params, num_points=100000, num_skip=100):
    """
    运行IFS迭代生成点集
    """
    probs = [param[6] for param in ifs_params]
    n_trans = len(ifs_params)
    total_iter = num_skip + num_points
    trans_indices = np.random.choice(n_trans, size=total_iter, p=probs)
    
    x, y = 0.0, 0.0
    points = []
    
    for i, idx in enumerate(trans_indices):
        params = ifs_params[idx]
        x, y = apply_transform((x, y), params)
        if i >= num_skip:
            points.append([x, y])
    
    return np.array(points)

def plot_ifs(points, title="IFS Fractal"):
    """
    绘制IFS分形
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.6, c='green', marker='.')
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 生成并绘制巴恩斯利蕨
    fern_params = get_fern_params()
    fern_points = run_ifs(fern_params)
    plot_ifs(fern_points, "Barnsley Fern")
    
    # 生成并绘制概率树
    tree_params = get_tree_params()
    tree_points = run_ifs(tree_params)
    plot_ifs(tree_points, "Probability Tree")
