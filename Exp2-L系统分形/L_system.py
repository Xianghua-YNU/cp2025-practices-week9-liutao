"""
项目2: L-System分形生成与绘图
实现L-System字符串生成与绘图功能。
"""
import matplotlib.pyplot as plt
import math

def apply_rules(axiom, rules, iterations):
    """
    生成L-System字符串
    :param axiom: 初始字符串（如"F"或"0"）
    :param rules: 替换规则字典（如{"F": "F+F--F+F"}）
    :param iterations: 迭代次数
    :return: 最终生成的字符串
    """
    current_str = axiom
    for _ in range(iterations):
        new_str = []
        for char in current_str:
            new_str.append(rules.get(char, char))  # 应用规则或保留原字符
        current_str = ''.join(new_str)
    return current_str

def draw_l_system(instructions, angle, step, start_pos=(0,0), start_angle=0, savefile=None):
    """
    根据L-System指令绘图
    :param instructions: 指令字符串（如"F+F--F+F"）
    :param angle: 转向角度（度）
    :param step: 步长
    :param start_pos: 起始坐标
    :param start_angle: 初始角度（0表示向右，90表示向上）
    :param savefile: 保存文件名（可选）
    """
    x, y = start_pos
    current_angle = start_angle
    stack = []
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')

    for cmd in instructions:
        if cmd in ['F', '0', '1']:  # 处理绘制指令
            rad = math.radians(current_angle)
            dx = step * math.cos(rad)
            dy = step * math.sin(rad)
            new_x = x + dx
            new_y = y + dy
            ax.plot([x, new_x], [y, new_y], color='black', linewidth=1)
            x, y = new_x, new_y
        elif cmd == '+':
            current_angle += angle
        elif cmd == '-':
            current_angle -= angle
        elif cmd == '[':
            stack.append((x, y, current_angle))
            current_angle += angle  # 左转
        elif cmd == ']':
            if stack:
                x, y, current_angle = stack.pop()
                current_angle -= angle  # 右转

    if savefile:
        plt.savefig(savefile, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # 1. 科赫曲线
    axiom = "F"
    rules = {"F": "F+F--F+F"}
    iterations = 3
    angle = 60
    step = 5
    instr = apply_rules(axiom, rules, iterations)
    draw_l_system(instr, angle, step, savefile="l_system_koch.png")

    # 2. 分形二叉树（调整初始角度为90度）
    axiom = "0"
    rules = {"1": "11", "0": "1[0]0"}
    iterations = 5
    angle = 45
    step = 5
    instr = apply_rules(axiom, rules, iterations)
    draw_l_system(instr, angle, step, start_angle=90, savefile="fractal_tree.png")
