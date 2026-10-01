# 观察降温速率、初始温度、迭代次数对解质量的影响
# 说明：支持自由组合 alpha, T, iterations 进行多维度对比实验
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 1. 生成槽位：5x6 网格，共 30 个位置
gx, gy = np.meshgrid(np.arange(5), np.arange(6))
slots = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)

num_slots = 30  # 槽位数
num_modules = 20  # 模块数
lambda1 = 1  # 惩罚系数

# 2. 生成模块之间的连接，简化版“网表”
edges = []
for i in range(num_modules):
    for j in range(i + 1, num_modules):
        if np.random.rand() < 0.15:
            edges.append((i, j))
if not edges:
    edges = [(0, 1), (2, 3), (4, 5)]


# 3. 添加罚函数
def density_penalty(perm):
    row_counts = np.zeros(6)
    for module in range(num_modules):
        slot_idx = perm[module]
        row = int(slots[slot_idx][1])  # y坐标就是行号
        row_counts[row] += 1
    avg = num_modules / 6
    return np.sum((row_counts - avg) ** 2)


# 4. 成本函数：所有连接线长之和，用曼哈顿距离（也就是L1范数诱导的距离）
def cost(perm):
    total = 0.0
    for i, j in edges:
        xi, yi = slots[perm[i]]
        xj, yj = slots[perm[j]]
        total += abs(xi - xj) + abs(yi - yj)
    penalty = lambda1 * density_penalty(perm)
    return total + penalty


# 5. 将模拟退火过程封装成函数（支持自由传入三个核心参数）
def run_sa(alpha, T, iterations):
    """
    运行单次模拟退火实验
    Parameters:
        alpha (float): 降温速率
        T (float): 初始温度
        iterations (int): 最大迭代次数
    Returns:
        tuple: (best, initial_cost, best_cost, history, record_step)
    """
    current = np.random.permutation(num_slots)
    current_cost = cost(current)
    initial_cost = current_cost
    best = current.copy()
    best_cost = current_cost

    curr_T = T
    history = []
    record_step = max(1, iterations // 200)  #可改成自适应版本，iterations//a1

    for k in range(iterations):
        i, j = np.random.choice(num_slots, 2, replace=False)
        new = current.copy()
        new[i], new[j] = new[j], new[i]
        new_cost = cost(new)

        # 【修复】防止温度过低导致除零溢出
        delta = new_cost - current_cost
        if delta < 0:
            accept = True
        else:
            exponent = -delta / curr_T if curr_T > 1e-300 else -700.0
            accept = np.random.rand() < np.exp(exponent)

        if accept:
            current = new
            current_cost = new_cost
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        curr_T *= alpha

        if k % record_step == 0:
            history.append(best_cost)

    return best, initial_cost, best_cost, history, record_step


# 6.================= 在这里自由定义你要对比的参数组合 =================
# 每一组代表一次独立实验，可随意增删改查
param_configs = [
    {'alpha': 0.95, 'T': 10.0, 'iterations': 20000},
    {'alpha': 0.99, 'T': 10.0, 'iterations': 20000},
    {'alpha': 0.999, 'T': 10.0, 'iterations': 20000},
    #  还可以改变T或迭代次数：
    # {'alpha': 0.99, 'T': 5.0,  'iterations': 20000},
    # {'alpha': 0.99, 'T': 20.0, 'iterations': 20000},
    # {'alpha': 0.99, 'T': 10.0, 'iterations': 5000},
]

colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
results = {}
labels = []

for idx, cfg in enumerate(param_configs):
    # 生成唯一标签用于图例和打印
    label = f"α={cfg['alpha']}, T={cfg['T']}, Iter={cfg['iterations']}"
    labels.append(label)

    best, initial_cost, best_cost, history, rec_step = run_sa(**cfg)
    results[label] = {
        'best': best,
        'initial_cost': initial_cost,
        'best_cost': best_cost,
        'history': history,
        'record_step': rec_step,
        'config': cfg
    }
    print(f"[{label}] 初始={initial_cost:.2f}, 最终={best_cost:.2f}, "
          f"改进={(initial_cost - best_cost) / initial_cost * 100:.1f}%")

# 7.布局对比图
fig, axes = plt.subplots(1, 3, figsize=(21, 6 ))
axes = np.atleast_2d(axes).flatten()

for idx, label in enumerate(labels):
    ax = axes[idx]
    best = results[label]['best']
    best_cost = results[label]['best_cost']
    color = colors[idx % len(colors)]

    for i, j in edges:
        xi, yi = slots[best[i]]
        xj, yj = slots[best[j]]
        ax.plot([xi, xj], [yi, yj], 'k-', alpha=0.25)

    x = slots[best][:, 0]
    y = slots[best][:, 1]
    ax.scatter(x, y, s=220, c='skyblue', edgecolors='k', zorder=3)
    for midx, (xi, yi) in enumerate(slots[best]):
        ax.text(xi, yi, str(midx), ha='center', va='center', fontsize=8, zorder=4)

    ax.invert_yaxis()  #翻转一下y轴
    ax.set_title(f'{label}\nFinal Cost={best_cost:.1f}', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


plt.suptitle('SA Placement: Multi-Parameter Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('placement_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. 曲线收敛图
plt.figure(figsize=(12, 6))
for idx, label in enumerate(labels):
    history = results[label]['history']
    color = colors[idx % len(colors)]
    plt.plot(history, label=label, color=color, linewidth=1.5)

base_record_step = results[labels[0]]['record_step']
plt.xlabel(f'Iteration (×{base_record_step})')
plt.ylabel('Cost')
plt.title('SA Convergence: Multi-Parameter Comparison')
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=150)
plt.show()
