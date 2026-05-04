import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# ================== 配置路径 ==================
BASE_DIR = r"outputs\3dgs"           # 你的根目录
DATASETS = ["405841_FRONT", "DL3DV-2", "Re10k-1"]
PLANS = ["PlanA", "PlanB"]           # 子文件夹名称

# ================== 解析训练日志（loss + 评估指标） ==================
def parse_train_log(log_path):
    iterations = []
    losses = []
    eval_iters = []
    eval_psnr = []
    train_psnr = []

    # 匹配进度条行，例如：
    # "Training progress:  79%|...| 23660/30000 [27:05<..., Loss=0.0197315, Depth Loss=0.0000000]"
    # 注意：Loss= 出现在 Depth Loss= 之前，我们只捕获第一个 Loss= 后的数字
    loss_pattern = re.compile(
        r"(\d+)/\d+.*?\sLoss=([0-9.]+)", re.I
    )

    test_pattern = re.compile(
        r"\[ITER\s+(\d+)\]\s+Evaluating test:\s+L1\s+[0-9.]+\s+PSNR\s+([0-9.]+)", re.I
    )
    train_pattern = re.compile(
        r"\[ITER\s+(\d+)\]\s+Evaluating train:\s+L1\s+[0-9.]+\s+PSNR\s+([0-9.]+)", re.I
    )

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for match in loss_pattern.finditer(content):
                it = int(match.group(1))
                loss = float(match.group(2))
                iterations.append(it)
                losses.append(loss)

            for match in test_pattern.finditer(content):
                eval_iters.append(int(match.group(1)))
                eval_psnr.append(float(match.group(2)))

            for match in train_pattern.finditer(content):
                train_psnr.append((int(match.group(1)), float(match.group(2))))

    except Exception as e:
        print(f"读取 {log_path} 失败: {e}")

    return (np.array(iterations), np.array(losses),
            np.array(eval_iters), np.array(eval_psnr),
            train_psnr)

# ================== 收集所有数据 ==================
def collect_all_data():
    all_losses = {}   # {(dataset, plan): (iters, losses)}
    all_psnr = {}     # {(dataset, plan): (eval_iters, eval_psnr)}
    final_metrics = {} # {(dataset, plan): {'PSNR': final_psnr, 'L1': final_l1}}

    for dataset in DATASETS:
        for plan in PLANS:
            plan_dir = os.path.join(BASE_DIR, plan, dataset)
            if not os.path.isdir(plan_dir):
                print(f"警告: 目录不存在 {plan_dir}，跳过")
                continue

            log_path = os.path.join(plan_dir, "logs", "train_console.log")
            if not os.path.exists(log_path):
                print(f"警告: 找不到 {log_path}")
                continue

            iters, losses, eval_iters, eval_psnr, train_psnr = parse_train_log(log_path)

            print(f"{dataset}/{plan}: 训练点 {len(iters)} 个, 评估点 {len(eval_iters)} 个")

            all_losses[(dataset, plan)] = (iters, losses)
            if len(eval_iters) > 0:
                all_psnr[(dataset, plan)] = (eval_iters, eval_psnr)
                # 取最后一个评估点作为最终质量
                final_metrics[(dataset, plan)] = {
                    'PSNR': eval_psnr[-1],
                    'Iteration': eval_iters[-1]
                }
                print(f"  -> 最终 PSNR: {eval_psnr[-1]:.2f} dB (iter {eval_iters[-1]})")
            else:
                # 如果没有测试评估，尝试使用最后的训练 PSNR（不推荐，仅备用）
                if train_psnr:
                    last_train_it, last_train_psnr = train_psnr[-1]
                    final_metrics[(dataset, plan)] = {
                        'PSNR': last_train_psnr,
                        'Iteration': last_train_it,
                        'Note': 'train PSNR'
                    }
                    print(f"  -> 最终 train PSNR: {last_train_psnr:.2f} dB (iter {last_train_it})")
                else:
                    print(f"  -> 未找到任何 PSNR 信息")

    return all_losses, all_psnr, final_metrics

# ================== 绘制 Loss 收敛曲线 ==================
def plot_convergence(all_losses):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 4))
    if len(DATASETS) == 1:
        axes = [axes]

    colors = {'PlanA': 'blue', 'PlanB': 'red'}
    for ax, dataset in zip(axes, DATASETS):
        y_min, y_max = float('inf'), float('-inf')
        for plan in PLANS:
            key = (dataset, plan)
            if key in all_losses:
                iters, losses = all_losses[key]
                if len(iters) > 0:
                    ax.plot(iters, losses, label=plan, color=colors[plan], alpha=0.8)
                    y_min = min(y_min, losses.min())
                    y_max = max(y_max, losses.max())
        # 动态设置 y 轴范围，让曲线展开（添加 5% 的上下边距）
        if y_min < float('inf') and y_max > float('-inf'):
            margin = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f"Training Loss: {dataset}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150)
    plt.show()

# ================== 绘制 PSNR 随迭代变化曲线 ==================
def plot_psnr_curves(all_psnr):
    if not all_psnr:
        print("没有 PSNR 评估点数据，跳过 PSNR 曲线图。")
        return

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 4))
    if len(DATASETS) == 1:
        axes = [axes]

    colors = {'PlanA': 'blue', 'PlanB': 'red'}
    for ax, dataset in zip(axes, DATASETS):
        for plan in PLANS:
            key = (dataset, plan)
            if key in all_psnr:
                eval_iters, eval_psnr = all_psnr[key]
                ax.plot(eval_iters, eval_psnr, 'o-', label=plan, color=colors[plan], markersize=4)
        ax.set_title(f"Test PSNR: {dataset}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("psnr_curves.png", dpi=150)
    plt.show()

# ================== 绘制最终 PSNR 柱状图 ==================
def plot_final_psnr(final_metrics):
    if not final_metrics:
        print("没有最终 PSNR 数据，跳过柱状图。")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DATASETS))
    width = 0.35
    colors = {'PlanA': 'cornflowerblue', 'PlanB': 'lightcoral'}

    for i, plan in enumerate(PLANS):
        values = []
        for dataset in DATASETS:
            key = (dataset, plan)
            val = final_metrics.get(key, {}).get('PSNR', None)
            values.append(val)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=plan, color=colors[plan])
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Final Test PSNR Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("final_psnr_comparison.png", dpi=150)
    plt.show()

# ================== 主程序 ==================
if __name__ == "__main__":
    loss_data, psnr_data, final_metrics = collect_all_data()

    if loss_data:
        plot_convergence(loss_data)
    if psnr_data:
        plot_psnr_curves(psnr_data)
    if final_metrics:
        plot_final_psnr(final_metrics)