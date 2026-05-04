import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# ================== 配置路径 ==================
BASE_DIR_3DGS = r"../../3dgs_outputs"  # 相对于脚本位置
BASE_DIR_DUST3R = r"../../dust3r_outputs"
DATASETS = ["405841_FRONT", "DL3DV-2", "Re10k-1"]
VARIANTS = ["dust3r_heavy", "dust3r_light", "dust3r_none"]

# 数据集名称映射
DATASET_MAPPING = {
    "dust3r_light": {"405841_FRONT": "405841"},
    "dust3r_heavy": {"405841_FRONT": "405841_FRONT"},
    "dust3r_none": {"405841_FRONT": "405841_FRONT"}
}

# ================== 解析训练日志（loss + 评估指标） ==================
def parse_train_log(log_path):
    iterations = []
    losses = []
    eval_iters = []
    eval_psnr = []
    train_psnr = []

    # 匹配进度条行，例如：
    # "Training progress: ... Loss=0.0197315, Depth Loss=0.0000000"
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

# ================== 解析 ATE RMSE JSON ==================
def parse_ate_rmse(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ate_data = {}
        for scene in data['scenes']:
            scene_name = scene['scene_rel']
            if scene_name == "405841/FRONT":
                scene_name = "405841_FRONT"
            ate_data[scene_name] = scene['ate_rmse']
        return ate_data
    except Exception as e:
        print(f"读取 {json_path} 失败: {e}")
        return {}

# ================== 收集所有数据 ==================
def collect_all_data():
    all_losses = {}   # {(dataset, variant): (iters, losses)}
    all_psnr = {}     # {(dataset, variant): (eval_iters, eval_psnr)}
    final_metrics = {} # {(dataset, variant): {'PSNR': final_psnr, 'Iteration': final_iter}}
    all_ate = {}      # {variant: {dataset: ate_rmse}}

    for variant in VARIANTS:
        ate_json = os.path.join(BASE_DIR_DUST3R, f"results_{variant}", "ate_rmse_summary.json")
        ate_data = parse_ate_rmse(ate_json)
        all_ate[variant] = ate_data

        for dataset in DATASETS:
            actual_dataset = DATASET_MAPPING.get(variant, {}).get(dataset, dataset)
            dataset_dir = os.path.join(BASE_DIR_3DGS, variant, actual_dataset)
            if not os.path.isdir(dataset_dir):
                print(f"警告: 目录不存在 {dataset_dir}，跳过")
                continue

            log_path = os.path.join(dataset_dir, "logs", "train_console.log")
            if not os.path.exists(log_path):
                print(f"警告: 找不到 {log_path}")
                continue

            iters, losses, eval_iters, eval_psnr, train_psnr = parse_train_log(log_path)

            print(f"{variant}/{dataset}: 训练点 {len(iters)} 个, 评估点 {len(eval_iters)} 个")

            all_losses[(dataset, variant)] = (iters, losses)
            if len(eval_iters) > 0:
                all_psnr[(dataset, variant)] = (eval_iters, eval_psnr)
                # 取最后一个评估点作为最终质量
                final_metrics[(dataset, variant)] = {
                    'PSNR': eval_psnr[-1],
                    'Iteration': eval_iters[-1]
                }
                print(f"  -> 最终 PSNR: {eval_psnr[-1]:.2f} dB (iter {eval_iters[-1]})")
            else:
                # 如果没有测试评估，尝试使用最后的训练 PSNR
                if train_psnr:
                    last_train_it, last_train_psnr = train_psnr[-1]
                    final_metrics[(dataset, variant)] = {
                        'PSNR': last_train_psnr,
                        'Iteration': last_train_it,
                        'Note': 'train PSNR'
                    }
                    print(f"  -> 最终 train PSNR: {last_train_psnr:.2f} dB (iter {last_train_it})")
                else:
                    print(f"  -> 未找到任何 PSNR 信息")

    return all_losses, all_psnr, final_metrics, all_ate

# ================== 绘制 Loss 收敛曲线 ==================
def plot_convergence(all_losses):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 4))
    if len(DATASETS) == 1:
        axes = [axes]

    colors = {'dust3r_heavy': 'red', 'dust3r_light': 'blue', 'dust3r_none': 'green'}
    for ax, dataset in zip(axes, DATASETS):
        y_min, y_max = float('inf'), float('-inf')
        for variant in VARIANTS:
            key = (dataset, variant)
            if key in all_losses:
                iters, losses = all_losses[key]
                if len(iters) > 0:
                    ax.plot(iters, losses, label=variant, color=colors[variant], alpha=0.8)
                    y_min = min(y_min, losses.min())
                    y_max = max(y_max, losses.max())
        # 动态设置 y 轴范围
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
    plt.close()

# ================== 绘制 PSNR 随迭代变化曲线 ==================
def plot_psnr_curves(all_psnr):
    if not all_psnr:
        print("没有 PSNR 评估点数据，跳过 PSNR 曲线图。")
        return

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5*len(DATASETS), 4))
    if len(DATASETS) == 1:
        axes = [axes]

    colors = {'dust3r_heavy': 'red', 'dust3r_light': 'blue', 'dust3r_none': 'green'}
    for ax, dataset in zip(axes, DATASETS):
        for variant in VARIANTS:
            key = (dataset, variant)
            if key in all_psnr:
                eval_iters, eval_psnr = all_psnr[key]
                ax.plot(eval_iters, eval_psnr, 'o-', label=variant, color=colors[variant], markersize=4)
        ax.set_title(f"Test PSNR: {dataset}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("psnr_curves.png", dpi=150)
    plt.close()

# ================== 绘制最终 PSNR 柱状图 ==================
def plot_final_psnr(final_metrics):
    if not final_metrics:
        print("没有最终 PSNR 数据，跳过柱状图。")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DATASETS))
    width = 0.25
    colors = {'dust3r_heavy': 'lightcoral', 'dust3r_light': 'cornflowerblue', 'dust3r_none': 'lightgreen'}

    for i, variant in enumerate(VARIANTS):
        values = []
        for dataset in DATASETS:
            key = (dataset, variant)
            val = final_metrics.get(key, {}).get('PSNR', None)
            values.append(val)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=variant, color=colors[variant])
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
    plt.close()

# ================== 绘制 ATE RMSE 比较 ==================
def plot_ate_rmse(all_ate):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DATASETS))
    width = 0.25
    colors = {'dust3r_heavy': 'lightcoral', 'dust3r_light': 'cornflowerblue', 'dust3r_none': 'lightgreen'}

    for i, variant in enumerate(VARIANTS):
        values = []
        for dataset in DATASETS:
            val = all_ate.get(variant, {}).get(dataset, 0)  # 用 0 如果没有
            values.append(val if val is not None else 0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=variant, color=colors[variant])
        for bar, v in zip(bars, values):
            if v != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("ATE RMSE")
    ax.set_title("ATE RMSE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("ate_rmse_comparison.png", dpi=150)
    plt.close()

# ================== 主程序 ==================
if __name__ == "__main__":
    loss_data, psnr_data, final_metrics, ate_data = collect_all_data()

    if loss_data:
        plot_convergence(loss_data)
    if psnr_data:
        plot_psnr_curves(psnr_data)
    if final_metrics:
        plot_final_psnr(final_metrics)
    if ate_data:
        plot_ate_rmse(ate_data)