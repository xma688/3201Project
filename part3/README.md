# Part 3 Course-Project Interface

这层现在不再走多端口服务，而是改成更适合作业的组织：

- `src/part3_stack/*.py` 放可复用函数
- `apps/*.py` 放按步骤执行的 CLI 入口
- `scripts/*.sh` 放最常用的 shell 包装

这样既能直接 `import`，也能一条命令一条命令地跑，不需要先起任何端口。

## 复用的现有代码

- `subsample_p2_frames.py`
- `run_dust3r_inference.py`
- `dust3r_to_colmap.py`
- `dust3r_to_colmap/`
- `gaussian-splatting/`
- `part3/DynamiCrafter/`
- `part3/DynamiCrafter512_interp.ckpt`

## 当前目录结构

```text
part3/
├── apps/
│   ├── prepare_scene.py
│   ├── generate_pseudo_views.py
│   ├── build_confidence.py
│   ├── build_hybrid_scene.py
│   ├── evaluate_part3.py
│   ├── run_part3_pipeline.py
│   ├── run_retention_experiment.py
│   ├── compare_part3_metrics.py
│   └── train_3dgs_confidence.py
├── configs/
│   └── project.json
├── scripts/
│   ├── run_part3_pipeline.sh
│   ├── train_part3_3dgs.sh
│   ├── eval_part3_3dgs.sh
│   ├── run_retention_experiment.sh
│   └── compare_sparse_vs_generated.sh
├── src/part3_stack/
│   ├── config.py
│   ├── geometry.py
│   ├── dynamicrafter_adapter.py
│   ├── confidence.py
│   ├── hybrid.py
│   └── pipeline.py
└── workspace/
    ├── runs/
    ├── pseudo_views/
    ├── confidence_maps/
    ├── hybrid_scenes/
    └── 3dgs_outputs/
```

## 对应 PDF Part 3 的四块要求

### 1. Pseudo-View Generation

- CLI: `apps/prepare_scene.py`
- CLI: `apps/generate_pseudo_views.py`
- 复用 `dust3r_to_colmap/`、`DynamiCrafter/`

### 2. Hybrid Optimization

- CLI: `apps/build_hybrid_scene.py`
- 复用 `gaussian-splatting/`
- 把 pseudo-view 追加回 COLMAP/3DGS 场景

### 3. Confidence Fusion

- CLI: `apps/build_confidence.py`
- 训练入口: `apps/train_3dgs_confidence.py`
- confidence mask 写进 RGBA 的 alpha 通道
- 当前 confidence 已拆成 `C_vis / C_reproj / C_feat / C_temp / C_patch`
- manifest 会记录 `clip_score / patch_keep_ratio / mean_feature_confidence / reprojection_valid_ratio`

### 4. Consistency-Aware Optimization

- `clip-level consistency` 用于 pseudo clip 采样重加权
- `patch pruning` 用于 pseudo supervision 的像素/patch 降权
- 训练时可开启 batch-local `online confidence refresh`

## 最推荐的可复用接口

如果你是想“把函数变成可复用接口”，最应该直接复用的是：

- `part3_stack.pipeline.prepare_scene`
- `part3_stack.pipeline.generate_pseudo_views`
- `part3_stack.pipeline.build_confidence`
- `part3_stack.pipeline.build_hybrid`
- `part3_stack.pipeline.evaluate_part3`
- `part3_stack.pipeline.run_part3_pipeline`

它们都在 [pipeline.py](/path/to/your/CVproj/part3/src/part3_stack/pipeline.py)。

一个最小调用例子：

```python
from part3_stack.pipeline import run_part3_pipeline

result = run_part3_pipeline(
    config_path="/path/to/your/CVproj/part3/configs/project.json",
    scene="Re10k-1",
)
print(result["hybrid"]["hybrid_scene_dir"])
```

## 直接运行方式

### 一步一步跑

```bash
python3 part3/apps/prepare_scene.py --scene Re10k-1
python3 part3/apps/generate_pseudo_views.py --trajectory-manifest /path/to/your/trajectory_manifest.json
python3 part3/apps/build_confidence.py --pseudo-manifest /path/to/your/pseudo_manifest.json
python3 part3/apps/build_hybrid_scene.py \
  --trajectory-manifest /path/to/your/trajectory_manifest.json \
  --pseudo-manifest /path/to/your/pseudo_manifest.json \
  --confidence-manifest /path/to/your/confidence_manifest.json
```

如果要显式控制消融项，可以这样跑 confidence：

```bash
python3 part3/apps/build_confidence.py \
  --pseudo-manifest /path/to/your/pseudo_manifest.json \
  --feature-backend dust3r \
  --feature-sigma 0.35 \
  --enable-clip-consistency \
  --enable-patch-pruning \
  --patch-size 16 \
  --patch-threshold 0.25
```

### 一条命令跑完整 pipeline

```bash
bash part3/scripts/run_part3_pipeline.sh Re10k-1
```

或者：

```bash
python3 part3/apps/run_part3_pipeline.py \
  --scene Re10k-1 \
  --prompt "novel view interpolation, geometrically consistent scene, stable camera motion"
```

这里已经不需要任何端口。

### 只测试 pseudo-view 保留率

如果只想决定 `num_intermediate_views` 和 `keep_ratio`，可以先跑到
`build_confidence.py` 为止，不构建 hybrid scene，也不训练 3DGS：

```bash
bash part3/scripts/run_retention_experiment.sh Re10k-1
```

这个入口默认使用 `configs/project_gen_full.json`、`max_pairs=0`、
`num_intermediate_views=4,6,8`、`keep_ratio=1.0`，并开启
`clip consistency + patch pruning`。统计规则是：

```text
clip_score >= 0.55
mean_raw_confidence >= 0.10
patch_keep_ratio >= 0.20
reprojection_valid_ratio >= 0.70
```

输出会写到：

```text
part3/workspace/runs/Re10k-1/retention_experiment/retention_summary.json
part3/workspace/runs/Re10k-1/retention_experiment/retention_summary.csv
part3/workspace/runs/Re10k-1/retention_experiment/retention_summary.md
```

如果只想重新统计已有 manifest，而不重新生成 pseudo views：

```bash
python3 part3/apps/run_retention_experiment.py --summary-only
```

默认还会在选定的 `N` 上派生一个 `keep_ratio=0.8` 验证 run；这个验证
复用 `keep_ratio=1.0` 已生成的 frames，不重新跑 DynamiCrafter。

## 做 PDF 要求里的最终对比

先跑 baseline sparse-only：

```bash
bash scripts/train_3dgs.sh Re10k-1 PlanB
bash scripts/eval_3dgs.sh Re10k-1 PlanB
```

再跑 Part 3 sparse+generated：

```bash
bash part3/scripts/train_part3_3dgs.sh /path/to/your/hybrid_scene Re10k-1_part3
bash part3/scripts/eval_part3_3dgs.sh /path/to/your/CVproj/part3/workspace/3dgs_outputs/Re10k-1_part3
```

训练时如果要使用 online confidence 和诊断图，可以把 confidence manifest 传进去：

```bash
bash part3/scripts/train_part3_3dgs.sh /path/to/your/hybrid_scene Re10k-1_part3 30000 \
  --confidence_manifest /path/to/your/confidence_manifest.json \
  --enable_online_confidence \
  --confidence_refresh_interval 200 \
  --diagnostics_interval 1000 \
  --diagnostics_debug_views 4
```

最后做指标对比：

```bash
python3 part3/apps/compare_part3_metrics.py \
  --baseline /path/to/your/CVproj/3dgs_outputs/colmap_sparse_3dgs/Re10k-1 \
  --part3 /path/to/your/CVproj/part3/workspace/3dgs_outputs/Re10k-1_part3 \
  --baseline-scene-dir /path/to/your/baseline_scene \
  --part3-scene-dir /path/to/your/hybrid_scene \
  --baseline-eval-meta /path/to/your/eval_meta_or_cameras_json \
  --part3-confidence-manifest /path/to/your/confidence_manifest.json \
  --output /path/to/your/part3_eval_summary.json
```

## 目前这版的边界

这版更偏课程作业友好，而不是部署友好。

已经直接落地的部分：

- 没有端口依赖
- CLI 命名更贴近步骤本身
- `src` 里保留可复用函数接口
- 复用 Part 2 / DUSt3R / COLMAP / 3DGS 路径
- DynamiCrafter 插帧接入
- 轨迹插值
- pseudo-view 写回 COLMAP 场景
- confidence-aware 3DGS 训练入口
- `C_reproj / C_feat / C_temp / C_patch` 组件化 confidence
- clip-level consistency 和 patch pruning
- online confidence refresh
- TensorBoard + PNG 训练诊断
- PSNR / SSIM / LPIPS + ATE / RPE + pseudo consistency 汇总

当前仍然是简化版的部分：

- `C_feat` 的 backend 接口叫 `dust3r`，但当前实现是轻量 dense descriptor 版本，后续可以无缝替换成真正 DUSt3R token 特征
- 深度图目前用于 `depth_drift_vs_coarse` 诊断，不默认加入新的 depth loss
