# Generative Sparse-View 3D Reconstruction

Course project for AIAA 3201 Project 4. The project studies the same scenes under three settings:

1. Dense posed reconstruction: compare COLMAP initialization with VGGT/foundation-model initialization for 3D Gaussian Splatting.
2. Sparse unposed reconstruction: subsample frames, hide camera poses, infer geometry with DUSt3R, export to COLMAP/3DGS, and evaluate ATE plus rendering metrics.
3. Generative sparse-view enhancement: generate pseudo-views with DynamiCrafter, attach interpolated/refined poses, build confidence masks, and train 3DGS with masked pseudo-view supervision.

The report sources and figures are under `anlysis_script_and_results/`. Large data, checkpoints, generated pseudo-views, DUSt3R outputs, and 3DGS training outputs are intentionally excluded from git by `.gitignore`.

## Repository Layout

```text
.
|-- scripts/                         # Part 1 / 3DGS helper shell scripts
|-- part3/
|   |-- apps/                        # Part 3 CLI entrypoints
|   |-- src/part3_stack/             # Manual confidence pipeline
|   |-- src/part3_stack_pretrained/  # MASt3R + SEA-RAFT confidence route
|   |-- scripts/                     # Part 3 shell wrappers
|   `-- configs/                     # Portable Part 3 config templates
|-- build_pairs.py                   # Sparse DUSt3R pair graph builder
|-- subsample_p2_frames.py           # Part 2 frame subsampling
|-- run_dust3r_inference.py          # DUSt3R inference and global alignment
|-- dust3r_to_colmap.py              # DUSt3R result to COLMAP/3DGS scene
|-- eval_ate_rmse.py                 # Sim(3)-aligned ATE RMSE
|-- gaussian-splatting/              # 3DGS codebase
|-- dust3r/                          # DUSt3R codebase
`-- vggt/                            # VGGT codebase
```

## Environments

The code uses separate environments because 3DGS CUDA extensions, DUSt3R/VGGT, and DynamiCrafter often need different dependency sets.

```bash
# 3DGS training/evaluation, CUDA 12.4
conda env create -f environment-3dgs-cu124.yml

# DUSt3R inference and Part 2 conversion utilities
conda env create -f environment-dust3r.yml

# DynamiCrafter pseudo-view generation
conda env create -f environment-dynamicrafter.yml
```

External system tools:

- `COLMAP` must be available as `colmap`.
- CUDA toolkit / compiler support is needed to build the 3DGS rasterization extensions.
- For VGGT, install the local package from `vggt/` and, if using `demo_colmap.py`, also install `vggt/requirements_demo.txt`.

Weights are not committed. Put them at paths you control, for example:

```text
/path/to/your/CVproj/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
/path/to/your/CVproj/part3/DynamiCrafter512_interp.ckpt
/path/to/your/CVproj/pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
/path/to/your/CVproj/pretrained/Tartan480x640-M.pth
```

For Part 3, the default config is `part3/configs/project.json`. Paths in that file are relative to the repository root. If your data or checkpoints live elsewhere, edit the config or pass `--config /path/to/your/project.json`.

## Data Layout

Expected mandatory data layout:

```text
data/
|-- 405841/FRONT/rgb/*.png
|-- 405841/FRONT/calib/*.txt
|-- 405841/FRONT/gt/*.txt
|-- DL3DV-2/rgb/*.png
|-- DL3DV-2/cameras.json
|-- DL3DV-2/intrinsics.json
|-- Re10k-1/images/*.png
|-- Re10k-1/cameras.json
`-- Re10k-1/intrinsics.json
```

Run commands from the repository root unless noted.

## Part 1: Dense COLMAP/VGGT + 3DGS

Plan A uses COLMAP to initialize 3DGS.

```bash
conda activate cvproj-3dgs

bash scripts/run_colmap.sh re10k sequential 1
bash scripts/inspect_colmap.sh re10k
bash scripts/organize_3dgs_scene.sh Re10k-1 0
```

Place the final COLMAP scene in one of the locations searched by `scripts/train_3dgs.sh`, for example:

```text
scenes_3dgs/PlanA/Re10k-1/{images,sparse/0}
```

Plan B uses the official VGGT COLMAP demo on the dense frames, then trains the same 3DGS code.

```bash
conda activate vggt
cd vggt
python demo_colmap.py \
  --scene_dir /path/to/your/CVproj/scenes_3dgs/PlanB/Re10k-1 \
  --weights /path/to/your/VGGT-1B/model.safetensors
cd ..
```

Train and evaluate either plan:

```bash
conda activate cvproj-3dgs
bash scripts/train_3dgs.sh Re10k-1 PlanA
bash scripts/eval_3dgs.sh Re10k-1 PlanA

bash scripts/train_3dgs.sh Re10k-1 PlanB
bash scripts/eval_3dgs.sh Re10k-1 PlanB
```

Repeat for `DL3DV-2` and `405841_FRONT`. The report compares convergence and final PSNR/SSIM/LPIPS between COLMAP and VGGT initialization.

## Part 2: Sparse Unposed DUSt3R

Subsample the dense data according to the project requirement: Waymo 1/10, DL3DV/Re10k 1/30. Ground-truth poses are saved only under `eval_meta/` for ATE evaluation.

```bash
conda activate dust3r

python subsample_p2_frames.py \
  --data_root data \
  --out_root data_p2_sparse \
  --save_eval_meta

python build_pairs.py \
  --root data_p2_sparse \
  --scene-graph swin-2 \
  --symmetrize
```

Run DUSt3R inference and global alignment:

```bash
python run_dust3r_inference.py \
  --root data_p2_sparse \
  --output-root outputs/dust3r/results_dust3r_light \
  --dust3r-repo dust3r \
  --weights /path/to/your/CVproj/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --scene-graph swin-2 \
  --min-conf-thr 3.0
```

Evaluate pose error:

```bash
python eval_ate_rmse.py \
  --results-root outputs/dust3r/results_dust3r_light \
  --data-root data_p2_sparse \
  --output outputs/dust3r/ate_rmse_summary.json
```

Export DUSt3R predictions to COLMAP/3DGS format:

```bash
python dust3r_to_colmap.py \
  --results-root outputs/dust3r/results_dust3r_light \
  --output-root outputs/dust3r_to_colmap/dust3r_light \
  --overwrite
```

Train 3DGS on a converted sparse scene:

```bash
conda activate cvproj-3dgs
cd gaussian-splatting
python train.py \
  -s ../outputs/dust3r_to_colmap/dust3r_light/Re10k-1 \
  -m ../outputs/3dgs/dust3r_light/Re10k-1 \
  --eval \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 1000 3000 7000 15000 30000
cd ..
```

## Part 3: Generated Pseudo-Views + Confidence

Part 3 consumes the Part 2 scene exported at:

```text
outputs/dust3r_to_colmap/dust3r_light/<scene>
```

Put the DynamiCrafter repo at `part3/DynamiCrafter/` and the interpolation checkpoint at `part3/DynamiCrafter512_interp.ckpt`, or update `part3/configs/project.json`.

Run the full manual route for one scene:

```bash
conda activate dust3r
bash part3/scripts/run_part3_pipeline.sh Re10k-1
```

For the final ablations, reuse one generated pseudo-view set and build `raw`, `conf`, and `full` variants:

```bash
SCENES="405841_FRONT DL3DV-2 Re10k-1" \
  bash part3/scripts/run_reuse_pseudo_ablation.sh
```

Train a generated-view hybrid scene:

```bash
conda activate cvproj-3dgs
bash part3/scripts/train_part3_3dgs.sh \
  part3/workspace/hybrid_scenes/Re10k-1_gen_full \
  Re10k-1_gen_full \
  30000 \
  --confidence_manifest part3/workspace/runs/Re10k-1/Re10k-1_gen_full/confidence/confidence_manifest.json \
  --enable_online_confidence

bash part3/scripts/eval_part3_3dgs.sh \
  part3/workspace/3dgs_outputs/Re10k-1_gen_full
```

Optional pretrained confidence route:

1. Put MASt3R and SEA-RAFT repos under `external/` or edit `part3/configs/project_pretrained_full.json`.
2. Put checkpoints under `pretrained/` or edit the same config.
3. Run:

```bash
SCENES="405841_FRONT DL3DV-2 Re10k-1" \
  bash part3/scripts/run_reuse_pseudo_pretrained_ablation.sh
```

## Analysis and Report Figures

The report figures are generated from saved logs:

```bash
python anlysis_script_and_results/part1/compare_plans.py
python anlysis_script_and_results/part2/analyze_part2.py
python part3/apps/compare_part3_metrics.py \
  --baseline /path/to/your/baseline_model \
  --part3 /path/to/your/part3_model \
  --output /path/to/your/part3_eval_summary.json
```

The final report summarizes:

- Part 1: dense COLMAP gives stronger 3DGS initialization than the current VGGT-to-COLMAP export.
- Part 2: DUSt3R provides usable sparse unposed poses, evaluated with Sim(3)-aligned ATE RMSE and 3DGS rendering metrics.
- Part 3: generated pseudo-views help most on weak sparse baselines, while confidence masks and consistency pruning keep pseudo supervision controllable.
