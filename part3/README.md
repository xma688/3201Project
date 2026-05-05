# Part 3 Interface

Part 3 adds generated pseudo-views to the sparse DUSt3R scene and trains 3DGS
with confidence-aware pseudo supervision.

The code is split into:

```text
part3/
|-- apps/                        # CLI entrypoints for each pipeline step
|-- configs/                     # portable config templates
|-- scripts/                     # common shell wrappers
|-- src/part3_stack/             # manual confidence route
`-- src/part3_stack_pretrained/  # MASt3R + SEA-RAFT confidence route
```

Generated files are written to `part3/workspace/` by default and are ignored by
Git. To move outputs to another disk, set:

```bash
export PART3_WORKSPACE_ROOT=path/to/your/part3/workspace
```

## External Requirements

The default configs expect:

```text
external/DynamiCrafter/
pretrained/DynamiCrafter512_interp.ckpt
outputs/dust3r_to_colmap/dust3r_light/<scene>
```

The optional pretrained confidence route additionally expects:

```text
external/MASt3R/
external/SEA-RAFT/
pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
pretrained/Tartan480x640-M.pth
```

Edit `part3/configs/*.json` if your repositories or checkpoints live
elsewhere.

## Manual Route

Run one full manual scene:

```bash
bash part3/scripts/run_part3_pipeline.sh Re10k-1
```

Build the final raw/conf/full ablation scenes by reusing one shared pseudo-view
set:

```bash
SCENES="405841_FRONT DL3DV-2 Re10k-1" \
  bash part3/scripts/run_reuse_pseudo_ablation.sh
```

This creates:

```text
part3/workspace/hybrid_scenes/<scene>_gen_raw
part3/workspace/hybrid_scenes/<scene>_gen_conf
part3/workspace/hybrid_scenes/<scene>_gen_full
```

## Pretrained Confidence Route

```bash
SCENES="405841_FRONT DL3DV-2 Re10k-1" \
  bash part3/scripts/run_reuse_pseudo_pretrained_ablation.sh
```

This route uses MASt3R for feature confidence and SEA-RAFT for temporal
confidence while preserving the same downstream hybrid-scene and 3DGS training
interface.

## Training and Evaluation

```bash
PART3_WORKSPACE_ROOT=path/to/your/part3/workspace \
CUDA_VISIBLE_DEVICES=0 \
bash part3/scripts/train_part3_3dgs.sh \
  part3/workspace/hybrid_scenes/Re10k-1_gen_full \
  Re10k-1_gen_full \
  30000 \
  --confidence_manifest part3/workspace/runs/Re10k-1/Re10k-1_gen_full/confidence/confidence_manifest.json \
  --enable_online_confidence
```

```bash
PART3_WORKSPACE_ROOT=path/to/your/part3/workspace \
bash part3/scripts/eval_part3_3dgs.sh \
  path/to/your/part3/workspace/3dgs_outputs/Re10k-1_gen_full
```

## Useful Entry Points

- `apps/prepare_scene.py`: interpolate target poses from a sparse scene.
- `apps/generate_pseudo_views.py`: run DynamiCrafter and save pseudo-views.
- `apps/build_confidence.py`: build manual masks and confidence manifests.
- `apps/generate_confidence_pretrained.py`: build MASt3R/SEA-RAFT masks.
- `apps/build_hybrid_scene.py`: add pseudo-views to a COLMAP/3DGS scene.
- `apps/train_3dgs_confidence.py`: 3DGS training entrypoint with pseudo masks.
- `apps/compare_part3_metrics.py`: summarize baseline vs Part 3 metrics.
