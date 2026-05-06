# Pretrained Checkpoints

Model checkpoints are intentionally excluded from GitHub. Download them from
the official model pages and place them here, or update the configs to point to
your own checkpoint directory.

Recommended filenames:

```text
pretrained/
|-- DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
|-- DynamiCrafter512_interp.ckpt
|-- MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
`-- Tartan480x640-M.pth
```

Download sources used by our Part 3 pipeline:

- DUSt3R weights: https://github.com/naver/dust3r
- DynamiCrafter interpolation checkpoint: https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/blob/main/model.ckpt
  - The downloaded file is named `model.ckpt`; rename or symlink it to
    `DynamiCrafter512_interp.ckpt`.
- MASt3R code and weights: https://github.com/naver/mast3r.git
- SEA-RAFT code: https://github.com/princeton-vl/SEA-RAFT.git
- SEA-RAFT weights: https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW

For the optional pretrained mask route, clone the MASt3R and SEA-RAFT code into
`external/MASt3R` and `external/SEA-RAFT`.  Our wrappers import their official
model modules directly.  In the tested setup, the existing `dust3r` environment
can run both backends, so no separate MASt3R/SEA-RAFT conda environment is
required unless your local environment is missing dependencies.

VGGT is not listed as a required local file here because the VGGT scripts
download weights automatically on first inference.  A local VGGT checkpoint can
still be passed explicitly with `--weights path/to/your/VGGT-1B/model.safetensors`.

If you store weights elsewhere, replace these values in commands or
`part3/configs/*.json` with `path/to/your/...`.
