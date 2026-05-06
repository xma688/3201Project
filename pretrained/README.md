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
- MASt3R weights: https://github.com/naver/mast3r
- SEA-RAFT weights: https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW

VGGT is not listed as a required local file here because the VGGT scripts
download weights automatically on first inference.  A local VGGT checkpoint can
still be passed explicitly with `--weights path/to/your/VGGT-1B/model.safetensors`.

If you store weights elsewhere, replace these values in commands or
`part3/configs/*.json` with `path/to/your/...`.
