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

If you store weights elsewhere, replace these values in commands or
`part3/configs/*.json` with `path/to/your/...`.
