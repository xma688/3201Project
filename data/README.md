# Data Directory

The real project data is not committed to GitHub. Place the course datasets
under this directory before running the reproduction scripts.

Expected layout:

```text
data/
|-- 405841/FRONT/
|   |-- rgb/*.png
|   |-- calib/*.txt
|   `-- gt/*.txt
|-- DL3DV-2/
|   |-- rgb/*.png
|   |-- cameras.json
|   `-- intrinsics.json
`-- Re10k-1/
    |-- images/*.png
    |-- cameras.json
    `-- intrinsics.json
```

If your data lives elsewhere, either create symlinks here or update the paths in
the relevant commands/configs. The README examples use repo-relative paths by
default.
