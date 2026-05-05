# Data Directory

The real project data is not committed to GitHub. Download the course datasets
from one of the project data mirrors, then place or symlink the extracted scenes
under this directory before running the reproduction scripts.

Dataset mirrors:

- Baidu Netdisk: <https://pan.baidu.com/s/1Sa18zCeYiYA2gWAllo11dg?pwd=p3bm#list/path=%2F>
- Google Drive: <https://drive.google.com/drive/folders/1euG7pnbFowljVWoNLcbCmil81IVsIEfM>

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
