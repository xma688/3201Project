# Outputs Directory

This directory is for generated artifacts and is ignored by Git.

Typical layout after running the project:

```text
outputs/
|-- colmap/
|-- dust3r/
|-- dust3r_to_colmap/
`-- 3dgs/
```

Part 3 also writes generated pseudo-views and hybrid 3DGS scenes to
`part3/workspace/` by default. To move that workspace to a larger disk, set:

```bash
export PART3_WORKSPACE_ROOT=/path/to/your/part3/workspace
```
