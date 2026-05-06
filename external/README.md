# External Repositories

Third-party repositories that are not vendored in this submission should be
cloned here when needed.

Recommended layout:

```text
external/
|-- DynamiCrafter/
|-- MASt3R/
`-- SEA-RAFT/
```

Recommended clone commands:

```bash
git clone https://github.com/Doubiiu/DynamiCrafter external/DynamiCrafter
git clone https://github.com/naver/mast3r.git external/MASt3R
git clone https://github.com/princeton-vl/SEA-RAFT.git external/SEA-RAFT
```

The Part 3 configs use these paths for the optional pretrained confidence
route:

```json
"dynami_crafter_root": "external/DynamiCrafter",
"repo_paths": {
  "mast3r": "external/MASt3R",
  "sea_raft": "external/SEA-RAFT"
}
```

Each external project keeps its own license.  Our pretrained Part 3 confidence
route imports MASt3R and SEA-RAFT official modules through wrappers, but in the
tested setup they run inside the existing `dust3r` environment; no separate
MASt3R/SEA-RAFT conda environment is required unless your local Python
environment is missing their dependencies.
