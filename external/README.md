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

The Part 3 configs use these paths for the optional pretrained confidence
route:

```json
"dynami_crafter_root": "external/DynamiCrafter",
"repo_paths": {
  "mast3r": "external/MASt3R",
  "sea_raft": "external/SEA-RAFT"
}
```

Each external project keeps its own license and installation instructions.
