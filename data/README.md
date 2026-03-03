# Data directory

This directory holds the private CASTLE data boundary and model assets used to
run experiments.

## Layout

```text
data/
  castle2024/
    raw/                      Private CASTLE frames (inputs for protocols)
    raw_dataset_index.csv     Inventory for the private raw frames
  models/                     Cached model assets for detectors / methods
```

Reviewed annotations and final protocol manifests (reportable evidence) live
under `outputs/01_protocol/`.

Raw `.webp` frames are private identifiable data and must not be redistributed
as public thesis evidence.
