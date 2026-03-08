# Data Format

Vision3D uses a **generic, ego-centric JSON + image format** that is
independent of any upstream dataset. All dataset-specific quirks are resolved
once during offline conversion so that the training pipeline only ever sees
clean, normalised data.

## On-disk directory layout

```
data/
└── <dataset_name>/          # e.g. nuscenes_v3d or dummy
    ├── train/
    │   ├── frame_000000.json
    │   ├── frame_000001.json
    │   └── ...
    ├── val/
    │   ├── frame_001000.json
    │   └── ...
    └── images/
        ├── front/
        │   ├── frame_000000.png
        │   └── ...
        ├── back/
        ├── left/
        ├── right/
        ├── front_left/
        └── front_right/
```

Each split directory contains one JSON file per frame. Images are stored
separately under `images/<camera_name>/` and are referenced by path inside
the JSON.

The `data/` root is **gitignored** – it must be populated by running a
converter script (see [NuScenes conversion](#nuscenes-conversion)) or the
dummy-data generator.

## Per-frame JSON schema

Each frame has exactly one JSON file. The schema is:

```json
{
  "frame_id": "unique_string_identifier",
  "timestamp": 1600000000.123456,
  "past_frame_ids": [
    "unique_string_identifier_minus_1",
    "unique_string_identifier_minus_2"
  ],
  "cameras": {
    "<camera_name>": {
      "image_path": "images/front/frame_000000.png",
      "intrinsics": [
        [800.0,   0.0, 400.0],
        [  0.0, 800.0, 300.0],
        [  0.0,   0.0,   1.0]
      ],
      "sensor2ego_translation": [1.5, 0.0, 1.2],
      "sensor2ego_rotation":    [1.0, 0.0, 0.0, 0.0]
    }
  },
  "annotations": [
    {
      "instance_id": "track_99",
      "class_name":  "car",
      "bbox_3d": [10.5, -2.1, 0.5, 4.5, 1.8, 1.5, 0.707, 0.707, 5.2, 0.0]
    }
  ],
  "metadata": {
    "weather": "clear",
    "occlusion": [0, 1, 0],
    "point_counts": [42, 7, 130]
  }
}
```

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | `string` | Globally unique identifier for this frame |
| `timestamp` | `float` | Unix timestamp in seconds |
| `past_frame_ids` | `string[]` | Ordered list of prior `frame_id` values (oldest last) used for temporal attention; may be empty |
| `cameras` | `object` | Dict of camera views keyed by camera name |
| `annotations` | `array` | Ground-truth 3-D bounding boxes (may be empty) |
| `metadata` | `object` | Arbitrary per-frame metadata used by `ImageFilter` |

### Camera object fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `image_path` | `string` | — | Path to the image file, **relative to `data_root`** |
| `intrinsics` | `float[][]` | `(3, 3)` | Pinhole camera matrix `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `sensor2ego_translation` | `float[]` | `(3,)` | Translation from sensor to ego frame (metres) |
| `sensor2ego_rotation` | `float[]` | `(4,)` | Unit quaternion `[w, x, y, z]` from sensor to ego frame |

> **Note:** Images must be **pre-undistorted** before conversion. The pipeline
> does not apply lens-distortion correction at runtime.

### Annotation fields

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | `string` | Unique tracking ID (consistent across frames for the same object) |
| `class_name` | `string` | Object class label, e.g. `"car"`, `"pedestrian"` |
| `bbox_3d` | `float[10]` | 10-DOF bounding box (see below) |

## Bounding box format

All bounding boxes use the **10-DOF ego-centric format**:

```
[x, y, z, w, l, h, sin(θ), cos(θ), vx, vy]
```

| Index | Parameter | Unit | Description |
|-------|-----------|------|-------------|
| 0 | `x` | m | Box centre, longitudinal (forward) axis |
| 1 | `y` | m | Box centre, lateral axis |
| 2 | `z` | m | Box centre, vertical axis |
| 3 | `w` | m | Width |
| 4 | `l` | m | Length |
| 5 | `h` | m | Height |
| 6 | `sin(θ)` | — | Sine of yaw angle |
| 7 | `cos(θ)` | — | Cosine of yaw angle |
| 8 | `vx` | m/s | Velocity along x-axis in ego frame |
| 9 | `vy` | m/s | Velocity along y-axis in ego frame |

Encoding orientation as `(sin θ, cos θ)` avoids angle-wrapping
discontinuities and keeps the representation smooth for regression losses.

## Coordinate system

- **Frame:** Ego-centric. The origin is the vehicle centre at the current
  timestamp.
- **Convention:** right-forward-up (standard for autonomous driving). Positive
  x points forward, positive y points left, positive z points up.
- **All transformations** (sensor-to-ego extrinsics, annotations) are resolved
  **offline** during dataset conversion. The training pipeline does not perform
  any coordinate-frame changes at runtime.

## Image format

| Property | Value |
|----------|-------|
| File format | PNG or JPG |
| Distortion | Pre-undistorted (no runtime correction) |
| Normalisation | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Default size | 900 × 1600 (H × W) — configurable via `image_size` |
| Tensor shape | `(C, H, W)` after loading |

## NuScenes conversion

After installing the `nuscenes` dependency group
(`uv sync --group nuscenes`), run the converter to produce the Vision3D
generic format:

```bash
python scripts/convert_nuscenes.py \
  --nuscenes-root /path/to/nuscenes \
  --output-dir    data/nuscenes_v3d \
  --version       v1.0-trainval
```

The script:
1. Reads each NuScenes sample via `nuscenes-devkit`.
2. Saves all 6 camera images as PNGs under `images/<camera>/`.
3. Converts calibration and annotations to the ego-centric format.
4. Writes one JSON per frame into `train/` or `val/`.

## Dummy dataset

For development and CI testing a synthetic dataset can be generated without
any real sensor data:

```bash
python scripts/generate_dummy_dataset.py \
  --output-dir data/dummy \
  --num-frames 50
```

Generated frames contain random RGB images and valid-but-random JSON metadata,
making them suitable for smoke-testing the full pipeline without downloading
any real dataset.
