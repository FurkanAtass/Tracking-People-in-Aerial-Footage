## Dataset Conversion Scripts

All conversion scripts extract only **person class** annotations and convert them to YOLO format. Test sets are **never sampled** to ensure reliable evaluation results.

### 1. VisDrone-DET Dataset

Converts VisDrone-DET dataset to YOLO format (person class only).

**Usage:**
```bash
cd datasets
python3 ../data/convert_visdrone_det_to_yolo.py [--visualize]
```

**Arguments:**
- `--visualize`: (optional) Show sample images with annotations from each split (train/valid/test)

**Expected Structure:**
```
datasets/
└── VisDrone-DET/
    ├── VisDrone2019-DET-train/
    ├── VisDrone2019-DET-val/
    └── VisDrone2019-DET-test-dev/
```

**Output:**
```
datasets/visdrone/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

**Features:**
- Combines "pedestrian" (class 1) and "people" (class 2) into single "person" class (class 0)

---

### 2. Okutama Action Dataset

Converts Okutama Action dataset to YOLO format with optional frame sampling.

**Usage:**
```bash
cd datasets
python3 ../data/convert_okutama_to_yolo.py [--sample_interval INTERVAL] [--visualize]
```

**Arguments:**
- `--sample_interval`: (optional, default: 15) Frame sampling interval. Example: 15 = take frames 1, 15, 30, 45, ...
  - **Note:** Test sets are NEVER sampled - all frames are used for reliable evaluation
- `--visualize`: (optional) Show sample images with annotations from each split

**Expected Structure (one of):**
```
datasets/
└── Okutama-Action-MOT/
    ├── TrainSetFrames/
    ├── TestSetFrames/
    └── ValidSetFrames/

OR

datasets/
├── TrainSetFrames/
└── TestSetFrames/

OR

datasets/
└── okutama/
    ├── Drone1/
    ├── Drone2/
    └── Labels/
```

**Output:**
```
datasets/okutama/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

**Features:**
- Frame sampling for train/valid sets (reduces dataset size)
- **Test sets always use ALL frames** (no sampling)

**Example:**
```bash
# Sample every 15th frame for train/valid, all frames for test
python3 ../data/convert_okutama_to_yolo.py --sample_interval 15 --visualize
```

---

### 3. MOT20 Challenge Dataset

Converts MOT20 Challenge dataset to YOLO format using detections (not ground truth).

**Usage:**
```bash
cd datasets
python3 ../data/convert_mot20_to_yolo.py [--sample_interval INTERVAL] [--visualize]
```

**Arguments:**
- `--sample_interval`: (optional, default: 15) Frame sampling interval. MOT20 uses 25 FPS, so:
  - Interval 25 = every 1 second
  - Interval 50 = every 2 seconds
  - **Note:** Test sets are NEVER sampled - all frames are used for reliable evaluation
- `--visualize`: (optional) Show sample images with annotations from each split

**Expected Structure:**
```
datasets/
└── MOT20/
    ├── train/
    │   ├── MOT20-01/
    │   ├── MOT20-02/
    │   └── ...
    ├── valid/
    │   └── MOT20-03/
    └── test/
        ├── MOT20-04/
        └── ...
```

**Output:**
```
datasets/mot-20/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

**Features:**
- Uses detections from `det/det.txt` (not ground truth)
- Frame sampling for train/valid sets (at 25 FPS)
- **Test sets always use ALL frames** (no sampling)

**Examples:**
```bash
# Sample every 15 frames (0.6 seconds) for train/valid
python3 ../data/convert_mot20_to_yolo.py --sample_interval 15 --visualize

# Sample every 25 frames (1 second) for train/valid
python3 ../data/convert_mot20_to_yolo.py --sample_interval 25 --visualize
```