# Tracking People in Aerial Footage

This project contains scripts for converting aerial person detection datasets to YOLO format, combining multiple datasets, and visualizing results.

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
datasets/visdrone-det/
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
- Skips conversion if dataset already exists
- Includes visualization option

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
- Handles multiple dataset structures automatically
- Skips conversion if dataset already exists

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
└── MOT20-Challenge/
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
datasets/mot20/
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
- Skips conversion if dataset already exists

**Examples:**
```bash
# Sample every 15 frames (0.6 seconds) for train/valid
python3 ../data/convert_mot20_to_yolo.py --sample_interval 15 --visualize

# Sample every 25 frames (1 second) for train/valid
python3 ../data/convert_mot20_to_yolo.py --sample_interval 25 --visualize
```

---

## Dataset Combination Script

### Combine Multiple YOLO Datasets

Combines multiple YOLO format datasets into a single dataset for training.

**Usage:**
```bash
cd datasets
python3 ../data/combine_yolo_datasets.py --datasets DATASET1 DATASET2 ... [--output OUTPUT]
```

**Arguments:**
- `--datasets`: List of dataset directory paths to combine (required)
- `--output`: (optional, default: combined) Output directory name

**Important Notes:**
- Only combines **train** and **valid** splits (test sets are excluded)
- Automatically handles filename conflicts by adding dataset prefixes
- Creates a single `dataset.yaml` file for the combined dataset

**Example:**
```bash
# Combine mot20, visdrone-det, and okutama datasets
python3 ../data/combine_yolo_datasets.py --datasets mot20 visdrone-det okutama --output all_datasets
```

**Output:**
```
datasets/all_datasets/
├── train/
│   ├── images/  (combined from all datasets)
│   └── labels/  (combined from all datasets)
├── valid/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

**Training with Combined Dataset:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='datasets/all_datasets/dataset.yaml', epochs=100)
```

---

## Visualization Scripts

### 1. Visualize YOLO Dataset

General visualization script for any YOLO format dataset.

**Usage:**
```bash
cd datasets
python3 ../data/visualize_yolo_dataset.py DATASET_ROOT [--num_samples N]
```

**Arguments:**
- `DATASET_ROOT`: Root directory of the YOLO dataset (required)
- `--num_samples`: (optional, default: 1) Number of random samples to show from each split

**Example:**
```bash
# Visualize mot20 dataset with 1 sample per split
python3 ../data/visualize_yolo_dataset.py mot20

# Visualize with 3 samples per split
python3 ../data/visualize_yolo_dataset.py visdrone-det --num_samples 3
```

**Features:**
- Randomly samples images from train/valid/test splits
- Displays bounding boxes with class names (if available in dataset.yaml)
- Color-coded boxes by class
- Works with any YOLO format dataset

---

### 2. Explore MOT20 Dataset

Explores and analyzes the MOT20 Challenge dataset structure.

**Usage:**
```bash
cd datasets
python3 ../data/explore_mot20.py [--samples_only] [--no_visualize]
```

**Arguments:**
- `--samples_only`: (optional) Skip analysis and show only visualization samples
- `--no_visualize`: (optional) Skip visualization (only print statistics)

**Example:**
```bash
# Full analysis with visualization
python3 ../data/explore_mot20.py

# Only show samples
python3 ../data/explore_mot20.py --samples_only

# Only analysis, no visualization
python3 ../data/explore_mot20.py --no_visualize
```

**Features:**
- Analyzes dataset structure (sequences, annotations, tracks, classes)
- Shows video metadata (FPS, resolution, duration)
- Displays sample frames with bounding boxes
- Provides detailed statistics by split

---

## Complete Workflow Example

### 1. Convert All Datasets

```bash
cd datasets

# Convert VisDrone-DET
python3 ../data/convert_visdrone_det_to_yolo.py --visualize

# Convert Okutama (with frame sampling)
python3 ../data/convert_okutama_to_yolo.py --sample_interval 15 --visualize

# Convert MOT20 (with frame sampling)
python3 ../data/convert_mot20_to_yolo.py --sample_interval 15 --visualize
```

### 2. Combine Datasets

```bash
python3 ../data/combine_yolo_datasets.py --datasets mot20 visdrone-det okutama --output combined
```

### 3. Visualize Combined Dataset

```bash
python3 ../data/visualize_yolo_dataset.py combined --num_samples 2
```

### 4. Train Model

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='datasets/combined/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## Important Notes

### Frame Sampling

- **Train/Valid sets:** Can be sampled to reduce dataset size (e.g., `--sample_interval 15`)
- **Test sets:** Always use ALL frames - never sampled for reliable evaluation

### Dataset Structure

All scripts expect datasets in the `datasets/` directory and create output in the same location.

### Skipping Re-conversion

All conversion scripts check if the dataset has already been converted and skip re-processing if found. To re-convert, delete the output directory first.

### Class Mapping

- All datasets map to single "person" class (class 0)
- VisDrone-DET: pedestrian (1) + people (2) → person (0)
- Okutama: Person → person (0)
- MOT20: person detections → person (0)

---

## Script Summary

| Script | Purpose | Frame Sampling | Test Set Sampling |
|--------|---------|----------------|-------------------|
| `convert_visdrone_det_to_yolo.py` | Convert VisDrone-DET | No | N/A (no sampling) |
| `convert_okutama_to_yolo.py` | Convert Okutama | Yes (train/valid) | **No** (all frames) |
| `convert_mot20_to_yolo.py` | Convert MOT20 | Yes (train/valid) | **No** (all frames) |
| `combine_yolo_datasets.py` | Combine datasets | N/A | Excludes test sets |
| `visualize_yolo_dataset.py` | Visualize any YOLO dataset | N/A | N/A |
| `explore_mot20.py` | Explore MOT20 structure | N/A | N/A |

---

## Requirements

- Python 3.11+
- Pillow (PIL)
- matplotlib
- tqdm
- ultralytics (for training)

Install dependencies:
```bash
pip install pillow matplotlib tqdm ultralytics
```
