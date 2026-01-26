# Tracking People in Aerial Footage

This project contains scripts for converting aerial person detection datasets to YOLO format, combining multiple datasets, and visualizing results.

## DATASET

Datasets used in this project are:
* VisDrone Object Detection[https://github.com/VisDrone/VisDrone-Dataset]
    Download: 
    * trainset(1.44GB)
    * valset(0.07GB)
    * testset-dev(0.28GB)
    Unzip each one and put inside "datasets/VisDrone-DET"
    Run 
    ```bash
    cd datasets
    python3 ../data/convert_visdrone_det_to_yolo.py
    python3 ../data/yolo_to_coco.py --dataset_root visdrone
    ```
    Optionally delete original dataset files.

* MOT20[https://motchallenge.net/data/MOT20/]
    Download all data (5GB).
    Unzip dataset.
    It only contains train and test sets. For validation set, create "valid" folder and move "train/MOT20-03" to valid folder.
    Run
    ```bash
    cd datasets
    python3 ../data/convert_mot20_to_yolo.py
    python3 ../data/yolo_to_coco.py --dataset_root mot-20
    ```
    Optionally delete original dataset files.

* Okutama-Action[http://okutama-action.org/]
    Download Training set (1280x720 frames & labels) (5.3GB)
    Test set (1280x720 frames & labels) (1.5GB)
    Unzip each one and put inside "datasets/Okutama-Action-MOT"
    It only contains train and test sets. For validation set, run the following code to split train set into train and validation with predefined folders and can be changed in the script:
    ``` bash
    python3 data/okutama_validation.py
    ```
    Run
    ```bash
    python3 ../data/convert_okutama_to_yolo.py
    python3 ../data/yolo_to_coco.py --dataset_root okutama
    ```
    Optionally delete original dataset files.


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

### Generate COCO Labels

Converts YOLO format labels to COCO format JSON files. Some models (e.g., DETR-based models) require COCO formatted annotations. This script converts the labels while keeping the original YOLO dataset structure intact, allowing you to use the same dataset for both YOLO and COCO-based training.

**Usage:**
```bash
cd datasets
python3 ../data/yolo_to_coco.py --dataset_root DATASET_PATH [--output_dir OUTPUT_DIR]
```

**Arguments:**
- `--dataset_root`: Path to dataset root directory (should contain train/images, train/labels, valid/images, valid/labels, and optionally test/images, test/labels)
- `--output_dir`: (optional, default: dataset_root) Directory to save JSON files

**Important Notes:**
- Converts YOLO format (normalized x_center, y_center, width, height) to COCO format (absolute x_min, y_min, width, height)
- Reads class names from `dataset.yaml` if available
- Generates `train.json` and `val.json` files (and `test.json` if test split exists)
- Original YOLO format files remain unchanged - only JSON annotations are created

**Example:**
```bash
# Convert combined dataset to COCO format
python3 ../data/yolo_to_coco.py --dataset_root combined

# Convert specific dataset and save JSONs to custom location
python3 ../data/yolo_to_coco.py --dataset_root visdrone --output_dir visdrone/coco_labels
```

**Output:**
```
datasets/combined/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── dataset.yaml
├── train.json  (COCO format annotations)
└── val.json    (COCO format annotations)
```

---


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
