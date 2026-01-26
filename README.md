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
    Unzip datasets and put in MOT20 folder.
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
    cd datasets
    python3 ../data/convert_okutama_to_yolo.py
    python3 ../data/yolo_to_coco.py --dataset_root okutama
    ```
    Optionally delete original dataset files.

* Combine datasets (only train and valid data) by using command:
```bash
python ../data/combine_yolo_datasets.py --datasets mot-20 okutama visdrone
```



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
