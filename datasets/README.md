# Download and Process Datasets

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