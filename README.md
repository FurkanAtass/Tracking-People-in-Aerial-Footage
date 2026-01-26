# Detecting People in Aerial Footage Captured by UAVs

## ENV SETUP
Install uv from original [document](https://docs.astral.sh/uv/getting-started/installation/)
Run "uv sync" to install all the dependencies.

## DATASETS
Datasets used in this project are:
* VisDrone
* Okutama-Action
* MOT20

Only the person/pedestrian classes are used. For Okutama-Action and MOT20, datasets are sampled because the tracking datasets contain many similar consecutive images; sampling reduces repetition and keeps the dataset diverse and manageable.

Processed dataset can be directly downloaded from [here](https://drive.google.com/file/d/1c0IpnPaGJYuL-cvad469-jxBcte0YUGl/view?usp=sharing)
or manually created by the instructions in [dataset/README.md](dataset/README.md)

Data processing scripts and detailed explanations are in [data/README.md](data/README.md)

## TRAIN
For training, there are three different folders for different model formats
* Ultralytics Models (ultralytics_models)
    - YoloV8 n/s/m
    - YoloV11 n/s/m
    - YoloV26 n/s/m/p2-n
    - RT-DETR-L
* RF-DETR (rf_detr)
* Faster R-CNN (faster_rcnn)

Each contains train and test scripts and each script includes usage examples.

Trained models and their respective logs can be downloaded directly from [here](https://drive.google.com/file/d/12nFpId9_SdC9MrAVFmI8GgMySXmpkAzY/view?usp=sharing)

## TEST

* Each model folder contains its own test script. To quickly evaluate a model after training, just run the corresponding test file with the necessary arguments as explained in the test scripts. Each script contains usage examples.
* Test datasets are not sampled and contain the full set of images for evaluation.
* Test results are logged to the console and follow standard object detection metrics (mAP, AR, etc). For details on each script, refer to comments in the code.

