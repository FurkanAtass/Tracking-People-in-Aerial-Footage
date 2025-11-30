#!/bin/bash

# Simple script to download and process VisDrone datasets
mkdir -p datasets
cd datasets

# Download train set
echo "Downloading train set..."
gdown "https://drive.google.com/uc?id=1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh" -O VisDrone2019-MOT-train.zip
unzip -q VisDrone2019-MOT-train.zip
rm VisDrone2019-MOT-train.zip
python3 ../data/convert_visdrone_to_yolo.py

# Download val set
echo "Downloading val set..."
gdown "https://drive.google.com/uc?id=1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu" -O VisDrone2019-MOT-val.zip
unzip -q VisDrone2019-MOT-val.zip
rm VisDrone2019-MOT-val.zip
python3 ../data/convert_visdrone_to_yolo.py

# Download test set
echo "Downloading test set..."
gdown "https://drive.google.com/uc?id=14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu" -O VisDrone2019-MOT-test-dev.zip
unzip -q VisDrone2019-MOT-test-dev.zip
rm VisDrone2019-MOT-test-dev.zip
python3 ../data/convert_visdrone_to_yolo.py

# Download Okutama Action dataset
echo "Downloading Okutama Action dataset..."
wget -O OkutamaAction.zip \
"https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE?e=2&preview=TrainSetFrames.zip&rlkey=7u7131amaul29amyr4jbnnu03&dl=1"

unzip OkutamaAction.zip

rm TestSetVideos.zip TrainSetVideos.zip FinalModels.zip Sample.zip Okutama.zip

unzip TrainSetFrames.zip

# Convert to YOLO format
python3 ../data/convert_okutama_to_yolo.py

# Combine both datasets into one
echo ""
echo "Combining VisDrone and Okutama datasets..."
python3 ../data/combine_datasets.py

echo ""
echo "Done!"
