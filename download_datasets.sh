#!/bin/bash

# Download VisDrone2019-MOT-train.zip
gdown "https://drive.google.com/uc?id=1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh" \
      -O VisDrone2019-MOT-train.zip
unzip VisDrone2019-MOT-train.zip
rm VisDrone2019-MOT-train.zip

# Download TrainSetFrames.zip
wget -O TrainSetFrames.zip \
"https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/APTyV-f01XLnJ0WFpZSBLOE?e=2&preview=TrainSetFrames.zip&rlkey=7u7131amaul29amyr4jbnnu03&dl=1"
rm TestSetVideos.zip TrainSetVideos.zip

unzip TrainSetFrames.zip