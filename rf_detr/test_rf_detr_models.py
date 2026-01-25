import supervision as sv
from rfdetr import RFDETRNano
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision, MeanAverageRecallResult, RecallResult
from PIL import Image
from rfdetr.datasets.coco_eval import CocoEvaluator

def print_coco_style_ar(mar):
    # ---- ALL objects ----
    print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {mar.mAR_at_1:.5f}")
    print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {mar.mAR_at_10:.5f}")
    print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {mar.mAR_at_100:.5f}")

    # ---- SMALL ----
    if mar.small_objects is not None:
        print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {mar.small_objects.mAR_at_100:.5f}")

    # ---- MEDIUM ----
    if mar.medium_objects is not None:
        print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {mar.medium_objects.mAR_at_100:.5f}")

    # ---- LARGE ----
    if mar.large_objects is not None:
        print(f"Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {mar.large_objects.mAR_at_100:.5f}")


import supervision as sv
from rfdetr import RFDETRNano
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision, MeanAverageRecall
from PIL import Image
import os

os.chdir("/content/Tracking-People-in-Aerial-Footage/faster_rcnn")
# from engine import evaluate # This import is no longer needed

TEST_IMAGE_DIR = "/content/Tracking-People-in-Aerial-Footage/datasets/okutama/test/images"
TEST_ANNOTATION_PATH = "/content/Tracking-People-in-Aerial-Footage/datasets/okutama/okutama_test.json"
MODEL_PATH = "/content/checkpoint_best_total.pth"

ds = sv.DetectionDataset.from_coco(
    images_directory_path=TEST_IMAGE_DIR,
    annotations_path=TEST_ANNOTATION_PATH,
)

model = RFDETRNano(pretrain_weights=MODEL_PATH)
model.optimize_for_inference()

targets = []
predictions = []

for path, image, annotations in tqdm(ds):
    image = Image.open(path)
    detections = model.predict(image, threshold=0)

    targets.append(annotations)
    predictions.append(detections)

map_metric = MeanAveragePrecision()
map_result = map_metric.update(predictions, targets).compute()

mAR_metric = MeanAverageRecall()
mAR_result = mAR_metric.update(predictions, targets).compute()

print("-"*50)
print(map_result)
print_coco_style_ar(mAR_result)

