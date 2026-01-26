from ultralytics import YOLO, RTDETR
import argparse

"""
Usage:
    python ultralytics_models/inference.py \
    --model-path Trained_Models/Ultralytics/yolov26-nano-p2/people-detection2/weights/best.pt \
    --model-type yolov26n \
    --input-image-path datasets/visdrone/test/images/0000006_00159_d_0000001.jpg \
    --output-image-path output.jpg
"""

def main():
    parser = argparse.ArgumentParser(description="Ultralytics model inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights (.pt file)")
    parser.add_argument("--model-type", type=str, required=True, help="Type of the model (e.g. yolo, rtdetr)")
    parser.add_argument("--input-image-path", type=str, required=True, help="Path to the source image")
    parser.add_argument("--output-image-path", type=str, required=True, help="Path to the output image")


    args = parser.parse_args()
    if args.model_type.startswith("rtdetr"):
        model = RTDETR(args.model_path)
    else:
        model = YOLO(args.model_path)


    results = model.predict(args.input_image_path)
    for result in results:
        result.save(args.output_image_path)

if __name__ == "__main__":
    main()