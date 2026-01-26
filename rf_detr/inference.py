from rfdetr import RFDETRNano
import supervision as sv
from PIL import Image
import argparse
import numpy as np

"""
Usage:
    python rf_detr/inference.py \
    --model-path Trained_Models/RF-DETR/checkpoint_best_total.pth \
    --input-image-path datasets/mot-20/test/images/MOT20-04_000001.jpg \
    --output-image-path output.jpg
"""

def main():
    parser = argparse.ArgumentParser(description="RF-DETR model inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights (.pth file)")
    parser.add_argument("--input-image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-image-path", type=str, required=True, help="Path to save the output image with bounding boxes")
    parser.add_argument("--class-name", type=str, default="person", help="Class name for labels (default: person)")

    args = parser.parse_args()

    # Load model
    model = RFDETRNano(pretrain_weights=args.model_path)
    model.optimize_for_inference()

    # Load image
    image = Image.open(args.input_image_path)

    # Perform inference
    detections = model.predict(image, threshold=0.5)

    # Calculate optimal text scale and line thickness
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    # Create annotators
    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale
    )
    
    detections_labels = [
        f"{args.class_name} {confidence:.2f}"
        for confidence
        in detections.confidence
    ]
    
    annotated_image = bbox_annotator.annotate(image.copy(), detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, detections_labels)

    annotated_image.save(args.output_image_path)
    
    print(f"Saved annotated image to {args.output_image_path}")
    print(f"Detected {len(detections)} objects")

if __name__ == "__main__":
    main()
