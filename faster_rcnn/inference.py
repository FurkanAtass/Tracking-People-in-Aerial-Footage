from torchvision import models, transforms
import torch
import cv2
from PIL import Image
import argparse
import numpy as np

"""
Usage:
    python faster_rcnn/inference.py \
    --model-path Trained_Models/Faster-RCNN/model_epoch_4.pth \
    --input-image-path datasets/mot-20/test/images/MOT20-04_000001.jpg \
    --output-image-path output.jpg \
    --threshold 0.8
"""

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN model inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights (.pth file)")
    parser.add_argument("--input-image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-image-path", type=str, required=True, help="Path to save the output image with bounding boxes")
    parser.add_argument("--threshold", type=float, default=0.8, help="Detection threshold (default: 0.8)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes including background (default: 2)")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Class names (background is index 0, person is index 1)
    label_list = ["", "person"]

    # Load the model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=args.num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image with OpenCV and convert to RGB
    image_bgr = cv2.imread(args.input_image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {args.input_image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract detection data
    boxes = predictions[0]['boxes'].cpu()
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu()

    # Draw bounding boxes and labels
    detection_count = 0
    for i in range(len(boxes)):
        if scores[i] > args.threshold:
            detection_count += 1
            box = boxes[i].numpy().astype(int)
            label = label_list[labels[i]]
            score = scores[i].item()

            # Draw bounding box
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            # Draw label with confidence
            text = f"{label}: {score:.2f}"
            cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(args.output_image_path, image_bgr)
    
    print(f"Saved annotated image to {args.output_image_path}")
    print(f"Detected {detection_count} objects above threshold {args.threshold}")

if __name__ == "__main__":
    main()
