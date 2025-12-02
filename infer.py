#!/usr/bin/env python3
"""
Simple inference script for trained YOLOv8 models.

Usage examples (from project root):

  # Use small model on a video file
  python infer.py \
    --weights results/earlystop_16batch_52epoch_small_v8/weights/best.pt \
    --source path/to/video.mp4

  # Use nano model on webcam 0
  python infer.py \
    --weights results/earlystop_16batch_52epoch_nano_v8/weights/best.pt \
    --source 0

  # Use medium model on a folder of images
  python infer.py \
    --weights results/earlystop_16batch_51epoch_medium_v8/weights/best.pt \
    --source datasets/visdrone/valid/images
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 inference script")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt file), e.g. results/.../weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Inference source: image/video path, directory, or camera index (e.g. '0'). "
            "Examples: '0', 'datasets/visdrone/test/images', 'video.mp4'"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="results/inference",
        help="Base directory to save inference results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name (subfolder under project)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results window during inference (may be slow over SSH)",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as YOLO-format txt files",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detection images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("=" * 60)
    print("YOLOv8 Inference")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Source:  {args.source}")

    # Load model
    model = YOLO(str(weights_path))

    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        save=True,       # save annotated images / video
        show=args.show,  # optionally display window
        save_txt=args.save_txt,
        save_conf=True,
        save_crop=args.save_crop,
        vid_stride=1,
    )

    # results is a list of Results objects
    save_dir = results[0].save_dir if results else Path(args.project) / args.name

    print("\nInference complete.")
    print(f"Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}