from ultralytics import YOLO, RTDETR
import argparse
from utils import load_model

"""
python train.py \
    --model_type yolov8n \
    --model_path yolov8n.pt \
    --is_p2 \
    --dataset_path datasets/combined/dataset.yaml \
    --epochs 300 \
    --imgsz 640 \
    --batch 16 \
    --experiment_name people-detection \
    --patience 100
"""

def main():
    parser = argparse.ArgumentParser(description='Train a YOLO model on a dataset')
    parser.add_argument('--model_type', type=str, default='yolov8n', help='Model type')
    parser.add_argument('--model_path', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--is_p2', action='store_true', help='Is the model a P2 model')
    parser.add_argument('--dataset_path', type=str, default='datasets/combined/dataset.yaml', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--experiment_name', type=str, default='people-detection', help='Experiment name')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')

    args = parser.parse_args()
    model = load_model(args.model_path, args.model_type, args.is_p2)

    model.train(
        data=args.dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.experiment_name,
        patience=args.patience,
        save=True,
        val=True,
        plots=True,
    )

if __name__ == "__main__":
    main()