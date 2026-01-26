from ultralytics import YOLO, RTDETR
import argparse

"""
Using a pretrained model:
    python ultralytics_models/train.py \
        --model_type yolov8n \
        --model_path yolov8n.pt \
        --dataset_path datasets/combined/dataset.yaml \
        --epochs 1 \
        --imgsz 640 \
        --batch 16 \
        --experiment_name people-detection \
        --patience 100

Using a P2 model:
    python ultralytics_models/train.py \
        --model_type yolov8n \
        --model_path ultralytics_models/yolo_configs/yolov26n-p2.yaml \
        --is_p2 \
        --dataset_path datasets/combined/dataset.yaml \
        --epochs 1 \
        --imgsz 640 \
        --batch 16 \
        --experiment_name people-detection \
        --patience 100
"""

def main():
    parser = argparse.ArgumentParser(description='Train a YOLO or RTDETR model on a dataset')
    parser.add_argument('--model_type', type=str, default='yolov8n', help='Model type or yaml file path')
    parser.add_argument('--model_path', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--is_p2', action='store_true', help='Is the model a P2 model')
    parser.add_argument('--dataset_path', type=str, default='datasets/combined/dataset.yaml', help='Dataset path')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--experiment_name', type=str, default='people-detection', help='Experiment name')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')

    args = parser.parse_args()
    if args.model_type.startswith('rtdetr-'):
        model = RTDETR(args.model_path)
    else:
        model = YOLO(args.model_path)

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