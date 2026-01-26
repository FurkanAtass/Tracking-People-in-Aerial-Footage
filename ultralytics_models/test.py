from ultralytics import YOLO, RTDETR
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Test a YOLO or RTDETR model on a dataset')
    parser.add_argument('--model_type', type=str, default='yolov8n', help='Model type or yaml file path')
    parser.add_argument('--model_path', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--dataset_path', type=str, default='datasets/combined/dataset.yaml', help='Dataset path')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path to save metrics')
    args = parser.parse_args()

    if args.model_type.startswith('rtdetr-'):
        model = RTDETR(args.model_path)
    else:
        model = YOLO(args.model_path)

    results = model.val(
        data=args.dataset_path,
        split='test',
    )

    box_metrics = results.box_metrics

    metrics_dict = {
    'map 50:95': float(box_metrics.ap),
    'map 50': float(box_metrics.mp),
    'mr': float(box_metrics.mr),
    'map50': float(box_metrics.map50),
    'map75': float(box_metrics.map75),
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics_dict, f)
        print(f"Metrics saved to {args.output}")

if __name__ == '__main__':
    main()