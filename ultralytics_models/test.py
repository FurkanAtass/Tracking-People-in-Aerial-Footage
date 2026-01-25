#!/usr/bin/env python3
"""
Test a pretrained model on a YOLO format test dataset and extract comprehensive metrics.

Usage:
    python test_model.py \
        --model-path path/to/model.pt \
        --model-type yolov8n \
        --dataset-path datasets/visdrone-det/dataset.yaml \
        --output-dir results/evaluation
"""

"""
python test_model_visdrone.py \
        --model-path Experiments/yolov11-nano/people-detection/weights/best.pt \
        --model-type yolo11n \
        --dataset-path datasets/visdrone/dataset.yaml \
        --output-dir results/visdrone-p2
"""
import argparse
import json
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import RTDETR, YOLO
from utils import load_model

try:
    import yaml
except ImportError:
    yaml = None


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_yaml_config(yaml_path: Path) -> Dict:
    """Load YAML configuration file."""
    if yaml:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f) or {}
    else:
        # Simple YAML parser for basic configs (fallback if PyYAML not available)
        config = {}
        with open(yaml_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            in_names = False
            names = {}
            for line in lines:
                original_line = line
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'names':
                        in_names = True
                        continue
                    elif in_names:
                        # Check if this is an indented line (part of names section)
                        if original_line.startswith(' ') or original_line.startswith('\t'):
                            try:
                                class_id = int(key)
                                names[class_id] = value
                            except ValueError:
                                pass
                        else:
                            # End of names section
                            in_names = False
                            if names:
                                config['names'] = names
                            # Process this line as regular config
                            if key and value:
                                config[key] = value
                    else:
                        config[key] = value
                elif in_names and not line:
                    in_names = False
                    if names:
                        config['names'] = names
            if in_names and names:
                config['names'] = names
        return config


def compute_mr_lamr_fppi(model, dataset_path: Path, conf_thresholds: List[float], iou_threshold: float = 0.5) -> Dict:
    """
    Compute Miss Rate (MR), Log-Average Miss Rate (LAMR), and FPPI.
    
    Args:
        model: YOLO model instance
        dataset_path: Path to dataset YAML file
        conf_thresholds: List of confidence thresholds to evaluate
        iou_threshold: IoU threshold for matching detections
    
    Returns:
        Dictionary with MR, LAMR, and FPPI values
    """
    # Load dataset config
    dataset_config = load_yaml_config(dataset_path)
    dataset_root = Path(dataset_path).parent / dataset_config.get('path', '')
    test_images_dir = dataset_root / dataset_config.get('test', 'test/images')
    
    if not test_images_dir.exists():
        # Try alternative paths
        test_images_dir = dataset_root / 'test' / 'images'
        if not test_images_dir.exists():
            print(f"Warning: Could not find test images directory. Skipping MR/LAMR/FPPI computation.")
            return {'miss_rate': None, 'lamr': None, 'fppi': None}
    
    # Get all test images
    image_files = sorted(list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png')))
    if not image_files:
        print(f"Warning: No test images found. Skipping MR/LAMR/FPPI computation.")
        return {'miss_rate': None, 'lamr': None, 'fppi': None}
    
    # Load ground truth labels
    labels_dir = dataset_root / 'test' / 'labels'
    if not labels_dir.exists():
        labels_dir = Path(dataset_config.get('test', 'test/images')).parent / 'labels'
        if not labels_dir.exists():
            print(f"Warning: Could not find test labels directory. Skipping MR/LAMR/FPPI computation.")
            return {'miss_rate': None, 'lamr': None, 'fppi': None}
    
    all_fppi = []
    all_mr = []
    
    # Evaluate at each confidence threshold
    for conf_thresh in conf_thresholds:
        total_fp = 0
        total_fn = 0
        total_tp = 0
        total_gt = 0
        
        for img_path in image_files:
            # Load ground truth
            label_path = labels_dir / (img_path.stem + '.txt')
            gt_boxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # YOLO format: class x_center y_center width height (normalized)
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert to xyxy format (absolute coordinates)
                            from PIL import Image
                            with Image.open(img_path) as img:
                                img_w, img_h = img.size
                            
                            x1 = (x_center - width / 2) * img_w
                            y1 = (y_center - height / 2) * img_h
                            x2 = (x_center + width / 2) * img_w
                            y2 = (y_center + height / 2) * img_h
                            
                            gt_boxes.append(np.array([x1, y1, x2, y2]))
            
            total_gt += len(gt_boxes)
            
            # Get predictions
            results = model.predict(str(img_path), conf=conf_thresh, verbose=False, imgsz=640)
            if len(results) == 0:
                total_fn += len(gt_boxes)
                continue
            
            pred_boxes = []
            pred_scores = []
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    for box, score in zip(boxes, scores):
                        pred_boxes.append(box)
                        pred_scores.append(score)
            
            # Match predictions with ground truth
            matched_gt = set()
            matched_pred = set()
            
            # Sort predictions by confidence (descending)
            pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
            
            for pred_idx in pred_indices:
                if pred_idx in matched_pred:
                    continue
                
                pred_box = pred_boxes[pred_idx]
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    # True positive
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                    total_tp += 1
                else:
                    # False positive
                    total_fp += 1
            
            # False negatives (unmatched ground truth)
            total_fn += len(gt_boxes) - len(matched_gt)
        
        # Compute metrics at this confidence threshold
        num_images = len(image_files)
        fppi = total_fp / num_images if num_images > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        mr = 1.0 - recall  # Miss Rate
        
        all_fppi.append(fppi)
        all_mr.append(mr)
    
    # Compute average MR (at default confidence, typically 0.25 or 0.5)
    avg_mr = np.mean(all_mr) if all_mr else None
    
    # Compute LAMR (Log-Average Miss Rate)
    # LAMR is computed as the average of log(MR) at 9 FPPI points: [0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000]
    target_fppi = [0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000]
    
    # Interpolate MR values at target FPPI points
    if len(all_fppi) > 1 and len(all_mr) > 1:
        # Sort by FPPI
        sorted_indices = np.argsort(all_fppi)
        sorted_fppi = np.array(all_fppi)[sorted_indices]
        sorted_mr = np.array(all_mr)[sorted_indices]
        
        # Interpolate MR at target FPPI points
        interpolated_mr = []
        for target_fp in target_fppi:
            if target_fp <= sorted_fppi[0]:
                mr_val = sorted_mr[0]
            elif target_fp >= sorted_fppi[-1]:
                mr_val = sorted_mr[-1]
            else:
                # Linear interpolation
                idx = np.searchsorted(sorted_fppi, target_fp)
                if idx == 0:
                    mr_val = sorted_mr[0]
                else:
                    # Linear interpolation
                    f1 = sorted_fppi[idx - 1]
                    f2 = sorted_fppi[idx]
                    m1 = sorted_mr[idx - 1]
                    m2 = sorted_mr[idx]
                    if f2 - f1 > 0:
                        mr_val = m1 + (m2 - m1) * (target_fp - f1) / (f2 - f1)
                    else:
                        mr_val = m1
            interpolated_mr.append(max(1e-10, mr_val))  # Avoid log(0)
        
        # Compute LAMR as average of log(MR) at target FPPI points
        lamr = np.mean([np.log10(mr) for mr in interpolated_mr]) if interpolated_mr else None
    else:
        lamr = None
    
    # FPPI at default confidence (0.5)
    default_conf_idx = conf_thresholds.index(0.5) if 0.5 in conf_thresholds else len(conf_thresholds) // 2
    fppi_at_default = all_fppi[default_conf_idx] if default_conf_idx < len(all_fppi) else None
    
    return {
        'miss_rate': float(avg_mr) if avg_mr is not None else None,
        'lamr': float(lamr) if lamr is not None else None,
        'fppi': float(fppi_at_default) if fppi_at_default is not None else None
    }


def extract_metrics(results, model_path: Path, dataset_path: Path, model_type: str, is_p2: bool, skip_mr_metrics: bool = False) -> Dict:
    """
    Extract all metrics from validation results.
    
    Args:
        results: Validation results from model.val()
        model_path: Path to model file
        dataset_path: Path to dataset YAML file
        model_type: Type of model (yolov8n, yolo11n, rtdetr-l)
    
    Returns:
        Dictionary with all metrics
    """
    metrics_dict = results.results_dict if hasattr(results, 'results_dict') else {}
    
    # Debug: Print available keys to help diagnose missing metrics
    print(f"\nDebug: Available keys in results_dict: {list(metrics_dict.keys()) if metrics_dict else 'None'}")
    if hasattr(results, 'box'):
        print(f"Debug: results.box attributes: {[attr for attr in dir(results.box) if not attr.startswith('_')]}")
        if hasattr(results.box, 'maps'):
            print(f"Debug: results.box.maps: {results.box.maps}")
    
    # Extract standard metrics - try multiple possible key formats
    # IMPORTANT: Use explicit None checks, not 'or', because 0.0 is a valid metric value
    precision = metrics_dict.get('metrics/precision(B)', None)
    if precision is None:
        precision = metrics_dict.get('precision', None)
    if precision is None and hasattr(results, 'box'):
        precision = getattr(results.box, 'p', None)
    
    recall = metrics_dict.get('metrics/recall(B)', None)
    if recall is None:
        recall = metrics_dict.get('recall', None)
    if recall is None and hasattr(results, 'box'):
        recall = getattr(results.box, 'r', None)
    
    map50 = metrics_dict.get('metrics/mAP50(B)', None)
    if map50 is None:
        map50 = metrics_dict.get('map50', None)
    if map50 is None and hasattr(results, 'box'):
        map50 = getattr(results.box, 'map50', None)
    
    map50_95 = metrics_dict.get('metrics/mAP50-95(B)', None)
    if map50_95 is None:
        map50_95 = metrics_dict.get('map50-95', None)
    if map50_95 is None and hasattr(results, 'box'):
        map50_95 = getattr(results.box, 'map', None)
    
    map75 = metrics_dict.get('metrics/mAP75(B)', None)
    if map75 is None:
        map75 = metrics_dict.get('map75', None)
    if map75 is None and hasattr(results, 'box'):
        map75 = getattr(results.box, 'map75', None)
    
    # Extract size-based AP metrics - try multiple possible key formats
    # These metrics are only available if the dataset has objects of different sizes
    # IMPORTANT: Use explicit None checks, not 'or', because 0.0 is a valid metric value
    ap_small = metrics_dict.get('metrics/mAP50-95(B)/small', None)
    if ap_small is None:
        ap_small = metrics_dict.get('metrics/mAP50-95(B)/s', None)
    if ap_small is None:
        ap_small = metrics_dict.get('map50-95/small', None)
    if ap_small is None:
        ap_small = metrics_dict.get('map50-95/s', None)
    if ap_small is None:
        ap_small = metrics_dict.get('metrics/mAP50-95(B)/small(B)', None)
    
    ap_medium = metrics_dict.get('metrics/mAP50-95(B)/medium', None)
    if ap_medium is None:
        ap_medium = metrics_dict.get('metrics/mAP50-95(B)/m', None)
    if ap_medium is None:
        ap_medium = metrics_dict.get('map50-95/medium', None)
    if ap_medium is None:
        ap_medium = metrics_dict.get('map50-95/m', None)
    if ap_medium is None:
        ap_medium = metrics_dict.get('metrics/mAP50-95(B)/medium(B)', None)
    
    ap_large = metrics_dict.get('metrics/mAP50-95(B)/large', None)
    if ap_large is None:
        ap_large = metrics_dict.get('metrics/mAP50-95(B)/l', None)
    if ap_large is None:
        ap_large = metrics_dict.get('map50-95/large', None)
    if ap_large is None:
        ap_large = metrics_dict.get('map50-95/l', None)
    if ap_large is None:
        ap_large = metrics_dict.get('metrics/mAP50-95(B)/large(B)', None)
    
    # Try accessing via results.box if available
    if hasattr(results, 'box'):
        box = results.box
        # Try maps attribute (array of [small, medium, large])
        if ap_small is None and hasattr(box, 'maps'):
            maps = box.maps
            if isinstance(maps, (list, tuple, np.ndarray)) and len(maps) >= 3:
                ap_small = float(maps[0]) if maps[0] is not None and not np.isnan(maps[0]) else None
                ap_medium = float(maps[1]) if maps[1] is not None and not np.isnan(maps[1]) else None
                ap_large = float(maps[2]) if maps[2] is not None and not np.isnan(maps[2]) else None
        
        # Try individual attributes
        if ap_small is None:
            ap_small = getattr(box, 'maps_s', None) or getattr(box, 'map_s', None) or getattr(box, 'small', None)
        if ap_medium is None:
            ap_medium = getattr(box, 'maps_m', None) or getattr(box, 'map_m', None) or getattr(box, 'medium', None)
        if ap_large is None:
            ap_large = getattr(box, 'maps_l', None) or getattr(box, 'map_l', None) or getattr(box, 'large', None)
    
    # Calculate F1 score
    f1 = None
    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
    
    # Recall@0.50 (same as recall at IoU=0.5, which is typically what's reported)
    recall_at_50 = recall
    
    # Compute MR, LAMR, FPPI (optional, can be slow)
    if skip_mr_metrics:
        print("\nSkipping MR, LAMR, and FPPI computation (--skip-mr-metrics flag set)")
        mr_metrics = {'miss_rate': None, 'lamr': None, 'fppi': None}
    else:
        print("\nComputing MR, LAMR, and FPPI metrics...")
        print("Note: This may take a while depending on dataset size...")
        
        try:
            conf_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            
            # Load model for MR/LAMR/FPPI computation
            model = load_model(model_path, model_type, is_p2)
            mr_metrics = compute_mr_lamr_fppi(model, dataset_path, conf_thresholds, iou_threshold=0.5)
            
            if mr_metrics['miss_rate'] is None and mr_metrics['lamr'] is None and mr_metrics['fppi'] is None:
                print("Warning: MR/LAMR/FPPI computation returned None values. Using approximation.")
                # Use simple approximation: MR = 1 - Recall
                mr_metrics = {
                    'miss_rate': float(1.0 - recall) if recall is not None else None,
                    'lamr': None,
                    'fppi': None
                }
        except Exception as e:
            print(f"Warning: Could not compute MR/LAMR/FPPI: {e}")
            import traceback
            traceback.print_exc()
            # Use simple approximation: MR = 1 - Recall
            mr_metrics = {
                'miss_rate': float(1.0 - recall) if recall is not None else None,
                'lamr': None,
                'fppi': None
            }
    
    return {
        'precision': float(precision) if precision is not None else None,
        'recall': float(recall) if recall is not None else None,
        'f1_score': float(f1) if f1 is not None else None,
        'map_50': float(map50) if map50 is not None else None,
        'map_50_95': float(map50_95) if map50_95 is not None else None,
        'ap_75': float(map75) if map75 is not None else None,
        'ap_small': float(ap_small) if ap_small is not None else None,
        'ap_medium': float(ap_medium) if ap_medium is not None else None,
        'ap_large': float(ap_large) if ap_large is not None else None,
        'recall_50': float(recall_at_50) if recall_at_50 is not None else None,
        'miss_rate': mr_metrics['miss_rate'],
        'lamr': mr_metrics['lamr'],
        'fppi': mr_metrics['fppi'],
    }


def prepare_dataset_yaml(dataset_path: Path) -> Tuple[Path, bool]:
    """
    Prepare dataset YAML file, adding train/val entries if missing.
    Ultralytics requires train and val keys even if only testing.
    
    Args:
        dataset_path: Path to original dataset YAML file
    
    Returns:
        Tuple of (yaml_path, is_temporary) where is_temporary indicates if a temp file was created
    """
    # Load original YAML
    dataset_config = load_yaml_config(dataset_path)
    
    # Check if train and val are missing
    has_train = 'train' in dataset_config
    has_val = 'val' in dataset_config
    
    # If both train and val exist, use original file
    if has_train and has_val:
        return dataset_path, False
    
    # Create temporary YAML with train/val entries
    # Use test directory for train/val if test exists, otherwise create dummy paths
    test_path = dataset_config.get('test', 'test/images')
    dataset_root = Path(dataset_path).parent / dataset_config.get('path', '')
    
    # Determine train/val paths
    if has_train:
        train_path = dataset_config['train']
    else:
        # Use test directory as fallback (won't be used since we specify split='test')
        train_path = test_path
    
    if has_val:
        val_path = dataset_config['val']
    else:
        # Use test directory as fallback (won't be used since we specify split='test')
        val_path = test_path
    
    # Create temporary YAML file
    temp_config = dataset_config.copy()
    temp_config['train'] = train_path
    temp_config['val'] = val_path
    
    # Create temporary file and write YAML manually
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='test_dataset_')
    try:
        with open(temp_fd, 'w') as f:
            # Write path
            if 'path' in temp_config:
                f.write(f"path: {temp_config['path']}\n")
            
            # Write train, val, test
            if 'train' in temp_config:
                f.write(f"train: {temp_config['train']}\n")
            if 'val' in temp_config:
                f.write(f"val: {temp_config['val']}\n")
            if 'test' in temp_config:
                f.write(f"test: {temp_config['test']}\n")
            
            # Write names
            if 'names' in temp_config:
                f.write("\n# Classes\n")
                f.write("names:\n")
                if isinstance(temp_config['names'], dict):
                    for class_id, class_name in sorted(temp_config['names'].items()):
                        f.write(f"  {class_id}: {class_name}\n")
                else:
                    # Handle list format
                    for idx, class_name in enumerate(temp_config['names']):
                        f.write(f"  {idx}: {class_name}\n")
            
            # Write nc
            if 'nc' in temp_config:
                f.write(f"\n# Number of classes\n")
                f.write(f"nc: {temp_config['nc']}\n")
        
        return Path(temp_path), True
    except Exception:
        # Clean up on error
        try:
            Path(temp_path).unlink()
        except:
            pass
        raise





def main():
    parser = argparse.ArgumentParser(
        description='Test a pretrained model on a YOLO format test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test YOLOv8n model
    python test_model.py \\
        --model-path yolov8n.pt \\
        --model-type yolov8n \\
        --dataset-path datasets/visdrone-det/dataset.yaml \\
        --output-dir results/evaluation

    # Test YOLO11n model
    python test_model.py \\
        --model-path yolo11n.pt \\
        --model-type yolo11n \\
        --dataset-path datasets/visdrone-det/dataset.yaml \\
        --output-dir results/evaluation

    # Test RT-DETR model
    python test_model.py \\
        --model-path rtdetr-l.pt \\
        --model-type rtdetr-l \\
        --dataset-path datasets/visdrone-det/dataset.yaml \\
        --output-dir results/evaluation
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to pretrained model file (.pt file)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
                 'yolov26n', 'yolov26s', 'yolov26m', 'yolov26l', 'yolov26x',
                 'rtdetr-l', 'rtdetr-x'],
        help='Type of model (yolov8n, yolo11n, rtdetr-l, etc.)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset YAML file (YOLO format)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory to save JSON results'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference (default: 640)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold for validation (default: 0.001)'
    )
    parser.add_argument(
        '--skip-mr-metrics',
        action='store_true',
        help='Skip computation of MR, LAMR, and FPPI (can be slow)'
    )
    parser.add_argument(
        'is_p2',
        action='store_true',
        help='Is the model a P2 model (default: False)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    is_p2 = args.is_p2
    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {dataset_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Model Testing Script")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Image Size: {args.imgsz}")
    print("=" * 80)
    
    # Prepare dataset YAML (add train/val if missing)
    print("\nPreparing dataset configuration...")
    yaml_path, is_temp = prepare_dataset_yaml(dataset_path)
    if is_temp:
        print("✓ Created temporary YAML with train/val entries (required by Ultralytics)")
    else:
        print("✓ Using original dataset YAML")
    
    try:
        # Load model
        print("\nLoading model...")
        model = load_model(model_path, args.model_type, is_p2)
        print("✓ Model loaded successfully")
        
        # Run validation
        print("\nRunning validation on test dataset...")
        results = model.val(
            data=str(yaml_path),
            split='test',
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=True,
            save=False,
            plots=False,
        )
        print("✓ Validation complete")
    finally:
        # Clean up temporary YAML file if created
        if is_temp and yaml_path.exists():
            yaml_path.unlink()
            print(f"✓ Cleaned up temporary YAML file")
    
    # Extract metrics
    print("\nExtracting metrics...")
    metrics = extract_metrics(results, model_path, dataset_path, args.model_type, is_p2, skip_mr_metrics=args.skip_mr_metrics)
    
    # Prepare output dictionary
    output_data = {
        'model_path': str(model_path),
        'model_type': args.model_type,
        'dataset_path': str(dataset_path),
        'image_size': args.imgsz,
        'metrics': metrics
    }
    
    # Save to JSON
    output_file = output_dir / f"{args.model_type}_evaluation_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"\nPrecision:        {metrics['precision']:.4f}" if metrics['precision'] else "Precision:        N/A")
    print(f"Recall:           {metrics['recall']:.4f}" if metrics['recall'] else "Recall:           N/A")
    print(f"F1-score:         {metrics['f1_score']:.4f}" if metrics['f1_score'] else "F1-score:         N/A")
    print(f"mAP@0.50:         {metrics['map_50']:.4f}" if metrics['map_50'] else "mAP@0.50:         N/A")
    print(f"mAP@0.50:0.95:    {metrics['map_50_95']:.4f}" if metrics['map_50_95'] else "mAP@0.50:0.95:    N/A")
    print(f"AP@0.75:          {metrics['ap_75']:.4f}" if metrics['ap_75'] else "AP@0.75:          N/A")
    print(f"AP_small:         {metrics['ap_small']:.4f}" if metrics['ap_small'] else "AP_small:         N/A (not available in results)")
    print(f"AP_medium:        {metrics['ap_medium']:.4f}" if metrics['ap_medium'] else "AP_medium:        N/A (not available in results)")
    print(f"AP_large:          {metrics['ap_large']:.4f}" if metrics['ap_large'] else "AP_large:          N/A (not available in results)")
    print(f"Recall@0.50:      {metrics['recall_50']:.4f}" if metrics['recall_50'] else "Recall@0.50:      N/A")
    print(f"Miss Rate (MR):   {metrics['miss_rate']:.4f}" if metrics['miss_rate'] is not None else "Miss Rate (MR):   N/A (use --skip-mr-metrics to skip, or computation failed)")
    print(f"LAMR:             {metrics['lamr']:.4f}" if metrics['lamr'] is not None else "LAMR:             N/A (use --skip-mr-metrics to skip, or computation failed)")
    print(f"FPPI:             {metrics['fppi']:.4f}" if metrics['fppi'] is not None else "FPPI:             N/A (use --skip-mr-metrics to skip, or computation failed)")
    print("=" * 80)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
