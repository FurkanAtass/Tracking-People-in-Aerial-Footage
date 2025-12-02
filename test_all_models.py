#!/usr/bin/env python3
"""
Test all trained models on VisDrone and Okutama test sets.
Generates a comprehensive CSV report with metrics, FPS, and model size.
"""

import os
import time
import csv
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm

def get_model_size(model_path):
    """Get model file size in MB."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)  # Convert to MB

def measure_fps(model, dataset_path, num_images=100, imgsz=640):
    """Measure FPS by running inference on test images."""
    import random
    from PIL import Image
    
    # Get sample images
    img_dir = Path(dataset_path) / 'test' / 'images'
    if not img_dir.exists():
        return None
    
    all_images = list(img_dir.glob('*.jpg'))
    if len(all_images) == 0:
        return None
    
    # Sample images for FPS measurement
    sample_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Warmup
    for img_path in sample_images[:5]:
        try:
            model.predict(str(img_path), imgsz=imgsz, verbose=False)
        except:
            pass
    
    # Measure FPS with progress bar
    start_time = time.time()
    for img_path in tqdm(sample_images, desc="      FPS measurement", leave=False):
        try:
            model.predict(str(img_path), imgsz=imgsz, verbose=False)
        except:
            continue
    end_time = time.time()
    
    elapsed = end_time - start_time
    fps = len(sample_images) / elapsed if elapsed > 0 else None
    return fps

def evaluate_model(model_path, dataset_yaml, dataset_name, imgsz=640):
    """Evaluate a model on a dataset and return metrics."""
    try:
        # Load model
        model = YOLO(str(model_path))
        
        # Get model size
        model_size_mb = get_model_size(model_path)
        
        # Run validation (Ultralytics shows its own progress)
        results = model.val(
            data=dataset_yaml,
            split='test',
            imgsz=imgsz,
            verbose=True,  # Show validation progress
            save=False,
        )
        
        # Extract metrics from results
        metrics = results.results_dict
        
        precision = metrics.get('metrics/precision(B)', None)
        recall = metrics.get('metrics/recall(B)', None)
        map50 = metrics.get('metrics/mAP50(B)', None)
        map50_95 = metrics.get('metrics/mAP50-95(B)', None)
        
        # Calculate F1 score
        f1 = None
        if precision is not None and recall is not None:
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
        
        # Measure FPS
        fps = measure_fps(model, Path(dataset_yaml).parent, num_images=50, imgsz=imgsz)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map50': map50,
            'map50_95': map50_95,
            'fps': fps,
            'model_size_mb': model_size_mb,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return {
            'precision': None,
            'recall': None,
            'f1': None,
            'map50': None,
            'map50_95': None,
            'fps': None,
            'model_size_mb': get_model_size(model_path) if model_path.exists() else None,
        }

def print_results_table(exp_name, visdrone_results, okutama_results, model_size_mb):
    """Print formatted results table for a model."""
    print("\n" + "="*80)
    print(f"Results for: {exp_name}")
    print("="*80)
    print(f"Model Size: {model_size_mb:.2f} MB" if model_size_mb else "Model Size: N/A")
    print("\n" + "-"*80)
    print(f"{'Metric':<20} {'VisDrone':<30} {'Okutama':<30}")
    print("-"*80)
    
    metrics = [
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1'),
        ('mAP50', 'map50'),
        ('mAP50-95', 'map50_95'),
        ('FPS', 'fps'),
    ]
    
    for metric_name, metric_key in metrics:
        vis_val = visdrone_results.get(metric_key)
        oku_val = okutama_results.get(metric_key)
        
        vis_str = f"{vis_val:.4f}" if vis_val is not None else "N/A"
        oku_str = f"{oku_val:.4f}" if oku_val is not None else "N/A"
        
        if metric_key == 'fps':
            vis_str = f"{vis_val:.2f}" if vis_val is not None else "N/A"
            oku_str = f"{oku_val:.2f}" if oku_val is not None else "N/A"
        
        print(f"{metric_name:<20} {vis_str:<30} {oku_str:<30}")
    
    print("="*80)

def find_all_models(results_dir):
    """Find all best.pt model files in results directory."""
    results_path = Path(results_dir)
    models = []
    
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        best_model = exp_dir / 'weights' / 'best.pt'
        if best_model.exists():
            models.append({
                'path': best_model,
                'experiment_name': exp_dir.name
            })
    
    return sorted(models, key=lambda x: x['experiment_name'])

def main():
    results_dir = Path('results')
    visdrone_yaml = 'datasets/visdrone/dataset.yaml'
    okutama_yaml = 'datasets/okutama/dataset.yaml'
    
    # Check if datasets exist
    if not Path(visdrone_yaml).exists():
        print(f"Error: {visdrone_yaml} not found")
        return
    
    if not Path(okutama_yaml).exists():
        print(f"Error: {okutama_yaml} not found")
        return
    
    # Find all models
    print("="*60)
    print("Finding trained models...")
    print("="*60)
    models = find_all_models(results_dir)
    
    if not models:
        print("No models found in results/ directory")
        return
    
    print(f"Found {len(models)} models:")
    for model in models:
        print(f"  - {model['experiment_name']}")
    
    # Test each model on both datasets
    print("\n" + "="*60)
    print("Testing models on test datasets...")
    print("="*60)
    
    all_results = []
    
    for model_info in tqdm(models, desc="Testing models", unit="model"):
        model_path = model_info['path']
        exp_name = model_info['experiment_name']
        
        # Determine input size based on model name
        if 'v2' in exp_name.lower() or 'v2Params' in exp_name:
            imgsz = 960
            print(f"\n{'='*80}")
            print(f"Testing: {exp_name} (using imgsz=960)")
            print(f"{'='*80}")
        else:
            imgsz = 640
            print(f"\n{'='*80}")
            print(f"Testing: {exp_name} (using imgsz=640)")
            print(f"{'='*80}")
        
        # Test on VisDrone
        print(f"\n  Dataset: VisDrone")
        visdrone_results = evaluate_model(model_path, visdrone_yaml, 'VisDrone', imgsz=imgsz)
        
        # Test on Okutama
        print(f"\n  Dataset: Okutama")
        okutama_results = evaluate_model(model_path, okutama_yaml, 'Okutama', imgsz=imgsz)
        
        # Print results immediately
        print_results_table(exp_name, visdrone_results, okutama_results, visdrone_results['model_size_mb'])
        
        # Store results
        all_results.append({
            'experiment_name': exp_name,
            'model_path': str(model_path),
            'model_size_mb': visdrone_results['model_size_mb'],  # Same for both
            'visdrone_precision': visdrone_results['precision'],
            'visdrone_recall': visdrone_results['recall'],
            'visdrone_f1': visdrone_results['f1'],
            'visdrone_map50': visdrone_results['map50'],
            'visdrone_map50_95': visdrone_results['map50_95'],
            'visdrone_fps': visdrone_results['fps'],
            'okutama_precision': okutama_results['precision'],
            'okutama_recall': okutama_results['recall'],
            'okutama_f1': okutama_results['f1'],
            'okutama_map50': okutama_results['map50'],
            'okutama_map50_95': okutama_results['map50_95'],
            'okutama_fps': okutama_results['fps'],
        })
    
    # Create CSV report
    csv_path = results_dir / 'model_evaluation_report.csv'
    
    print("\n" + "="*60)
    print("Generating CSV report...")
    print("="*60)
    
    # Define CSV columns
    fieldnames = [
        'experiment_name',
        'model_size_mb',
        'visdrone_precision',
        'visdrone_recall',
        'visdrone_f1',
        'visdrone_map50',
        'visdrone_map50_95',
        'visdrone_fps',
        'okutama_precision',
        'okutama_recall',
        'okutama_f1',
        'okutama_map50',
        'okutama_map50_95',
        'okutama_fps',
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            writer.writerow(result)
    
    print(f"\n✓ Report saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Model':<40} {'Size(MB)':<10} {'VisDrone mAP50':<15} {'Okutama mAP50':<15}")
    print("-" * 80)
    
    for result in all_results:
        vis_map = f"{result['visdrone_map50']:.4f}" if result['visdrone_map50'] else "N/A"
        oku_map = f"{result['okutama_map50']:.4f}" if result['okutama_map50'] else "N/A"
        size = f"{result['model_size_mb']:.2f}" if result['model_size_mb'] else "N/A"
        
        print(f"{result['experiment_name']:<40} {size:<10} {vis_map:<15} {oku_map:<15}")
    
    print("\n✓ Evaluation complete!")

if __name__ == '__main__':
    main()

