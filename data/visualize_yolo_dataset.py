#!/usr/bin/env python3
"""
General visualization script for YOLO format datasets.
Randomly samples images from each split (train/valid/test) and displays them with bounding boxes.
"""

import argparse
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_yolo_annotations(label_path, img_width, img_height):
    """Load YOLO format annotations and convert to absolute coordinates."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert from normalized YOLO format to absolute coordinates
                    bb_width = width * img_width
                    bb_height = height * img_height
                    bb_left = center_x * img_width - bb_width / 2
                    bb_top = center_y * img_height - bb_height / 2
                    
                    boxes.append({
                        'class_id': class_id,
                        'bb_left': bb_left,
                        'bb_top': bb_top,
                        'bb_width': bb_width,
                        'bb_height': bb_height,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        print(f"  ⚠️  Error reading label file {label_path}: {e}")
    
    return boxes

def get_class_names(dataset_root):
    """Try to load class names from dataset.yaml if it exists."""
    yaml_path = dataset_root / 'dataset.yaml'
    class_names = {}
    
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                content = f.read()
                # Simple parsing of YAML names section
                if 'names:' in content:
                    in_names = False
                    for line in content.split('\n'):
                        if 'names:' in line:
                            in_names = True
                            continue
                        if in_names and ':' in line and not line.strip().startswith('#'):
                            # Parse line like "  0: person" or "  1: car"
                            parts = line.split(':')
                            if len(parts) == 2:
                                try:
                                    class_id = int(parts[0].strip())
                                    class_name = parts[1].strip()
                                    class_names[class_id] = class_name
                                except ValueError:
                                    pass
                        elif in_names and line.strip() and not line.strip().startswith('#'):
                            # Check if we've moved past names section
                            if not line.strip().startswith(' ') and ':' in line:
                                break
        except Exception as e:
            print(f"  ⚠️  Could not parse dataset.yaml: {e}")
    
    return class_names

def visualize_yolo_dataset(dataset_root, num_samples_per_split=1):
    """Visualize random samples from a YOLO format dataset."""
    dataset_root = Path(dataset_root)
    
    if not dataset_root.exists():
        print(f"Error: Dataset root directory not found: {dataset_root}")
        return
    
    # Try to find splits
    splits = []
    for split_name in ['train', 'valid', 'val', 'test']:
        split_dir = dataset_root / split_name
        if split_dir.exists():
            img_dir = split_dir / 'images'
            label_dir = split_dir / 'labels'
            if img_dir.exists() and label_dir.exists():
                splits.append(split_name)
    
    if not splits:
        print(f"Error: No valid splits found in {dataset_root}")
        print("Expected structure: dataset_root/{train,valid,test}/images/ and labels/")
        return
    
    print("="*70)
    print(f"Visualizing YOLO Dataset: {dataset_root}")
    print("="*70)
    print(f"Found splits: {', '.join(splits)}")
    
    # Load class names if available
    class_names = get_class_names(dataset_root)
    if class_names:
        print(f"Class names: {class_names}")
    else:
        print("No class names found in dataset.yaml, using class IDs")
    
    # Collect samples from each split
    samples = []
    for split_name in splits:
        img_dir = dataset_root / split_name / 'images'
        label_dir = dataset_root / split_name / 'labels'
        
        # Get all image files
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        if not img_files:
            print(f"  ⚠️  No images found in {split_name}")
            continue
        
        # Randomly sample images
        num_samples = min(num_samples_per_split, len(img_files))
        sampled_files = random.sample(img_files, num_samples)
        
        for img_file in sampled_files:
            label_file = label_dir / (img_file.stem + '.txt')
            
            if not label_file.exists():
                print(f"  ⚠️  Label file not found: {label_file.name}")
                continue
            
            try:
                img = Image.open(img_file)
                img_width, img_height = img.size
                
                boxes = load_yolo_annotations(label_file, img_width, img_height)
                
                samples.append({
                    'split': split_name,
                    'img_path': img_file,
                    'img': img,
                    'boxes': boxes,
                    'img_width': img_width,
                    'img_height': img_height
                })
                
                print(f"  ✓ {split_name}/{img_file.name}: {len(boxes)} objects")
            except Exception as e:
                print(f"  ⚠️  Error loading {img_file.name}: {e}")
    
    if not samples:
        print("\n⚠️  No valid samples found to visualize")
        return
    
    # Create visualization
    print("\n" + "="*70)
    print("Generating Visualization...")
    print("="*70)
    
    num_samples = len(samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    fig.suptitle(f'YOLO Dataset Visualization: {dataset_root.name}', 
                 fontsize=16, fontweight='bold')
    
    # Color palette for different classes
    colors = plt.cm.tab20(range(20))
    
    for idx, sample in enumerate(samples):
        ax = axes[idx]
        
        # Display image
        ax.imshow(sample['img'])
        ax.set_title(f'{sample["split"].upper()}\n{sample["img_path"].name}\n{len(sample["boxes"])} objects', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw bounding boxes
        for box in sample['boxes']:
            class_id = box['class_id']
            bb_left = box['bb_left']
            bb_top = box['bb_top']
            bb_width = box['bb_width']
            bb_height = box['bb_height']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Get class name or use ID
            if class_id in class_names:
                class_label = class_names[class_id]
            else:
                class_label = f'Class {class_id}'
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bb_left, bb_top), bb_width, bb_height,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add class label (only if not too many boxes to avoid clutter)
            if len(sample['boxes']) <= 50:
                ax.text(bb_left, bb_top - 5, class_label, 
                       color=color, fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Add object count
        ax.text(10, 20, f'{len(sample["boxes"])} objects', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Visualization displayed")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO format dataset with random samples from each split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize dataset with 1 sample per split
  python visualize_yolo_dataset.py datasets/mot20
  
  # Visualize with 3 samples per split
  python visualize_yolo_dataset.py datasets/visdrone-det --num_samples 3
  
  # Visualize from current directory
  python visualize_yolo_dataset.py .
        """
    )
    parser.add_argument('dataset_root', type=str,
                       help='Root directory of the YOLO dataset (should contain train/, valid/, test/ subdirectories)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of random samples to show from each split (default: 1)')
    args = parser.parse_args()
    
    visualize_yolo_dataset(args.dataset_root, num_samples_per_split=args.num_samples)

if __name__ == '__main__':
    main()
