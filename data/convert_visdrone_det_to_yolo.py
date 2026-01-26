import os
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# VisDrone class mapping: pedestrian (1) and people (2) -> person (0)
VISDRONE_TO_YOLO = {1: 0, 2: 0}

def convert_bbox_to_yolo(bbox_left, bbox_top, bbox_width, bbox_height, img_width, img_height):
    """Convert VisDrone bbox to YOLO normalized format."""
    center_x = (bbox_left + bbox_width / 2.0) / img_width
    center_y = (bbox_top + bbox_height / 2.0) / img_height
    width = bbox_width / img_width
    height = bbox_height / img_height
    # Clamp to [0, 1]
    return max(0, min(1, center_x)), max(0, min(1, center_y)), max(0, min(1, width)), max(0, min(1, height))

def parse_visdrone_det_annotation(line):
    """Parse VisDrone-DET annotation line.
    Format: bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
    """
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None
    try:
        return {
            'bbox_left': int(parts[0]),
            'bbox_top': int(parts[1]),
            'bbox_width': int(parts[2]),
            'bbox_height': int(parts[3]),
            'category': int(parts[5])  # object_category is at index 5
        }
    except (ValueError, IndexError):
        return None

def get_person_annotations(anno_path, img_width, img_height):
    """Get person annotations from annotation file. Returns list of bboxes."""
    person_boxes = []
    if anno_path.exists():
        with open(anno_path, 'r') as f:
            for line in f:
                anno = parse_visdrone_det_annotation(line)
                if anno and anno['category'] in VISDRONE_TO_YOLO:
                    person_boxes.append({
                        'bbox_left': anno['bbox_left'],
                        'bbox_top': anno['bbox_top'],
                        'bbox_width': anno['bbox_width'],
                        'bbox_height': anno['bbox_height']
                    })
    return person_boxes

def process_image(img_path, anno_path, output_path, split):
    """Process a single image and its annotation."""
    # Get image dimensions
    try:
        with Image.open(img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        return False, None
    
    # Get person annotations
    person_boxes = get_person_annotations(anno_path, img_width, img_height)
    
    # Only process images that have person annotations
    if not person_boxes:
        return False, None
    
    # Convert to YOLO format
    yolo_lines = []
    for box in person_boxes:
        center_x, center_y, width, height = convert_bbox_to_yolo(
            box['bbox_left'], box['bbox_top'],
            box['bbox_width'], box['bbox_height'],
            img_width, img_height
        )
        yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Copy image
    img_name = img_path.name
    output_img_path = output_path / split / 'images' / img_name
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, output_img_path)
    
    # Write label file
    label_name = img_path.stem + '.txt'
    output_label_path = output_path / split / 'labels' / label_name
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # Return image info for visualization
    sample_info = {
        'img_path': img_path,
        'boxes': person_boxes,
        'split': split
    }
    
    return True, sample_info

def process_split(split_name, visdrone_det_path, output_path, collect_samples=False):
    """Process a single split (train, val, or test)."""
    images_dir = visdrone_det_path / 'images'
    annotations_dir = visdrone_det_path / 'annotations'
    
    if not images_dir.exists():
        print(f"  No images directory found for {split_name}, skipping...")
        return 0, None
    
    # Get all image files
    image_files = sorted(images_dir.glob('*.jpg'))
    print(f"  Found {len(image_files)} images")
    
    processed_count = 0
    samples = []  # Collect samples for visualization
    
    for img_path in tqdm(image_files, desc=f"  Converting {split_name}"):
        # Find corresponding annotation file
        anno_path = annotations_dir / (img_path.stem + '.txt')
        
        success, sample_info = process_image(img_path, anno_path, output_path, split_name)
        if success:
            processed_count += 1
            if collect_samples and sample_info:
                samples.append(sample_info)
    
    # Return a random sample if requested
    sample = None
    if collect_samples and samples:
        sample = random.choice(samples)
    
    return processed_count, sample

def is_already_converted(output_path):
    """Check if the dataset has already been converted."""
    if not output_path.exists():
        return False
    
    # Check for expected structure
    expected_splits = ['train', 'valid', 'test']
    for split in expected_splits:
        img_dir = output_path / split / 'images'
        label_dir = output_path / split / 'labels'
        
        # Check if directories exist
        if not img_dir.exists() or not label_dir.exists():
            return False
        
        # Check if there are actual files
        img_files = list(img_dir.glob('*.jpg'))
        label_files = list(label_dir.glob('*.txt'))
        
        if len(img_files) == 0 or len(label_files) == 0:
            return False
    
    return True

def load_samples_from_converted(output_path):
    """Load sample images from already converted dataset for visualization."""
    samples_dict = {}
    splits = {'train': 'train', 'valid': 'valid', 'test': 'test'}
    
    for split_key, split_name in splits.items():
        img_dir = output_path / split_name / 'images'
        label_dir = output_path / split_name / 'labels'
        
        if not img_dir.exists() or not label_dir.exists():
            continue
        
        # Get random image
        img_files = list(img_dir.glob('*.jpg'))
        if not img_files:
            continue
        
        img_path = random.choice(img_files)
        label_path = label_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            continue
        
        # Load image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue
        
        # Load annotations from YOLO format
        person_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class center_x center_y width height
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert back to absolute coordinates for visualization
                    bbox_width = width * img_width
                    bbox_height = height * img_height
                    bbox_left = center_x * img_width - bbox_width / 2.0
                    bbox_top = center_y * img_height - bbox_height / 2.0
                    
                    person_boxes.append({
                        'bbox_left': int(bbox_left),
                        'bbox_top': int(bbox_top),
                        'bbox_width': int(bbox_width),
                        'bbox_height': int(bbox_height)
                    })
        
        if person_boxes:
            samples_dict[split_key] = {
                'img_path': img_path,
                'boxes': person_boxes,
                'split': split_name
            }
    
    return samples_dict

def visualize_samples(samples_dict, visdrone_det_base):
    """Visualize one sample image from each split with annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('VisDrone-DET Sample Images with Person Annotations', fontsize=16, fontweight='bold')
    
    splits_order = ['train', 'valid', 'test']
    split_display_names = {'train': 'Train', 'valid': 'Validation', 'test': 'Test'}
    
    for idx, split_name in enumerate(splits_order):
        sample = samples_dict.get(split_name)
        if sample is None:
            axes[idx].text(0.5, 0.5, f'No {split_display_names[split_name]} samples\nwith person annotations', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(f'{split_display_names[split_name]} Set', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            continue
        
        # Load image
        img_path = sample['img_path']
        img = Image.open(img_path)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].set_title(f'{split_display_names[split_name]} Set\n{img_path.name}', 
                          fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        
        # Draw bounding boxes
        img_width, img_height = img.size
        for box in sample['boxes']:
            x1 = box['bbox_left']
            y1 = box['bbox_top']
            width = box['bbox_width']
            height = box['bbox_height']
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[idx].add_patch(rect)
        
        # Add annotation count
        axes[idx].text(10, 20, f'{len(sample["boxes"])} persons', 
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                      color='white', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("\n✓ Displayed sample images with annotations")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert VisDrone-DET to YOLO format')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Show sample images with annotations from each split (default: False)')
    args = parser.parse_args()
    
    # Script runs from datasets/ directory, so look in current directory
    datasets_dir = Path('.')
    output_path = datasets_dir / 'visdrone'
    
    # Look for VisDrone-DET directory
    visdrone_det_base = datasets_dir / 'VisDrone-DET'
    
    if not visdrone_det_base.exists():
        print("No VisDrone-DET dataset found in current directory")
        print(f"Expected: {visdrone_det_base}")
        return
    
    # Check if dataset is already converted
    if is_already_converted(output_path):
        print("="*60)
        print("Dataset already converted!")
        print("="*60)
        print(f"Found existing converted dataset at: {output_path}")
        print("\nSkipping conversion...")
        
        # Get statistics from existing dataset
        total_stats = {}
        splits_check = {'train': 'train', 'valid': 'valid', 'test': 'test'}
        for split_key, split_name in splits_check.items():
            img_dir = output_path / split_name / 'images'
            if img_dir.exists():
                total_stats[split_name] = len(list(img_dir.glob('*.jpg')))
        
        # Load samples for visualization if requested
        samples_dict = {}
        if args.visualize:
            print("\nLoading samples from converted dataset for visualization...")
            samples_dict = load_samples_from_converted(output_path)
        
        # Ensure dataset.yaml exists
        yaml_path = output_path / 'dataset.yaml'
        if not yaml_path.exists():
            print("\nCreating missing dataset.yaml...")
            yaml_path_str = 'datasets/visdrone-det'
            yaml_content = f"""# YOLOv8 dataset configuration for VisDrone-DET (person class only)
path: {yaml_path_str}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes (single class: person = pedestrian + people combined)
names:
  0: person

# Number of classes
nc: 1
"""
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            print(f"✓ Created dataset.yaml at {yaml_path}")
        else:
            print(f"\n✓ Dataset.yaml already exists at {yaml_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("Existing Dataset Summary")
        print("="*60)
        for split_name, count in total_stats.items():
            print(f"{split_name.upper()}: {count} images")
        
        # Visualize samples if requested
        if args.visualize and samples_dict:
            print("\n" + "="*60)
            print("Generating visualization...")
            print("="*60)
            visualize_samples(samples_dict, visdrone_det_base)
        
        return
    
    print("="*60)
    print("Converting VisDrone-DET to YOLO format (person class only)")
    print("="*60)
    if args.visualize:
        print("Visualization mode: Will show sample images after conversion")
    print()
    
    # Process train, val, and test sets
    splits = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'valid',
        'VisDrone2019-DET-test-dev': 'test'
    }
    
    total_stats = {}
    samples_dict = {}  # Store samples for visualization
    
    for dataset_dir_name, split_name in splits.items():
        visdrone_det_path = visdrone_det_base / dataset_dir_name
        
        if not visdrone_det_path.exists():
            print(f"\n⚠️  {dataset_dir_name} not found, skipping...")
            continue
        
        print(f"\nProcessing {split_name} set from {dataset_dir_name}...")
        count, sample = process_split(split_name, visdrone_det_path, output_path, 
                                      collect_samples=args.visualize)
        total_stats[split_name] = count
        if sample:
            samples_dict[split_name] = sample
        print(f"  ✓ Processed {count} images with person annotations")
    
    # Create dataset.yaml file
    yaml_path = output_path / 'dataset.yaml'
    # Path should be relative to project root (datasets/visdrone-det)
    yaml_path_str = 'datasets/visdrone-det'
    yaml_content = f"""# YOLOv8 dataset configuration for VisDrone-DET (person class only)
path: {yaml_path_str}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes (single class: person = pedestrian + people combined)
names:
  0: person

# Number of classes
nc: 1
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n✓ Created dataset.yaml at {yaml_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    for split_name, count in total_stats.items():
        print(f"{split_name.upper()}: {count} images")
    
    print(f"\n✓ Dataset structure created at: {output_path}")
    print(f"  - {output_path}/train/images/ and labels/")
    print(f"  - {output_path}/valid/images/ and labels/")
    print(f"  - {output_path}/test/images/ and labels/")
    
    # Visualize samples if requested
    if args.visualize and samples_dict:
        print("\n" + "="*60)
        print("Generating visualization...")
        print("="*60)
        visualize_samples(samples_dict, visdrone_det_base)

if __name__ == '__main__':
    main()
