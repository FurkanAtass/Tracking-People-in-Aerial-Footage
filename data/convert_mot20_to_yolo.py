#!/usr/bin/env python3
"""
Convert MOT20 Challenge dataset to YOLO format using detections.
Only processes person class (class=1) from det/det.txt files.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import configparser

def parse_seqinfo(seqinfo_path):
    """Parse seqinfo.ini file."""
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    
    if 'Sequence' in config:
        seq = config['Sequence']
        return {
            'name': seq.get('name', ''),
            'imDir': seq.get('imDir', 'img1'),
            'frameRate': float(seq.get('frameRate', 25)),
            'seqLength': int(seq.get('seqLength', 0)),
            'imWidth': int(seq.get('imWidth', 0)),
            'imHeight': int(seq.get('imHeight', 0)),
            'imExt': seq.get('imExt', '.jpg')
        }
    return None

def parse_mot_detection(line):
    """Parse MOT detection line.
    Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility,...
    Note: In MOT20 detection files, class can be -1 (unknown/any class) or 1 (person).
    We accept all detections since they are typically all person detections.
    """
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None
    
    try:
        frame = int(parts[0])
        bb_left = float(parts[2])
        bb_top = float(parts[3])
        bb_width = float(parts[4])
        bb_height = float(parts[5])
        conf = float(parts[6])
        class_id = int(parts[7])
        
        # Accept all detections (class=-1 means unknown/any class, class=1 means person, class=0 might be ignored)
        # In MOT20 detection files, detections are typically all person detections
        # Accept class -1 (unknown/any), 0 (might be ignored but still valid), and 1 (person)
        if class_id not in [-1, 0, 1]:
            return None
        
        # Skip invalid bounding boxes
        if bb_width <= 0 or bb_height <= 0:
            return None
        
        return {
            'frame': frame,
            'bb_left': bb_left,
            'bb_top': bb_top,
            'bb_width': bb_width,
            'bb_height': bb_height,
            'conf': conf,
            'class': class_id
        }
    except (ValueError, IndexError):
        return None

def convert_bbox_to_yolo(bb_left, bb_top, bb_width, bb_height, img_width, img_height):
    """Convert MOT bbox to YOLO normalized format."""
    center_x = (bb_left + bb_width / 2.0) / img_width
    center_y = (bb_top + bb_height / 2.0) / img_height
    width = bb_width / img_width
    height = bb_height / img_height
    # Clamp to [0, 1]
    return max(0, min(1, center_x)), max(0, min(1, center_y)), max(0, min(1, width)), max(0, min(1, height))

def load_detections_by_frame(det_path):
    """Load detections grouped by frame number."""
    detections_by_frame = {}
    if not det_path.exists():
        return detections_by_frame
    
    with open(det_path, 'r') as f:
        for line in f:
            det = parse_mot_detection(line)
            if det:
                frame = det['frame']
                if frame not in detections_by_frame:
                    detections_by_frame[frame] = []
                detections_by_frame[frame].append(det)
    
    return detections_by_frame

def process_sequence(seq_path, output_path, split, seqinfo, sample_interval=None, collect_samples=False):
    """Process a single MOT20 sequence."""
    img_dir = seq_path / seqinfo['imDir']
    det_path = seq_path / 'det' / 'det.txt'
    
    if not img_dir.exists() or not det_path.exists():
        return 0, None
    
    # Load detections grouped by frame
    detections_by_frame = load_detections_by_frame(det_path)
    
    if not detections_by_frame:
        return 0, None
    
    # Filter frames based on sampling interval
    # IMPORTANT: Do NOT sample test sets - use all frames for reliable evaluation
    if sample_interval and sample_interval > 1 and split != 'test':
        filtered_frames = {
            frame: dets for frame, dets in detections_by_frame.items()
            if frame == 1 or frame % sample_interval == 0
        }
        detections_by_frame = filtered_frames
    
    if not detections_by_frame:
        return 0, None
    
    processed_count = 0
    sample_info = None
    
    img_width = seqinfo['imWidth']
    img_height = seqinfo['imHeight']
    im_ext = seqinfo['imExt']
    
    # Process each frame
    for frame_num in sorted(detections_by_frame.keys()):
        # Find corresponding image
        img_name = f"{frame_num:06d}{im_ext}"
        img_path = img_dir / img_name
        
        if not img_path.exists():
            continue
        
        detections = detections_by_frame[frame_num]
        if not detections:
            continue
        
        # Convert detections to YOLO format
        yolo_lines = []
        boxes_for_viz = []
        
        for det in detections:
            bb_left = det['bb_left']
            bb_top = det['bb_top']
            bb_width = det['bb_width']
            bb_height = det['bb_height']
            
            # Skip if box has invalid dimensions
            if bb_width <= 0 or bb_height <= 0:
                continue
            
            # Store original coordinates for visualization (no clamping) - same as explore code
            boxes_for_viz.append({
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': det['conf']
            })
            
            # For YOLO conversion, use original coordinates directly
            # The convert_bbox_to_yolo function will clamp normalized coordinates to [0,1]
            # This matches how the explore/visualization code shows ALL boxes
            center_x, center_y, width, height = convert_bbox_to_yolo(
                bb_left, bb_top, bb_width, bb_height,
                img_width, img_height
            )
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Create unique image name (sequence_frame.jpg)
        seq_name = seqinfo['name']
        yolo_img_name = f"{seq_name}_{frame_num:06d}{im_ext}"
        yolo_label_name = f"{seq_name}_{frame_num:06d}.txt"
        
        # Copy image
        output_img_path = output_path / split / 'images' / yolo_img_name
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_img_path)
        
        # Write label file
        if yolo_lines:
            output_label_path = output_path / split / 'labels' / yolo_label_name
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            processed_count += 1
            
            # Store sample info for visualization (store first processed frame with detections)
            if collect_samples and sample_info is None and boxes_for_viz:
                sample_info = {
                    'img_path': img_path,
                    'boxes': boxes_for_viz,
                    'split': split,
                    'seq_name': seq_name,
                    'frame_num': frame_num,
                    'img_width': img_width,
                    'img_height': img_height
                }
    
    return processed_count, sample_info

def visualize_samples(samples_dict):
    """Visualize sample images with bounding boxes."""
    if not samples_dict:
        return
    
    num_samples = len(samples_dict)
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    fig.suptitle('MOT20 Dataset - Sample Frames with Detections', fontsize=16, fontweight='bold')
    
    for idx, (split_name, sample) in enumerate(samples_dict.items()):
        ax = axes[idx]
        
        # Load image
        img = Image.open(sample['img_path'])
        ax.imshow(img)
        ax.set_title(f'{split_name.upper()}/{sample["seq_name"]}\nFrame {sample["frame_num"]}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw ALL bounding boxes (no limit) - same as explore code
        for box in sample['boxes']:
            bb_left = box['bb_left']
            bb_top = box['bb_top']
            bb_width = box['bb_width']
            bb_height = box['bb_height']
            
            rect = patches.Rectangle(
                (bb_left, bb_top), bb_width, bb_height,
                linewidth=2, edgecolor='lime', facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Optionally show confidence score (limit label display to avoid clutter)
            if len(sample['boxes']) <= 50 and box['conf'] < 1.0:  # Only show if not too many boxes
                ax.text(bb_left, bb_top - 5, f'{box["conf"]:.2f}', 
                       color='lime', fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Add detection count (show actual number of boxes drawn)
        boxes_drawn = len(sample['boxes'])
        ax.text(10, 20, f'{boxes_drawn} detections', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def load_samples_from_converted(output_path):
    """Load random sample images and annotations from converted dataset."""
    samples_dict = {}
    
    for split in ['train', 'valid', 'test']:
        img_dir = output_path / split / 'images'
        label_dir = output_path / split / 'labels'
        
        if not img_dir.exists() or not label_dir.exists():
            continue
        
        # Get random image from this split
        img_files = list(img_dir.glob('*.jpg'))
        if not img_files:
            continue
        
        img_file = random.choice(img_files)
        label_file = label_dir / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        # Load image
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
            
            # Load annotations
            boxes = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert back to absolute coordinates
                        bb_width = width * img_width
                        bb_height = height * img_height
                        bb_left = center_x * img_width - bb_width / 2
                        bb_top = center_y * img_height - bb_height / 2
                        
                        boxes.append({
                            'bb_left': bb_left,
                            'bb_top': bb_top,
                            'bb_width': bb_width,
                            'bb_height': bb_height
                        })
            
            if boxes:
                # Parse sequence name and frame from filename
                name_parts = img_file.stem.split('_')
                seq_name = name_parts[0]
                frame_num = int(name_parts[1]) if len(name_parts) > 1 else 0
                
                samples_dict[split] = {
                    'img_path': img_file,
                    'boxes': boxes,
                    'split': split,
                    'seq_name': seq_name,
                    'frame_num': frame_num,
                    'img_width': img_width,
                    'img_height': img_height
                }
        except Exception as e:
            print(f"  ⚠️  Error loading sample from {img_file}: {e}")
            continue
    
    return samples_dict

def is_already_converted(output_path):
    """Check if the dataset has already been converted."""
    if not output_path.exists():
        return False
    
    # Check for expected directory structure
    expected_dirs = [
        output_path / 'train' / 'images',
        output_path / 'train' / 'labels',
        output_path / 'valid' / 'images',
        output_path / 'valid' / 'labels',
        output_path / 'test' / 'images',
        output_path / 'test' / 'labels'
    ]
    
    for dir_path in expected_dirs:
        if not dir_path.exists():
            return False
        # Check if directory has files
        if not list(dir_path.glob('*')):
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert MOT20 dataset to YOLO format (detections only)')
    parser.add_argument('--sample_interval', type=int, default=15,
                       help='Frame sampling interval. MOT20 uses 25 FPS, so interval 25 = every 1 second, 50 = every 2 seconds, etc. (e.g., 25 = take frames 1, 25, 50, 75, ...). If not specified, all frames are processed.')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Show sample images with annotations from each split (default: False)')
    args = parser.parse_args()
    
    datasets_dir = Path('.')
    output_path = datasets_dir / 'mot-20'
    mot20_base = datasets_dir / 'MOT20'
    
    if not mot20_base.exists():
        print("No MOT20 dataset found in current directory")
        print(f"Expected: {mot20_base}")
        return
    
    # Check if already converted
    if is_already_converted(output_path):
        print("="*60)
        print("Dataset already converted!")
        print("="*60)
        print(f"Found existing converted dataset at: {output_path}")
        print("\nSkipping conversion...")
        
        total_stats = {}
        for split in ['train', 'valid', 'test']:
            img_dir = output_path / split / 'images'
            if img_dir.exists():
                total_stats[split] = len(list(img_dir.glob('*.jpg')))
        
        samples_dict = {}
        if args.visualize:
            print("\nLoading samples from converted dataset for visualization...")
            samples_dict = load_samples_from_converted(output_path)
        
        yaml_path = output_path / 'dataset.yaml'
        if not yaml_path.exists():
            print("\nCreating missing dataset.yaml...")
            yaml_path_str = 'datasets/mot20'
            yaml_content = f"""# YOLOv8 dataset configuration for MOT20 (person class only - detections)
path: {yaml_path_str}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes (single class: person)
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
        
        print("\n" + "="*60)
        print("Existing Dataset Summary")
        print("="*60)
        for split_name, count in total_stats.items():
            print(f"{split_name.upper()}: {count} images")
        
        if args.visualize and samples_dict:
            print("\n" + "="*60)
            print("Generating visualization...")
            print("="*60)
            visualize_samples(samples_dict)
        
        return
    
    print("="*60)
    print("Converting MOT20 to YOLO format (detections, person class only)")
    print("="*60)
    if args.sample_interval and args.sample_interval > 1:
        print(f"Frame sampling interval: {args.sample_interval}")
        print(f"  At 25 FPS, this means sampling every {args.sample_interval/25:.1f} seconds")
        print(f"  Frames taken: 1, {args.sample_interval}, {2*args.sample_interval}, {3*args.sample_interval}, ...")
        print(f"  NOTE: Test sets will use ALL frames (no sampling) for reliable evaluation")
    if args.visualize:
        print("Visualization mode: Will show sample images after conversion")
    print()
    
    splits = ['train', 'valid', 'test']
    total_stats = {}
    samples_dict = {}
    
    for split in splits:
        split_dir = mot20_base / split
        if not split_dir.exists():
            print(f"\n⚠️  {split} directory not found, skipping...")
            continue
        
        # Use sampling only for train/valid, not for test sets
        if split == 'test':
            print(f"\nProcessing {split} set (no sampling - using all frames for reliable evaluation)...")
            sample_interval_for_split = None
        else:
            print(f"\nProcessing {split} set...")
            sample_interval_for_split = args.sample_interval
        
        split_count = 0
        split_sample = None
        
        # Process each sequence in this split
        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('MOT20-')])
        
        for seq_path in tqdm(sequences, desc=f"  Converting {split}"):
            seqinfo_path = seq_path / 'seqinfo.ini'
            if not seqinfo_path.exists():
                print(f"    ⚠️  seqinfo.ini not found in {seq_path.name}, skipping...")
                continue
            
            seqinfo = parse_seqinfo(seqinfo_path)
            if not seqinfo:
                print(f"    ⚠️  Could not parse seqinfo.ini in {seq_path.name}, skipping...")
                continue
            
            count, sample = process_sequence(seq_path, output_path, split, seqinfo, 
                                            sample_interval_for_split, args.visualize)
            split_count += count
            if sample and split_sample is None:
                split_sample = sample
        
        total_stats[split] = split_count
        if split_sample:
            samples_dict[split] = split_sample
        print(f"  ✓ Processed {split_count} images with person detections")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    yaml_path = output_path / 'dataset.yaml'
    yaml_path_str = 'datasets/mot20'
    yaml_content = f"""# YOLOv8 dataset configuration for MOT20 (person class only - detections)
path: {yaml_path_str}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes (single class: person)
names:
  0: person

# Number of classes
nc: 1
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n✓ Created dataset.yaml at {yaml_path}")
    
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    for split_name, count in total_stats.items():
        print(f"{split_name.upper()}: {count} images")
    
    print(f"\n✓ Dataset structure created at: {output_path}")
    print(f"  - {output_path}/train/images/ and labels/")
    print(f"  - {output_path}/valid/images/ and labels/")
    print(f"  - {output_path}/test/images/ and labels/")
    
    if args.visualize and samples_dict:
        print("\n" + "="*60)
        print("Generating visualization...")
        print("="*60)
        visualize_samples(samples_dict)

if __name__ == '__main__':
    main()
