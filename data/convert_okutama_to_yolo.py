import os
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set random seed for train/valid split
random.seed(42)

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """Convert (xmin, ymin, xmax, ymax) to YOLO normalized format."""
    # Calculate center and dimensions
    center_x = (xmin + xmax) / 2.0 / img_width
    center_y = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    # Clamp to [0, 1]
    return max(0, min(1, center_x)), max(0, min(1, center_y)), max(0, min(1, width)), max(0, min(1, height))

def parse_okutama_label(line):
    """Parse Okutama label line.
    Format: track_id xmin ymin xmax ymax frame lost occluded generated "Person" "Action"...
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    
    try:
        track_id = int(parts[0])
        xmin = int(parts[1])
        ymin = int(parts[2])
        xmax = int(parts[3])
        ymax = int(parts[4])
        frame = int(parts[5])
        lost = int(parts[6])
        occluded = int(parts[7])
        generated = int(parts[8])
        label = parts[9].strip('"')
        
        # Only process "Person" labels
        if label != "Person":
            return None
        
        # Skip if lost or occluded (or we can keep them, but typically skip lost)
        if lost == 1:
            return None
        
        return {
            'track_id': track_id,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'frame': frame,
            'lost': lost,
            'occluded': occluded,
            'generated': generated
        }
    except (ValueError, IndexError):
        return None

def find_image_path(sequence_name, frame_num, base_path):
    """Find image file for a given sequence and frame number."""
    # Search in Drone1 and Drone2, Morning and Noon
    for drone in ['Drone1', 'Drone2']:
        for time in ['Morning', 'Noon']:
            img_dir = base_path / drone / time / 'Extracted-Frames-1280x720' / sequence_name
            if img_dir.exists():
                # Try different frame number formats
                for fmt in [f'{frame_num}.jpg', f'{frame_num:04d}.jpg', f'{frame_num:05d}.jpg']:
                    img_path = img_dir / fmt
                    if img_path.exists():
                        return img_path
    return None

def process_sequence(sequence_name, label_path, images_base_path, output_path, split, sample_interval=None, collect_samples=False):
    """Process a single sequence.
    
    Args:
        sequence_name: Name of the sequence
        label_path: Path to the label file
        images_base_path: Base path for images
        output_path: Output directory
        split: Split name (train/valid/test)
        sample_interval: If provided, only process frames at this interval (e.g., 15 = frames 1, 15, 30, ...)
        collect_samples: If True, return sample info for visualization
    """
    if not label_path.exists():
        return 0, None
    
    # Read annotations and group by frame
    annotations_by_frame = {}
    with open(label_path, 'r') as f:
        for line in f:
            anno = parse_okutama_label(line)
            if anno is None:
                continue
            
            frame = anno['frame']
            if frame not in annotations_by_frame:
                annotations_by_frame[frame] = []
            annotations_by_frame[frame].append(anno)
    
    if not annotations_by_frame:
        return 0, None
    
    # Filter frames based on sampling interval
    # IMPORTANT: Do NOT sample test sets - use all frames for reliable evaluation
    if sample_interval and sample_interval > 1 and split != 'test':
        # Filter: keep frame 1 and frames that are multiples of sample_interval
        # This gives us frames 1, sample_interval, 2*sample_interval, etc.
        # Example: sample_interval=15 gives frames 1, 15, 30, 45, ...
        filtered_frames = {
            frame: annotations for frame, annotations in annotations_by_frame.items()
            if frame == 1 or frame % sample_interval == 0
        }
        annotations_by_frame = filtered_frames
    
    if not annotations_by_frame:
        return 0, None
    
    processed_count = 0
    sample_info = None  # Store sample for visualization
    
    # Process each frame
    for frame_num, annotations in sorted(annotations_by_frame.items()):
        # Find corresponding image
        img_path = find_image_path(sequence_name, frame_num, images_base_path)
        if img_path is None:
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception:
            continue
        
        # Note: Labels are for 3840x2160, but images are 1280x720
        # Scale factor: 1280/3840 = 1/3, 720/2160 = 1/3
        scale_x = img_width / 3840.0
        scale_y = img_height / 2160.0
        
        # Create unique filename
        yolo_img_name = f"{sequence_name}_{frame_num:06d}.jpg"
        yolo_label_name = f"{sequence_name}_{frame_num:06d}.txt"
        
        # Copy image
        output_img_path = output_path / split / 'images' / yolo_img_name
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, output_img_path)
        
        # Convert annotations to YOLO format and collect boxes for visualization
        yolo_lines = []
        boxes_for_viz = []  # Store boxes in absolute coordinates for visualization
        
        for anno in annotations:
            # Scale coordinates from 3840x2160 to image size
            xmin = anno['xmin'] * scale_x
            ymin = anno['ymin'] * scale_y
            xmax = anno['xmax'] * scale_x
            ymax = anno['ymax'] * scale_y
            
            # Store for visualization
            boxes_for_viz.append({
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            })
            
            center_x, center_y, width, height = convert_bbox_to_yolo(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        if yolo_lines:
            output_label_path = output_path / split / 'labels' / yolo_label_name
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            processed_count += 1
            
            # Store sample info for visualization (store first processed frame)
            if collect_samples and sample_info is None and boxes_for_viz:
                sample_info = {
                    'img_path': output_img_path,  # Use output path for already converted images
                    'boxes': boxes_for_viz,
                    'split': split
                }
    
    return processed_count, sample_info

def process_split(split_name, datasets_dir, output_path, sample_interval=None, collect_samples=False, has_separate_valid=False):
    """Process train, valid, or test split.
    
    Args:
        split_name: Name of the split (train/test/valid)
        datasets_dir: Base datasets directory
        output_path: Output directory
        sample_interval: Frame sampling interval (e.g., 15 = frames 1, 15, 30, ...)
        collect_samples: If True, collect sample info for visualization
        has_separate_valid: If True, ValidSetFrames exists separately (don't split train)
    """
    # Try to find the dataset structure
    # Option 1: Okutama-Action-MOT/TrainSetFrames/TestSetFrames structure
    okutama_action_dir = datasets_dir / 'Okutama-Action-MOT'
    if okutama_action_dir.exists():
        if split_name == 'train':
            frames_dir = okutama_action_dir / 'TrainSetFrames'
        elif split_name == 'valid':
            frames_dir = okutama_action_dir / 'ValidSetFrames'
        elif split_name == 'test':
            frames_dir = okutama_action_dir / 'TestSetFrames'
        else:
            return 0, {}
    # Option 2: TrainSetFrames/TestSetFrames directly in datasets/
    elif split_name == 'train':
        frames_dir = datasets_dir / 'TrainSetFrames'
    elif split_name == 'test':
        frames_dir = datasets_dir / 'TestSetFrames'
    elif split_name == 'valid':
        frames_dir = datasets_dir / 'ValidSetFrames'
    else:
        return 0, {}
    
    # Option 3: Direct okutama/ structure (if TrainSetFrames/TestSetFrames not found)
    if not frames_dir.exists():
        # Check if we have direct okutama structure
        okutama_dir = datasets_dir / 'okutama'
        if okutama_dir.exists() and (okutama_dir / 'Drone1').exists():
            frames_dir = okutama_dir
            print(f"  Using direct okutama structure for {split_name}")
        else:
            print(f"  {split_name} set not found, skipping...")
            return 0, {}
    
    # Find labels directory
    labels_dir = frames_dir / 'Labels' / 'SingleActionTrackingLabels' / '3840x2160'
    if not labels_dir.exists():
        # Try alternative label paths
        labels_dir = frames_dir / 'Labels' / '3840x2160'
    if not labels_dir.exists():
        labels_dir = frames_dir / 'Labels'
    
    images_base = frames_dir
    
    if not labels_dir.exists():
        print(f"  Labels directory not found for {split_name}, skipping...")
        return 0, {}
    
    # Get all label files
    label_files = sorted(labels_dir.glob('*.txt'))
    print(f"  Found {len(label_files)} sequences in {split_name} set")
    if sample_interval and sample_interval > 1:
        print(f"  Using frame sampling interval: {sample_interval} (frames 1, {sample_interval}, {2*sample_interval}, {3*sample_interval}, ...)")
    
    # For train split, check if ValidSetFrames exists separately
    # If ValidSetFrames exists separately, don't split train into train/valid
    if split_name == 'train':
        if has_separate_valid:
            # ValidSetFrames exists separately, so process train as-is
            train_count = 0
            train_sample = None
            for label_file in tqdm(label_files, desc="  Converting train"):
                seq_name = label_file.stem
                count, sample = process_sequence(seq_name, label_file, images_base, output_path, 'train', sample_interval, collect_samples)
                train_count += count
                if sample and train_sample is None:
                    train_sample = sample
            
            print(f"  Train: {train_count} images")
            samples_dict = {}
            if train_sample:
                samples_dict['train'] = train_sample
            return train_count, samples_dict
        else:
            # No separate ValidSetFrames, split train into train/valid (80/20)
            label_files_list = list(label_files)
            random.shuffle(label_files_list)
            
            split_idx = int(len(label_files_list) * 0.8)
            train_files = label_files_list[:split_idx]
            valid_files = label_files_list[split_idx:]
            
            # Process train files
            train_count = 0
            train_sample = None
            for label_file in tqdm(train_files, desc="  Converting train"):
                seq_name = label_file.stem
                count, sample = process_sequence(seq_name, label_file, images_base, output_path, 'train', sample_interval, collect_samples)
                train_count += count
                if sample and train_sample is None:
                    train_sample = sample
            
            # Process valid files
            valid_count = 0
            valid_sample = None
            for label_file in tqdm(valid_files, desc="  Converting valid"):
                seq_name = label_file.stem
                count, sample = process_sequence(seq_name, label_file, images_base, output_path, 'valid', sample_interval, collect_samples)
                valid_count += count
                if sample and valid_sample is None:
                    valid_sample = sample
            
            print(f"  Train: {train_count} images, Valid: {valid_count} images")
            samples_dict = {}
            if train_sample:
                samples_dict['train'] = train_sample
            if valid_sample:
                samples_dict['valid'] = valid_sample
            return train_count + valid_count, samples_dict
    else:
        # Process test/valid files (depending on split_name)
        split_count = 0
        split_sample = None
        for label_file in tqdm(label_files, desc=f"  Converting {split_name}"):
            seq_name = label_file.stem
            count, sample = process_sequence(seq_name, label_file, images_base, output_path, split_name, sample_interval, collect_samples)
            split_count += count
            if sample and split_sample is None:
                split_sample = sample
        
        print(f"  {split_name.capitalize()}: {split_count} images")
        samples_dict = {}
        if split_sample:
            samples_dict[split_name] = split_sample
        return split_count, samples_dict

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
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class center_x center_y width height
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert back to absolute coordinates (xmin, ymin, xmax, ymax)
                    bbox_width = width * img_width
                    bbox_height = height * img_height
                    bbox_left = center_x * img_width - bbox_width / 2.0
                    bbox_top = center_y * img_height - bbox_height / 2.0
                    
                    boxes.append({
                        'xmin': int(bbox_left),
                        'ymin': int(bbox_top),
                        'xmax': int(bbox_left + bbox_width),
                        'ymax': int(bbox_top + bbox_height)
                    })
        
        if boxes:
            samples_dict[split_key] = {
                'img_path': img_path,
                'boxes': boxes,
                'split': split_name
            }
    
    return samples_dict

def visualize_samples(samples_dict):
    """Visualize one sample image from each split with annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Okutama Action Sample Images with Person Annotations', fontsize=16, fontweight='bold')
    
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
        
        # Draw bounding boxes (Okutama uses xmin, ymin, xmax, ymax format)
        for box in sample['boxes']:
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            
            width = xmax - xmin
            height = ymax - ymin
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
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
    parser = argparse.ArgumentParser(description='Convert Okutama Action dataset to YOLO format')
    parser.add_argument('--sample_interval', type=int, default=15,
                       help='Frame sampling interval (e.g., 15 = take frames 1, 15, 30, 45, ...). If not specified, all frames are processed.')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Show sample images with annotations from each split (default: False)')
    args = parser.parse_args()
    
    # Script runs from datasets/ directory
    datasets_dir = Path('.')
    output_path = datasets_dir / 'okutama'
    
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
            yaml_content = """# YOLOv8 dataset configuration for Okutama Action (person class only)
path: datasets/okutama  # dataset root dir
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
            visualize_samples(samples_dict)
        
        return
    
    print("Converting Okutama Action dataset to YOLO format...")
    print(f"Output: {output_path}")
    if args.sample_interval and args.sample_interval > 1:
        print(f"Frame sampling interval: {args.sample_interval} (frames 1, {args.sample_interval}, {2*args.sample_interval}, {3*args.sample_interval}, ...)")
    if args.visualize:
        print("Visualization mode: Will show sample images after conversion")
    
    # Check for different dataset structures
    # Option 1: Okutama-Action-MOT/ structure (preferred)
    okutama_action_dir = datasets_dir / 'Okutama-Action-MOT'
    has_okutama_action = okutama_action_dir.exists() and (
        (okutama_action_dir / 'TrainSetFrames').exists() or 
        (okutama_action_dir / 'TestSetFrames').exists() or
        (okutama_action_dir / 'ValidSetFrames').exists()
    )
    
    # Option 2: Direct TrainSetFrames/TestSetFrames
    has_train_test = (datasets_dir / 'TrainSetFrames').exists() or (datasets_dir / 'TestSetFrames').exists()
    
    # Option 3: Direct okutama/ structure
    has_okutama_raw = (datasets_dir / 'okutama' / 'Drone1').exists() and (datasets_dir / 'okutama' / 'Labels').exists()
    
    all_samples_dict = {}  # Collect samples from all splits
    
    if has_okutama_action:
        # Check if ValidSetFrames exists separately
        has_separate_valid_set = (okutama_action_dir / 'ValidSetFrames').exists()
        
        # Process train set (will split into train/valid if ValidSetFrames doesn't exist)
        print("\nProcessing train set...")
        train_total, train_samples = process_split('train', datasets_dir, output_path, args.sample_interval, args.visualize, has_separate_valid_set)
        if train_samples:
            all_samples_dict.update(train_samples)
        
        # Process valid set if it exists separately
        if has_separate_valid_set:
            print("\nProcessing valid set...")
            valid_total, valid_samples = process_split('valid', datasets_dir, output_path, args.sample_interval, args.visualize, False)
            if valid_samples:
                all_samples_dict.update(valid_samples)
        
        # Process test set (no sampling - use all frames for reliable evaluation)
        print("\nProcessing test set (no sampling - using all frames)...")
        test_total, test_samples = process_split('test', datasets_dir, output_path, None, args.visualize, False)
        if test_samples:
            all_samples_dict.update(test_samples)
    elif has_train_test:
        # Check if ValidSetFrames exists
        has_separate_valid_set = (datasets_dir / 'ValidSetFrames').exists()
        
        # Process train set (will split into train/valid if ValidSetFrames doesn't exist)
        print("\nProcessing train set...")
        train_total, train_samples = process_split('train', datasets_dir, output_path, args.sample_interval, args.visualize, has_separate_valid_set)
        if train_samples:
            all_samples_dict.update(train_samples)
        
        # Process valid set if it exists separately
        if has_separate_valid_set:
            print("\nProcessing valid set...")
            valid_total, valid_samples = process_split('valid', datasets_dir, output_path, args.sample_interval, args.visualize, False)
            if valid_samples:
                all_samples_dict.update(valid_samples)
        
        # Process test set (no sampling - use all frames for reliable evaluation)
        print("\nProcessing test set (no sampling - using all frames)...")
        test_total, test_samples = process_split('test', datasets_dir, output_path, None, args.visualize, False)
        if test_samples:
            all_samples_dict.update(test_samples)
    elif has_okutama_raw:
        # All data is in okutama/, need to process and split
        print("\nProcessing combined dataset (will split into train/valid/test)...")
        okutama_dir = datasets_dir / 'okutama'
        labels_dir = okutama_dir / 'Labels' / 'SingleActionTrackingLabels' / '3840x2160'
        if not labels_dir.exists():
            labels_dir = okutama_dir / 'Labels' / '3840x2160'
        if not labels_dir.exists():
            labels_dir = okutama_dir / 'Labels'
        
        if not labels_dir.exists():
            print("  Error: Labels directory not found")
            return
        
        label_files = sorted(labels_dir.glob('*.txt'))
        print(f"  Found {len(label_files)} sequences")
        if args.sample_interval and args.sample_interval > 1:
            print(f"  Using frame sampling interval: {args.sample_interval} (frames 1, {args.sample_interval}, {2*args.sample_interval}, {3*args.sample_interval}, ...)")
            print(f"  NOTE: Test set will use ALL frames (no sampling) for reliable evaluation")
        
        # Split: 70% train, 15% valid, 15% test
        label_files_list = list(label_files)
        random.shuffle(label_files_list)
        
        train_idx = int(len(label_files_list) * 0.7)
        valid_idx = train_idx + int(len(label_files_list) * 0.15)
        
        train_files = label_files_list[:train_idx]
        valid_files = label_files_list[train_idx:valid_idx]
        test_files = label_files_list[valid_idx:]
        
        # Process train
        train_count = 0
        train_sample = None
        for label_file in tqdm(train_files, desc="  Converting train"):
            seq_name = label_file.stem
            count, sample = process_sequence(seq_name, label_file, okutama_dir, output_path, 'train', args.sample_interval, args.visualize)
            train_count += count
            if sample and train_sample is None:
                train_sample = sample
        
        # Process valid
        valid_count = 0
        valid_sample = None
        for label_file in tqdm(valid_files, desc="  Converting valid"):
            seq_name = label_file.stem
            count, sample = process_sequence(seq_name, label_file, okutama_dir, output_path, 'valid', args.sample_interval, args.visualize)
            valid_count += count
            if sample and valid_sample is None:
                valid_sample = sample
        
        # Process test (no sampling - use all frames for reliable evaluation)
        test_count = 0
        test_sample = None
        for label_file in tqdm(test_files, desc="  Converting test (no sampling)"):
            seq_name = label_file.stem
            count, sample = process_sequence(seq_name, label_file, okutama_dir, output_path, 'test', None, args.visualize)
            test_count += count
            if sample and test_sample is None:
                test_sample = sample
        
        print(f"  Train: {train_count} images, Valid: {valid_count} images, Test: {test_count} images")
        
        # Collect samples
        if train_sample:
            all_samples_dict['train'] = train_sample
        if valid_sample:
            all_samples_dict['valid'] = valid_sample
        if test_sample:
            all_samples_dict['test'] = test_sample
    else:
        print("Error: Could not find Okutama dataset structure")
        print("Expected one of:")
        print("  - Okutama-Action-MOT/TrainSetFrames/, Okutama-Action-MOT/TestSetFrames/, Okutama-Action-MOT/ValidSetFrames/")
        print("  - TrainSetFrames/ and TestSetFrames/ directories in datasets/, OR")
        print("  - okutama/Drone1/, okutama/Drone2/, okutama/Labels/ structure")
        return
    
    # Create dataset.yaml file
    yaml_path = output_path / 'dataset.yaml'
    yaml_content = """# YOLOv8 dataset configuration for Okutama Action (person class only)
path: datasets/okutama  # dataset root dir
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
    print(f"\nCreated dataset.yaml at {yaml_path}")
    
    # Remove original directories (only if they exist and are not the output directory)
    print("\nRemoving original dataset directories...")
    for old_dir in ['TrainSetFrames', 'TestSetFrames']:
        old_path = datasets_dir / old_dir
        if old_path.exists() and old_path != output_path:
            print(f"  Removing {old_dir}...")
            shutil.rmtree(old_path)
    
    # Also check if okutama has raw data that needs cleaning
    okutama_raw = datasets_dir / 'okutama'
    if okutama_raw.exists() and okutama_raw != output_path:
        # Check if it has Drone1/Drone2/Labels (raw structure)
        if (okutama_raw / 'Drone1').exists() and (okutama_raw / 'Labels').exists():
            # Check if we already have converted data
            if (output_path / 'train' / 'images').exists():
                print(f"  Removing raw okutama data (Drone1/Drone2/Labels)...")
                for item in ['Drone1', 'Drone2', 'Labels']:
                    item_path = okutama_raw / item
                    if item_path.exists():
                        shutil.rmtree(item_path)
    
    print("\nDone! Dataset structure:")
    print(f"  {output_path}/train/images/")
    print(f"  {output_path}/train/labels/")
    print(f"  {output_path}/valid/images/")
    print(f"  {output_path}/valid/labels/")
    print(f"  {output_path}/test/images/")
    print(f"  {output_path}/test/labels/")
    
    # Visualize samples if requested
    if args.visualize and all_samples_dict:
        print("\n" + "="*60)
        print("Generating visualization...")
        print("="*60)
        visualize_samples(all_samples_dict)
    elif args.visualize:
        # Try loading from already converted dataset
        print("\n" + "="*60)
        print("Loading samples from converted dataset for visualization...")
        print("="*60)
        samples_from_converted = load_samples_from_converted(output_path)
        if samples_from_converted:
            visualize_samples(samples_from_converted)
        else:
            print("No samples found for visualization")

if __name__ == '__main__':
    main()

