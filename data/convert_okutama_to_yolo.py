import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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

def process_sequence(sequence_name, label_path, images_base_path, output_path, split):
    """Process a single sequence."""
    if not label_path.exists():
        return 0
    
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
        return 0
    
    processed_count = 0
    
    # Process each frame
    for frame_num, annotations in annotations_by_frame.items():
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
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for anno in annotations:
            # Scale coordinates from 3840x2160 to image size
            xmin = anno['xmin'] * scale_x
            ymin = anno['ymin'] * scale_y
            xmax = anno['xmax'] * scale_x
            ymax = anno['ymax'] * scale_y
            
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
    
    return processed_count

def process_split(split_name, datasets_dir, output_path):
    """Process train or test split."""
    # Try to find the dataset structure
    # Option 1: TrainSetFrames/TestSetFrames structure
    if split_name == 'train':
        frames_dir = datasets_dir / 'TrainSetFrames'
    elif split_name == 'test':
        frames_dir = datasets_dir / 'TestSetFrames'
    else:
        return 0
    
    # Option 2: Direct okutama/ structure (if TrainSetFrames/TestSetFrames not found)
    if not frames_dir.exists():
        # Check if we have direct okutama structure
        okutama_dir = datasets_dir / 'okutama'
        if okutama_dir.exists() and (okutama_dir / 'Drone1').exists():
            frames_dir = okutama_dir
            print(f"  Using direct okutama structure for {split_name}")
        else:
            print(f"  {split_name} set not found, skipping...")
            return 0
    
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
        return 0
    
    # Get all label files
    label_files = sorted(labels_dir.glob('*.txt'))
    print(f"  Found {len(label_files)} sequences in {split_name} set")
    
    # For train split, split into train/valid
    if split_name == 'train':
        # Shuffle with seed
        label_files_list = list(label_files)
        random.shuffle(label_files_list)
        
        # 80/20 split
        split_idx = int(len(label_files_list) * 0.8)
        train_files = label_files_list[:split_idx]
        valid_files = label_files_list[split_idx:]
        
        # Process train files
        train_count = 0
        for label_file in tqdm(train_files, desc="  Converting train"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, images_base, output_path, 'train')
            train_count += count
        
        # Process valid files
        valid_count = 0
        for label_file in tqdm(valid_files, desc="  Converting valid"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, images_base, output_path, 'valid')
            valid_count += count
        
        print(f"  Train: {train_count} images, Valid: {valid_count} images")
        return train_count + valid_count
    else:
        # Process test files
        test_count = 0
        for label_file in tqdm(label_files, desc=f"  Converting {split_name}"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, images_base, output_path, 'test')
            test_count += count
        
        print(f"  Test: {test_count} images")
        return test_count

def main():
    # Script runs from datasets/ directory
    datasets_dir = Path('.')
    output_path = datasets_dir / 'okutama'
    
    print("Converting Okutama Action dataset to YOLO format...")
    print(f"Output: {output_path}")
    
    # Check if we have separate TrainSetFrames/TestSetFrames or combined okutama structure
    has_train_test = (datasets_dir / 'TrainSetFrames').exists() and (datasets_dir / 'TestSetFrames').exists()
    has_okutama_raw = (datasets_dir / 'okutama' / 'Drone1').exists() and (datasets_dir / 'okutama' / 'Labels').exists()
    
    if has_train_test:
        # Process train set (will split into train/valid)
        print("\nProcessing train set...")
        train_total = process_split('train', datasets_dir, output_path)
        
        # Process test set
        print("\nProcessing test set...")
        test_total = process_split('test', datasets_dir, output_path)
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
        for label_file in tqdm(train_files, desc="  Converting train"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, okutama_dir, output_path, 'train')
            train_count += count
        
        # Process valid
        valid_count = 0
        for label_file in tqdm(valid_files, desc="  Converting valid"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, okutama_dir, output_path, 'valid')
            valid_count += count
        
        # Process test
        test_count = 0
        for label_file in tqdm(test_files, desc="  Converting test"):
            seq_name = label_file.stem
            count = process_sequence(seq_name, label_file, okutama_dir, output_path, 'test')
            test_count += count
        
        print(f"  Train: {train_count} images, Valid: {valid_count} images, Test: {test_count} images")
    else:
        print("Error: Could not find Okutama dataset structure")
        print("Expected either:")
        print("  - TrainSetFrames/ and TestSetFrames/ directories, OR")
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

if __name__ == '__main__':
    main()

