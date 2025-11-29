import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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

def parse_visdrone_annotation(line):
    """Parse VisDrone annotation line."""
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None
    try:
        return {
            'frame_id': int(parts[0]),
            'bbox_left': int(parts[2]),
            'bbox_top': int(parts[3]),
            'bbox_width': int(parts[4]),
            'bbox_height': int(parts[5]),
            'category': int(parts[7])
        }
    except (ValueError, IndexError):
        return None

def process_sequence(seq_name, visdrone_path, output_path, split):
    """Process a single sequence."""
    seq_path = visdrone_path / 'sequences' / seq_name
    anno_path = visdrone_path / 'annotations' / f'{seq_name}.txt'
    
    if not seq_path.exists() or not anno_path.exists():
        return 0
    
    # Read annotations
    annotations_by_frame = {}
    with open(anno_path, 'r') as f:
        for line in f:
            anno = parse_visdrone_annotation(line)
            if anno and anno['category'] in VISDRONE_TO_YOLO:
                frame_id = anno['frame_id']
                if frame_id not in annotations_by_frame:
                    annotations_by_frame[frame_id] = []
                annotations_by_frame[frame_id].append(anno)
    
    # Process images
    processed_count = 0
    for img_file in sorted(seq_path.glob('*.jpg')):
        try:
            frame_id = int(img_file.stem)
        except ValueError:
            continue
        
        if frame_id not in annotations_by_frame:
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception:
            continue
        
        # Copy image
        yolo_img_name = f"{seq_name}_{frame_id:07d}.jpg"
        output_img_path = output_path / split / 'images' / yolo_img_name
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, output_img_path)
        
        # Convert annotations
        yolo_lines = []
        for anno in annotations_by_frame[frame_id]:
            center_x, center_y, width, height = convert_bbox_to_yolo(
                anno['bbox_left'], anno['bbox_top'],
                anno['bbox_width'], anno['bbox_height'],
                img_width, img_height
            )
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        if yolo_lines:
            output_label_path = output_path / split / 'labels' / f"{seq_name}_{frame_id:07d}.txt"
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            processed_count += 1
    
    return processed_count

def main():
    # Script runs from datasets/ directory, so look in current directory
    datasets_dir = Path('.')
    output_path = datasets_dir / 'visdrone'
    
    # Find VisDrone directory (train, val, or test-dev)
    visdrone_dirs = list(datasets_dir.glob('VisDrone2019-MOT-*'))
    if not visdrone_dirs:
        print("No VisDrone dataset found in current directory")
        return
    
    for visdrone_path in visdrone_dirs:
        if not visdrone_path.is_dir():
            continue
        
        # Determine split name
        if 'train' in visdrone_path.name:
            split = 'train'
        elif 'val' in visdrone_path.name:
            split = 'valid'
        elif 'test' in visdrone_path.name:
            split = 'test'
        else:
            continue
        
        print(f"Processing {split} set from {visdrone_path.name}...")
        
        sequences_path = visdrone_path / 'sequences'
        if not sequences_path.exists():
            print(f"  No sequences directory found, skipping...")
            continue
        
        sequences = sorted([d.name for d in sequences_path.iterdir() if d.is_dir()])
        print(f"  Found {len(sequences)} sequences")
        
        total_processed = 0
        for seq_name in tqdm(sequences, desc=f"  Converting {split}"):
            count = process_sequence(seq_name, visdrone_path, output_path, split)
            total_processed += count
        
        print(f"  Processed {total_processed} images")
        
        # Remove original directory
        print(f"  Removing {visdrone_path.name}...")
        shutil.rmtree(visdrone_path)
    
    # Create dataset.yaml file
    yaml_path = output_path / 'dataset.yaml'
    # Path should be relative to project root (datasets/visdrone)
    yaml_path_str = 'datasets/visdrone'
    yaml_content = f"""# YOLOv8 dataset configuration for VisDrone (person class only)
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
    print(f"\nCreated dataset.yaml at {yaml_path}")
    
    print("\nDone! Dataset structure:")
    print(f"  {output_path}/train/images/")
    print(f"  {output_path}/train/labels/")
    print(f"  {output_path}/valid/images/")
    print(f"  {output_path}/valid/labels/")
    print(f"  {output_path}/test/images/")
    print(f"  {output_path}/test/labels/")

if __name__ == '__main__':
    main()
