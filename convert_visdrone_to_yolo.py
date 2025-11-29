import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# VisDrone class mapping (category column index 7)
# 0: ignored regions
# 1: pedestrian
# 2: people
# 3-11: other classes (filtered out)

# YOLO class mapping (pedestrian and people combined into one class)
# 0: person (pedestrian + people)
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian -> class 0
    2: 0,  # people -> class 0 (same as pedestrian)
}

def convert_bbox_to_yolo(bbox_left, bbox_top, bbox_width, bbox_height, img_width, img_height):
    """Convert VisDrone bbox format to YOLO normalized format.
    
    VisDrone: (left, top, width, height) in pixels
    YOLO: (center_x, center_y, width, height) normalized 0-1
    """
    # Calculate center coordinates
    center_x = (bbox_left + bbox_width / 2.0) / img_width
    center_y = (bbox_top + bbox_height / 2.0) / img_height
    
    # Normalize width and height
    width = bbox_width / img_width
    height = bbox_height / img_height
    
    # Clamp values to [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def parse_visdrone_annotation(line):
    """Parse a line from VisDrone annotation file.
    
    Format: <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
    """
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None
    
    try:
        frame_id = int(parts[0])
        target_id = int(parts[1])
        bbox_left = int(parts[2])
        bbox_top = int(parts[3])
        bbox_width = int(parts[4])
        bbox_height = int(parts[5])
        score = float(parts[6])
        category = int(parts[7])
        truncation = int(parts[8]) if len(parts) > 8 else 0
        occlusion = int(parts[9]) if len(parts) > 9 else 0
        
        return {
            'frame_id': frame_id,
            'target_id': target_id,
            'bbox_left': bbox_left,
            'bbox_top': bbox_top,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'category': category,
            'truncation': truncation,
            'occlusion': occlusion
        }
    except (ValueError, IndexError):
        return None

def process_sequence(seq_name, visdrone_path, output_path, split='train'):
    """Process a single sequence from VisDrone dataset."""
    seq_path = visdrone_path / 'sequences' / seq_name
    anno_path = visdrone_path / 'annotations' / f'{seq_name}.txt'
    
    if not seq_path.exists() or not anno_path.exists():
        print(f"Warning: Missing sequence {seq_name}")
        return 0
    
    # Read annotations and group by frame_id
    annotations_by_frame = {}
    with open(anno_path, 'r') as f:
        for line in f:
            anno = parse_visdrone_annotation(line)
            if anno is None:
                continue
            
            # Filter for pedestrian (1) and people (2) only
            if anno['category'] not in VISDRONE_TO_YOLO:
                continue
            
            frame_id = anno['frame_id']
            if frame_id not in annotations_by_frame:
                annotations_by_frame[frame_id] = []
            annotations_by_frame[frame_id].append(anno)
    
    # Process images
    image_files = sorted(seq_path.glob('*.jpg'))
    processed_count = 0
    
    for img_file in image_files:
        # Extract frame number from filename (e.g., 0000001.jpg -> 1)
        try:
            frame_id = int(img_file.stem)
        except ValueError:
            continue
        
        # Skip if no annotations for this frame
        if frame_id not in annotations_by_frame:
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            continue
        
        # Create unique filename for YOLO dataset
        yolo_img_name = f"{seq_name}_{frame_id:07d}.jpg"
        yolo_label_name = f"{seq_name}_{frame_id:07d}.txt"
        
        # Copy image
        output_img_path = output_path / 'images' / split / yolo_img_name
        shutil.copy2(img_file, output_img_path)
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for anno in annotations_by_frame[frame_id]:
            center_x, center_y, width, height = convert_bbox_to_yolo(
                anno['bbox_left'],
                anno['bbox_top'],
                anno['bbox_width'],
                anno['bbox_height'],
                img_width,
                img_height
            )
            
            # Map VisDrone class to YOLO class
            yolo_class = VISDRONE_TO_YOLO[anno['category']]
            
            yolo_lines.append(f"{yolo_class} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Write YOLO annotation file
        if yolo_lines:
            output_label_path = output_path / 'labels' / split / yolo_label_name
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            processed_count += 1
    
    return processed_count

def main():
    # Paths
    visdrone_path = Path('datasets/VisDrone2019-MOT-train')
    output_path = Path('datasets/yolo_visdrone')
    
    if not visdrone_path.exists():
        print(f"Error: VisDrone dataset not found at {visdrone_path}")
        print("Please run download_datasets.sh first")
        return
    
    # Create output directories
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    # Get all sequences
    sequences_path = visdrone_path / 'sequences'
    if not sequences_path.exists():
        print(f"Error: Sequences directory not found at {sequences_path}")
        return
    
    sequences = [d.name for d in sequences_path.iterdir() if d.is_dir()]
    sequences.sort()
    
    print(f"Found {len(sequences)} sequences")
    print("Converting VisDrone to YOLOv8 format (pedestrian and people combined as one class)...")
    print(f"Output directory: {output_path}")
    
    total_processed = 0
    for seq_name in tqdm(sequences, desc="Processing sequences"):
        count = process_sequence(seq_name, visdrone_path, output_path, split='train')
        total_processed += count
    
    print(f"\nConversion complete!")
    print(f"Processed {total_processed} images")
    print(f"Output structure:")
    print(f"  {output_path / 'images' / 'train'} - {len(list((output_path / 'images' / 'train').glob('*.jpg')))} images")
    print(f"  {output_path / 'labels' / 'train'} - {len(list((output_path / 'labels' / 'train').glob('*.txt')))} labels")
    print(f"\nYOLO class mapping:")
    print(f"  0: person (pedestrian + people)")

if __name__ == '__main__':
    main()

