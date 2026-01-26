"""
Convert YOLO format dataset to COCO format JSON files.

YOLO format: class_id x_center y_center width height (all normalized 0-1)
COCO format: JSON with categories, images, and annotations
"""
import os
import json
from pathlib import Path
from PIL import Image
import yaml


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox (normalized x_center, y_center, width, height) 
    to COCO format bbox (x_min, y_min, width, height in absolute coordinates).
    
    Args:
        yolo_bbox: list [x_center, y_center, width, height] normalized 0-1
        img_width: image width in pixels
        img_height: image height in pixels
    
    Returns:
        list [x_min, y_min, width, height] in absolute coordinates
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to absolute coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert center coordinates to top-left coordinates
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    
    # Ensure coordinates are within image bounds
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    width = max(0, min(width, img_width - x_min))
    height = max(0, min(height, img_height - y_min))
    
    return [float(x_min), float(y_min), float(width), float(height)]


def convert_yolo_to_coco(dataset_root, output_dir=None):
    """
    Convert YOLO format dataset to COCO format JSON files.
    
    Args:
        dataset_root: Path to dataset root (should contain train/images, train/labels, valid/images, valid/labels)
        output_dir: Directory to save JSON files (default: dataset_root)
    """
    dataset_root = Path(dataset_root)
    if output_dir is None:
        output_dir = dataset_root
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read dataset.yaml to get class names
    yaml_path = dataset_root / "dataset.yaml"
    class_names = ["person"]  # default
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            if 'names' in yaml_data:
                if isinstance(yaml_data['names'], dict):
                    # Convert dict to list maintaining order
                    names_dict = yaml_data['names']
                    max_id = max(names_dict.keys()) if names_dict else -1
                    class_names = [names_dict.get(i, f"class_{i}") for i in range(max_id + 1)]
                elif isinstance(yaml_data['names'], list):
                    class_names = yaml_data['names']
    
    # Create categories (COCO format uses 0-indexed, but 0 is reserved for background in some frameworks)
    # DETR uses 0-indexed classes, so we use 0, 1, 2, ...
    categories = [{"supercategory": "none", "name": name, "id": idx} 
                  for idx, name in enumerate(class_names)]
    
    phases = ["train", "valid", "test"]
    
    for phase in phases:
        images_dir = dataset_root / phase / "images"
        labels_dir = dataset_root / phase / "labels"
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {phase}")
            continue
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping {phase}")
            continue
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        image_files.sort()
        
        json_file = {
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        image_id = 0
        annotation_id = 0
        
        for img_path in image_files:
            # Read image to get dimensions
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue
            
            # Get corresponding label file
            label_path = labels_dir / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"Warning: Label file {label_path} not found, skipping image {img_path.name}")
                continue
            
            # Add image entry
            image_entry = {
                "file_name": img_path.name,
                "height": img_height,
                "width": img_width,
                "id": image_id
            }
            json_file["images"].append(image_entry)
            
            # Read YOLO annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO bbox to COCO bbox
                coco_bbox = yolo_to_coco_bbox([x_center, y_center, width, height], 
                                             img_width, img_height)
                
                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]
                
                # Skip invalid boxes
                if area <= 0 or coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                    continue
                
                # Create annotation entry
                annotation_entry = {
                    "id": annotation_id,
                    "bbox": coco_bbox,
                    "segmentation": [
                        [coco_bbox[0], coco_bbox[1],
                         coco_bbox[0] + coco_bbox[2], coco_bbox[1],
                         coco_bbox[0] + coco_bbox[2], coco_bbox[1] + coco_bbox[3],
                         coco_bbox[0], coco_bbox[1] + coco_bbox[3]]
                    ],
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": class_id,
                    "iscrowd": 0,
                    "area": area
                }
                json_file["annotations"].append(annotation_entry)
                annotation_id += 1
            
            image_id += 1
        
        # Save JSON file
        output_filename = f"{phase}.json"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(json_file, f, indent=2)
        
        print(f"Processed {len(json_file['images'])} {phase} images...")
        print(f"Created {output_path} with {len(json_file['annotations'])} annotations")
    
    print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to COCO format')
    parser.add_argument('--dataset_root', type=str, default='combined',
                       help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save JSON files (default: dataset_root)')
    
    args = parser.parse_args()
    convert_yolo_to_coco(args.dataset_root, args.output_dir)
