#!/usr/bin/env python3
"""
Combine multiple YOLO format datasets into a single dataset.
This allows training on multiple datasets using a single dataset.yaml file.
"""

import argparse
import json
import shutil
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

def copy_files_with_rename(src_dir, dst_dir, file_type='*.jpg', prefix=None, counter=0):
    """Copy files from source to destination, ensuring unique names."""
    if not src_dir.exists():
        return 0, counter
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob(file_type))
    
    copied = 0
    for src_file in files:
        # Use prefix to make filenames unique
        if prefix:
            dst_file = dst_dir / f"{prefix}_{src_file.name}"
        else:
            dst_file = dst_dir / src_file.name
        
        # If still exists, add counter
        if dst_file.exists():
            dst_file = dst_dir / f"{prefix}_{counter:06d}_{src_file.name}"
            counter += 1
        
        shutil.copy2(src_file, dst_file)
        copied += 1
    
    return copied, counter

def combine_coco_json_files(dataset_paths, output_path, split_name):
    """
    Combine COCO format JSON annotation files from multiple datasets.
    
    Args:
        dataset_paths: List of dataset paths
        output_path: Output directory for combined dataset
        split_name: Split name ('train' or 'valid')
    
    Returns:
        True if JSON files were combined, False otherwise
    """
    json_files_found = []
    
    # Collect all JSON files for this split
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        
        # Check for _annotations.coco.json in split folder
        json_paths = []
        if (dataset_path / split_name / '_annotations.coco.json').exists():
            json_paths.append(dataset_path / split_name / '_annotations.coco.json')
        elif split_name == 'valid' and (dataset_path / 'val' / '_annotations.coco.json').exists():
            json_paths.append(dataset_path / 'val' / '_annotations.coco.json')
        
        for json_path in json_paths:
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    json_files_found.append({
                        'path': json_path,
                        'data': json_data,
                        'dataset_name': dataset_path.name
                    })
            except Exception as e:
                print(f"  Warning: Could not read {json_path}: {e}")
                continue
    
    if not json_files_found:
        return False
    
    # Combine JSON files
    combined_categories = []
    combined_images = []
    combined_annotations = []
    
    # Use categories from first file (should be same across all)
    if json_files_found:
        combined_categories = json_files_found[0]['data'].get('categories', [])
    
    # Track IDs to ensure uniqueness
    image_id_offset = 0
    annotation_id_offset = 0
    
    for json_info in json_files_found:
        json_data = json_info['data']
        dataset_name = json_info['dataset_name']
        
        # Map old image IDs to new image IDs (scoped to this dataset)
        image_id_mapping = {}
        
        # Process images first to build the mapping
        for img in json_data.get('images', []):
            old_image_id = img['id']
            new_image_id = image_id_offset + old_image_id
            
            # Update file_name to match the renamed image files
            # Images are copied to output_path/split_name/images/ with dataset prefix
            # So file_name should be "images/dataset_filename.jpg"
            original_filename = img['file_name']
            # Remove 'images/' prefix if present to get just the filename
            if original_filename.startswith('images/'):
                filename_only = original_filename.replace('images/', '')
            else:
                filename_only = original_filename
            
            # Add dataset prefix to match the renamed files, and keep images/ prefix
            new_filename = f"images/{dataset_name}_{filename_only}"
            
            new_image = {
                'id': new_image_id,
                'file_name': new_filename,
                'height': img['height'],
                'width': img['width']
            }
            combined_images.append(new_image)
            
            # Store mapping for annotations
            image_id_mapping[old_image_id] = new_image_id
        
        # Process annotations using the mapping
        for ann in json_data.get('annotations', []):
            old_image_id = ann['image_id']
            if old_image_id in image_id_mapping:
                new_annotation = {
                    'id': annotation_id_offset + ann['id'],
                    'image_id': image_id_mapping[old_image_id],
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann.get('iscrowd', 0),
                    'ignore': ann.get('ignore', 0),
                    'segmentation': ann.get('segmentation', [])
                }
                combined_annotations.append(new_annotation)
        
        # Update offsets for next dataset
        max_image_id = max([img['id'] for img in json_data.get('images', [])], default=-1)
        max_annotation_id = max([ann['id'] for ann in json_data.get('annotations', [])], default=-1)
        image_id_offset += max_image_id + 1
        annotation_id_offset += max_annotation_id + 1
    
    # Create combined JSON
    combined_json = {
        'categories': combined_categories,
        'images': combined_images,
        'annotations': combined_annotations
    }
    
    # Save combined JSON
    output_json_path = output_path / split_name / '_annotations.coco.json'
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(combined_json, f, indent=2)
    
    print(f"  {split_name.upper()}: Combined {len(json_files_found)} JSON files -> {len(combined_images)} images, {len(combined_annotations)} annotations")
    
    return True

def combine_yolo_datasets(dataset_paths, output_path):
    """
    Combine multiple YOLO format datasets into one.
    
    Args:
        dataset_paths: List of paths to YOLO datasets
        output_path: Output directory for combined dataset
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"Combining {len(dataset_paths)} YOLO Datasets")
    print("="*70)
    
    # Verify all datasets exist
    valid_datasets = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            continue
        
        # Check for YOLO structure
        has_train = (dataset_path / 'train' / 'images').exists()
        has_valid = (dataset_path / 'valid' / 'images').exists() or (dataset_path / 'val' / 'images').exists()
        
        if not (has_train or has_valid):
            print(f"⚠️  Invalid YOLO structure in: {dataset_path}")
            print(f"    Expected: {dataset_path}/{{train,valid}}/{{images,labels}}/")
            continue
        
        valid_datasets.append(dataset_path)
        print(f"✓ Found dataset: {dataset_path}")
    
    if not valid_datasets:
        print("\n⚠️  No valid datasets found to combine")
        return False
    
    print(f"\nCombining {len(valid_datasets)} datasets...")
    
    # Process each split (skip test sets)
    splits = ['train', 'valid', 'val']
    total_stats = {}
    counter = 0
    
    for split in splits:
        split_name = 'valid' if split == 'val' else split  # Normalize val -> valid
        total_images = 0
        total_labels = 0
        
        for idx, dataset_path in enumerate(valid_datasets):
            dataset_name_part = dataset_path.name or f"dataset_{idx}"
            
            # Check for both 'valid' and 'val' directories
            split_dirs = []
            if (dataset_path / split / 'images').exists():
                split_dirs.append(split)
            elif split == 'valid' and (dataset_path / 'val' / 'images').exists():
                split_dirs.append('val')
            
            for actual_split in split_dirs:
                img_dir = dataset_path / actual_split / 'images'
                label_dir = dataset_path / actual_split / 'labels'
                
                if not img_dir.exists():
                    continue
                
                output_img_dir = output_path / split_name / 'images'
                output_label_dir = output_path / split_name / 'labels'
                
                # Copy images and labels with dataset prefix
                prefix = dataset_name_part
                img_count, counter = copy_files_with_rename(
                    img_dir, output_img_dir, '*.jpg', prefix, counter
                )
                label_count, counter = copy_files_with_rename(
                    label_dir, output_label_dir, '*.txt', prefix, counter
                )
                
                total_images += img_count
                total_labels += label_count
                
                if img_count > 0:
                    print(f"  {split_name.upper()}: Added {img_count} images from {dataset_name_part}")
        
        if total_images > 0:
            total_stats[split_name] = {'images': total_images, 'labels': total_labels}
            
            # Combine COCO JSON files for this split
            print(f"\n  Combining COCO JSON annotations for {split_name}...")
            combine_coco_json_files(valid_datasets, output_path, split_name)
    
    if not total_stats:
        print("\n⚠️  No data was combined")
        return False
    
    # Create dataset.yaml
    # Try to load class names from first dataset
    class_names = {0: 'person'}  # Default
    nc = 1
    
    first_dataset = valid_datasets[0]
    yaml_path = first_dataset / 'dataset.yaml'
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                content = f.read()
                if 'names:' in content:
                    in_names = False
                    for line in content.split('\n'):
                        if 'names:' in line:
                            in_names = True
                            continue
                        if in_names and ':' in line and not line.strip().startswith('#'):
                            parts = line.split(':')
                            if len(parts) == 2:
                                try:
                                    class_id = int(parts[0].strip())
                                    class_name = parts[1].strip()
                                    class_names[class_id] = class_name
                                except ValueError:
                                    pass
                        elif in_names and line.strip() and not line.strip().startswith('#'):
                            if not line.strip().startswith(' ') and ':' in line:
                                break
                if 'nc:' in content:
                    for line in content.split('\n'):
                        if 'nc:' in line and not line.strip().startswith('#'):
                            try:
                                nc = int(line.split('nc:')[1].strip().split()[0])
                            except:
                                pass
        except:
            pass
    
    # Write combined dataset.yaml
    output_yaml_path = output_path / 'dataset.yaml'
    
    # Build names section
    names_lines = []
    for class_id in sorted(class_names.keys()):
        names_lines.append(f"  {class_id}: {class_names[class_id]}")
    
    yaml_content = f"""# YOLOv8 dataset configuration for Combined Dataset
# Combined from: {', '.join([d.name for d in valid_datasets])}
path: datasets/{output_path.name}  # dataset root dir
"""
    
    if 'train' in total_stats:
        yaml_content += "train: train/images  # train images (relative to 'path')\n"
    if 'valid' in total_stats:
        yaml_content += "val: valid/images  # val images (relative to 'path')\n"
    elif 'val' in total_stats:
        yaml_content += "val: val/images  # val images (relative to 'path')\n"
    
    yaml_content += f"""
# Classes
names:
{chr(10).join(names_lines)}

# Number of classes
nc: {nc}
"""
    
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created dataset.yaml at {output_yaml_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Combined Dataset Summary")
    print("="*70)
    
    for split_name, stats in sorted(total_stats.items()):
        print(f"{split_name.upper()}: {stats['images']} images, {stats['labels']} labels")
    
    total_imgs = sum(s['images'] for s in total_stats.values())
    total_lbls = sum(s['labels'] for s in total_stats.values())
    print(f"\nTOTAL: {total_imgs} images, {total_lbls} labels")
    
    print(f"\n✓ Combined dataset created at: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple YOLO format datasets into one for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine mot20 and visdrone-det datasets
  python combine_yolo_datasets.py --datasets mot20 visdrone-det --output combined
  
  # Combine multiple datasets
  python combine_yolo_datasets.py --datasets mot20 visdrone-det okutama --output all_datasets
        """
    )
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='List of dataset directory paths to combine')
    parser.add_argument('--output', type=str, default='combined',
                       help='Output directory for combined dataset (default: combined)')
    args = parser.parse_args()
    
    combine_yolo_datasets(args.datasets, args.output)

if __name__ == '__main__':
    main()
