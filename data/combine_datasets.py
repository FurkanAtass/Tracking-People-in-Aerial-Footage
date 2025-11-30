#!/usr/bin/env python3
"""
Combine VisDrone and Okutama datasets into a single YOLO format dataset.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def copy_files(src_dir, dst_dir, file_type='*.jpg'):
    """Copy files from source to destination, renaming if conflicts."""
    if not src_dir.exists():
        return 0
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob(file_type))
    
    copied = 0
    for src_file in tqdm(files, desc=f"  Copying {file_type}", leave=False):
        dst_file = dst_dir / src_file.name
        
        # If file exists, add prefix to avoid conflicts
        if dst_file.exists():
            # Add dataset prefix
            if 'visdrone' in str(src_dir).lower():
                prefix = 'visdrone_'
            elif 'okutama' in str(src_dir).lower():
                prefix = 'okutama_'
            else:
                prefix = 'combined_'
            
            dst_file = dst_dir / f"{prefix}{src_file.name}"
        
        shutil.copy2(src_file, dst_file)
        copied += 1
    
    return copied

def combine_datasets():
    """Combine VisDrone and Okutama datasets."""
    datasets_dir = Path('.')  # Script runs from datasets/ directory
    output_path = datasets_dir / 'combined'
    
    visdrone_path = datasets_dir / 'visdrone'
    okutama_path = datasets_dir / 'okutama'
    
    print("="*60)
    print("Combining VisDrone and Okutama datasets")
    print("="*60)
    
    # Check if datasets exist
    if not visdrone_path.exists():
        print(f"⚠️  VisDrone dataset not found at {visdrone_path}")
        return
    
    if not okutama_path.exists():
        print(f"⚠️  Okutama dataset not found at {okutama_path}")
        return
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # VisDrone
        visdrone_img_dir = visdrone_path / split / 'images'
        visdrone_label_dir = visdrone_path / split / 'labels'
        
        if visdrone_img_dir.exists():
            print(f"  Adding VisDrone {split}...")
            output_img_dir = output_path / split / 'images'
            output_label_dir = output_path / split / 'labels'
            
            img_count = copy_files(visdrone_img_dir, output_img_dir, '*.jpg')
            label_count = copy_files(visdrone_label_dir, output_label_dir, '*.txt')
            print(f"    Images: {img_count}, Labels: {label_count}")
        
        # Okutama
        okutama_img_dir = okutama_path / split / 'images'
        okutama_label_dir = okutama_path / split / 'labels'
        
        if okutama_img_dir.exists():
            print(f"  Adding Okutama {split}...")
            output_img_dir = output_path / split / 'images'
            output_label_dir = output_path / split / 'labels'
            
            img_count = copy_files(okutama_img_dir, output_img_dir, '*.jpg')
            label_count = copy_files(okutama_label_dir, output_label_dir, '*.txt')
            print(f"    Images: {img_count}, Labels: {label_count}")
    
    # Create dataset.yaml
    yaml_path = output_path / 'dataset.yaml'
    yaml_content = """# YOLOv8 dataset configuration for Combined VisDrone + Okutama (person class only)
path: datasets/combined  # dataset root dir
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
    
    print(f"\n✓ Created dataset.yaml at {yaml_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Combined Dataset Summary")
    print("="*60)
    
    for split in splits:
        split_path = output_path / split
        if split_path.exists():
            img_count = len(list((split_path / 'images').glob('*.jpg')))
            label_count = len(list((split_path / 'labels').glob('*.txt')))
            print(f"{split.upper()}: {img_count} images, {label_count} labels")
    
    print(f"\n✓ Combined dataset created at: {output_path}")
    print(f"  Dataset YAML: {yaml_path}")

if __name__ == '__main__':
    combine_datasets()

