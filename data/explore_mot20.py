#!/usr/bin/env python3
"""
Explore and visualize MOT20 Challenge dataset.
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import configparser
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def parse_mot_annotation(line):
    """Parse MOT format annotation line.
    Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility
    """
    parts = line.strip().split(',')
    if len(parts) < 9:
        return None
    
    try:
        return {
            'frame': int(parts[0]),
            'id': int(parts[1]),
            'bb_left': float(parts[2]),
            'bb_top': float(parts[3]),
            'bb_width': float(parts[4]),
            'bb_height': float(parts[5]),
            'conf': float(parts[6]),
            'class': int(parts[7]),
            'visibility': float(parts[8])
        }
    except (ValueError, IndexError):
        return None

def analyze_sequence(seq_path, split_name):
    """Analyze a single MOT20 sequence."""
    seqinfo_path = seq_path / 'seqinfo.ini'
    gt_path = seq_path / 'gt' / 'gt.txt'
    det_path = seq_path / 'det' / 'det.txt'
    img_dir = seq_path / 'img1'
    
    if not seqinfo_path.exists():
        return None
    
    seqinfo = parse_seqinfo(seqinfo_path)
    if not seqinfo:
        return None
    
    # Count images
    if img_dir.exists():
        img_count = len(list(img_dir.glob('*.jpg')))
    else:
        img_count = 0
    
    # Parse ground truth annotations
    gt_annotations = []
    track_ids = set()
    class_ids = Counter()
    frames_with_annotations = set()
    
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                anno = parse_mot_annotation(line)
                if anno:
                    # Only count class 1 (person) - MOT20 typically uses class 1 for pedestrians
                    if anno['class'] == 1:
                        gt_annotations.append(anno)
                        track_ids.add(anno['id'])
                        class_ids[anno['class']] += 1
                        frames_with_annotations.add(anno['frame'])
    
    # Parse detections if available
    det_count = 0
    if det_path.exists():
        with open(det_path, 'r') as f:
            for line in f:
                anno = parse_mot_annotation(line)
                if anno and anno['class'] == 1:
                    det_count += 1
    
    return {
        'name': seqinfo['name'],
        'split': split_name,
        'path': seq_path,
        'frameRate': seqinfo['frameRate'],
        'seqLength': seqinfo['seqLength'],
        'imWidth': seqinfo['imWidth'],
        'imHeight': seqinfo['imHeight'],
        'img_count': img_count,
        'gt_annotations': len(gt_annotations),
        'detections': det_count,
        'unique_tracks': len(track_ids),
        'frames_with_annotations': len(frames_with_annotations),
        'class_distribution': dict(class_ids),
        'seqinfo': seqinfo,
        'gt_path': gt_path,
        'img_dir': img_dir
    }

def analyze_dataset_structure(datasets_dir):
    """Analyze MOT20 dataset structure."""
    mot20_base = datasets_dir / 'MOT20-Challenge' / 'MOT20'
    
    if not mot20_base.exists():
        return None
    
    sequences_info = []
    splits = ['train', 'test']
    
    for split in splits:
        split_dir = mot20_base / split
        if not split_dir.exists():
            continue
        
        # Find all sequences in this split
        for seq_dir in sorted(split_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.startswith('MOT20-'):
                seq_info = analyze_sequence(seq_dir, split)
                if seq_info:
                    sequences_info.append(seq_info)
    
    if not sequences_info:
        return None
    
    # Aggregate statistics
    total_annotations = sum(s['gt_annotations'] for s in sequences_info)
    total_tracks = sum(s['unique_tracks'] for s in sequences_info)
    total_images = sum(s['img_count'] for s in sequences_info)
    total_frames = sum(s['seqLength'] for s in sequences_info)
    
    # Statistics by split
    split_stats = defaultdict(lambda: {
        'sequences': 0,
        'annotations': 0,
        'tracks': 0,
        'images': 0,
        'frames': 0
    })
    
    for seq in sequences_info:
        split = seq['split']
        split_stats[split]['sequences'] += 1
        split_stats[split]['annotations'] += seq['gt_annotations']
        split_stats[split]['tracks'] += seq['unique_tracks']
        split_stats[split]['images'] += seq['img_count']
        split_stats[split]['frames'] += seq['seqLength']
    
    # Resolution distribution
    resolutions = Counter()
    for seq in sequences_info:
        res = (seq['imWidth'], seq['imHeight'])
        resolutions[res] += seq['seqLength']
    
    return {
        'sequences': sequences_info,
        'total_sequences': len(sequences_info),
        'total_annotations': total_annotations,
        'total_tracks': total_tracks,
        'total_images': total_images,
        'total_frames': total_frames,
        'split_stats': dict(split_stats),
        'resolutions': resolutions
    }

def load_annotations_for_frame(gt_path, frame_num):
    """Load all annotations for a specific frame."""
    annotations = []
    if not gt_path.exists():
        return annotations
    
    with open(gt_path, 'r') as f:
        for line in f:
            anno = parse_mot_annotation(line)
            if anno and anno['frame'] == frame_num and anno['class'] == 1:
                annotations.append(anno)
    
    return annotations

def visualize_sample_frame(seq_info):
    """Extract and visualize a sample frame from a sequence."""
    img_dir = seq_info['img_dir']
    gt_path = seq_info['gt_path']
    seqinfo = seq_info['seqinfo']
    
    if not img_dir.exists():
        return None
    
    # Get available frames
    img_files = sorted(img_dir.glob(f'*{seqinfo["imExt"]}'))
    if not img_files:
        return None
    
    # Try to find a frame with annotations
    max_attempts = 50
    for attempt in range(max_attempts):
        # Pick a random frame from the middle of the sequence (more likely to have objects)
        if len(img_files) > 100:
            start_idx = len(img_files) // 4
            end_idx = 3 * len(img_files) // 4
            img_file = random.choice(img_files[start_idx:end_idx])
        else:
            img_file = random.choice(img_files)
        
        # Extract frame number from filename (e.g., 000001.jpg -> 1)
        frame_num = int(img_file.stem)
        
        # Load annotations for this frame
        annotations = load_annotations_for_frame(gt_path, frame_num)
        
        if annotations:
            # Load image
            try:
                img = Image.open(img_file)
                return img, annotations, frame_num
            except Exception as e:
                print(f"  ⚠️  Error loading image {img_file}: {e}")
                continue
    
    # If no frame with annotations found, return first frame anyway
    img_file = img_files[0]
    frame_num = int(img_file.stem)
    annotations = load_annotations_for_frame(gt_path, frame_num)
    try:
        img = Image.open(img_file)
        return img, annotations, frame_num
    except:
        return None

def create_visualization(structure_info, sample_data_list):
    """Create comprehensive visualization of MOT20 dataset."""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Split statistics (bar chart)
    ax1 = plt.subplot(2, 3, 1)
    splits = list(structure_info['split_stats'].keys())
    seq_counts = [structure_info['split_stats'][s]['sequences'] for s in splits]
    anno_counts = [structure_info['split_stats'][s]['annotations'] for s in splits]
    
    x = range(len(splits))
    width = 0.35
    ax1.bar([i - width/2 for i in x], seq_counts, width, label='Sequences', alpha=0.8)
    ax1.bar([i + width/2 for i in x], anno_counts, width, label='Annotations (×1000)', alpha=0.8)
    ax1.set_xlabel('Split')
    ax1.set_ylabel('Count')
    ax1.set_title('Sequences and Annotations by Split', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Format y-axis for annotations (show in thousands)
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Annotations (×1000)', color='orange')
    ax1_twin.set_ylim([c/1000 for c in ax1.get_ylim()])
    
    # 2. Resolution distribution
    ax2 = plt.subplot(2, 3, 2)
    resolutions = structure_info['resolutions']
    if resolutions:
        res_labels = [f"{w}×{h}" for (w, h), _ in resolutions.most_common()]
        res_counts = [count for _, count in resolutions.most_common()]
        ax2.pie(res_counts, labels=res_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Resolution Distribution\n(by frame count)', fontsize=12, fontweight='bold')
    
    # 3. Sequence statistics
    ax3 = plt.subplot(2, 3, 3)
    sequences = structure_info['sequences']
    seq_names = [s['name'] for s in sequences]
    seq_tracks = [s['unique_tracks'] for s in sequences]
    seq_frames = [s['seqLength'] for s in sequences]
    
    x = range(len(seq_names))
    ax3.bar([i - 0.2 for i in x], seq_tracks, 0.4, label='Tracks', alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.bar([i + 0.2 for i in x], seq_frames, 0.4, label='Frames', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Sequence')
    ax3.set_ylabel('Unique Tracks', color='blue')
    ax3_twin.set_ylabel('Frames', color='orange')
    ax3.set_title('Tracks and Frames per Sequence', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(seq_names, rotation=45, ha='right')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4-6. Sample frames with annotations
    for idx, (img, annotations, frame_num, seq_name, split_name) in enumerate(sample_data_list[:3]):
        ax = plt.subplot(2, 3, 4 + idx)
        
        # Display frame
        ax.imshow(img)
        ax.set_title(f'{split_name.upper()}/{seq_name}\nFrame {frame_num}', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Draw bounding boxes - use different colors for different tracks
        colors = plt.cm.tab20(range(20))  # Cycle through colors for tracks
        
        track_ids = list(set(a['id'] for a in annotations))
        track_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(track_ids)}
        
        for anno in annotations:
            track_id = anno['id']
            bb_left = anno['bb_left']
            bb_top = anno['bb_top']
            bb_width = anno['bb_width']
            bb_height = anno['bb_height']
            
            color = track_color_map[track_id]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bb_left, bb_top), bb_width, bb_height,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add track ID label
            ax.text(bb_left, bb_top - 5, f'ID:{track_id}', 
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Add annotation count
        ax.text(10, 20, f'{len(annotations)} objects\n{len(track_ids)} tracks', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white', fontsize=10, fontweight='bold')
    
    plt.suptitle('MOT20 Challenge Dataset Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def main():
    parser = argparse.ArgumentParser(description='Explore MOT20 Challenge Dataset')
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                       help='Path to datasets directory (default: datasets)')
    parser.add_argument('--samples_only', action='store_true',
                       help='Skip analysis and show only visualization samples')
    args = parser.parse_args()
    
    # Convert to Path
    if args.datasets_dir == 'datasets':
        if Path('datasets').exists():
            datasets_dir = Path('datasets')
        elif Path('MOT20-Challenge').exists():
            datasets_dir = Path('.')
        else:
            datasets_dir = Path('..')
    else:
        datasets_dir = Path(args.datasets_dir)
    
    # Analyze structure
    if args.samples_only:
        print("="*70)
        print("Visualization Mode (Skipping Analysis)")
        print("="*70)
        print("Extracting sample frames for visualization...")
        structure_info = analyze_dataset_structure(datasets_dir)
        if structure_info is None:
            print("Error: Could not find MOT20 dataset")
            return
    else:
        print("="*70)
        print("Analyzing MOT20 Challenge Dataset...")
        print("="*70)
        
        structure_info = analyze_dataset_structure(datasets_dir)
        
        if structure_info is None:
            print("Error: Could not find MOT20 dataset")
            print(f"Expected structure: {datasets_dir}/MOT20-Challenge/MOT20/")
            return
        
        # Print summary
        print("\n" + "="*70)
        print("Dataset Summary")
        print("="*70)
        print(f"Total Sequences: {structure_info['total_sequences']}")
        print(f"Total Annotations: {structure_info['total_annotations']:,}")
        print(f"Total Tracks: {structure_info['total_tracks']:,}")
        print(f"Total Images: {structure_info['total_images']:,}")
        print(f"Total Frames: {structure_info['total_frames']:,}")
        
        # Print split information
        print("\n" + "="*70)
        print("Split Statistics")
        print("="*70)
        for split in ['train', 'test']:
            if split in structure_info['split_stats']:
                stats = structure_info['split_stats'][split]
                print(f"\n{split.upper()}:")
                print(f"  Sequences: {stats['sequences']}")
                print(f"  Annotations: {stats['annotations']:,}")
                print(f"  Tracks: {stats['tracks']:,}")
                print(f"  Images: {stats['images']:,}")
                print(f"  Frames: {stats['frames']:,}")
        
        # Print sequence details
        print("\n" + "="*70)
        print("Sequence Details")
        print("="*70)
        for seq in structure_info['sequences']:
            print(f"\n{seq['name']} ({seq['split']}):")
            print(f"  Resolution: {seq['imWidth']}×{seq['imHeight']}")
            print(f"  Frame Rate: {seq['frameRate']} FPS")
            print(f"  Length: {seq['seqLength']} frames")
            print(f"  Images: {seq['img_count']}")
            print(f"  Annotations: {seq['gt_annotations']:,}")
            print(f"  Unique Tracks: {seq['unique_tracks']}")
            if seq['detections'] > 0:
                print(f"  Detections: {seq['detections']:,}")
        
        # Print resolution distribution
        if structure_info['resolutions']:
            print("\n" + "="*70)
            print("Resolution Distribution")
            print("="*70)
            for (w, h), count in structure_info['resolutions'].most_common():
                percentage = (count / structure_info['total_frames']) * 100
                print(f"  {w}×{h}: {count:,} frames ({percentage:.1f}%)")
    
    # Visualization
    print("\n" + "="*70)
    print("Extracting Sample Frames for Visualization...")
    print("="*70)
    
    sample_data_list = []
    sequences = structure_info['sequences']
    
    # Get samples from different splits - prioritize train sequences (they have annotations)
    train_seqs = [s for s in sequences if s['split'] == 'train' and s['gt_annotations'] > 0]
    test_seqs = [s for s in sequences if s['split'] == 'test']
    
    # Try to get samples from train sequences first (they have annotations)
    random.shuffle(train_seqs)
    selected_seqs = train_seqs[:min(3, len(train_seqs))]
    
    # If we need more samples and have test sequences, add one from test
    if len(selected_seqs) < 3 and test_seqs:
        random.shuffle(test_seqs)
        selected_seqs.append(test_seqs[0])
    
    for seq in selected_seqs:
        result = visualize_sample_frame(seq)
        if result:
            img, annotations, frame_num = result
            sample_data_list.append((img, annotations, frame_num, seq['name'], seq['split']))
            print(f"  ✓ Got sample from {seq['split']}/{seq['name']} (frame {frame_num}, {len(annotations)} objects)")
    
    if sample_data_list:
        print(f"\n✓ Extracted {len(sample_data_list)} sample frames")
        print("\n" + "="*70)
        print("Generating Visualization...")
        print("="*70)
        
        fig = create_visualization(structure_info, sample_data_list)
        plt.show()
        print("\n✓ Visualization displayed")
    else:
        print("⚠️  Could not extract sample frames for visualization")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == '__main__':
    main()
