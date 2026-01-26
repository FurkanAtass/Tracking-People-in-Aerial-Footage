import os
import shutil

def main():
    datasets_dir = "datasets/Okutama-Action-MOT"
    valid_dir = f"{datasets_dir}/ValidSetFrames"
    train_dir = f"{datasets_dir}/TrainSetFrames"

    os.makedirs(valid_dir, exist_ok=True)
    folders_to_move = [
        "1.1.2",
        "1.1.4",
        "1.2.5",
        "1.2.6",
        "2.1.2",
        "2.1.5",
        "2.2.4",
        "2.2.6",
    ]
    for folder in folders_to_move:
        try:
            ids = folder.split(".")
            drone_id = ids[0]
            time = "Morning" if ids[1] == "1" else "Noon"

            source_folder = f"{train_dir}/Drone{drone_id}/{time}/Extracted-Frames-1280x720/{folder}"
            target_folder = f"{valid_dir}/Drone{drone_id}/{time}/Extracted-Frames-1280x720"
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(source_folder, target_folder)

            source_file = f"{train_dir}/Labels/SingleActionTrackingLabels/3840x2160/{folder}.txt"
            target_file = f"{valid_dir}/Labels/SingleActionTrackingLabels/3840x2160/{folder}.txt"
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.move(source_file, target_file)
        except Exception as e:
            print(f"Error moving {folder}: {e}")
            continue

if __name__ == "__main__":
    main()