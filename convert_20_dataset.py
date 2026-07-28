import os
import cv2
import numpy as np
from config import TOTAL_FRAMES, FEATURES_PATH, LABELS_PATH, REVERSE_LABEL_DICT


def process_collected_data_folder():
    outputs_dir = os.path.join("collected_data", "outputs")
    if not os.path.exists(outputs_dir):
        print(f"❌ Error: Cannot find '{outputs_dir}'.")
        return

    features_list = []
    labels_list = []
    total_processed = 0

    print("📁 Extracting PIXEL frames for 3D-CNN Model...\n")
    subfolders = [f.path for f in os.scandir(outputs_dir) if f.is_dir()]

    for folder in subfolders:
        word_str = os.path.basename(folder).lower().split('_')[0]
        if word_str not in REVERSE_LABEL_DICT:
            continue

        class_id = REVERSE_LABEL_DICT[word_str]
        frames = []

        for i in range(TOTAL_FRAMES):
            img_path = os.path.join(folder, f"{i}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Ensure standard 2.0 dimensions
                    img = cv2.resize(img, (112, 80))
                    frames.append(img)

        if len(frames) < 10:
            continue

        # Pad frames if short
        while len(frames) < TOTAL_FRAMES:
            frames.append(frames[-1])
        frames = frames[:TOTAL_FRAMES]

        # Normalize pixels to 0-1
        seq_arr = np.array(frames, dtype=np.float32) / 255.0

        features_list.append(seq_arr)
        labels_list.append(class_id)
        total_processed += 1

        if total_processed % 50 == 0:
            print(f"⚡ Extracted {total_processed} sequences...")

    if len(features_list) > 0:
        np.save(FEATURES_PATH, np.array(features_list, dtype=np.float32))
        np.save(LABELS_PATH, np.array(labels_list, dtype=np.int64))
        print(
            f"\n✅ Dataset conversion complete! Processed {total_processed} samples.")
    else:
        print("\n❌ No valid samples processed.")


if __name__ == "__main__":
    process_collected_data_folder()
