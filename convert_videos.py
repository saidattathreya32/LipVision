import os
import cv2
import numpy as np
from config import TOTAL_FRAMES, FEATURES_PATH, LABELS_PATH
from data_processor import LipFeatureExtractor


def convert_video_folder(video_root_dir):
    """
    Expects folder structure:
    video_root_dir/
       ├── hello/
       │    ├── clip1.mp4
       │    └── clip2.mp4
       ├── dog/
       ...
    """
    extractor = LipFeatureExtractor()
    features_list = []
    labels_list = []

    # Map subfolder names to class indices
    from config import REVERSE_LABEL_DICT

    print(f"📁 Processing video files from: {video_root_dir}")

    for word_str, class_id in REVERSE_LABEL_DICT.items():
        word_folder = os.path.join(video_root_dir, word_str)
        if not os.path.exists(word_folder):
            print(f"⚠️ Skipping missing folder: {word_folder}")
            continue

        video_files = [f for f in os.listdir(
            word_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(
            f"Processing word '{word_str.upper()}' ({len(video_files)} videos)...")

        for vid_file in video_files:
            vid_path = os.path.join(word_folder, vid_file)
            cap = cv2.VideoCapture(vid_path)

            sequence_coords = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                coords, _ = extractor.extract_landmarks(frame)
                if coords is not None:
                    sequence_coords.append(coords)

            cap.release()

            # Format to 22 frames
            if len(sequence_coords) >= 6:
                seq_arr = np.array(sequence_coords)
                if len(seq_arr) < TOTAL_FRAMES:
                    pad = np.tile(
                        seq_arr[-1], (TOTAL_FRAMES - len(seq_arr), 1))
                    seq_arr = np.vstack([seq_arr, pad])
                else:
                    seq_arr = seq_arr[:TOTAL_FRAMES]

                features = extractor.compute_temporal_features(seq_arr)
                features_list.append(features)
                labels_list.append(class_id)

    if len(features_list) > 0:
        np.save(FEATURES_PATH, np.array(features_list, dtype=np.float32))
        np.save(LABELS_PATH, np.array(labels_list, dtype=np.int64))
        print(
            f"\n✅ Successfully processed {len(features_list)} total samples!")
        print(f"Saved to '{FEATURES_PATH}' and '{LABELS_PATH}'")
    else:
        print("❌ No valid video sequences processed. Check your folder paths.")


if __name__ == "__main__":
    # Specify the directory containing your word folders
    convert_video_folder("collected_data")
