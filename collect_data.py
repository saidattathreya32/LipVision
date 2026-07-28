# Webcam script to collect your custom lip-reading dataset
import os
import cv2
import numpy as np
from config import TOTAL_FRAMES, LABEL_DICT, REVERSE_LABEL_DICT, LAR_THRESHOLD, SILENCE_FRAMES_LIMIT, MIN_SPEECH_FRAMES, DATA_DIR, FEATURES_PATH, LABELS_PATH
from data_processor import LipFeatureExtractor


def collect_dataset(samples_per_word=10):
    os.makedirs(DATA_DIR, exist_ok=True)
    extractor = LipFeatureExtractor()
    cap = cv2.VideoCapture(0)

    features_list = []
    labels_list = []

    print("\n--- LipRead-3.0 Data Collector ---")
    for word_id, word_str in LABEL_DICT.items():
        print(
            f"\nTarget Word: [{word_str.upper()}] | Collecting {samples_per_word} samples.")

        collected = 0
        while collected < samples_per_word:
            print(
                f"Press 'SPACE' when ready to speak '{word_str}' (Sample {collected+1}/{samples_per_word})...")

            # Wait for user trigger
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"Word: {word_str.upper()} ({collected}/{samples_per_word})", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(frame, "Press SPACE and SPEAK", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # Spacebar
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Capture video sequence
            current_sequence = []
            is_speaking = False
            silence_counter = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                coords, lar = extractor.extract_landmarks(frame)
                if coords is not None:
                    if lar > LAR_THRESHOLD:
                        is_speaking = True
                        silence_counter = 0
                        current_sequence.append(coords)
                        cv2.putText(frame, "RECORDING...", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "TALK NOW", (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if is_speaking:
                            silence_counter += 1
                            current_sequence.append(coords)

                            if silence_counter > SILENCE_FRAMES_LIMIT:
                                break

                cv2.imshow("Data Collector", frame)
                cv2.waitKey(1)

            # Pad or truncate sequence to 22 frames
            if len(current_sequence) >= MIN_SPEECH_FRAMES:
                seq_arr = np.array(current_sequence)
                if len(seq_arr) < TOTAL_FRAMES:
                    pad = np.tile(
                        seq_arr[-1], (TOTAL_FRAMES - len(seq_arr), 1))
                    seq_arr = np.vstack([seq_arr, pad])
                else:
                    seq_arr = seq_arr[:TOTAL_FRAMES]

                features = extractor.compute_temporal_features(seq_arr)
                features_list.append(features)
                labels_list.append(word_id)
                collected += 1
                print(f"Sample {collected} saved successfully.")

    cap.release()
    cv2.destroyAllWindows()

    np.save(FEATURES_PATH, np.array(features_list, dtype=np.float32))
    np.save(LABELS_PATH, np.array(labels_list, dtype=np.int64))
    print(f"\nDataset saved successfully: {FEATURES_PATH} and {LABELS_PATH}")


if __name__ == "__main__":
    # Adjust number of samples per word as needed
    collect_dataset(samples_per_word=5)
