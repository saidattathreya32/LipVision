import sys
import os
import cv2
import dlib
import torch
import numpy as np
import argparse
import shutil
import time
import threading

# Optimize for Intel CPUs
torch.backends.mkldnn.enabled = True
torch.set_num_threads(4)

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_AVSR_DIR = os.path.join(BASE_DIR, "auto_avsr")

if os.path.exists(os.path.join(AUTO_AVSR_DIR, "auto_avsr-main")):
    AUTO_AVSR_DIR = os.path.join(AUTO_AVSR_DIR, "auto_avsr-main")

if AUTO_AVSR_DIR not in sys.path:
    sys.path.insert(0, AUTO_AVSR_DIR)


def align_to_25fps_by_time(frames, timestamps):
    """Resample frames to 25 FPS based on elapsed time."""
    if len(frames) < 5 or not timestamps:
        return frames

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return frames

    target_len = int(round(duration * 25.0))

    # Cap at 75 frames (3 seconds) to prevent UI blocking
    if target_len > 75:
        target_len = 75

    indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
    return [frames[i] for i in indices]


def run_live_vsr():
    try:
        from lightning import ModelModule
        from datamodule.transforms import VideoTransform
    except ModuleNotFoundError as e:
        print(f"Dependency error: {e}")
        return

    device = torch.device("cpu")
    print("Hardware mode: CPU")

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args(args=[])
    setattr(args, 'modality', 'video')

    # Ensure sentencepiece model is in the correct directory
    spm_dir = os.path.join(BASE_DIR, "spm", "unigram")
    os.makedirs(spm_dir, exist_ok=True)
    local_unigram = os.path.join(BASE_DIR, "unigram5000.model")
    target_unigram = os.path.join(spm_dir, "unigram5000.model")

    if os.path.exists(local_unigram) and not os.path.exists(target_unigram):
        shutil.copy(local_unigram, target_unigram)

    model_path = None
    pth_files = [f for f in os.listdir(
        BASE_DIR) if f.startswith("vsr_") and f.endswith(".pth")]

    if pth_files:
        model_path = os.path.join(BASE_DIR, pth_files[0])
    else:
        print("Error: Model weights (.pth) not found.")
        return

    print("Loading model...")
    ckpt = torch.load(model_path, map_location="cpu")
    modelmodule = ModelModule(args)
    modelmodule.model.load_state_dict(ckpt)
    modelmodule.to(device)
    modelmodule.eval()

    video_transform = VideoTransform(subset="test")

    print("Initializing tracker...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(
        BASE_DIR, 'shape_predictor_68_face_landmarks.dat'))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = {
        "predicted_sentence": "Ready...",
        "is_processing": False,
        "lock": threading.Lock()
    }

    def decode_worker(speech_frames, timestamps):
        try:
            aligned_frames = align_to_25fps_by_time(speech_frames, timestamps)

            video_tensor = torch.tensor(np.array(aligned_frames)).unsqueeze(-1)
            video_tensor = video_tensor.permute((0, 3, 1, 2)).float()
            video_tensor = video_transform(video_tensor).to(device)

            with torch.no_grad():
                transcript = modelmodule(video_tensor)

            result_text = transcript[0].upper() if isinstance(
                transcript, list) else str(transcript).upper()

            if not result_text.strip():
                result_text = "[Unclear]"

            print(f"Transcribed: {result_text}")

            with state["lock"]:
                state["predicted_sentence"] = result_text
                state["is_processing"] = False

        except Exception as e:
            print(f"Inference error: {e}")
            with state["lock"]:
                state["is_processing"] = False

    frame_buffer = []
    time_buffer = []
    is_speaking = False
    silence_counter = 0
    frame_counter = 0
    tracked_face = None

    smooth_x, smooth_y, smooth_size = None, None, None
    alpha = 0.4

    print("System ready. Press 'q' to exit.")

    while cap.isOpened():
        time.sleep(0.01)

        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        current_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run detection every 3 frames to maintain FPS
        if frame_counter % 3 == 0 or tracked_face is None:
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = detector(small_gray)

            if len(faces) > 0:
                f = max(faces, key=lambda rect: rect.width() * rect.height())
                tracked_face = dlib.rectangle(
                    int(f.left() * 2), int(f.top() * 2),
                    int(f.right() * 2), int(f.bottom() * 2)
                )
            else:
                tracked_face = None

        if tracked_face is not None:
            landmarks = predictor(gray, tracked_face)

            x_coords = [landmarks.part(i).x for i in range(48, 68)]
            y_coords = [landmarks.part(i).y for i in range(48, 68)]

            for cx, cy in zip(x_coords, y_coords):
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            mouth_height = float(
                abs(landmarks.part(66).y - landmarks.part(62).y))
            mouth_width = float(
                abs(landmarks.part(54).x - landmarks.part(48).x)) + 1e-6
            mar = mouth_height / mouth_width

            raw_x = int(np.mean(x_coords))
            raw_y = int(np.mean(y_coords))
            raw_size = int(max(mouth_width, mouth_height) * 1.8)

            # EMA smoothing for the bounding box
            if smooth_x is None:
                smooth_x, smooth_y, smooth_size = raw_x, raw_y, raw_size
            else:
                smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)
                smooth_size = int(alpha * raw_size + (1 - alpha) * smooth_size)

            half = smooth_size // 2
            y1, y2 = max(
                0, smooth_y - half), min(frame.shape[0], smooth_y + half)
            x1, x2 = max(
                0, smooth_x - half), min(frame.shape[1], smooth_x + half)

            if (y2 - y1) > 10 and (x2 - x1) > 10:
                mouth_roi = gray[y1:y2, x1:x2]
                mouth_resized = cv2.resize(mouth_roi, (96, 96))

                if mar > 0.055:
                    cv2.putText(frame, "Talking", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    is_speaking = True
                    silence_counter = 0
                    frame_buffer.append(mouth_resized)
                    time_buffer.append(current_time)
                else:
                    cv2.putText(frame, "Idle", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if is_speaking:
                        silence_counter += 1
                        frame_buffer.append(mouth_resized)
                        time_buffer.append(current_time)

                        if silence_counter > 8:
                            is_speaking = False

                            speech_frames = frame_buffer[:-8] if len(
                                frame_buffer) > 8 else frame_buffer
                            speech_times = time_buffer[:-8] if len(
                                time_buffer) > 8 else time_buffer

                            with state["lock"]:
                                already_processing = state["is_processing"]

                            if len(speech_frames) >= 5 and not already_processing:
                                with state["lock"]:
                                    state["is_processing"] = True

                                worker = threading.Thread(
                                    target=decode_worker,
                                    args=(list(speech_frames),
                                          list(speech_times))
                                )
                                worker.daemon = True
                                worker.start()

                            frame_buffer.clear()
                            time_buffer.clear()
                            silence_counter = 0
        else:
            cv2.putText(frame, "No face detected", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            smooth_x, smooth_y, smooth_size = None, None, None

        with state["lock"]:
            curr_sentence = state["predicted_sentence"]
            processing_now = state["is_processing"]

        if processing_now:
            cv2.putText(frame, "Processing...", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.putText(frame, f"Text: {curr_sentence}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('Live VSR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_vsr()
