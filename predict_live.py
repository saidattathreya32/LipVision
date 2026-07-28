import cv2
import torch
import numpy as np
import dlib
from transformers import AutoProcessor, AutoModelForCTC  # The 3.0 Upgrade!


def run_live_vsr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load a Pre-Trained Open-Vocabulary Visual Speech Model
    print("⏳ Downloading/Loading Pre-Trained VSR Transformer...")
    processor = AutoProcessor.from_pretrained(
        "tiiuae/visper")  # Example VSR model
    model = AutoModelForCTC.from_pretrained("tiiuae/visper").to(device)
    model.eval()

    # 2. Setup Dlib AI Tracker (Keeps your flawless lip tracking)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    frame_buffer = []
    predicted_sentence = "WAITING..."
    is_speaking = False
    silence_counter = 0

    print("\n--- 🚀 LipRead-3.0 Open-Vocabulary Engine Active ---")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            mar = 0.0

            if len(faces) > 0:
                face = max(faces, key=lambda rect: rect.width()
                           * rect.height())
                landmarks = predictor(gray, face)

                x_coords = [landmarks.part(i).x for i in range(48, 68)]
                y_coords = [landmarks.part(i).y for i in range(48, 68)]

                for cx, cy in zip(x_coords, y_coords):
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                # Mouth Aspect Ratio for VAD
                mouth_height = float(
                    abs(landmarks.part(66).y - landmarks.part(62).y))
                mouth_width = float(
                    abs(landmarks.part(54).x - landmarks.part(48).x)) + 1e-6
                mar = mouth_height / mouth_width

                # Standardized crop for the Transformer (usually 96x96 for AV-HuBERT)
                pad_w = int(mouth_width * 0.4)
                pad_h = int(mouth_width * 0.4)
                mx, my = max(0, min(x_coords) - pad_w), max(0,
                                                            min(y_coords) - pad_h)
                mw, mh = min(frame.shape[1], max(
                    x_coords) + pad_w) - mx, min(frame.shape[0], max(y_coords) + pad_h) - my

                if mw > 0 and mh > 0:
                    mouth_roi = gray[my:my+mh, mx:mx+mw]
                    mouth_resized = cv2.resize(
                        mouth_roi, (96, 96))  # VSR standard size
                    normalized_mouth = mouth_resized.astype(np.float32) / 255.0

                    if mar > 0.05:
                        cv2.putText(frame, "Talking", (30, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        is_speaking = True
                        silence_counter = 0
                        frame_buffer.append(normalized_mouth)
                    else:
                        cv2.putText(frame, "Not talking", (30, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if is_speaking:
                            silence_counter += 1
                            frame_buffer.append(normalized_mouth)

                            # Trigger when you stop speaking (Wait 15 frames)
                            if silence_counter > 15:
                                is_speaking = False
                                speech_frames = frame_buffer[:-12] if len(
                                    frame_buffer) > 12 else frame_buffer

                                if len(speech_frames) >= 10:
                                    # IN LIPREAD 3.0 WE DO NOT PAD TO A FIXED LENGTH!
                                    # Transformers accept dynamic sequences of any length.
                                    input_video = torch.tensor(
                                        np.array(speech_frames)).unsqueeze(0).to(device)

                                    # 3. Predict the character sequence
                                    logits = model(input_video).logits
                                    predicted_ids = torch.argmax(
                                        logits, dim=-1)

                                    # 4. Decode characters into a full English sentence
                                    transcription = processor.batch_decode(predicted_ids)[
                                        0]

                                    predicted_sentence = transcription.upper()
                                    print(
                                        f"\n🗣️ *** TRANSCRIBED: {predicted_sentence} ***\n")

                                frame_buffer.clear()
                                silence_counter = 0

            cv2.putText(frame, f"Text: {predicted_sentence}", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
            cv2.imshow('LipRead-3.0 Open Vocabulary', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_vsr()
