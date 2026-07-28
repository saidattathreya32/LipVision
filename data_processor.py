import cv2
import numpy as np
import mediapipe as mp
from config import LIP_LANDMARKS, FEATURE_DIM


class LipFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, 0.0

        landmarks = results.multi_face_landmarks[0].landmark

        lip_pts = np.array([[landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w]
                            for idx in LIP_LANDMARKS], dtype=np.float32)

        center = np.mean(lip_pts, axis=0)
        lip_pts -= center

        mouth_width = np.linalg.norm(lip_pts[0] - lip_pts[10]) + 1e-6
        lip_pts /= mouth_width

        top_lip_center = lip_pts[5]
        bottom_lip_center = lip_pts[15]
        vertical_dist = np.linalg.norm(top_lip_center - bottom_lip_center)
        lar = vertical_dist / (np.linalg.norm(lip_pts[0] - lip_pts[10]) + 1e-6)

        return lip_pts.flatten(), lar

    @staticmethod
    def compute_temporal_features(sequence_coords):
        seq_len, raw_dim = sequence_coords.shape
        velocity = np.zeros_like(sequence_coords)

        velocity[1:] = sequence_coords[1:] - sequence_coords[:-1]

        distances = []
        for i in range(seq_len):
            pts = sequence_coords[i].reshape(-1, 3)
            d1 = np.linalg.norm(pts[5] - pts[15])
            d2 = np.linalg.norm(pts[0] - pts[10])
            d3 = np.linalg.norm(pts[21] - pts[27])
            d4 = np.linalg.norm(pts[16] - pts[22])
            distances.append([d1, d2, d3, d4, d1/(d2+1e-6),
                             d3/(d4+1e-6), d1-d3, d2-d4])

        distances = np.array(distances, dtype=np.float32)
        full_features = np.hstack([sequence_coords, velocity, distances])
        return full_features
