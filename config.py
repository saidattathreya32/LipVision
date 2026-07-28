# Central settings, thresholds, and mappings
import os

# Sequence Dimensions
TOTAL_FRAMES = 22
NUM_CLASSES = 13

# Label mapping based on the original 13-word dictionary
LABEL_DICT = {
    0: 'a', 1: 'bye', 2: 'can', 3: 'cat', 4: 'demo',
    5: 'dog', 6: 'hello', 7: 'here', 8: 'is', 9: 'lips',
    10: 'my', 11: 'read', 12: 'you'
}
REVERSE_LABEL_DICT = {v: k for k, v in LABEL_DICT.items()}

# MediaPipe Face Mesh Lip Landmark Indices (32 key points)
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,  # Outer contour
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14  # Inner contour
]

NUM_POINTS = len(LIP_LANDMARKS)
COORDS_PER_POINT = 3  # (x, y, z)
RAW_FEATURE_DIM = NUM_POINTS * COORDS_PER_POINT  # 96 values per frame

# Total feature dimension including velocity & relative distances
FEATURE_DIM = (RAW_FEATURE_DIM * 2) + 8  # 200 inputs per frame

# Speech Trigger (Voice Activity Detection via Lip Aspect Ratio - LAR)
LAR_THRESHOLD = 0.15          # Open mouth threshold
SILENCE_FRAMES_LIMIT = 4       # Consecutive closed-mouth frames before prediction
MIN_SPEECH_FRAMES = 6         # Minimum valid frames to register speech

# Paths
DATA_DIR = "dataset"
FEATURES_PATH = "features.npy"
LABELS_PATH = "labels.npy"
MODEL_WEIGHTS_PATH = "lipread_v3_best.pth"
