"""Shared MediaPipe landmark-to-image preprocessing for training and inference."""
import cv2
import numpy as np

IMAGE_SIZE = 128

# MediaPipe Hands landmark indices and connections.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def landmarks_to_image(hand_landmarks, image_size=IMAGE_SIZE):
    """Render normalized 21-point hand landmarks as a 3-channel CNN input image."""
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    # Translation and scale normalization make the representation less position-dependent.
    xy = points[:, :2]
    xy = xy - xy.min(axis=0, keepdims=True)
    span = xy.max(axis=0) - xy.min(axis=0)
    scale = max(float(span.max()), 1e-6)
    xy = xy / scale
    margin = 12
    usable = image_size - 2 * margin
    px = np.clip((xy[:, 0] * usable + margin).astype(int), 0, image_size - 1)
    py = np.clip((xy[:, 1] * usable + margin).astype(int), 0, image_size - 1)

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # Draw bones, then joints. Distinct channel values preserve visual structure.
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, (px[a], py[a]), (px[b], py[b]), (90, 180, 255), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(zip(px, py)):
        cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 6, (40, 140, 220), 1, cv2.LINE_AA)
    return canvas
