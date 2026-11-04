"""Facial geometry used by the monitor: eye/mouth aspect ratios and head pose.

All functions take landmarks as an (N, 2) or (N, 3) array in pixel coordinates,
indexed with the MediaPipe Face Mesh topology (468 points).
"""
from __future__ import annotations

import numpy as np

# Face Mesh indices (p1..p6 ordering from Soukupova & Cech, 2016).
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
# Mouth corners and three vertical pairs (upper lip / lower lip).
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((13, 14), (81, 178), (311, 402))
# Points used for head pose, with a generic 3D face model (millimetres).
POSE_POINTS = (1, 152, 33, 263, 61, 291)  # nose tip, chin, eye corners, mouth corners
POSE_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). Drops towards 0 when the eye closes."""
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width)


def mean_ear(lm: np.ndarray) -> float:
    return (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2.0


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    """MAR = mean vertical lip gap / mouth width. Rises sharply during a yawn."""
    width = _dist(lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    gaps = [_dist(lm[a], lm[b]) for a, b in MOUTH_PAIRS]
    return float(np.mean(gaps) / width)


def head_pose(lm: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees from a PnP fit against a generic face model."""
    import cv2  # local import keeps the pure-math helpers dependency-light

    image_pts = np.array([lm[i][:2] for i in POSE_POINTS], dtype=np.float64)
    focal = float(frame_w)
    camera = np.array([[focal, 0, frame_w / 2], [0, focal, frame_h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_MODEL, image_pts, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    (pitch, yaw, roll), *_ = cv2.RQDecomp3x3(rot)
    # Image y points down while the face model's y points up, so a frontal face reads pitch ~ +-180.
    pitch = pitch - 180.0 if pitch > 90 else pitch + 180.0 if pitch < -90 else pitch
    return float(yaw), float(pitch), float(roll)
