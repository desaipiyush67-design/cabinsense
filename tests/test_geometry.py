import numpy as np

from cabinsense.geometry import LEFT_EYE, MOUTH_CORNERS, MOUTH_PAIRS, eye_aspect_ratio, mouth_aspect_ratio


def make_eye(height: float) -> np.ndarray:
    lm = np.zeros((468, 3))
    p1, p2, p3, p4, p5, p6 = LEFT_EYE
    lm[p1], lm[p4] = (0, 0, 0), (30, 0, 0)          # eye corners, width 30
    lm[p2], lm[p3] = (10, -height / 2, 0), (20, -height / 2, 0)
    lm[p6], lm[p5] = (10, height / 2, 0), (20, height / 2, 0)
    return lm


def test_ear_open_vs_closed():
    assert abs(eye_aspect_ratio(make_eye(9), LEFT_EYE) - 0.3) < 1e-9
    assert eye_aspect_ratio(make_eye(1), LEFT_EYE) < 0.05


def test_ear_degenerate_width_is_zero():
    assert eye_aspect_ratio(np.zeros((468, 3)), LEFT_EYE) == 0.0


def test_mar_grows_with_mouth_opening():
    def mouth(gap):
        lm = np.zeros((468, 3))
        lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]] = (0, 0, 0), (50, 0, 0)
        for x, (a, b) in zip((15, 25, 35), MOUTH_PAIRS):
            lm[a], lm[b] = (x, -gap / 2, 0), (x, gap / 2, 0)
        return lm
    assert mouth_aspect_ratio(mouth(2)) < 0.1
    assert mouth_aspect_ratio(mouth(40)) > 0.6
