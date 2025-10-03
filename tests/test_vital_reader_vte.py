import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vital_reader


np = pytest.importorskip("numpy")


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    # Convert BGR to RGB ordering before applying standard luma weights.
    rgb = image[..., ::-1]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(rgb, weights, axes=([-1], [0])).astype(np.float32)


def test_read_vte_roi_enhances_blue_roi(monkeypatch):
    base_color = np.array([200, 220, 255], dtype=np.uint8)
    roi = np.full((32, 64, 3), base_color, dtype=np.uint8)

    # Simulate darker digits painted on the light-blue background.
    roi[:, 20:24, :] = np.array([160, 180, 220], dtype=np.uint8)
    roi[8:24, 30:34, :] = np.array([30, 60, 120], dtype=np.uint8)

    captured: dict[str, object] = {}

    def fake_read_easy(image, allow_dot=False):
        captured["image"] = image
        captured["allow_dot"] = allow_dot
        return "12.3", []

    monkeypatch.setattr(vital_reader, "read_easy", fake_read_easy)

    result = vital_reader.read_vte_roi(roi)

    assert result == "12.3"
    assert captured["allow_dot"] is True

    processed = captured["image"]
    assert isinstance(processed, np.ndarray)
    assert processed.shape == roi.shape

    original_gray = _to_grayscale(roi)
    processed_gray = _to_grayscale(processed)

    if vital_reader.cv2 is None:
        assert processed is roi
        assert np.allclose(processed_gray, original_gray)
    else:
        assert processed is not roi
        assert processed_gray.max() - processed_gray.min() > original_gray.max() - original_gray.min()
