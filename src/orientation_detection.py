"""
Image orientation detection and correction using an ONNX model.

The model classifies an image into one of four rotation classes:
  0  -> 0°   (upright, no correction needed)
  1  -> 90°  (rotated 90° CW,  correct by rotating 90° CCW)
  2  -> 180° (upside-down,     correct by rotating 180°)
  3  -> 270° (rotated 270° CW, correct by rotating 90° CW)
"""

import os
from typing import Tuple

import numpy as np
import onnxruntime
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Input size expected by the orientation model (height x width)
_MODEL_INPUT_SIZE = 384

# ImageNet normalisation constants (channel-wise mean / std)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class index → detected CW rotation angle in degrees
_CLASS_TO_ANGLE = {0: 0, 1: 90, 2: 180, 3: 270}

# Correction: CCW rotation needed to undo each detected CW angle
# PIL.Image.rotate() is CCW-positive, so we negate the CW angle.
_CORRECTION_ANGLE = {0: 0, 90: 270, 180: 180, 270: 90}

# Default model path: <project_root>/models/orientation_model_v2_0.9882.onnx
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "orientation_model_v2_0.9882.onnx",
)

# Module-level session cache (lazy-loaded, one per process)
_ort_session: onnxruntime.InferenceSession = None

def _get_ort_session(model_path: str) -> onnxruntime.InferenceSession:
    """Lazily load and cache the ONNX inference session."""
    global _ort_session
    if _ort_session is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Orientation ONNX model not found at: {model_path}"
            )

        preferred_providers = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = onnxruntime.get_available_providers()
        providers = [p for p in preferred_providers if p in available] or ["CPUExecutionProvider"]

        _ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

    return _ort_session


# Internal helpers
def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize, centre-crop, and normalise a PIL image into an ONNX-ready tensor.

    Returns shape: (1, 3, H, W)  dtype: float32
    """
    # Resize to slightly larger than model input, then centre-crop
    resize_to = _MODEL_INPUT_SIZE + 32
    img = image.resize((resize_to, resize_to), Image.Resampling.BILINEAR)

    # Centre crop
    left = (resize_to - _MODEL_INPUT_SIZE) // 2
    top  = (resize_to - _MODEL_INPUT_SIZE) // 2
    img = img.crop((left, top, left + _MODEL_INPUT_SIZE, top + _MODEL_INPUT_SIZE))

    # Convert to float32 in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)

    # ImageNet normalisation
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD           # (H, W, 3)

    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]          # (1, 3, H, W)

    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_as_bgr(image: "np.ndarray | str") -> np.ndarray:
    """Ensure the input is a cv2 BGR uint8 array."""
    if isinstance(image, str):
        import cv2
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"cv2.imread could not read image at: {image}")
        return img
    return image


# Public API
def detect_and_fix_orientation(image: "np.ndarray | str", model_path: str = _DEFAULT_MODEL_PATH) -> Tuple[np.ndarray, int]:
    """ Detect the rotation of an image and return it in the upright (0°) orientation as a BGR numpy array. """
    # Ensure we have a BGR numpy array
    bgr_image = _load_as_bgr(image)
    
    # Convert BGR numpy array to RGB PIL Image for preprocessing
    pil_image = Image.fromarray(bgr_image[:, :, ::-1].astype(np.uint8), mode="RGB")

    # Pre-process into the model's input tensor
    input_tensor = _preprocess(pil_image)

    # Run ONNX inference
    session = _get_ort_session(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # Decode class prediction → angle
    logits = outputs[0]                                     # shape: (1, 4)
    predicted_idx = int(np.argmax(logits, axis=1)[0])
    detected_angle = _CLASS_TO_ANGLE[predicted_idx]

    # Return as-is when already upright
    if detected_angle == 0:
        return bgr_image, detected_angle

    # Apply correction rotation (PIL rotates CCW, expand=True keeps full image)
    correction = _CORRECTION_ANGLE[detected_angle]
    corrected_pil = pil_image.rotate(correction, expand=True)
    
    # Convert back to BGR numpy array
    corrected_bgr = np.array(corrected_pil)[:, :, ::-1]

    return corrected_bgr, detected_angle