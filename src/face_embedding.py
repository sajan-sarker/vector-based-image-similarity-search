"""
Face detection, alignment, and embedding generation.

Uses the InsightFace `buffalo_l` model pack:
  - Detection:   RetinaFace-10GF 
  - Alignment:   5-point landmark similarity transform
  - Recognition: ArcFace-trained ResNet50@WebFace600K

All three stages run through one FaceAnalysis().get() call.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDING_DIMENSION = 512
MODEL_PACK = "buffalo_l"

_app: Optional[FaceAnalysis] = None


def _get_app() -> FaceAnalysis:
    """Lazily load and cache the InsightFace model pack (once per process)."""
    global _app
    if _app is None:
        _app = FaceAnalysis(
            name=MODEL_PACK,
            providers=["CPUExecutionProvider"],

        )
        _app.prepare(ctx_id=-1)  # load in CPU;
    return _app


def _normalize(vector: np.ndarray) -> np.ndarray:
    """ L2-normalize a vector and convert to float32 for Pinecone. """ 
    vector = vector.astype(np.float32)  # convert to float32 for Pinecone
    norm = np.linalg.norm(vector)       # compute L2 norm of the vector
    if norm == 0:
        raise ValueError("Zero-norm embedding; the model returned a degenerate result")
    return vector / norm


def generate_face_embedding(image_path: str) -> np.ndarray:
    """
    Run detection -> alignment -> embedding on a single local image.

    Returns a 512-dim, L2-normalized float32 embedding for the most
    prominent face in the image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image (unsupported or corrupt file): {path}")

    faces = _get_app().get(image)

    if len(faces) == 0:
        raise ValueError(f"No face detected: {path}")

    if len(faces) > 1:
        # This gallery/search use case has no "claimed identity" signal to
        # disambiguate multiple faces the way a 1:1 verification flow would.
        # We use the largest face and warn, rather than silently guessing;
        # switch this to a hard rejection if your gallery must guarantee
        # exactly one face per image.
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        print(f"Warning: {path} contains {len(faces)} faces; using the largest one.")

    return _normalize(faces[0].embedding)
