"""
Face angle (pose) detection.

Each detected Face object exposes a `pose` attribute:
    pose[0]  ->  yaw   (left/right head rotation, degrees)
    pose[1]  ->  pitch (up/down head rotation,   degrees)
    pose[2]  ->  roll  (head tilt,                degrees)

Angle convention (InsightFace / 3DMM):
    yaw > 0  ->  face turned to the RIGHT of the camera (subject's left)
    yaw < 0  ->  face turned to the LEFT  of the camera (subject's right)

The function returns one of three strings:
    "straight"  - face is approximately front-on
    "left"      - face is turned to the LEFT  (from the viewer's perspective)
    "right"     - face is turned to the RIGHT (from the viewer's perspective)
"""

from pathlib import Path
from typing import Literal, Any


import cv2

# -- Tunable thresholds -------------------------------------------------------
# Absolute yaw (degrees) beyond which a face is considered turned.
# Faces with |yaw| < YAW_THRESHOLD are classified as "straight".
YAW_THRESHOLD: float = 20.0

FaceAngle = Literal["straight", "left", "right"]


def detect_face_angle(face: Any) -> Any:
    """
    Detect the horizontal facing direction of the most prominent face.
    "straight" | "left" | "right"
        Orientation of the face from the viewer's point of view:
        - "straight" -> face is looking directly at the camera  (|yaw| < threshold)
        - "left"     -> face is turned toward the viewer's left  (yaw < -threshold)
        - "right"    -> face is turned toward the viewer's right (yaw > +threshold)
    """
    # `pose` is a (3,) array: [yaw, pitch, roll] in degrees.
    yaw: float = float(face.pose[0])

    if yaw > YAW_THRESHOLD:
        return "right"
    elif yaw < -YAW_THRESHOLD:
        return "left"
    else:
        return "straight"