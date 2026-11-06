"""
core.config — Constants and one-time initialisation (MediaPipe, average face).

Everything in this module is pure Python — no PyQt imports.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import h5py
import logging

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger('iTemplate2')

# ---------------------------------------------------------------------------
# MediaPipe Initialization
# ---------------------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable Metal (GPU) in sub-processes to avoid contention during parallel extraction
if "__main__" not in sys.modules or os.environ.get("MP_FORCE_CPU", "0") == "1":
    os.environ["TF_METAL_ENABLED"] = "0"
else:
    os.environ["TF_METAL_ENABLED"] = "1"

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# `current` points to the project root (one level up from core/)
current = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

base_options = python.BaseOptions(
    model_asset_path=os.path.join(current, 'face_landmarker.task')
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=5,
)

detector = None
_detector_init_error = None

def get_face_detector():
    """Lazy initialization of the global FaceLandmarker.

    Returns
    -------
    FaceLandmarker or None
        The shared detector instance, or None if initialization failed.
    """
    global detector, _detector_init_error
    if detector is not None:
        return detector
    try:
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        _detector_init_error = str(e)
        logger.warning("Failed to initialize MediaPipe FaceLandmarker: %s", e)
        return None
    return detector

# ---------------------------------------------------------------------------
# Average Face Data
# ---------------------------------------------------------------------------
average_face = None
avg_face_image = None
avg_landmarks = None
_h5_load_error = "average_face_data.h5 is missing or contains invalid data."
delaunay_tri = None
width = 1024   # default; overwritten below if h5 data is found
height = 1024

h5File = os.path.join(current, 'average_face_data.h5')
if os.path.exists(h5File):
    with h5py.File(h5File, 'r') as hf:
        average_face = hf['average_face'][:]
        avg_face_image = cv2.cvtColor(average_face, cv2.COLOR_BGRA2RGBA)
        avg_landmarks = hf['avg_landmarks'][:]
        delaunay_tri = hf['delaunay_tri'][:]
    height, width = avg_face_image.shape[:2]
else:
    logger.warning("average_face_data.h5 not found at %s", h5File)
    _h5_load_error = "average_face_data.h5 not found at the project root."


def check_assets():
    """Validate that all required model files are present and initialized.

    Returns a dict:
        {
            "ok": bool,           # True if all assets are ready
            "detector": bool,     # MediaPipe detector available
            "avg_face": bool,     # Average face template loaded
            "errors": [str],      # List of problem descriptions
        }
    """
    errors = []
    det = get_face_detector()
    det_ok = det is not None
    if not det_ok:
        errors.append(
            f"MediaPipe FaceLandmarker failed to initialize: {_detector_init_error}\n"
            "Ensure 'face_landmarker.task' is in the iTemplate2 root directory.\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )
    af_ok = delaunay_tri is not None and avg_landmarks is not None
    if not af_ok:
        errors.append(_h5_load_error)

    return {
        "ok": det_ok and af_ok,
        "detector": det_ok,
        "avg_face": af_ok,
        "errors": errors,
    }


def create_detector(num_faces=1):
    """Create a new FaceLandmarker configured for the specified face count.

    Parameters
    ----------
    num_faces : int
        Maximum number of faces to detect per frame.

    Returns
    -------
    FaceLandmarker
        A newly created detector instance.
    """
    opts = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(opts)
