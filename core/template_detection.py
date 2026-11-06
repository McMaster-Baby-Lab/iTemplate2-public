"""Isolated MediaPipe image detection entry point.

Native detector failures must not terminate the GUI process.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np


def detect_image(path):
    import mediapipe as mp
    from core.config import create_detector, current
    from core.template_manager import file_hash
    image = cv2.imread(path)
    if image is None:
        raise ValueError('The selected image could not be read.')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with create_detector(num_faces=5) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = [np.array([(v.x, v.y, v.z) for v in face], dtype=np.float32)
             for face in result.face_landmarks]
    if not faces:
        raise ValueError('No face detected. Choose a clear, unobstructed face image.')
    provenance = {
        'source_kind': 'uploaded_image', 'source_name': Path(path).name,
        'name': Path(path).name, 'source_hash': file_hash(path),
        'landmark_provenance': 'detected',
        'detector_provenance': {'mediapipe_version': mp.__version__,
                                'model_hash': file_hash(Path(current) / 'face_landmarker.task')},
    }
    return image, faces, provenance


def main():
    source, output = sys.argv[1:]
    try:
        image, faces, provenance = detect_image(source)
        np.savez_compressed(Path(output) / 'detected.npz', image=image, faces=np.asarray(faces))
        (Path(output) / 'provenance.json').write_text(json.dumps(provenance))
    except Exception as exc:
        (Path(output) / 'error.txt').write_text(str(exc))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
