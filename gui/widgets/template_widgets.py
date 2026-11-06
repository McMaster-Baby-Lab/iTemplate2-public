"""Small image-only template preview and isolated upload detection worker."""
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QComboBox, QHBoxLayout
from core.template_manager import TemplateManager, file_hash

class TemplateDetectionWorker(QThread):
    result = pyqtSignal(object, object, object)
    error = pyqtSignal(str)
    def __init__(self,path,parent=None):
        super().__init__(parent); self.path=path
    def run(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        try:
            with tempfile.TemporaryDirectory(prefix='itemplate-detection-') as directory:
                root = Path(__file__).resolve().parents[2]
                env = dict(os.environ, MP_FORCE_CPU='1', TF_METAL_ENABLED='0')
                completed = subprocess.run([sys.executable,'-m','core.template_detection',self.path,directory],
                                           cwd=root,env=env,capture_output=True,timeout=120)
                if completed.returncode != 0:
                    error = Path(directory)/'error.txt'
                    message = error.read_text() if error.exists() else 'MediaPipe could not run on this system. Check the detector/runtime installation, or choose a stimulus frame with extracted landmarks.'
                    raise ValueError(message)
                with np.load(Path(directory)/'detected.npz',allow_pickle=False) as result:
                    image, faces = result['image'].copy(), result['faces'].copy()
                provenance = json.loads((Path(directory)/'provenance.json').read_text())
                self.result.emit(image,list(faces),provenance)
        except Exception as exc:
            self.error.emit(str(exc))

class TemplatePreviewDialog(QDialog):
    def __init__(self,image,faces,parent=None,face_ids=None):
        super().__init__(parent); self.image=image; self.faces=faces
        self.setWindowTitle('Preview Template Face'); self.setMinimumWidth(450)
        layout=QVBoxLayout(self)
        label=QLabel('Select one face. Choose a clear, frontal frame; existing template AOIs will need review.')
        label.setWordWrap(True);layout.addWidget(label)
        self.combo=QComboBox();self.combo.addItems([f'Face {fid}' for fid in (face_ids or list(range(1,len(faces)+1)))]);layout.addWidget(self.combo)
        self.preview=QLabel();self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter);layout.addWidget(self.preview)
        row=QHBoxLayout();apply=QPushButton('Use This Template');cancel=QPushButton('Cancel')
        apply.clicked.connect(self.accept);cancel.clicked.connect(self.reject);row.addWidget(apply);row.addWidget(cancel);layout.addLayout(row)
        self.combo.currentIndexChanged.connect(self.refresh);self.refresh()
    def refresh(self):
        img=self.image.copy();h,w=img.shape[:2]
        for x,y in self.faces[self.combo.currentIndex()][:,:2]:
            cv2.circle(img,(int(x*w),int(y*h)),1,(0,255,255),-1)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pix=QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy())
        self.preview.setPixmap(pix.scaled(550,400,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
    def selected_landmarks(self): return self.faces[self.combo.currentIndex()]
