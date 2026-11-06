"""
About View — Application information and credits.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt
from .theme import text_accent


class AboutView(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        # Title
        title = QLabel("iTemplate2")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("A face-template-based eye-movement data processing pipeline")
        subtitle.setObjectName("hint-label")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        version = QLabel("Version 2.00")
        version.setStyleSheet("font-size: 13px; color: #aaa; margin-bottom: 10px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        # Separator
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setObjectName("about-separator")
        layout.addWidget(sep)

        # Description
        desc = QLabel(
            "iTemplate2 co-registers eye-tracking gaze data onto a normalised face template, "
            "enabling researchers to analyse where participants looked on dynamic face stimuli "
            "(videos) or static face images.\n\n"
            "The pipeline extracts facial landmarks from stimuli, maps raw gaze coordinates "
            "into template-normalised space, and provides tools for visualisation, heatmap "
            "generation, and area-of-interest (AOI) analysis."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; line-height: 1.5;")
        layout.addWidget(desc)

        # Features
        features_title = QLabel("Key Features")
        features_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(features_title)

        features = QLabel(
            "• Face landmark extraction using MediaPipe\n"
            "• Automatic gaze-to-template coordinate transformation\n"
            "• Support for both video and image stimuli\n"
            "• Interactive heatmap visualisation with per-participant normalisation\n"
            "• Dual-context AOI analysis (template face and stimulus images)\n"
            "• Flexible data import with column mapping and stimulus name cleaning\n"
            "• Temporal preprocessing with onset/offset trial segmentation\n"
            "• Export to CSV with column selection"
        )
        features.setWordWrap(True)
        features.setStyleSheet("font-size: 13px; line-height: 1.6;")
        layout.addWidget(features)

        # Developed by
        dev_title = QLabel("Developed by")
        dev_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(dev_title)

        dev_info = QLabel(
            "Dr. Gabriel (Naiqi) Xiao with contributions from Dr. Rafael Román-Caballero, "
            "Dr. Wei Fang, Sara Ripley, Dr. Danyang Han, and Dr. Laurel J. Trainor\n"
            "McMaster Baby Lab\n"
            "McMaster University"
        )
        dev_info.setWordWrap(True)
        dev_info.setStyleSheet("font-size: 13px;")
        layout.addWidget(dev_info)

        # Links
        links_title = QLabel("Links")
        links_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(links_title)

        link = QLabel(
            '<a href="https://github.com/McMaster-Baby-Lab/iTemplate2-public" '
            f'style="color: {text_accent};">GitHub Repository</a>'
        )
        link.setOpenExternalLinks(True)
        link.setStyleSheet("font-size: 13px;")
        layout.addWidget(link)

        # Spacer
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)
