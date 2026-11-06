"""
Settings page widget — displays and edits all project-level parameters
in a single scrollable form.  Changes are written to project_settings.json
on Save and propagated to individual step widgets via signals.
"""

import multiprocessing

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QSpinBox, QComboBox,
    QCheckBox, QScrollArea,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from .button_styles import button_style
from .theme import status_icon_success
from .dialog_helpers import show_warning


class SettingsView(QWidget):
    """Centralized settings page shown in the sidebar."""

    # Emitted after Save so main.py can push values to other widgets
    settings_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self._project_settings = None  # assigned by main.py
        self._current_project_path = ""
        self._loading = False
        self._save_debounce_timer = QTimer(self)
        self._save_debounce_timer.setSingleShot(True)
        self._save_debounce_timer.setInterval(700)
        self._save_debounce_timer.timeout.connect(
            lambda: self._on_save(silent=True, autosave=True)
        )

        outer = QVBoxLayout()
        self.setLayout(outer)

        lbl_title = QLabel("Project Settings")
        lbl_title.setObjectName("title-label")
        outer.addWidget(lbl_title)

        # ── Scrollable form ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        form_widget = QWidget()
        form_layout = QVBoxLayout()
        form_widget.setLayout(form_layout)

        # ── 1. Face Extraction ────────────────────────────────
        vp_group = QGroupBox("Face Extraction")
        vp_grid = QGridLayout()

        vp_grid.addWidget(QLabel("Parallel Workers:"), 0, 0)
        self.spin_vp_workers = QSpinBox()
        self.spin_vp_workers.setRange(1, multiprocessing.cpu_count())
        self.spin_vp_workers.setValue(min(4, multiprocessing.cpu_count()))
        vp_grid.addWidget(self.spin_vp_workers, 0, 1)

        vp_group.setLayout(vp_grid)
        form_layout.addWidget(vp_group)

        # ── 2. Temporal Preprocessing ────────────────────────────
        tp_group = QGroupBox("Temporal Preprocessing")
        tp_grid = QGridLayout()

        tp_grid.addWidget(QLabel("Primary temporal column:"), 0, 0)
        self.txt_tp_primary = QLineEdit()
        self.txt_tp_primary.setPlaceholderText("e.g. TIMESTAMP")
        tp_grid.addWidget(self.txt_tp_primary, 0, 1)

        tp_grid.addWidget(QLabel("Secondary temporal column:"), 1, 0)
        self.txt_tp_secondary = QLineEdit()
        self.txt_tp_secondary.setPlaceholderText("e.g. Trial_Index (optional)")
        tp_grid.addWidget(self.txt_tp_secondary, 1, 1)

        tp_grid.addWidget(QLabel("Onset marker column:"), 2, 0)
        self.txt_tp_onset_col = QLineEdit()
        tp_grid.addWidget(self.txt_tp_onset_col, 2, 1)

        tp_grid.addWidget(QLabel("Onset markers (comma-sep):"), 3, 0)
        self.txt_tp_onset_markers = QLineEdit()
        self.txt_tp_onset_markers.setPlaceholderText("e.g. stim_onset, trial_start")
        tp_grid.addWidget(self.txt_tp_onset_markers, 3, 1)

        tp_grid.addWidget(QLabel("Offset mode:"), 4, 0)
        self.combo_tp_offset_mode = QComboBox()
        self.combo_tp_offset_mode.setMinimumWidth(150)
        self.combo_tp_offset_mode.setToolTip("Choose how trial offsets are determined")
        self.combo_tp_offset_mode.addItems(["marker", "duration"])
        tp_grid.addWidget(self.combo_tp_offset_mode, 4, 1)

        tp_grid.addWidget(QLabel("Offset marker column:"), 5, 0)
        self.txt_tp_offset_col = QLineEdit()
        tp_grid.addWidget(self.txt_tp_offset_col, 5, 1)

        tp_grid.addWidget(QLabel("Offset markers (comma-sep):"), 6, 0)
        self.txt_tp_offset_markers = QLineEdit()
        tp_grid.addWidget(self.txt_tp_offset_markers, 6, 1)

        tp_grid.addWidget(QLabel("Offset duration:"), 7, 0)
        self.txt_tp_duration = QLineEdit()
        self.txt_tp_duration.setPlaceholderText("e.g. 5.0")
        tp_grid.addWidget(self.txt_tp_duration, 7, 1)

        tp_grid.addWidget(QLabel("Duration unit:"), 8, 0)
        self.combo_tp_duration_unit = QComboBox()
        self.combo_tp_duration_unit.setMinimumWidth(180)
        self.combo_tp_duration_unit.setToolTip("Unit for the offset duration value")
        self.combo_tp_duration_unit.addItems(["seconds (s)", "milliseconds (ms)"])
        tp_grid.addWidget(self.combo_tp_duration_unit, 8, 1)

        tp_group.setLayout(tp_grid)
        form_layout.addWidget(tp_group)

        # ── 3. Coordinate Setup ──────────────────────────────────
        cs_group = QGroupBox("Coordinate Setup")
        cs_grid = QGridLayout()

        cs_grid.addWidget(QLabel("Screen Width (px):"), 0, 0)
        self.txt_cs_screen_w = QLineEdit()
        self.txt_cs_screen_w.setPlaceholderText("e.g. 1920")
        self.txt_cs_screen_w.setValidator(QIntValidator(1, 16384))
        cs_grid.addWidget(self.txt_cs_screen_w, 0, 1)

        cs_grid.addWidget(QLabel("Screen Height (px):"), 0, 2)
        self.txt_cs_screen_h = QLineEdit()
        self.txt_cs_screen_h.setPlaceholderText("e.g. 1080")
        self.txt_cs_screen_h.setValidator(QIntValidator(1, 16384))
        cs_grid.addWidget(self.txt_cs_screen_h, 0, 3)

        self.chk_cs_centered = QCheckBox("Stimuli centered on screen")
        self.chk_cs_centered.setChecked(True)
        cs_grid.addWidget(self.chk_cs_centered, 1, 0, 1, 2)

        cs_grid.addWidget(QLabel("Center X (px):"), 2, 0)
        self.txt_cs_center_x = QLineEdit()
        self.txt_cs_center_x.setPlaceholderText("auto = screen_w / 2")
        self.txt_cs_center_x.setValidator(QDoubleValidator(-999, 999, 2))
        cs_grid.addWidget(self.txt_cs_center_x, 2, 1)

        cs_grid.addWidget(QLabel("Center Y (px):"), 2, 2)
        self.txt_cs_center_y = QLineEdit()
        self.txt_cs_center_y.setPlaceholderText("auto = screen_h / 2")
        self.txt_cs_center_y.setValidator(QDoubleValidator(-999, 999, 2))
        cs_grid.addWidget(self.txt_cs_center_y, 2, 3)

        cs_grid.addWidget(QLabel("Frame Rate (FPS):"), 3, 0)
        self.txt_cs_fps = QLineEdit()
        self.txt_cs_fps.setPlaceholderText("auto-detected")
        self.txt_cs_fps.setValidator(QDoubleValidator(0.1, 9999, 2))
        cs_grid.addWidget(self.txt_cs_fps, 3, 1)

        cs_grid.addWidget(QLabel("Gaze Response Lag (ms):"), 3, 2)
        self.txt_cs_delay = QLineEdit("0")
        self.txt_cs_delay.setValidator(QIntValidator(0, 9999))
        cs_grid.addWidget(self.txt_cs_delay, 3, 3)

        cs_group.setLayout(cs_grid)
        form_layout.addWidget(cs_group)

        # ── 4. Transformation ────────────────────────────────────────
        pr_group = QGroupBox("Transformation")
        pr_grid = QGridLayout()

        pr_grid.addWidget(QLabel("Stimuli Type:"), 0, 0)
        self.combo_pr_type = QComboBox()
        self.combo_pr_type.setMinimumWidth(150)
        self.combo_pr_type.setToolTip("Type of stimuli being processed")
        self.combo_pr_type.addItems(["video", "image"])
        pr_grid.addWidget(self.combo_pr_type, 0, 1)

        pr_grid.addWidget(QLabel("Parallel Workers:"), 1, 0)
        self.spin_pr_workers = QSpinBox()
        self.spin_pr_workers.setRange(1, multiprocessing.cpu_count())
        self.spin_pr_workers.setValue(min(4, multiprocessing.cpu_count()))
        pr_grid.addWidget(self.spin_pr_workers, 1, 1)

        pr_group.setLayout(pr_grid)
        form_layout.addWidget(pr_group)

        # ── 5. Visualization ─────────────────────────────────────
        vis_group = QGroupBox("Visualization (Heatmap)")
        vis_grid = QGridLayout()

        vis_grid.addWidget(QLabel("Bins:"), 0, 0)
        self.txt_vis_bins = QLineEdit()
        self.txt_vis_bins.setPlaceholderText("100")
        self.txt_vis_bins.setValidator(QIntValidator(1, 200))
        vis_grid.addWidget(self.txt_vis_bins, 0, 1)

        vis_grid.addWidget(QLabel("Sigma:"), 0, 2)
        self.txt_vis_sigma = QLineEdit()
        self.txt_vis_sigma.setPlaceholderText("2")
        self.txt_vis_sigma.setValidator(QDoubleValidator(0.1, 100, 2))
        vis_grid.addWidget(self.txt_vis_sigma, 0, 3)

        self.chk_vis_scatter = QCheckBox("Show scatter")
        self.chk_vis_scatter.setChecked(False)
        vis_grid.addWidget(self.chk_vis_scatter, 1, 0, 1, 2)

        self.chk_vis_normalize = QCheckBox("Normalize by participant")
        self.chk_vis_normalize.setChecked(True)
        vis_grid.addWidget(self.chk_vis_normalize, 1, 2, 1, 2)

        vis_group.setLayout(vis_grid)
        form_layout.addWidget(vis_group)

        # ── Spacer ───────────────────────────────────────────────
        form_layout.addStretch()

        scroll.setWidget(form_widget)
        outer.addWidget(scroll, 1)

        # ── Save button ─────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.setStyleSheet(button_style("primary", padding="10px 30px"))
        self.btn_save.clicked.connect(lambda: self._on_save(silent=False, autosave=False))
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.lbl_status)
        
        self._connect_signals()

    def _connect_signals(self):
        """Connect all input widgets to auto-save."""
        # VP
        self.spin_vp_workers.valueChanged.connect(self._auto_save_triggered)
        
        # TP
        for w in [self.txt_tp_primary, self.txt_tp_secondary, self.txt_tp_onset_col,
                  self.txt_tp_onset_markers, self.txt_tp_offset_col, 
                  self.txt_tp_offset_markers, self.txt_tp_duration]:
            w.editingFinished.connect(self._auto_save_triggered)
        
        self.combo_tp_offset_mode.currentIndexChanged.connect(self._auto_save_triggered)
        self.combo_tp_duration_unit.currentIndexChanged.connect(self._auto_save_triggered)
        
        # CS
        for w in [self.txt_cs_screen_w, self.txt_cs_screen_h, self.txt_cs_center_x,
                  self.txt_cs_center_y, self.txt_cs_fps, self.txt_cs_delay]:
            w.textChanged.connect(self._auto_save_triggered)
        self.chk_cs_centered.toggled.connect(self._auto_save_triggered)
        
        # Processing
        self.combo_pr_type.currentIndexChanged.connect(self._auto_save_triggered)
        self.spin_pr_workers.valueChanged.connect(self._auto_save_triggered)
        
        # Visualization
        self.txt_vis_bins.editingFinished.connect(self._auto_save_triggered)
        self.txt_vis_sigma.editingFinished.connect(self._auto_save_triggered)
        self.chk_vis_scatter.toggled.connect(self._auto_save_triggered)
        self.chk_vis_normalize.toggled.connect(self._auto_save_triggered)

    def _auto_save_triggered(self):
        if self._loading:
            return
        if self._project_settings is None or not self._current_project_path:
            return
        self._save_debounce_timer.start()

    # ── public helpers ────────────────────────────────────────────

    def set_settings(self, project_settings):
        """Store a reference to the shared ProjectSettings instance."""
        self._project_settings = project_settings

    def set_project_path(self, path):
        """Called when the active project changes — refresh UI from current settings."""
        self._current_project_path = path
        if self._project_settings is None:
            return
        self._populate_from_settings()
        self.lbl_status.setText("")

    def refresh_from_settings(self):
        """Re-read values from the shared ProjectSettings (no disk I/O)."""
        self._populate_from_settings()

    # ── Save ──────────────────────────────────────────────────────

    def _on_save(self, silent=False, autosave=False):
        if self._project_settings is None or not self._current_project_path:
            if not silent:
                show_warning(self, "No Project Open",
                                    "Select a project in Project Manager first.")
            return
        self._write_to_settings()
        self._project_settings.save()
        self.lbl_status.setText("Autosaved" if autosave else f"{status_icon_success} Settings saved")
        self.lbl_status.setObjectName("status-success-label")
        self.settings_changed.emit()

    # ── Mapping: UI ↔ ProjectSettings ─────────────────────────────

    def _populate_from_settings(self):
        s = self._project_settings
        if s is None:
            return
            
        self._loading = True
        try:
            # Video Preprocessor
            self.spin_vp_workers.setValue(
                s.get("video_preprocessor", "parallel_workers",
                      min(4, multiprocessing.cpu_count())))

            # Temporal Preprocessing
            self.txt_tp_primary.setText(
                s.get("temporal_preprocessing", "primary_temporal_col", ""))
            self.txt_tp_secondary.setText(
                s.get("temporal_preprocessing", "secondary_temporal_col", ""))
            self.txt_tp_onset_col.setText(
                s.get("temporal_preprocessing", "onset_marker_col", ""))
            markers = s.get("temporal_preprocessing", "onset_markers", [])
            self.txt_tp_onset_markers.setText(", ".join(markers) if markers else "")
            mode = s.get("temporal_preprocessing", "offset_mode", "marker")
            idx = self.combo_tp_offset_mode.findText(mode)
            if idx >= 0:
                self.combo_tp_offset_mode.setCurrentIndex(idx)
            self.txt_tp_offset_col.setText(
                s.get("temporal_preprocessing", "offset_marker_col", ""))
            off_markers = s.get("temporal_preprocessing", "offset_markers", [])
            self.txt_tp_offset_markers.setText(", ".join(off_markers) if off_markers else "")
            self.txt_tp_duration.setText(
                s.get("temporal_preprocessing", "offset_duration", ""))
            unit = s.get("temporal_preprocessing", "offset_duration_unit", "seconds (s)")
            u_idx = self.combo_tp_duration_unit.findText(unit)
            if u_idx >= 0:
                self.combo_tp_duration_unit.setCurrentIndex(u_idx)

            # Coordinate Setup
            self.txt_cs_screen_w.setText(
                str(s.get("coordinate_setup", "screen_width", "")))
            self.txt_cs_screen_h.setText(
                str(s.get("coordinate_setup", "screen_height", "")))
            self.chk_cs_centered.setChecked(
                s.get("coordinate_setup", "stimuli_centered", True))
            self.txt_cs_center_x.setText(
                str(s.get("coordinate_setup", "center_x", "")))
            self.txt_cs_center_y.setText(
                str(s.get("coordinate_setup", "center_y", "")))
            self.txt_cs_fps.setText(
                str(s.get("coordinate_setup", "frame_rate_fps", "")))
            self.txt_cs_delay.setText(
                str(s.get("coordinate_setup", "gaze_response_lag", "0")))

            # Processing
            stype = s.get("processing", "stimuli_type", "video")
            t_idx = self.combo_pr_type.findText(stype)
            if t_idx >= 0:
                self.combo_pr_type.setCurrentIndex(t_idx)
            self.spin_pr_workers.setValue(
                s.get("processing", "parallel_workers",
                      min(4, multiprocessing.cpu_count())))

            # Visualization
            self.txt_vis_bins.setText(
                str(s.get("visualization", "bins", "")))
            self.txt_vis_sigma.setText(
                str(s.get("visualization", "sigma", "")))
            self.chk_vis_scatter.setChecked(
                s.get("visualization", "show_scatter", False))
            self.chk_vis_normalize.setChecked(
                s.get("visualization", "normalize_by_participant", True))
        finally:
            self._loading = False

    def _write_to_settings(self):
        s = self._project_settings
        if s is None:
            return

        # Optimize: temporarily disable auto-save to avoid multiple disk writes
        was_auto = s.auto_save
        s.auto_save = False
        try:
            # Video Preprocessor
            s.set("video_preprocessor", "parallel_workers",
                  self.spin_vp_workers.value())

            # Temporal Preprocessing
            s.set("temporal_preprocessing", "primary_temporal_col",
                  self.txt_tp_primary.text().strip())
            s.set("temporal_preprocessing", "secondary_temporal_col",
                  self.txt_tp_secondary.text().strip())
            s.set("temporal_preprocessing", "onset_marker_col",
                  self.txt_tp_onset_col.text().strip())
            raw = self.txt_tp_onset_markers.text().strip()
            s.set("temporal_preprocessing", "onset_markers",
                  [m.strip() for m in raw.split(",") if m.strip()] if raw else [])
            s.set("temporal_preprocessing", "offset_mode",
                  self.combo_tp_offset_mode.currentText())
            s.set("temporal_preprocessing", "offset_marker_col",
                  self.txt_tp_offset_col.text().strip())
            raw_off = self.txt_tp_offset_markers.text().strip()
            s.set("temporal_preprocessing", "offset_markers",
                  [m.strip() for m in raw_off.split(",") if m.strip()] if raw_off else [])
            s.set("temporal_preprocessing", "offset_duration",
                  self.txt_tp_duration.text().strip())
            s.set("temporal_preprocessing", "offset_duration_unit",
                  self.combo_tp_duration_unit.currentText())

            # Coordinate Setup
            s.set("coordinate_setup", "screen_width",
                  self.txt_cs_screen_w.text().strip())
            s.set("coordinate_setup", "screen_height",
                  self.txt_cs_screen_h.text().strip())
            s.set("coordinate_setup", "stimuli_centered",
                  self.chk_cs_centered.isChecked())
            s.set("coordinate_setup", "center_x",
                  self.txt_cs_center_x.text().strip())
            s.set("coordinate_setup", "center_y",
                  self.txt_cs_center_y.text().strip())
            s.set("coordinate_setup", "frame_rate_fps",
                  self.txt_cs_fps.text().strip())
            s.set("coordinate_setup", "gaze_response_lag",
                  self.txt_cs_delay.text().strip())

            # Processing
            s.set("processing", "stimuli_type",
                  self.combo_pr_type.currentText())
            s.set("processing", "parallel_workers",
                  self.spin_pr_workers.value())

            # Visualization
            s.set("visualization", "bins",
                  self.txt_vis_bins.text().strip())
            s.set("visualization", "sigma",
                  self.txt_vis_sigma.text().strip())
            s.set("visualization", "show_scatter",
                  self.chk_vis_scatter.isChecked())
            s.set("visualization", "normalize_by_participant",
                  self.chk_vis_normalize.isChecked())
        finally:
            s.auto_save = was_auto
