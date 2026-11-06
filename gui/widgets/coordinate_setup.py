from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QCheckBox,
                             QGroupBox, QGridLayout, QFileDialog, QApplication,
                             QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QCloseEvent, QIntValidator, QDoubleValidator
from PyQt6.QtGui import QColor
from .button_styles import button_style
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm
from .processing_utils import ProcessingButton, ProcessingWorker, ProcessingOverlay
from .theme import (
    status_warning, status_info,
    status_icon_success, status_icon_error,
    status_icon_warning, status_icon_info)
import os
import logging
import cv2
import json
import subprocess
import numpy as np
import pandas as pd
from core.io_utils import atomic_save_parquet
from core.spatial_prep import normalize_screen, normalize_wearable, assign_frames
from core.data_resolver import is_imported_data_file

logger = logging.getLogger('iTemplate2')


class _StimuliScanWorker(QThread):
    """Scan stimuli folder for video/image dimensions in background."""
    progress = pyqtSignal(str, int, int, float, int, str)  # name, w, h, fps, frames, type
    finished = pyqtSignal()

    def __init__(self, stimuli_dir):
        super().__init__()
        self.stimuli_dir = stimuli_dir

    @staticmethod
    def _get_fps_ffprobe(fpath):
        """Fallback: get FPS from ffprobe when cv2 returns garbage."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", fpath,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            r_frame_rate = stream.get("r_frame_rate", "")
            if r_frame_rate and "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                fps = int(num) / max(int(den), 1)
                if 1 <= fps <= 1000:
                    return round(fps, 2)
            # Fallback to avg_frame_rate
            avg_frame_rate = stream.get("avg_frame_rate", "")
            if avg_frame_rate and "/" in avg_frame_rate:
                num, den = avg_frame_rate.split("/")
                fps = int(num) / max(int(den), 1)
                if 1 <= fps <= 1000:
                    return round(fps, 2)
        except Exception as e:
            logger.error("ffprobe failed for %s: %s", fpath, e)
        return None

    def run(self):
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = sorted(os.listdir(self.stimuli_dir))
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
                continue
            fpath = os.path.join(self.stimuli_dir, fname)
            base_name = os.path.splitext(fname)[0]
            width, height, fps, frames = 0, 0, 0, 0
            ftype = "image" if ext in IMAGE_EXTS else "video"
            try:
                if ftype == "video":
                    cap = cv2.VideoCapture(fpath)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps_raw = cap.get(cv2.CAP_PROP_FPS)
                        if isinstance(fps_raw, (int, float)) and 1 <= fps_raw <= 1000:
                            fps = float(round(fps_raw, 2))
                        else:
                            fps = self._get_fps_ffprobe(fpath)
                            if fps is None or not (1 <= fps <= 1000):
                                fps = 0.0
                        
                        try:
                            frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            frames = int(frames_raw) if (isinstance(frames_raw, (int, float)) and frames_raw > 0) else 0
                        except:
                            frames = 0
                    cap.release()
                else:
                    img = cv2.imread(fpath)
                    if img is not None:
                        height, width = img.shape[:2]
            except Exception as e:
                logger.error("Error reading %s: %s", fname, e)
                continue
            self.progress.emit(base_name, width, height, fps, frames, ftype)
        self.finished.emit()


class CoordinateSetup(QWidget):
    data_changed = pyqtSignal()  # Signal for when data is modified (e.g. frames assigned)

    def __init__(self):
        super().__init__()
        
        self.current_project_path = ""
        self.gaze_data_path = ""
        self.stimuli_dir = ""
        self.stim_dimensions = {}  # {stim_name: {'width': w, 'height': h, 'fps': fps, 'type': t, 'frames': f}}
        self._project_settings = None
        self._syncing_selection = False
        self._loading = False
        self._pending_stimuli_scan = False
        self._overlay = ProcessingOverlay(self)

        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        lbl_title = QLabel("Step 2.5b: Coordinate Setup, Normalization, & Video Frame Index Assignment")
        lbl_title.setObjectName("title-label")
        layout.addWidget(lbl_title)

        # === Top-of-page notice banner ===
        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.hide()
        layout.addWidget(self.lbl_notice)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        # === Section 1: Stimulus Dimensions ===
        stim_group = QGroupBox("Stimulus Dimensions")
        stim_layout = QVBoxLayout()
        
        stim_btn_layout = QHBoxLayout()
        self.btn_select_stimuli = QPushButton("Select and Scan Stimuli Folder")
        self.btn_select_stimuli.clicked.connect(self.select_stimuli_dir)
        stim_btn_layout.addWidget(self.btn_select_stimuli)
        stim_layout.addLayout(stim_btn_layout)
        
        self.stim_table = QTableWidget()
        self.stim_table.setColumnCount(5)
        self.stim_table.setHorizontalHeaderLabels(["Stimulus Name", "Type", "Width", "Height", "FPS"])
        self.stim_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stim_layout.addWidget(self.stim_table)
        
        self.lbl_stim_summary = QLabel("")
        stim_layout.addWidget(self.lbl_stim_summary)
        
        stim_group.setLayout(stim_layout)
        layout.addWidget(stim_group)
        
        # === Eye-Tracker Mode Tabs ===
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setStyleSheet("")

        # -- Tab 0: Screen-based --
        screen_tab = QWidget()
        screen_tab.setStyleSheet("")
        screen_layout = QGridLayout()
        
        screen_layout.addWidget(QLabel("Screen Width (px):"), 0, 0)
        self.txt_screen_w = QLineEdit()
        self.txt_screen_w.setPlaceholderText("e.g. 1920")
        self.txt_screen_w.setValidator(QIntValidator(1, 16384))
        self.txt_screen_w.setToolTip("Screen width in pixels (1–16384)")
        screen_layout.addWidget(self.txt_screen_w, 0, 1)
        
        screen_layout.addWidget(QLabel("Screen Height (px):"), 0, 2)
        self.txt_screen_h = QLineEdit()
        self.txt_screen_h.setPlaceholderText("e.g. 1080")
        self.txt_screen_h.setValidator(QIntValidator(1, 16384))
        self.txt_screen_h.setToolTip("Screen height in pixels (1–16384)")
        screen_layout.addWidget(self.txt_screen_h, 0, 3)
        
        self.chk_centered = QCheckBox("Stimuli centered on screen")
        self.chk_centered.setChecked(True)
        self.chk_centered.stateChanged.connect(self.toggle_center_fields)
        screen_layout.addWidget(self.chk_centered, 1, 0, 1, 2)
        
        self.lbl_center_x = QLabel("Center X (px):")
        self.txt_center_x = QLineEdit()
        self.txt_center_x.setPlaceholderText("auto = screen_w / 2")
        self.txt_center_x.setValidator(QDoubleValidator(-999, 999, 2))
        self.txt_center_x.setToolTip("Center X coordinate in pixels (-999–999, up to 2 decimals)")
        self.txt_center_x.setEnabled(False)
        screen_layout.addWidget(self.lbl_center_x, 2, 0)
        screen_layout.addWidget(self.txt_center_x, 2, 1)
        
        self.lbl_center_y = QLabel("Center Y (px):")
        self.txt_center_y = QLineEdit()
        self.txt_center_y.setPlaceholderText("auto = screen_h / 2")
        self.txt_center_y.setValidator(QDoubleValidator(-999, 999, 2))
        self.txt_center_y.setToolTip("Center Y coordinate in pixels (-999–999, up to 2 decimals)")
        self.txt_center_y.setEnabled(False)
        screen_layout.addWidget(self.lbl_center_y, 2, 2)
        screen_layout.addWidget(self.txt_center_y, 2, 3)
        
        screen_tab.setLayout(screen_layout)
        self.mode_tabs.addTab(screen_tab, "Screen-based Eye Tracker")

        # -- Tab 1: Wearable --
        wearable_tab = QWidget()
        wearable_tab.setStyleSheet("")
        wearable_layout = QGridLayout()

        lbl_wearable_info = QLabel(
            "Wearable / Scene-based Eye Tracker will normalize gaze coordinates to the scene video dimensions.\n"
            "Press the 'Normalize Coordinates' button to normalize gaze coordinates to the scene video dimensions."
        )
        lbl_wearable_info.setWordWrap(True)
        lbl_wearable_info.setObjectName("description-label")
        wearable_layout.addWidget(lbl_wearable_info, 0, 0, 1, 2)

        wearable_tab.setLayout(wearable_layout)
        self.mode_tabs.addTab(wearable_tab, "Wearable / Scene-based Eye Tracker")

        self.mode_tabs.setMaximumHeight(144)
        layout.addWidget(self.mode_tabs)
        
        # === Section 3: Data File Selection ===
        data_group = QGroupBox("Gaze Data")
        data_layout = QHBoxLayout()
        
        self.btn_select_data = QPushButton("Select Parquet File")
        self.btn_select_data.clicked.connect(self.select_data_file)
        data_layout.addWidget(self.btn_select_data)
        
        self.lbl_data_path = QLabel("No file selected")
        data_layout.addWidget(self.lbl_data_path)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        

        
        # === Normalize Button ===
        self.btn_normalize = ProcessingButton("Normalize Coordinates")
        self.btn_normalize.setStyleSheet(button_style("primary", padding="10px"))
        self.btn_normalize.clicked.connect(self.run_normalization)
        layout.addWidget(self.btn_normalize)
        
        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)
        
        # === Section 4: Video Frame Index (only for video stimuli) ===
        self.frame_group = QGroupBox("Video Frame Index Assignment")
        frame_layout = QVBoxLayout()
        
        self.lbl_frame_desc = QLabel(
            "For video stimuli, assign a frame index to each gaze point based on TIMESTAMP.\n"
            "Requires: Participant_ID, stimulusName, TIMESTAMP columns, and trial_event markers."
        )
        frame_layout.addWidget(self.lbl_frame_desc)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.txt_fps = QLineEdit()
        self.txt_fps.setPlaceholderText("auto-detected from stimuli scan")
        self.txt_fps.setValidator(QDoubleValidator(0.1, 9999, 2))
        self.txt_fps.setToolTip("Frames per second: positive number (0.1–9999, up to 2 decimals)")
        fps_layout.addWidget(self.txt_fps)
        
        fps_layout.addWidget(QLabel("Gaze Response Lag (ms):"))
        self.txt_video_delay = QLineEdit("0")
        self.txt_video_delay.setValidator(QIntValidator(0, 9999))
        self.txt_video_delay.setToolTip(
            "Gaze response lag in milliseconds (0–9999). "
            "Typical values: 200–400 ms for the eye movement to begin after a head turn."
        )
        self.txt_video_delay.editingFinished.connect(self._on_settings_edited)
        fps_layout.addWidget(self.txt_video_delay)
        
        frame_layout.addLayout(fps_layout)
        
        self.btn_assign_frames = ProcessingButton("Assign Frame Index")
        self.btn_assign_frames.setStyleSheet(button_style("secondary", padding="8px"))
        self.btn_assign_frames.clicked.connect(self.assign_frame_index)
        frame_layout.addWidget(self.btn_assign_frames)
        
        self.lbl_frame_status = QLabel("")
        frame_layout.addWidget(self.lbl_frame_status)
        
        self.frame_group.setLayout(frame_layout)
        self.frame_group.setEnabled(False)
        layout.addWidget(self.frame_group)
        layout.addStretch()

        # Connect other signals for auto-save
        self.txt_screen_w.textChanged.connect(self._on_settings_edited)
        self.txt_screen_h.textChanged.connect(self._on_settings_edited)
        self.chk_centered.toggled.connect(self._on_settings_edited)
        self.txt_center_x.editingFinished.connect(self._on_settings_edited)
        self.txt_center_y.editingFinished.connect(self._on_settings_edited)
        self.txt_fps.editingFinished.connect(self._on_settings_edited)
        self.mode_tabs.currentChanged.connect(self._on_et_mode_changed)

    # ------------------------------------------------------------------
    # ET Mode
    # ------------------------------------------------------------------
    def _is_wearable(self):
        return self.mode_tabs.currentIndex() == 1

    def _on_et_mode_changed(self):
        """Update descriptions based on active tab."""
        if self._is_wearable():
            self.lbl_frame_desc.setText(
                "For scene videos, assign a frame index to each gaze point based on TIMESTAMP.\n"
                "Matches Participant_ID to scene video filename."
            )
        else:
            self.lbl_frame_desc.setText(
                "For video stimuli, assign a frame index to each gaze point based on TIMESTAMP.\n"
                "Requires: Participant_ID, stimulusName, TIMESTAMP columns, and trial_event markers."
            )
        self._on_settings_edited()
    
    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings
        self._loading = True

        selected = settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != self.gaze_data_path:
            self._syncing_selection = True
            try:
                self._set_gaze_data_file(selected, update_settings=False)
            finally:
                self._syncing_selection = False
        
        # Block signals during load
        widgets_to_block = [
            self.mode_tabs,
            self.txt_screen_w, self.txt_screen_h, self.chk_centered,
            self.txt_center_x, self.txt_center_y, self.txt_fps,
            self.txt_video_delay,
        ]
        for w in widgets_to_block:
            w.blockSignals(True)
        try:
            # ET mode
            et_mode = settings.get("coordinate_setup", "et_mode", "screen_based")
            self.mode_tabs.setCurrentIndex(1 if et_mode == "wearable" else 0)
            self._on_et_mode_changed()

            sw = settings.get("coordinate_setup", "screen_width", "")
            if sw:
                self.txt_screen_w.setText(str(sw))
            sh = settings.get("coordinate_setup", "screen_height", "")
            if sh:
                self.txt_screen_h.setText(str(sh))
            centered = settings.get("coordinate_setup", "stimuli_centered", True)
            self.chk_centered.setChecked(centered)
            cx = settings.get("coordinate_setup", "center_x", "")
            if cx:
                self.txt_center_x.setText(str(cx))
            cy = settings.get("coordinate_setup", "center_y", "")
            if cy:
                self.txt_center_y.setText(str(cy))
            fps = settings.get("coordinate_setup", "frame_rate_fps", "")
            if fps:
                self.txt_fps.setText(str(fps))
            delay = settings.get("coordinate_setup", "gaze_response_lag", "0")
            self.txt_video_delay.setText(str(delay))
            # Restore stimuli folder and defer any expensive scan until this page is shown.
            stim_dir = settings.get("coordinate_setup", "stimuli_dir", "")
            if stim_dir and os.path.isdir(stim_dir):
                self.stimuli_dir = stim_dir
                self._pending_stimuli_scan = True
                if self.isVisible():
                    self._maybe_run_pending_stimuli_scan()
                else:
                    self.lbl_stim_summary.setText(
                        f"Stimuli folder ready: {os.path.basename(stim_dir)}/ (scan will run when this page opens)"
                    )
        finally:
            self._loading = False
            for w in widgets_to_block:
                w.blockSignals(False)
            # Ensure center fields state updates
            self.toggle_center_fields(None)

    def showEvent(self, event):
        super().showEvent(event)
        self._maybe_run_pending_stimuli_scan()

    def _maybe_run_pending_stimuli_scan(self):
        if not self._pending_stimuli_scan:
            return
        if not self.stimuli_dir or not os.path.isdir(self.stimuli_dir):
            self._pending_stimuli_scan = False
            return
        self._pending_stimuli_scan = False
        self.scan_stimuli()

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        settings.set("coordinate_setup", "et_mode",
                     "wearable" if self._is_wearable() else "screen_based")
        settings.set("coordinate_setup", "screen_width",
                     self.txt_screen_w.text().strip())
        settings.set("coordinate_setup", "screen_height",
                     self.txt_screen_h.text().strip())
        settings.set("coordinate_setup", "stimuli_centered",
                     self.chk_centered.isChecked())
        settings.set("coordinate_setup", "center_x",
                     self.txt_center_x.text().strip())
        settings.set("coordinate_setup", "center_y",
                     self.txt_center_y.text().strip())
        settings.set("coordinate_setup", "frame_rate_fps",
                     self.txt_fps.text().strip())
        settings.set("coordinate_setup", "gaze_response_lag",
                     self.txt_video_delay.text().strip())
        if self.stimuli_dir:
            settings.set("coordinate_setup", "stimuli_dir", self.stimuli_dir)

    def _on_settings_edited(self):
        """Auto-save settings when user edits any field."""
        if self._loading:
            return
        if self._project_settings:
            self.save_settings(self._project_settings)

    def set_project_path(self, path):
        self.current_project_path = path
        
        # Auto-populate stimuli folder path (but don't scan yet — it's expensive)
        stimuli_dir = os.path.join(path, "stimuli")
        if os.path.isdir(stimuli_dir):
            self.stimuli_dir = stimuli_dir
        
        if self._project_settings:
            selected = self._project_settings.resolve_working_data_file("")
            if selected and os.path.isfile(selected):
                self._set_gaze_data_file(selected, update_settings=False)
    
    def refresh_data(self):
        """Reload the last used project path to refresh file lists."""
        logger.debug("CoordinateSetup.refresh_data called. Path=%s", self.current_project_path)
        if self.current_project_path:
            self.set_project_path(self.current_project_path)

    def toggle_center_fields(self, state):
        custom = not self.chk_centered.isChecked()
        self.txt_center_x.setEnabled(custom)
        self.txt_center_y.setEnabled(custom)
        if not custom:
            self.txt_center_x.clear()
            self.txt_center_y.clear()
    
    def select_stimuli_dir(self):
        default_dir = self.current_project_path or ""
        d = QFileDialog.getExistingDirectory(self, "Select Stimuli Folder", default_dir)
        if d:
            self.stimuli_dir = d
            self.scan_stimuli()
            self._on_settings_edited()  # persist the new stimuli_dir
    
    def select_data_file(self):
        start = self.current_project_path if self.current_project_path else ""
        f, _ = QFileDialog.getOpenFileName(self, "Select Gaze Data File", start, "Parquet Files (*.parquet)")
        if f:
            self._set_gaze_data_file(f, update_settings=True)

    def _set_global_selected_file(self, path):
        if not self._project_settings or self._syncing_selection:
            return
        if self.current_project_path and is_imported_data_file(self.current_project_path, path):
            self._project_settings.set("global", "source_data_file", path)
        self._project_settings.set("global", "selected_data_file", path)

    def _set_gaze_data_file(self, path, update_settings=False):
        if not path or not os.path.isfile(path):
            return
        self.gaze_data_path = path
        self.lbl_data_path.setText(os.path.basename(path))
        self._update_processing_notice(path)
        if update_settings:
            self._set_global_selected_file(path)

    def _update_processing_notice(self, path):
        # Check if processing was already done
        try:
            import pyarrow.parquet as pq
            cols = pq.read_schema(path).names
            notices = []
            if '_video_frame_index' in cols:
                notices.append(f"{status_icon_success} Video frame index already assigned.")
            if 'stimuli_x' in cols and 'stimuli_y' in cols:
                notices.append(f"{status_icon_success} Coordinate normalization already done.")
            if notices:
                self.lbl_notice.setText(
                    f"{status_icon_info} Previously processed:\n" + "\n".join(notices) +
                    "\nYou can skip this step or re-run to overwrite."
                )
                self.lbl_notice.show()
            else:
                self.lbl_notice.hide()
        except Exception:
            self.lbl_notice.hide()
    
    def scan_stimuli(self):
        """Scan the stimuli folder and extract dimensions for each file."""
        if not self.stimuli_dir or not os.path.isdir(self.stimuli_dir):
            show_warning(self, "No Folder Selected", "Please select a stimuli folder first.")
            return

        self.stim_table.setRowCount(0)
        self.stim_dimensions.clear()
        self.lbl_stim_summary.setText("Scanning stimuli...")
        self._overlay.set_status("Scanning Stimuli")
        self._overlay.hide_progress()
        self._overlay.show()

        self._scan_worker = _StimuliScanWorker(self.stimuli_dir)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_progress(self, name, width, height, fps, frames, ftype):
        self.stim_dimensions[name] = {
            'width': width, 'height': height, 'fps': fps, 'type': ftype,
            'frames': frames
        }
        row = self.stim_table.rowCount()
        self.stim_table.insertRow(row)
        self.stim_table.setItem(row, 0, QTableWidgetItem(name))
        self.stim_table.setItem(row, 1, QTableWidgetItem(ftype))
        self.stim_table.setItem(row, 2, QTableWidgetItem(str(width)))
        self.stim_table.setItem(row, 3, QTableWidgetItem(str(height)))
        self.stim_table.setItem(row, 4, QTableWidgetItem(f"{fps:.2f}" if (ftype == "video" and 0 < fps < 10000) else "—"))

    def _on_scan_finished(self):
        self._overlay.hide()
        n = len(self.stim_dimensions)
        self.lbl_stim_summary.setText(f"Found {n} stimuli in {os.path.basename(self.stimuli_dir)}/")
        video_fps_values = [d['fps'] for d in self.stim_dimensions.values()
                            if d['type'] == 'video' and d['fps'] > 0]
        if video_fps_values:
            common_fps = max(set(video_fps_values), key=video_fps_values.count)
            self.txt_fps.setText(str(common_fps))
            self.frame_group.setEnabled(True)
        else:
            self.frame_group.setEnabled(False)
    
    def _get_screen_params(self):
        """Parse and validate screen parameters. Returns (screen_w, screen_h, center_x, center_y) or None."""
        try:
            screen_w = int(self.txt_screen_w.text().strip())
            screen_h = int(self.txt_screen_h.text().strip())
        except ValueError:
            show_warning(self, "Invalid Input", "Please enter valid screen width and height in pixels.")
            return None
        
        if self.chk_centered.isChecked():
            center_x = screen_w / 2
            center_y = screen_h / 2
        else:
            try:
                center_x = float(self.txt_center_x.text().strip())
                center_y = float(self.txt_center_y.text().strip())
            except ValueError:
                show_warning(self, "Invalid Input", "Please enter valid center X and Y coordinates in pixels.")
                return None
        
        return screen_w, screen_h, center_x, center_y
    
    def run_normalization(self):
        """Normalize gaze coordinates per stimulus using the screen and stimulus dimensions."""
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file before normalizing coordinates.")
            return

        if not self.stim_dimensions:
            show_warning(self, "No Stimuli Found", "Please scan the stimuli folder first.")
            return
        
        self.btn_normalize.start_processing("Normalizing")
        self.lbl_status.setText("Normalizing coordinates...")
        self._overlay.set_status("Normalizing Coordinates")
        self._overlay.hide_progress()
        self._overlay.show()
        
        if self._is_wearable():
            scene_col = None
            scene_kw = None
            if self._project_settings:
                scene_col = self._project_settings.get(
                    "temporal_preprocessing", "kw_filter_col", "") or None
                scene_kw = self._project_settings.get(
                    "temporal_preprocessing", "kw_filter_value", "") or None
            self._norm_worker = ProcessingWorker(
                self._do_normalization_wearable,
                self.gaze_data_path, self.stim_dimensions, scene_col, scene_kw
            )
        else:
            params = self._get_screen_params()
            if params is None:
                self.btn_normalize.stop_processing()
                return
            screen_w, screen_h, center_x, center_y = params
            self._norm_worker = ProcessingWorker(
                self._do_normalization,
                self.gaze_data_path, self.stim_dimensions,
                center_x, center_y
            )
        
        self.btn_normalize.connect_worker(self._norm_worker)
        self._norm_worker.finished.connect(self._on_normalization_done)
        self._norm_worker.error.connect(self._on_normalization_error)
        self._norm_worker.start()
    
    # ── Screen-based normalization ──
    @staticmethod
    def _do_normalization(gaze_path, stim_dimensions, center_x, center_y, progress_callback=None):
        """Normalization for screen-based ET — delegates to core.spatial_prep."""
        return normalize_screen(gaze_path, stim_dimensions, center_x, center_y, progress_callback)

    # ── Wearable normalization ──
    @staticmethod
    def _do_normalization_wearable(gaze_path, stim_dimensions, scene_col, scene_keyword, progress_callback=None):
        """Normalization for wearable ET — delegates to core.spatial_prep."""
        return normalize_wearable(gaze_path, stim_dimensions, scene_col, scene_keyword, progress_callback)
    
    def _on_normalization_done(self, result):
        self._overlay.hide()
        self.btn_normalize.stop_processing()
        matched = result['matched']
        total = result['total']
        unmatched = result['unmatched']
        
        msg = f"{status_icon_success} Normalization complete!\n\nStimuli matched: {matched}/{total}"
        if unmatched:
            msg += f"\n\nUnmatched stimuli ({len(unmatched)}):\n" + "\n".join(f"  • {s}" for s in unmatched[:10])
            if len(unmatched) > 10:
                msg += f"\n  ... and {len(unmatched) - 10} more"
        
        self.lbl_status.setText(f"{status_icon_success} Done. Matched {matched}/{total} stimuli.")
        self.lbl_status.setObjectName("status-success-label")
        show_success(self, "Normalization Complete", msg)
    
    def _on_normalization_error(self, err_msg):
        self._overlay.hide()
        self.btn_normalize.stop_processing()
        show_error(self, "Normalization Failed", f"Coordinate normalization failed:\n{err_msg}")
        self.lbl_status.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_status.setObjectName("status-error-label")
    
    def assign_frame_index(self):
        """Assign _video_frame_index to each gaze point for video stimuli."""
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file before assigning frame indices.")
            return

        try:
            frame_rate = float(self.txt_fps.text().strip())
            if frame_rate <= 0:
                raise ValueError
        except ValueError:
            show_warning(self, "Invalid Frame Rate", "Please enter a valid frame rate (positive number).")
            return

        try:
            delay = float(self.txt_video_delay.text().strip()) if self.txt_video_delay.text().strip() else 0
        except ValueError:
            show_warning(self, "Invalid Value", "Please enter a valid gaze response lag in milliseconds (e.g., 0 for no lag).")
            return
        
        self.btn_assign_frames.start_processing("Assigning frames")
        self.lbl_frame_status.setText("Assigning frame indices...")
        self._overlay.set_status("Assigning Frame Indices")
        self._overlay.hide_progress()
        self._overlay.show()
        
        wearable = self._is_wearable()
        scene_col = None
        scene_kws = None
        if self._project_settings:
            scene_col = self._project_settings.get(
                "temporal_preprocessing", "kw_include_col", "") or None
            scene_kws = self._project_settings.get(
                "temporal_preprocessing", "kw_include_values", []) or None
        
        self._frame_worker = ProcessingWorker(
            self._do_assign_frames,
            self.gaze_data_path, frame_rate, delay, self.stim_dimensions,
            wearable, scene_col, scene_kws
        )
        self.btn_assign_frames.connect_worker(self._frame_worker)
        self._frame_worker.finished.connect(self._on_frames_done)
        self._frame_worker.error.connect(self._on_frames_error)
        self._frame_worker.start()
    
    @staticmethod
    def _do_assign_frames(gaze_path, frame_rate, delay, stim_dimensions,
                          wearable=False, scene_col=None, scene_keywords=None,
                          progress_callback=None):
        """Assign _video_frame_index — delegates to core.spatial_prep."""
        return assign_frames(gaze_path, frame_rate, delay, stim_dimensions,
                             wearable, scene_col, scene_keywords, progress_callback)
    
    def _on_frames_done(self, result):
        self._overlay.hide()
        self.btn_assign_frames.stop_processing()
        n_valid = result['n_valid']
        max_frame = result['max_frame']
        fps = result['fps']
        
        msg = (f"{status_icon_success} Frame index assigned!\n\n"
               f"FPS: {fps}\n"
               f"Valid frames: {n_valid}\n"
               f"Max frame index: {max_frame}")
        
        self.lbl_frame_status.setText(f"{status_icon_success} Done. {n_valid} frames assigned (max: {max_frame})")
        self.lbl_frame_status.setObjectName("status-success-label")
        show_success(self, "Frame Index Assigned", msg)
        self.data_changed.emit()
    
    def _on_frames_error(self, err_msg):
        self._overlay.hide()
        self.btn_assign_frames.stop_processing()
        show_error(self, "Frame Assignment Failed", f"Frame index assignment failed:\n{err_msg}")
        self.lbl_frame_status.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_frame_status.setObjectName("status-error-label")

    def closeEvent(self, event: QCloseEvent):
        for attr in ('_norm_worker', '_frame_worker', '_scan_worker'):
            if hasattr(self, attr):
                w = getattr(self, attr)
                if w.isRunning():
                    w.wait()
