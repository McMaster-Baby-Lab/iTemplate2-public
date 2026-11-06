import sys
import os
import logging
import multiprocessing


from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QGroupBox, QRadioButton, QButtonGroup, QFileDialog,
                             QTextEdit, QSpinBox, QLineEdit,
                             QDoubleSpinBox, QDialog, QListWidget, QListWidgetItem,
                             QCheckBox, QMessageBox)
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPixmap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .processing_utils import ProcessingButton, ProcessingOverlay
from .button_styles import button_style
from .theme import log_bg, log_text, text_secondary
from .theme import FACE_COLORS, face_color_hex
from .theme import status_success, status_error, gaze_red
from .theme import status_icon_success, status_icon_error, status_icon_warning, status_icon_info
import pandas as pd
from core.data_resolver import is_imported_data_file

logger = logging.getLogger('iTemplate2')


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class TransformWorker(QThread):
    """Runs the two-stage transformation pipeline in a background thread."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # completed, total
    finished = pyqtSignal(dict)       # result dict
    error = pyqtSignal(str)

    def __init__(self, gaze_path, stimuli_folder, landmark_folder,
                 stimuli_type, num_workers, face_ids=None,
                 max_gap_frames=None, template=None, project_path=None,
                 contour_margin=0.0):
        super().__init__()
        self.gaze_path = gaze_path
        self.stimuli_folder = stimuli_folder
        self.landmark_folder = landmark_folder
        self.stimuli_type = stimuli_type   # 'video' or 'image'
        self.num_workers = num_workers
        self.face_ids = face_ids or [1]
        self.max_gap_frames = max_gap_frames
        self.template = template
        self.contour_margin = contour_margin
        self.project_path = project_path or os.path.dirname(os.path.dirname(gaze_path))

    def _log(self, msg):
        """Emit to both the UI console and the file logger."""
        logger.info(msg)
        self.log.emit(msg)

    def run(self):
        try:
            from core.transformation import (read_frame_index_data, find_missing_and_replace,
                                              transform_coordinates_parallel)
            from core.io_utils import resolve_file_in_folder, atomic_save_parquet

            self._log("Loading data...")
            df = pd.read_parquet(self.gaze_path)
            from core.template_manager import (TemplateManager, dataset_manifest, file_hash,
                                               save_dataset, replay_review_decisions)
            from pathlib import Path
            import uuid
            from datetime import datetime, timezone
            template = self.template or TemplateManager(self.project_path).load()
            previous = dataset_manifest(self.gaze_path)
            for column in previous.get('aoi_columns',[]):
                if column in df: df.drop(columns=column,inplace=True)
            source_hash = previous.get('source_dataset_hash') or file_hash(self.gaze_path)
            if 'gaze_sample_id' not in df:
                df['gaze_sample_id'] = [f'{source_hash}:{i}' for i in range(len(df))]
            if df['gaze_sample_id'].isna().any() or not df['gaze_sample_id'].is_unique:
                raise ValueError('Gaze sample IDs must be unique and non-missing.')
            if '_source_video_frame_index' in df:
                df['_video_frame_index'] = df['_source_video_frame_index']
            elif '_video_frame_index' in df:
                df['_source_video_frame_index'] = df['_video_frame_index']
            if not {'stimuli_x','stimuli_y'}.issubset(df.columns):
                raise ValueError('Prepared gaze data must contain original stimulus coordinates.')
            lm_hashes = {}
            for asset in sorted(Path(self.landmark_folder).glob('*')):
                if asset.is_file() and asset.suffix in ('.parquet','.csv'):
                    lm_hashes[asset.name] = file_hash(asset)
            manifest = dict(template_id=template.template_id, template_hash=template.content_hash,
                            transform_run_id=uuid.uuid4().hex, source_dataset_hash=source_hash,
                            landmark_hashes=lm_hashes, face_ids=self.face_ids, stimuli_type=self.stimuli_type,
                            max_gap_frames=self.max_gap_frames, num_workers=self.num_workers,
                            contour_margin=self.contour_margin,
                            stimuli_folder=os.path.relpath(self.stimuli_folder,self.project_path),
                            landmark_folder=os.path.relpath(self.landmark_folder,self.project_path),
                            coordinate_convention='normalized_xy',
                            prepared_input=os.path.relpath(self.gaze_path,self.project_path),
                            created_at=datetime.now(timezone.utc).isoformat(), application_version='iTemplate2')
            self._log(f'Template: {template.provenance.get("name",template.template_id)}')
            total_rows = len(df)
            self._log(f"  ✓ Loaded {total_rows:,} rows, {len(df.columns)} columns")

            # ── Determine the stimulus name column ──
            if 'stimulusName' in df.columns:
                stim_col = 'stimulusName'
            elif 'videoName' in df.columns:
                stim_col = 'videoName'
            elif 'STIMULI_NAME' in df.columns:
                stim_col = 'STIMULI_NAME'
            else:
                self.error.emit("No stimulus name column found (stimulusName / videoName / STIMULI_NAME).")
                return

            unique_stims = df[stim_col].dropna().unique().tolist()
            n_stims = len(unique_stims)
            for stimulus in unique_stims:
                if resolve_file_in_folder(stimulus,self.stimuli_folder) is None:
                    raise ValueError(f'Stimulus file not found: {stimulus}')
                if template.template_id != 'default' and resolve_file_in_folder(stimulus,self.landmark_folder,suffix='_landmarks',extensions=['.parquet','.csv']) is None:
                    raise ValueError(f'Extract facial landmarks before applying this template: {stimulus}')
            self._log(f"  Found {n_stims} unique stimuli in '{stim_col}'\n")

            # =================================================================
            # Stage 1 — Fix missing frame indices (video only)
            # =================================================================
            if self.stimuli_type == 'video' and '_video_frame_index' in df.columns:
                gap_info = (
                    f" (max gap={self.max_gap_frames} frames)"
                    if self.max_gap_frames is not None else ""
                )
                self._log(f"━━━ Stage 1: Missing Frame Index Replacement{gap_info}━━━")
                replaced_total = 0

                for i, stim_name in enumerate(unique_stims):
                    # Resolve the _transformation.csv file (handles missing extensions)
                    csv_path = resolve_file_in_folder(
                        stim_name, self.landmark_folder,
                        suffix='_transformation', extensions=['.csv']
                    )

                    if csv_path is None:
                        self._log(f"  [{i+1}/{n_stims}] {stim_name} — no transformation CSV, skipped")
                        self.progress.emit(i + 1, n_stims)
                        continue

                    frame_data = read_frame_index_data(csv_path)
                    if frame_data is None:
                        self._log(f"  [{i+1}/{n_stims}] {stim_name} — no Frame_Index column, skipped")
                        self.progress.emit(i + 1, n_stims)
                        continue

                    missing_df = find_missing_and_replace(
                        frame_data.copy(), max_gap=self.max_gap_frames
                    )

                    if missing_df is not None and len(missing_df) > 0:
                        # Vectorized replacement: build a mapping dict and apply in one pass
                        remap = dict(zip(missing_df['Missing_Index'], missing_df['Replacement_Index']))
                        stim_mask = df[stim_col] == stim_name
                        orig = df.loc[stim_mask, '_video_frame_index']
                        mapped = orig.map(remap)
                        df.loc[stim_mask, '_video_frame_index'] = mapped.fillna(orig)
                        replaced_total += len(missing_df)
                        self._log(f"  [{i+1}/{n_stims}] {stim_name} — {len(missing_df)} frames remapped")
                    else:
                        self._log(f"  [{i+1}/{n_stims}] {stim_name} — no missing frames")

                    self.progress.emit(i + 1, n_stims)

                self._log(f"\n  ✓ Stage 1 complete: {replaced_total} total frame indices remapped\n")
            else:
                self._log("━━━ Stage 1: Skipped (no _video_frame_index or image mode) ━━━\n")

            # =================================================================
            # Stage 2 — Coordinate Transformation (parallel)
            # =================================================================
            self._log("━━━ Stage 2: Coordinate Transformation ━━━")
            self._log(f"  Stimuli folder: {self.stimuli_folder}")
            self._log(f"  Landmark folder: {self.landmark_folder}")
            self._log(f"  Workers: {self.num_workers}")
            self._log(f"  Type: {self.stimuli_type}")
            self._log(f"  Face contour margin: {self.contour_margin:.0%}")
            self._log(f"  Processing {n_stims} stimuli in parallel...\n")

            df = transform_coordinates_parallel(
                df,
                type=self.stimuli_type,
                stimuliFolder=self.stimuli_folder,
                landmarkFolder=self.landmark_folder,
                num_workers=self.num_workers,
                face_ids=self.face_ids,
                template=template,
                contour_margin=self.contour_margin,
            )

            df = df.reset_index(drop=True)
            # Remove any level_ columns from groupby artifacts
            df = df.loc[:, ~df.columns.str.startswith('level')]

            self._log("  ✓ Stage 2 complete: coordinate transformation finished\n")

            # =================================================================
            # Save output
            # =================================================================
            project_dir = self.project_path
            output_dir = os.path.join(project_dir, "transformed_data")
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.gaze_path))[0]
            # Strip existing suffix to avoid chaining (e.g. _transformed_transformed)
            if base_name.endswith('_transformed'):
                base_name = base_name[:-len('_transformed')]
            output_name = f"{base_name.split('__')[0]}__{template.template_id[:12]}__{manifest['transform_run_id'][:12]}_transformed.parquet"
            output_path = os.path.join(output_dir, output_name)

            df = replay_review_decisions(self.project_path,df,manifest,self.landmark_folder,template)
            df['template_id'] = template.template_id
            save_dataset(df, output_path, manifest)

            # Long-format schema: one consolidated column regardless of face count
            n_transformed = int(df['transformed_x'].notna().sum()) if 'transformed_x' in df.columns else 0

            self._log("━━━ Pipeline Complete ━━━")
            self._log(f"  Rows: {len(df):,}")
            self._log(f"  Transformed points: {n_transformed:,}")
            for region in ('on_face', 'off_face', 'unmapped'):
                count = int(df['gaze_region'].eq(region).sum())
                self._log(f"  {region}: {count:,}")
            self._log("  Off-face coordinates are for visualization; no face ID is assigned.")
            self._log("  Output columns: transformed_x / transformed_y / face_id / face_label / gaze_region")
            self._log(f"  Saved to: {output_path}")

            self.finished.emit({
                'output_path': output_path,
                'template_id': template.template_id,
                'total_rows': len(df),
                'n_transformed': int(n_transformed),
            })

        except Exception as e:
            import traceback
            self._log(f"\n✗ Error: {e}")
            self._log(traceback.format_exc())
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Face selection dialog
# ---------------------------------------------------------------------------
class FaceSelectDialog(QDialog):
    """Lets the user pick which face IDs to include in transformation."""

    def __init__(self, available_ids, selected_ids=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Faces to Transform")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Choose the faces that should be co-registered onto the face template. "
            "These IDs were detected during Step 1 (Facial Feature Extraction)."
        )
        info.setWordWrap(True)
        info.setObjectName("hint-label")
        layout.addWidget(info)

        self.list_widget = QListWidget()
        for fid in sorted(available_ids):
            item = QListWidgetItem(f"Face {fid}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if selected_ids is None or fid in selected_ids
                else Qt.CheckState.Unchecked
            )
            hex_color = face_color_hex(fid)
            item.setBackground(QColor(hex_color).lighter(140))
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setToolTip("Check all files in the list")
        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.setToolTip("Uncheck all files in the list")
        btn_select_all.clicked.connect(lambda: self._set_all_states(Qt.CheckState.Checked))
        btn_deselect_all.clicked.connect(lambda: self._set_all_states(Qt.CheckState.Unchecked))
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_deselect_all)
        btn_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _set_all_states(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def selected_ids(self):
        ids = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                text = item.text().replace("Face ", "")
                ids.append(int(text))
        return sorted(ids)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ProcessingView(QWidget):
    data_changed = pyqtSignal()  # Signal when transformation completes
    template_selection_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.current_project_path = ""
        self.gaze_data_path = ""
        self._active_template = None
        self._queued_template_run = False
        self.template_change_guard = lambda: True
        self._detected_stimuli_type = None   # Set from coordinate_setup scan

        layout = QVBoxLayout()
        self.setLayout(layout)

        lbl_title = QLabel("Step 3: Run Transformation Pipeline")
        lbl_title.setObjectName("title-label")
        layout.addWidget(lbl_title)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        layout.addWidget(self.lbl_notice)

        # ── Data file ──
        data_group = QGroupBox("Gaze Data File")
        data_layout = QHBoxLayout()
        self.btn_select_data = QPushButton("Select Parquet File")
        self.btn_select_data.clicked.connect(self.select_data_file)
        data_layout.addWidget(self.btn_select_data)
        self.lbl_data_path = QLabel("No file selected")
        data_layout.addWidget(self.lbl_data_path, 1)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # ── Stimuli type ──
        type_group = QGroupBox("Stimuli Type")
        type_layout = QHBoxLayout()
        self.radio_video = QRadioButton("Video")
        self.radio_image = QRadioButton("Image")
        self.radio_video.setChecked(True)
        self.type_btn_group = QButtonGroup()
        self.type_btn_group.addButton(self.radio_video)
        self.type_btn_group.addButton(self.radio_image)
        type_layout.addWidget(self.radio_video)
        type_layout.addWidget(self.radio_image)
        type_layout.addStretch()
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # ── Folder paths ──
        folder_group = QGroupBox("Folder Paths")
        folder_layout = QVBoxLayout()

        # Stimuli folder
        stim_row = QHBoxLayout()
        stim_row.addWidget(QLabel("Stimuli folder:"))
        self.txt_stimuli_folder = QLineEdit()
        self.txt_stimuli_folder.setPlaceholderText("Path to original video/image files")
        stim_row.addWidget(self.txt_stimuli_folder, 1)
        btn_browse_stim = QPushButton("Browse")
        btn_browse_stim.setToolTip("Browse for the stimuli folder containing original videos/images")
        btn_browse_stim.clicked.connect(lambda: self._browse_folder(self.txt_stimuli_folder))
        stim_row.addWidget(btn_browse_stim)
        folder_layout.addLayout(stim_row)

        # Landmark / processed data folder
        lm_row = QHBoxLayout()
        lm_row.addWidget(QLabel("Landmark folder:"))
        self.txt_landmark_folder = QLineEdit()
        self.txt_landmark_folder.setPlaceholderText("Path to processed landmark data (_transformation.parquet or .csv)")
        lm_row.addWidget(self.txt_landmark_folder, 1)
        btn_browse_lm = QPushButton("Browse")
        btn_browse_lm.setToolTip("Browse for the folder with processed landmark data")
        btn_browse_lm.clicked.connect(lambda: self._browse_folder(self.txt_landmark_folder))
        lm_row.addWidget(btn_browse_lm)
        folder_layout.addLayout(lm_row)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        template_group = QGroupBox("Template Face")
        template_layout = QHBoxLayout(template_group)
        self.template_preview = QLabel()
        self.template_preview.setFixedSize(110, 110)
        template_layout.addWidget(self.template_preview)
        details = QVBoxLayout()
        self.lbl_template = QLabel("Default template")
        self.lbl_template.setWordWrap(True)
        details.addWidget(self.lbl_template)
        buttons = QHBoxLayout()
        self.btn_template_stimulus = QPushButton("Choose from Stimulus")
        self.btn_template_upload = QPushButton("Upload Face Image")
        self.btn_template_reset = QPushButton("Reset to Default")
        self.btn_template_stimulus.clicked.connect(self._choose_template_stimulus)
        self.btn_template_upload.clicked.connect(self._upload_template)
        self.btn_template_reset.clicked.connect(self._reset_template)
        for button in (self.btn_template_stimulus, self.btn_template_upload, self.btn_template_reset):
            buttons.addWidget(button)
        details.addLayout(buttons)
        template_layout.addLayout(details, 1)
        layout.addWidget(template_group)

        # ── Workers ──
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Parallel Workers:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, multiprocessing.cpu_count())
        self.spin_workers.setValue(min(4, multiprocessing.cpu_count()))
        self.spin_workers.setToolTip(
            "Number of stimuli processed simultaneously.\n"
            f"Max = {multiprocessing.cpu_count()} (your CPU cores)."
        )
        worker_layout.addWidget(self.spin_workers)
        worker_layout.addStretch()
        layout.addLayout(worker_layout)

        # Max gap fill row
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Max Gap Fill (seconds):"))
        self.spin_max_gap = QDoubleSpinBox()
        self.spin_max_gap.setRange(0.1, 60.0)
        self.spin_max_gap.setValue(2.0)
        self.spin_max_gap.setDecimals(1)
        self.spin_max_gap.setSingleStep(0.5)
        self.spin_max_gap.setToolTip(
            "Gaps larger than this (in seconds) will NOT be auto-filled.\n"
            "Gaze rows in those gaps remain as NaN.\n"
            "Set to 60.0 to disable (fill all gaps)."
        )
        gap_layout.addWidget(self.spin_max_gap)
        gap_layout.addStretch()
        layout.addLayout(gap_layout)

        # Face contour margin row
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Face Contour Margin (%):"))
        self.spin_contour_margin = QSpinBox()
        self.spin_contour_margin.setRange(0, 30)
        self.spin_contour_margin.setValue(0)
        self.spin_contour_margin.setToolTip(
            "Enlarges each stimulus face outline by this percentage before mapping,\n"
            "so gaze just outside the face (e.g. from tracker error) is warped\n"
            "onto the template face. The template landmarks are not changed.\n"
            "0 = use the detected outline."
        )
        margin_layout.addWidget(self.spin_contour_margin)
        margin_layout.addStretch()
        layout.addLayout(margin_layout)
        margin_tip = QLabel(
            "💡 Pro tip: eye trackers have some error, so looks at the face edge can land just outside it. "
            "The margin enlarges each face outline by this percentage, so those gaze points are mapped "
            "onto the template face instead of being counted as off-face. Preview the margin in Step 1 "
            "(\"Show Contour Margin\"). The value is saved with the project and in each output file; "
            "changing it requires re-running this step."
        )
        margin_tip.setWordWrap(True)
        margin_tip.setObjectName("hint-label")
        layout.addWidget(margin_tip)

        # Face info row
        face_info_layout = QHBoxLayout()
        face_info_layout.addWidget(QLabel("Faces to transform:"))
        self.lbl_face_info = QLabel("Auto (all detected)")
        self.lbl_face_info.setObjectName("face-info-label")
        face_info_layout.addWidget(self.lbl_face_info, 1)
        self.btn_select_faces = QPushButton("Select...")
        self.btn_select_faces.setToolTip("Choose which detected faces to include in transformation")
        self.btn_select_faces.clicked.connect(self._open_face_selector)
        face_info_layout.addWidget(self.btn_select_faces)
        layout.addLayout(face_info_layout)

        # ── Run button ──
        self.btn_run = ProcessingButton("Run Transformation Pipeline")
        self.btn_run.setStyleSheet(button_style("primary", padding="10px"))
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        # ── Log console ──
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            f"background-color: {log_bg}; color: {log_text}; font-family: Menlo, monospace; font-size: 12px;"
        )
        self.log_console.setMinimumHeight(200)
        layout.addWidget(self.log_console)



        # ── Result label ──
        self.lbl_result = QLabel("")
        layout.addWidget(self.lbl_result)
        
        self._project_settings = None
        self._syncing_selection = False
        self._overlay = ProcessingOverlay(self)

        # Face selection state
        self._available_face_ids = []   # All face IDs found in landmark data
        self._selected_face_ids = None  # None = use all, list = specific subset
        
        # Connect signals for auto-save
        self.type_btn_group.buttonToggled.connect(self._on_settings_edited)
        self.spin_workers.valueChanged.connect(self._on_settings_edited)
        self.spin_max_gap.valueChanged.connect(self._on_settings_edited)
        self.spin_contour_margin.valueChanged.connect(self._on_settings_edited)
        self.txt_stimuli_folder.textChanged.connect(self._update_onboarding_state)
        self.txt_stimuli_folder.editingFinished.connect(self._on_settings_edited)
        self.txt_landmark_folder.textChanged.connect(self._update_onboarding_state)
        self.txt_landmark_folder.editingFinished.connect(self._on_settings_edited)
        self._update_onboarding_state()

    # ------------------------------------------------------------------
    # Project path & auto-population
    # ------------------------------------------------------------------
    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings
        self._load_template_selection()

        selected = settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != self.gaze_data_path:
            self._syncing_selection = True
            try:
                self._set_gaze_data_file(selected, update_settings=False)
            finally:
                self._syncing_selection = False
        
        # Block signals on the specific widgets that connect to _on_settings_edited
        widgets_to_block = [self.type_btn_group, self.spin_workers, self.spin_max_gap,
                            self.spin_contour_margin, self.radio_video, self.radio_image]
        for w in widgets_to_block:
            w.blockSignals(True)
        try:
            stype = settings.get("processing", "stimuli_type", "video")
            if stype == "image":
                self.radio_image.setChecked(True)
            else:
                self.radio_video.setChecked(True)
            workers = settings.get("processing", "parallel_workers", None)
            if workers is not None:
                self.spin_workers.setValue(int(workers))
            max_gap = settings.get("processing", "max_gap_seconds", 2.0)
            self.spin_max_gap.setValue(float(max_gap))
            self.spin_contour_margin.setValue(int(settings.get("processing", "contour_margin_pct", 0)))
        finally:
            for w in widgets_to_block:
                w.blockSignals(False)

        # Auto-populate folder paths if still empty
        if self.current_project_path:
            if not self.txt_stimuli_folder.text().strip():
                stimuli_dir = os.path.join(self.current_project_path, "stimuli")
                if os.path.isdir(stimuli_dir):
                    self.txt_stimuli_folder.setText(stimuli_dir)
            if not self.txt_landmark_folder.text().strip():
                processed_dir = os.path.join(self.current_project_path, "processed_stimuli_data")
                if os.path.isdir(processed_dir):
                    self.txt_landmark_folder.setText(processed_dir)
        self._update_onboarding_state()

        # Update face info label
        self._available_face_ids = self._detect_available_faces()
        for key, field in (('stimuli_folder',self.txt_stimuli_folder),('landmark_folder',self.txt_landmark_folder)):
            saved = settings.get('processing',key,'')
            if saved:
                field.setText(saved if os.path.isabs(saved) else os.path.join(self.current_project_path,saved))

        saved_face_ids = settings.get("video_preprocessor", "selected_face_ids", None)
        if saved_face_ids:
            saved_list = sorted(saved_face_ids)
            available_list = sorted(self._available_face_ids)
            if saved_list != available_list:
                self._selected_face_ids = saved_list
            else:
                self._selected_face_ids = None
        else:
            self._selected_face_ids = None
        self._update_face_info()

    def _on_settings_edited(self, *args):
        """Auto-save settings when UI changes."""
        if self._project_settings:
            # save_settings() calls .set() which now triggers save()
            self.save_settings(self._project_settings)

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        settings.set("processing", "stimuli_type",
                     "image" if self.radio_image.isChecked() else "video")
        settings.set("processing", "parallel_workers",
                     self.spin_workers.value())
        settings.set("processing", "max_gap_seconds",
                     self.spin_max_gap.value())
        settings.set("processing", "contour_margin_pct",
                     self.spin_contour_margin.value())

        for key, field in (('stimuli_folder',self.txt_stimuli_folder),('landmark_folder',self.txt_landmark_folder)):
            path = field.text().strip()
            if path and self.current_project_path:
                settings.set('processing',key,os.path.relpath(path,self.current_project_path))

    def set_project_path(self, path):
        changed_project = self.current_project_path != path
        self.current_project_path = path
        if changed_project:
            self._active_template = None
            self._queued_template_run = False

        if self._project_settings:
            selected = self._project_settings.resolve_working_data_file("")
            if selected and os.path.isfile(selected):
                self._set_gaze_data_file(selected, update_settings=False)

        # Auto-populate stimuli folder
        stimuli_dir = os.path.join(path, "stimuli")
        if os.path.isdir(stimuli_dir):
            self.txt_stimuli_folder.setText(stimuli_dir)

        # Auto-populate landmark folder
        processed_dir = os.path.join(path, "processed_stimuli_data")
        if os.path.isdir(processed_dir):
            self.txt_landmark_folder.setText(processed_dir)
        self._update_onboarding_state()
        self._selected_face_ids = None
        self._available_face_ids = self._detect_available_faces()
        self._update_face_info()

    def set_stimuli_type(self, has_video):
        """Called from main.py when coordinate_setup scans stimuli."""
        self._detected_stimuli_type = 'video' if has_video else 'image'
        if has_video:
            self.radio_video.setChecked(True)
        else:
            self.radio_image.setChecked(True)

    def refresh_data(self):
        """Reload project path to refresh file lists."""
        logger.debug("ProcessingView.refresh_data called. Path=%s", self.current_project_path)
        if self.current_project_path:
            self.set_project_path(self.current_project_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def select_data_file(self):
        start_dir = os.path.join(self.current_project_path, "imported_parquet") \
            if self.current_project_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Gaze Data Parquet", start_dir, "Parquet Files (*.parquet)"
        )
        if path:
            self._set_gaze_data_file(path, update_settings=True)

    def _set_global_selected_file(self, path):
        if not self._project_settings or self._syncing_selection:
            return
        if self.current_project_path and is_imported_data_file(self.current_project_path, path):
            self._project_settings.set("global", "source_data_file", path)
        self._project_settings.set("global", "selected_data_file", path)

    def _set_global_transformed_file(self, path):
        if not self._project_settings or self._syncing_selection:
            return
        self._project_settings.set("global", "selected_transformed_file", path)

    def _set_gaze_data_file(self, path, update_settings=False):
        if not path or not os.path.isfile(path):
            return
        self.gaze_data_path = path
        self.lbl_data_path.setText(os.path.basename(path))
        if update_settings:
            QTimer.singleShot(0,self._run_pending_template)
            self._set_global_selected_file(path)
        self._update_onboarding_state()

    def _browse_folder(self, line_edit):
        # Use the current text as start directory if valid, otherwise fall back
        # to the appropriate project subfolder
        current_text = line_edit.text().strip()
        if current_text and os.path.isdir(current_text):
            start_dir = current_text
        elif self.current_project_path:
            if line_edit is self.txt_stimuli_folder:
                candidate = os.path.join(self.current_project_path, "stimuli")
            elif line_edit is self.txt_landmark_folder:
                candidate = os.path.join(self.current_project_path, "processed_stimuli_data")
            else:
                candidate = self.current_project_path
            start_dir = candidate if os.path.isdir(candidate) else self.current_project_path
        else:
            start_dir = ""
        path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if path:
            line_edit.setText(path)
            self._on_settings_edited()
        self._update_onboarding_state()

    def _update_onboarding_state(self):
        has_project = bool(self.current_project_path)
        has_data = bool(self.gaze_data_path and os.path.isfile(self.gaze_data_path))
        has_stimuli = bool(os.path.isdir(self.txt_stimuli_folder.text().strip()))
        has_landmarks = bool(os.path.isdir(self.txt_landmark_folder.text().strip()))
        self.btn_run.setEnabled(has_data and has_stimuli and has_landmarks)
        if getattr(self,'_queued_template_run',False) and has_data and has_stimuli and has_landmarks:
            QTimer.singleShot(0,self._run_pending_template)
        for button in (self.btn_template_stimulus,self.btn_template_reset):
            button.setEnabled(has_project)
        if not (getattr(self,'_template_detection',None) and self._template_detection.isRunning()):
            self.btn_template_upload.setEnabled(has_project)

        if not has_project:
            self.lbl_notice.setText("Open a project first in Project Manager.")
            self.lbl_notice.show()
        elif not has_data:
            self.lbl_notice.setText("Select a gaze parquet file to transform.")
            self.lbl_notice.show()
        elif not has_stimuli:
            self.lbl_notice.setText("Choose a valid stimuli folder (usually project/stimuli).")
            self.lbl_notice.show()
        elif not has_landmarks:
            self.lbl_notice.setText("Choose a valid landmark folder (usually project/processed_stimuli_data).")
            self.lbl_notice.show()
        else:
            self.lbl_notice.hide()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _load_template_selection(self):
        from core.template_manager import TemplateManager
        if not self.current_project_path:
            return
        tid = self._project_settings.get('processing','active_template_id','default') if self._project_settings else 'default'
        # A status left as 'processing' by a crashed or closed session is stale.
        if self._project_settings and self._project_settings.get('processing','template_processing_status','ready') == 'processing' \
                and not (getattr(self,'_worker',None) and self._worker.isRunning()):
            self._project_settings.set('processing','template_processing_status','interrupted')
        try:
            self._active_template = TemplateManager(self.current_project_path).load(tid)
            self._show_template()
        except Exception as exc:
            self._active_template = None
            self.lbl_template.setText(f'Template unavailable: {exc}')
            self.template_preview.clear()

    def _show_template(self):
        import cv2
        template = self._active_template
        rgb = cv2.cvtColor(template.image, cv2.COLOR_BGRA2RGBA)
        pix = QPixmap.fromImage(QImage(rgb.data,template.width,template.height,rgb.strides[0],QImage.Format.Format_RGBA8888).copy())
        self.template_preview.setPixmap(pix.scaled(110,110,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        state = self._project_settings.get('processing','template_processing_status','ready') if self._project_settings else 'ready'
        self.lbl_template.setText(f"{template.provenance.get('name',template.template_id)} — {state}")

    def _choose_template_stimulus(self):
        if not (self.current_project_path and self.template_change_guard()):
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle('Choose a Template Face from a Stimulus')
        box.setText('You will pick the template face on the Face Extraction page.')
        box.setInformativeText(
            '1. Open the stimulus that contains the face you want.\n'
            '2. Use the playback controls (or the Frame box) to find a clear, frontal frame.\n'
            '3. Under "Detected Faces", check exactly one face. Landmarks must already be '
            'extracted for that frame.\n'
            '4. Click "Use Selected Face as Template", just below the frame controls, then '
            'confirm the preview.\n\n'
            'You will return here and the gaze data is re-transformed onto the new template. '
            'Existing results are kept, and AOIs drawn on another template will need review. '
            'Click Cancel on the Face Extraction page to leave without changes.')
        go = box.addButton('Go to Face Extraction', QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(go)
        box.exec()
        if box.clickedButton() is go:
            self.template_selection_requested.emit()

    def _upload_template(self):
        if not self.current_project_path or not self.template_change_guard():
            return
        if getattr(self,'_template_detection',None) and self._template_detection.isRunning():
            return
        path, _ = QFileDialog.getOpenFileName(self,'Upload Face Image',self.current_project_path,'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path: return
        from .template_widgets import TemplateDetectionWorker
        self._template_detection = TemplateDetectionWorker(path,self)
        self.btn_template_upload.setEnabled(False)
        selection_project = self.current_project_path
        self._overlay.set_status('Detecting template landmarks')
        self._overlay.hide_progress()
        self._overlay.show()
        self._template_detection.result.connect(lambda image,faces,provenance:
            self.accept_template_candidate(image,faces,provenance) if self.current_project_path == selection_project else None)
        self._template_detection.error.connect(lambda msg: show_error(self,'Template Detection',msg))
        self._template_detection.finished.connect(lambda: self.btn_template_upload.setEnabled(True))
        self._template_detection.finished.connect(self._finish_template_detection)
        self._template_detection.start()

    def _finish_template_detection(self):
        if not (getattr(self,'_worker',None) and self._worker.isRunning()):
            self._overlay.hide()

    def accept_template_candidate(self,image,faces,provenance):
        self._overlay.hide()
        from .template_widgets import TemplatePreviewDialog
        from core.template_manager import TemplateManager
        face_ids = [provenance['face_id']] if len(faces) == 1 and 'face_id' in provenance else None
        dialog = TemplatePreviewDialog(image,faces,self,face_ids=face_ids)
        if dialog.exec() != QDialog.DialogCode.Accepted: return False
        try:
            provenance = dict(provenance)
            if provenance.get('source_kind') == 'uploaded_image':
                provenance['detected_face_index'] = dialog.combo.currentIndex()+1
            candidate = TemplateManager(self.current_project_path).create(image,dialog.selected_landmarks(),provenance)
            self._apply_template(candidate)
            return True
        except Exception as exc:
            show_error(self,'Invalid Template',str(exc))
            return False

    def _reset_template(self):
        from core.template_manager import TemplateManager
        if not self.current_project_path or not self.template_change_guard(): return
        if self._active_template is not None and self._active_template.template_id == 'default': return
        self._apply_template(TemplateManager(self.current_project_path).load())

    def _apply_template(self,template):
        self._active_template = template
        if self._project_settings:
            self._project_settings.set('processing','active_template_id',template.template_id)
            self._project_settings.set('processing','template_processing_status','pending')
        self._show_template()
        self._queued_template_run = True
        QTimer.singleShot(0,self._run_pending_template)

    def _run_pending_template(self):
        if not self._queued_template_run: return
        if getattr(self,'_worker',None) and self._worker.isRunning():
            QTimer.singleShot(100,self._run_pending_template)
            return
        if not (self.gaze_data_path and os.path.isfile(self.gaze_data_path)
                and os.path.isdir(self.txt_stimuli_folder.text().strip())
                and os.path.isdir(self.txt_landmark_folder.text().strip())):
            self.lbl_template.setText(self.lbl_template.text().split(' — ')[0]+' — transformation pending: select processing inputs')
            return
        self._queued_template_run = False
        self.run_pipeline()

    def run_pipeline(self):
        if hasattr(self, '_worker') and self._worker.isRunning():
            show_warning(self, "Pipeline Already Running", "A transformation pipeline is already running. Please wait for it to complete.")
            return
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file before running the transformation.")
            return

        stimuli_folder = self.txt_stimuli_folder.text().strip()
        if not stimuli_folder or not os.path.isdir(stimuli_folder):
            show_warning(self, "No Stimuli Folder", "Please select a stimuli folder to locate the source videos.")
            return

        landmark_folder = self.txt_landmark_folder.text().strip()
        if not landmark_folder or not os.path.isdir(landmark_folder):
            show_warning(self, "No Landmark Folder", "Please select a valid landmark folder containing the face extraction results.")
            return

        if self._active_template is None:
            self._load_template_selection()
        if self._active_template is None:
            show_error(self, 'Template Unavailable', 'Select a valid template before processing.')
            return
        if not self.template_change_guard(): return
        if self._project_settings:
            self._project_settings.set('processing','template_processing_status','processing')
        self._show_template()
        stimuli_type = 'video' if self.radio_video.isChecked() else 'image'
        num_workers = self.spin_workers.value()

        # Convert max gap seconds to frames
        max_gap_seconds = self.spin_max_gap.value()
        max_gap_frames = None  # disabled by default
        if max_gap_seconds < 60.0:  # 60.0 = disabled
            default_fps = 30.0
            if self._project_settings:
                default_fps = self._project_settings.get("processing", "default_fps", 30.0)
            max_gap_frames = int(max_gap_seconds * default_fps)

        # Determine face IDs: local selection > settings > auto-detect > default
        face_ids = None  # ensure always defined before the fallback check
        if self._selected_face_ids is not None:
            face_ids = self._selected_face_ids
        elif self._project_settings:
            face_ids = self._project_settings.get("video_preprocessor", "selected_face_ids", None)
        if not face_ids:
            face_ids = self._available_face_ids if self._available_face_ids else [1]

        if not face_ids:
            show_warning(self, "No Faces Available", "No faces have been detected yet. Run Step 1 (Face Extraction) first.")
            self.btn_run.stop_processing()
            self.spin_workers.setEnabled(True)
            return

        self.log_console.clear()
        self.btn_run.set_progress(0)
        self.lbl_result.setText("")

        self.btn_run.start_processing("Processing")
        self.spin_workers.setEnabled(False)
        self._overlay.set_status("Running Transformation Pipeline")
        self._overlay.hide_progress()
        self._overlay.show()

        self._worker = TransformWorker(
            self.gaze_data_path,
            stimuli_folder,
            landmark_folder,
            stimuli_type,
            num_workers,
            face_ids,
            max_gap_frames,
            template=self._active_template,
            project_path=self.current_project_path,
            contour_margin=self.spin_contour_margin.value() / 100.0,
        )
        self.log_console.append(f"Transforming faces: {face_ids}")
        self._worker.log.connect(self._on_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._worker.finished.connect(lambda _: QTimer.singleShot(100,self._run_pending_template))
        self._worker.error.connect(lambda _: QTimer.singleShot(100,self._run_pending_template))

    def _on_log(self, msg):
        self.log_console.append(msg)
        # Auto-scroll to bottom
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, completed, total):
        pct = int((completed / max(total, 1)) * 100)
        self.btn_run.set_progress(pct)

    def _on_finished(self, result):
        self._overlay.hide()
        self.btn_run.stop_processing()
        self.spin_workers.setEnabled(True)

        n_rows = result['total_rows']
        n_trans = result['n_transformed']
        out_path = result['output_path']

        self.lbl_result.setText(
            f"{status_icon_success} Done — {n_trans:,} / {n_rows:,} points transformed. "
            f"Saved to: {os.path.basename(out_path)}"
        )
        self.lbl_result.setObjectName("status-success-label")

        show_success(
            self, "Transformation Complete",
            f"Transformation complete!\n\n"
            f"Rows: {n_rows:,}\n"
            f"Transformed points: {n_trans:,}\n"
            f"Output: {out_path}"
        )
        if self._project_settings:
            self._project_settings.set('processing','last_successful_template_id',result.get('template_id','default'))
            status = 'pending' if self._queued_template_run else 'ready'
            self._project_settings.set('processing','template_processing_status',status)
            self._project_settings.set("global", "selected_transformed_file", out_path)
        self._show_template()
        self.data_changed.emit()

    def _on_error(self, err_msg):
        self._overlay.hide()
        self.btn_run.stop_processing()
        self.spin_workers.setEnabled(True)

        self.lbl_result.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_result.setObjectName("status-error-label")
        if self._project_settings:
            self._project_settings.set('processing','template_processing_status','failed')
        if self._active_template: self._show_template()
        show_error(self, "Transformation Failed", f"The coordinate transformation failed:\n{err_msg}")

    # ------------------------------------------------------------------
    # Face selection
    # ------------------------------------------------------------------
    def _detect_available_faces(self):
        """Scan processed_stimuli_data for landmark files to find face IDs."""
        if not self.current_project_path:
            return []
        processed_dir = os.path.join(self.current_project_path, "processed_stimuli_data")
        if not os.path.isdir(processed_dir):
            return []
        try:
            landmark_files = sorted(
                f for f in os.listdir(processed_dir)
                if '_landmarks' in f and f.endswith(('.parquet', '.csv'))
            )
            if not landmark_files:
                return []
            df = pd.read_parquet(os.path.join(processed_dir, landmark_files[0])) if landmark_files[0].endswith('.parquet') else pd.read_csv(os.path.join(processed_dir, landmark_files[0]))
            if 'Face_ID' in df.columns:
                return sorted(df['Face_ID'].unique().tolist())
        except Exception:
            pass
        return [1]

    def _update_face_info(self):
        """Refresh the face info label based on current state."""
        if not self._available_face_ids:
            self.lbl_face_info.setText("Auto (all detected)")
            self.btn_select_faces.setEnabled(False)
            return

        self.btn_select_faces.setEnabled(True)
        if self._selected_face_ids is None:
            if len(self._available_face_ids) == 1:
                self.lbl_face_info.setText(f"Face {self._available_face_ids[0]}")
            else:
                self.lbl_face_info.setText(
                    f"Faces {', '.join(str(f) for f in self._available_face_ids)} (all)"
                )
        else:
            if not self._selected_face_ids:
                self.lbl_face_info.setText("None selected")
                self.lbl_face_info.setObjectName("face-info-error-label")
            else:
                self.lbl_face_info.setText(f"Faces {', '.join(str(f) for f in self._selected_face_ids)}")
                self.lbl_face_info.setObjectName("face-info-label")

    def _open_face_selector(self):
        """Open the face selection dialog."""
        if not self._available_face_ids:
            self._available_face_ids = self._detect_available_faces()
        if not self._available_face_ids:
            show_warning(self, "No Faces Detected", "No faces have been detected. Run Step 1 (Face Extraction) to detect faces in your stimuli.")
            return

        current = self._selected_face_ids if self._selected_face_ids is not None else self._available_face_ids
        dialog = FaceSelectDialog(self._available_face_ids, current, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.selected_ids()
            if not selected:
                show_warning(self, "No Faces Selected", "Select at least one face to include in the transformation.")
                return
            if selected == self._available_face_ids:
                self._selected_face_ids = None
            else:
                self._selected_face_ids = selected
            self._update_face_info()
            if self._project_settings:
                # Save None when all faces are selected (Auto) so that if new
                # faces appear in a later session they are not silently excluded.
                self._project_settings.set("video_preprocessor", "selected_face_ids",
                                          self._selected_face_ids)

    def closeEvent(self, event: QCloseEvent):
        if getattr(self, '_template_detection', None) and self._template_detection.isRunning():
            self._template_detection.wait()
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.wait()
