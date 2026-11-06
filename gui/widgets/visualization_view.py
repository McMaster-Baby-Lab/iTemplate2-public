from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QSlider,
                             QHBoxLayout, QFileDialog, QComboBox,
                             QGroupBox, QCheckBox, QScrollArea, QProgressDialog,
                             QDoubleSpinBox, QListWidget, QListWidgetItem, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit)
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
import cv2
import pandas as pd
import numpy as np
import os
import re
import sys
import io

# matplotlib — use non-interactive Agg backend for rendering to pixmaps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from .processing_utils import ProcessingOverlay
from .theme import THEME, status_success, status_error, status_warning, status_info, status_icon_success, status_icon_error, status_icon_warning, status_icon_info, bg_primary, log_bg, text_secondary, HINT_STYLE
from .theme import FACE_COLORS, face_color_hex, face_color_bgr
from .theme import GAZE_OFF_FACE_BGR as OFF_FACE_BGR, GAZE_UNMAPPED_BGR as UNMAPPED_BGR, bgr_to_hex
OFF_FACE_HEX = bgr_to_hex(OFF_FACE_BGR)

# Shared template assets from the core layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.config import average_face, avg_face_image, width, height
    from core.io_utils import resolve_file_in_folder
except ImportError:
    average_face = None
    avg_face_image = None
    width = 1024
    height = 1024
    resolve_file_in_folder = None


class HeatmapWorker(QThread):
    """Background thread for histogram computation and matplotlib rendering."""
    finished = pyqtSignal(bytes, bytes)   # (display_png, export_png)
    error = pyqtSignal(str)

    def __init__(self, df, avg_face_bgr, face_w, face_h,
                 n_bins, sigma, normalize, group_cols, show_scatter,
                 highlight_off_face=False):
        super().__init__()
        self.df = df
        self.avg_face_bgr = avg_face_bgr
        self.face_w = face_w
        self.face_h = face_h
        self.n_bins = n_bins
        self.sigma = sigma
        self.normalize = normalize
        self.group_cols = group_cols
        self.show_scatter = show_scatter
        self.highlight_off_face = highlight_off_face

    def run(self):
        try:
            import math
            df = self.df
            face_w, face_h = self.face_w, self.face_h
            n_bins = self.n_bins
            sigma = self.sigma

            # ── Grouping ──
            if not self.group_cols:
                groups = {'All Data': df}
            elif len(self.group_cols) == 1:
                groups = {str(k): g for k, g in df.groupby(self.group_cols[0])}
            else:
                groups = {}
                for keys, g in df.groupby(self.group_cols):
                    label = ' \u00d7 '.join(str(k) for k in keys)
                    groups[label] = g

            # ── Histogram + Gaussian smoothing per group ──
            heatmap_results = {}
            for label, gdf in groups.items():
                if len(gdf) < 5:
                    continue
                if self.normalize and 'Participant_ID' in gdf.columns:
                    pid_col = 'Participant_ID'
                    pids = gdf[pid_col].dropna().unique()
                    avg_hist = np.zeros((n_bins, n_bins), dtype=np.float64)
                    n_pids = 0
                    for pid in pids:
                        pdf = gdf[gdf[pid_col] == pid]
                        if len(pdf) < 2:
                            continue
                        h, _, _ = np.histogram2d(
                            pdf['fix_x'].values, pdf['fix_y'].values,
                            bins=n_bins, range=[[0, face_w], [0, face_h]]
                        )
                        h_sum = h.sum()
                        if h_sum > 0:
                            h = h / h_sum
                        avg_hist += h
                        n_pids += 1
                    if n_pids > 0:
                        avg_hist /= n_pids
                    heatmap = avg_hist
                else:
                    heatmap, _, _ = np.histogram2d(
                        gdf['fix_x'].values, gdf['fix_y'].values,
                        bins=n_bins, range=[[0, face_w], [0, face_h]]
                    )
                heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
                heatmap_results[label] = (heatmap_smooth, gdf)

            if not heatmap_results:
                self.error.emit("No groups with \u22655 data points.")
                return

            # ── Matplotlib figure rendering ──
            avg_face_rgb = cv2.cvtColor(self.avg_face_bgr, cv2.COLOR_BGR2RGB)
            n = len(heatmap_results)

            all_densities = [d for d, _ in heatmap_results.values()]
            vmax = max(d.max() for d in all_densities)

            max_cols = min(n, 5)
            ncols = max_cols
            nrows = math.ceil(n / ncols)

            fig_width = 4.0 * ncols + 1.0
            fig_height = 4.0 * nrows + 0.5
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(fig_width, fig_height),
                                     dpi=150, squeeze=False)

            last_im = None
            items = list(heatmap_results.items())
            for idx in range(nrows * ncols):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                if idx < n:
                    label, (density, gdf) = items[idx]
                    ax.imshow(avg_face_rgb,
                              extent=[0, face_w, face_h, 0], aspect='equal')
                    im = ax.imshow(density.T, origin='upper',
                                   extent=[0, face_w, face_h, 0],
                                   aspect='equal', cmap='jet', alpha=0.55,
                                   vmin=0, vmax=vmax)
                    last_im = im
                    if self.show_scatter:
                        off = (gdf['gaze_region'].eq('off_face').fillna(False)
                               if self.highlight_off_face and 'gaze_region' in gdf
                               else pd.Series(False, index=gdf.index))
                        ax.scatter(gdf.loc[~off, 'fix_x'], gdf.loc[~off, 'fix_y'],
                                   s=1, c='white', alpha=0.1,
                                   edgecolors='none')
                        if off.any():
                            ax.scatter(gdf.loc[off, 'fix_x'], gdf.loc[off, 'fix_y'],
                                       s=1, c=OFF_FACE_HEX, alpha=0.4,
                                       edgecolors='none')
                    ax.set_title(f"{label} (n={len(gdf):,})",
                                 fontsize=10, fontweight='bold', pad=6)
                    ax.axis('off')
                else:
                    ax.set_visible(False)

            fig.subplots_adjust(left=0.02, right=0.88, top=0.93, bottom=0.03,
                                wspace=0.08, hspace=0.18)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label('Density', fontsize=10)
            if self.show_scatter and self.highlight_off_face:
                from matplotlib.lines import Line2D
                handles = [
                    Line2D([], [], marker='o', linestyle='none', markersize=6,
                           markerfacecolor='white', markeredgecolor='0.4', label='On-face'),
                    Line2D([], [], marker='o', linestyle='none', markersize=6,
                           markerfacecolor=OFF_FACE_HEX, markeredgecolor='none', label='Off-face'),
                ]
                fig.legend(handles=handles, loc='lower left',
                           bbox_to_anchor=(0.88, 0.0), fontsize=9, frameon=False)
            fig.patch.set_facecolor('white')

            # Render to PNG bytes (display + export resolutions)
            buf_display = io.BytesIO()
            fig.savefig(buf_display, format='png', bbox_inches='tight',
                        dpi=200, facecolor='white')
            buf_export = io.BytesIO()
            fig.savefig(buf_export, format='png', bbox_inches='tight',
                        dpi=300, facecolor='white')
            plt.close(fig)

            self.finished.emit(buf_display.getvalue(), buf_export.getvalue())
        except Exception as e:
            self.error.emit(str(e))


class OverlapWorker(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, data, landmark_path, face_ids, is_video, max_frame=None, template=None,
                 contour_margin=0.0):
        super().__init__()
        self.data, self.landmark_path = data, landmark_path
        self.face_ids, self.is_video = face_ids, is_video
        self.max_frame = max_frame
        self.template = template
        self.contour_margin = contour_margin

    def run(self):
        try:
            from core.overlap_review import scan_overlaps
            path = self.landmark_path
            landmarks = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
            if self.is_video and self.max_frame is not None and 'Frame_Index' in landmarks:
                indices = pd.to_numeric(landmarks.Frame_Index, errors='coerce')
                landmarks = landmarks[indices.between(1, self.max_frame)]
            self.result.emit(scan_overlaps(self.data, landmarks, self.face_ids,
                                          self.is_video, self.isInterruptionRequested, template=self.template,
                                          contour_margin=self.contour_margin))
        except InterruptedError:
            pass
        except Exception as exc:
            self.error.emit(str(exc))


class VisualizationView(QWidget):
    def __init__(self):
        super().__init__()
        
        self.cap = None
        self.data = None
        self.data_grouped = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.is_video = False
        self.static_image = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.current_project_path = ""
        self.stimuli_folder = ""
        self.video_writer = None
        
        # Average face + file resolver
        self.avg_face_resized = None
        self._resolve_file = resolve_file_in_folder or self._default_resolve_file_in_folder
        self.avg_face_bgr = self._load_average_face_bgr()
        self._template = None
        self._contour_margin = 0.0
        
        self._overlap_worker = None
        self._overlap_generation = 0
        self._pending_overlap_scan = False
        self._overlap_candidates = {}
        self._overlap_outlines = {}
        self._overlap_rows = []
        self._review_sample = None
        self._review_dirty = False
        self._build_ui()

        # Store full-resolution pixmaps for responsive rescaling of top panels
        self._left_full_pixmap = None
        self._right_full_pixmap = None
        
        self._project_settings = None
        self._syncing_selection = False
        self._overlay = ProcessingOverlay(self)
        
        # Connect signals for auto-save
        self.spin_bins.editingFinished.connect(self._on_settings_edited)
        self.spin_sigma.editingFinished.connect(self._on_settings_edited)
        self.chk_scatter.toggled.connect(self._on_settings_edited)
        self.chk_normalize.toggled.connect(self._on_settings_edited)

        # Multi-face state
        self._transform_face_ids = []     # Face IDs found in transformed data
        self._gaze_regions = set()        # gaze_region values present in loaded data
        self._selected_vis_face = "all"   # "all" or integer face ID
    
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)
        
        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.hide()
        layout.addWidget(self.lbl_notice)
        
        # ── Top: Data Loading ──
        data_group = QGroupBox("Data && Stimulus")
        data_layout = QVBoxLayout()
        
        # Data file row
        data_row = QHBoxLayout()
        self.btn_load_data = QPushButton("Load Transformed Data")
        self.btn_load_data.clicked.connect(self.load_data)
        data_row.addWidget(self.btn_load_data)
        self.lbl_data = QLabel("No data loaded")
        data_row.addWidget(self.lbl_data, 1)
        data_layout.addLayout(data_row)
        
        # Stimulus selector
        stim_row = QHBoxLayout()
        stim_row.addWidget(QLabel("Stimulus:"))
        self.combo_stimulus = QComboBox()
        self.combo_stimulus.setMinimumWidth(250)
        self.combo_stimulus.currentTextChanged.connect(self.on_stimulus_changed)
        stim_row.addWidget(self.combo_stimulus, 1)
        
        self.btn_load_video = QPushButton("Load Custom Video")
        self.btn_load_video.clicked.connect(self.load_video_manual)
        stim_row.addWidget(self.btn_load_video)
        data_layout.addLayout(stim_row)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # ── Color Control ──
        color_row = QHBoxLayout()
        self.chk_unique_colors = QCheckBox("Show unique participant colors")
        self.chk_unique_colors.setToolTip("Color gaze points based on Participant_ID")
        self.chk_unique_colors.toggled.connect(lambda: self.show_frame(self.current_frame_idx))
        color_row.addWidget(self.chk_unique_colors)

        self.chk_highlight_off_face = QCheckBox("Highlight off-face gaze")
        self.chk_highlight_off_face.setToolTip(
            "Draw off-face gaze points in magenta (and unmapped samples in gray)\n"
            "with a color legend, here and in the heatmap scatter.\n"
            "Uses the gaze_region column written during transformation.")
        self.chk_highlight_off_face.setEnabled(False)
        self.chk_highlight_off_face.toggled.connect(lambda: self.show_frame(self.current_frame_idx))
        color_row.addWidget(self.chk_highlight_off_face)

        # Face selector dropdown (hidden until multi-face data loaded)
        self.lbl_face_selector = QLabel("Transform Face:")
        self.lbl_face_selector.setObjectName("hint-label")
        self.lbl_face_selector.hide()
        color_row.addWidget(self.lbl_face_selector)

        self.combo_face = QComboBox()
        self.combo_face.setMinimumWidth(120)
        self.combo_face.currentIndexChanged.connect(self._on_face_selector_changed)
        self.combo_face.hide()
        color_row.addWidget(self.combo_face)

        color_row.addStretch()
        layout.addLayout(color_row)
        
        # ── Dual-Panel Display ──
        panels_layout = QHBoxLayout()
        
        # Left: original
        left_panel = QVBoxLayout()
        self.lbl_left_title = QLabel("Original Gaze")
        self.lbl_left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_left_title.setObjectName("panel-title-blue")
        left_panel.addWidget(self.lbl_left_title)
        self.lbl_left = QLabel()
        self.lbl_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_left.setMinimumSize(160, 120)
        self.lbl_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_left.setObjectName("image-panel")
        left_panel.addWidget(self.lbl_left)
        panels_layout.addLayout(left_panel)
        
        # Right: transformed
        right_panel = QVBoxLayout()
        self.lbl_right_title = QLabel("Transformed Gaze")
        self.lbl_right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_right_title.setObjectName("panel-title-warning")
        right_panel.addWidget(self.lbl_right_title)
        self.lbl_right = QLabel()
        self.lbl_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_right.setMinimumSize(160, 120)
        self.lbl_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_right.setObjectName("image-panel")
        right_panel.addWidget(self.lbl_right)
        panels_layout.addLayout(right_panel)
        
        layout.addLayout(panels_layout, 1)
        
        # ── Seek Bar ──
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.valueChanged.connect(self._seek_frame)
        layout.addWidget(self.slider)
        
        # ── Playback Controls ──
        controls_layout = QHBoxLayout()

        self.btn_step_back = QPushButton("◀ Step")
        self.btn_step_back.setMinimumSize(64, 28)
        self.btn_step_back.clicked.connect(self.step_backward)
        self.btn_step_back.setAccessibleName("Step backward")
        controls_layout.addWidget(self.btn_step_back)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setMinimumSize(80, 28)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setAccessibleName("Play/Pause")
        controls_layout.addWidget(self.btn_play)

        self.btn_step_fwd = QPushButton("Step ▶")
        self.btn_step_fwd.setMinimumSize(64, 28)
        self.btn_step_fwd.clicked.connect(self.step_forward)
        self.btn_step_fwd.setAccessibleName("Step forward")
        controls_layout.addWidget(self.btn_step_fwd)
        
        controls_layout.addSpacing(16)
        
        self.lbl_frame = QLabel("Frame: 0 / 0")
        controls_layout.addWidget(self.lbl_frame)
        
        controls_layout.addStretch()
        
        self.btn_export = QPushButton("Export Video")
        self.btn_export.setMinimumSize(120, 28)
        self.btn_export.clicked.connect(self.export_video)
        controls_layout.addWidget(self.btn_export)
        
        layout.addLayout(controls_layout)

        # Sample-level review shares the original video and playback controls.
        self.overlap_group = QGroupBox("Review overlapping faces")
        self.overlap_group.setCheckable(True)
        self.overlap_group.setChecked(False)
        self.overlap_group.setEnabled(False)
        review_layout = QVBoxLayout(self.overlap_group)
        self.overlap_body = QWidget()
        body = QVBoxLayout(self.overlap_body)
        body.setContentsMargins(0, 0, 0, 0)
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Candidate face IDs:"))
        self.review_faces = QLineEdit()
        self.review_faces.setPlaceholderText("e.g. 1,2 (same faces selected for transformation)")
        settings_row.addWidget(self.review_faces, 1)
        self.btn_scan_overlap = QPushButton("Detect overlaps")
        self.btn_scan_overlap.clicked.connect(self._scan_overlaps)
        settings_row.addWidget(self.btn_scan_overlap)
        self.btn_save_review = QPushButton("Save reviewed data…")
        self.btn_save_review.clicked.connect(self._save_overlap_review)
        self.btn_save_review.setEnabled(False)
        settings_row.addWidget(self.btn_save_review)
        layout.addLayout(settings_row)
        self.lbl_overlap = QLabel("Detect samples within multiple facial meshes. Select a row to seek; choose a face or exclude. Gold marks the selected gaze; outlines identify candidate faces.")
        self.lbl_overlap.setWordWrap(True)
        layout.addWidget(self.lbl_overlap)
        self.overlap_table = QTableWidget(0, 9)
        self.overlap_table.setHorizontalHeaderLabels([
            "Sample row", "Participant", "Video time (s)", "Frame", "Source TIMESTAMP", "Candidate faces", "Assignment", "Section (frames)", "Gaze match"])
        self.overlap_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.overlap_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.overlap_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overlap_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.overlap_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.overlap_table.setMinimumHeight(130)
        self.overlap_table.setMaximumHeight(230)
        self.overlap_table.itemSelectionChanged.connect(self._select_overlap_row)
        body.addWidget(self.overlap_table)
        review_layout.addWidget(self.overlap_body)
        self.overlap_body.hide()
        self.overlap_group.toggled.connect(self.overlap_body.setVisible)
        self.overlap_group.toggled.connect(self._toggle_overlap_review)
        layout.addWidget(self.overlap_group)

        # ── Distribution Analysis Section ──
        dist_group = QGroupBox("Distribution Analysis (Heatmap)")
        self._distribution_group = dist_group
        dist_layout = QVBoxLayout()

        # Controls row
        dist_controls = QHBoxLayout()

        # Multi-select grouping variable list
        group_vbox = QVBoxLayout()
        group_vbox.addWidget(QLabel("Group by (select 1–2):" ))
        self.list_group_cols = QListWidget()
        self.list_group_cols.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_group_cols.setMaximumHeight(120)
        self.list_group_cols.setMaximumWidth(184)
        group_vbox.addWidget(self.list_group_cols)
        dist_controls.addLayout(group_vbox)

        # Right side: params + buttons
        right_col = QVBoxLayout()

        # Histogram parameter row
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Bins:"))
        self.spin_bins = QDoubleSpinBox()
        self.spin_bins.setValue(100)
        self.spin_bins.setRange(10, 500)
        self.spin_bins.setDecimals(0)
        self.spin_bins.setFixedWidth(64)
        self.spin_bins.setToolTip("Number of histogram bins (default: 100)")
        self.spin_bins.setAccessibleName("Histogram bins")
        param_row.addWidget(self.spin_bins)

        param_row.addWidget(QLabel("Sigma:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.setRange(0.1, 50.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setFixedWidth(64)
        self.spin_sigma.setToolTip("Gaussian smoothing sigma (default: 2)")
        self.spin_sigma.setAccessibleName("Gaussian smoothing sigma")
        param_row.addWidget(self.spin_sigma)

        self.chk_scatter = QCheckBox("Show scatter")
        self.chk_scatter.setChecked(False)
        param_row.addWidget(self.chk_scatter)

        self.chk_normalize = QCheckBox("Normalize by participant")
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setToolTip(
            "Average per-participant histograms so each participant\n"
            "contributes equally regardless of number of data points")
        param_row.addWidget(self.chk_normalize)

        param_row.addStretch()
        right_col.addLayout(param_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_generate_dist = QPushButton("Generate Heatmaps")
        self.btn_generate_dist.clicked.connect(self._generate_distribution)
        btn_row.addWidget(self.btn_generate_dist)

        self.btn_export_dist = QPushButton("Export Image")
        self.btn_export_dist.clicked.connect(self._export_heatmap_image)
        self.btn_export_dist.setEnabled(False)
        btn_row.addWidget(self.btn_export_dist)

        btn_row.addSpacing(16)
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_in.setMinimumWidth(32)
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_zoom_in.setAccessibleName("Zoom in")
        self.btn_zoom_in.clicked.connect(lambda: self._dist_zoom_by(1.25))
        btn_row.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_zoom_out.setMinimumWidth(32)
        self.btn_zoom_out.setToolTip("Zoom out")
        self.btn_zoom_out.setAccessibleName("Zoom out")
        self.btn_zoom_out.clicked.connect(lambda: self._dist_zoom_by(0.8))
        btn_row.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("Fit")
        self.btn_zoom_reset.setMinimumWidth(32)
        self.btn_zoom_reset.setToolTip("Reset zoom to fit window")
        self.btn_zoom_reset.setAccessibleName("Reset zoom")
        self.btn_zoom_reset.clicked.connect(self._dist_zoom_reset)
        btn_row.addWidget(self.btn_zoom_reset)

        btn_row.addStretch()
        right_col.addLayout(btn_row)

        dist_controls.addLayout(right_col)
        dist_controls.addStretch()
        dist_layout.addLayout(dist_controls)

        # Scrollable heatmap area (allows panning when zoomed in)
        self.lbl_dist_image = QLabel()
        self.lbl_dist_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dist_image.setObjectName("bg-primary-panel")
        self.lbl_dist_image.setScaledContents(False)

        self.dist_scroll = QScrollArea()
        self.dist_scroll.setWidget(self.lbl_dist_image)
        self.dist_scroll.setWidgetResizable(False)
        self.dist_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dist_scroll.setMinimumHeight(300)
        self.dist_scroll.setObjectName("bg-primary-panel")
        dist_layout.addWidget(self.dist_scroll, 1)

        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group, 1)

        # Store the full-resolution pixmap for responsive rescaling
        self._dist_full_pixmap = None
        self._dist_fig_png = None
        self._dist_zoom = 0  # 0 = fit-to-window mode
    
    # ------------------------------------------------------------------
    # Project path & auto-load
    # ------------------------------------------------------------------
    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings
        
        selected = settings.resolve_transformed_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != getattr(self, "_data_path", None):
            self._syncing_selection = True
            try:
                self._load_data_file(selected)
            finally:
                self._syncing_selection = False
        
        # Block signals to avoid auto-save loop on load
        self.blockSignals(True)
        try:
            # 1. Update Transformed Gaze label with delay info
            delay = settings.get("coordinate_setup", "gaze_response_lag", "0")
            try:
                val = float(delay)
                # If val is integer like 200.0, show 200
                if val.is_integer():
                    val = int(val)
                txt = f"{val}ms"
            except ValueError:
                txt = "0ms"
                
            self._right_title_delay = txt
            self.lbl_right_title.setText(f"Transformed Gaze (Delay: {txt})")
            
            # 2. Update visualization controls
            bins = settings.get("visualization", "bins", "")
            if bins:
                try:
                    self.spin_bins.setValue(float(bins))
                except ValueError:
                    pass
            sigma = settings.get("visualization", "sigma", "")
            if sigma:
                try:
                    self.spin_sigma.setValue(float(sigma))
                except ValueError:
                    pass
            self.chk_scatter.setChecked(
                settings.get("visualization", "show_scatter", False))
            self.chk_normalize.setChecked(
                settings.get("visualization", "normalize_by_participant", True))
        finally:
            self.blockSignals(False)

    def _on_settings_edited(self, *args):
        """Auto-save settings when UI changes."""
        if self._project_settings:
            # save_settings() calls .set() which now triggers save()
            self.save_settings(self._project_settings)

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        settings.set("visualization", "bins",
                     str(int(self.spin_bins.value())))
        settings.set("visualization", "sigma",
                     str(self.spin_sigma.value()))
        settings.set("visualization", "show_scatter",
                     self.chk_scatter.isChecked())
        settings.set("visualization", "normalize_by_participant",
                     self.chk_normalize.isChecked())

    def refresh_data(self):
        """Reload project path to refresh file lists."""
        if self.current_project_path:
            self.set_project_path(self.current_project_path)

    def set_project_path(self, path):
        if not self._allow_review_discard():
            return
        self._review_dirty = False
        self._clear_overlap_review()
        self.current_project_path = path
        self.stimuli_folder = os.path.join(path, "stimuli")

        # ── Reset all previews & figures ──
        # Video / gaze panels
        if self.cap:
            self.cap.release()
            self.cap = None
        self.data = None
        self.lbl_left.clear()
        self.lbl_right.clear()
        self._left_full_pixmap = None
        self._right_full_pixmap = None
        self.slider.setValue(0)
        self.slider.setMaximum(0)
        self.lbl_frame.setText("Frame: 0 / 0")

        # Distribution heatmap
        self.lbl_dist_image.clear()
        self._dist_full_pixmap = None
        self._dist_fig_png = None
        self.btn_export_dist.setEnabled(False)
        self.list_group_cols.clear()
        self._dist_zoom = 0

        # Reset face state
        self._transform_face_ids = []
        self._selected_vis_face = "all"
        self.combo_face.hide()
        self.lbl_face_selector.hide()

        # Explicit selection is authoritative; never select a run by modification time.
        self._set_notice(True)
        if self._project_settings:
            selected = self._project_settings.resolve_transformed_data_file("")
            if selected:
                self._load_data_file(selected)
                self._set_notice(False)
            else:
                expected = self._project_settings.expected_transformed_data_file("")
                if expected:
                    self.lbl_notice.setText(
                        f"{status_icon_warning} Expected transformed data not found:\n"
                        f"{os.path.relpath(expected, self.current_project_path)}\n"
                        "Run Step 3 (Transformation) for the selected source dataset, or load a different transformed file."
                    )
                    self.lbl_notice.show()
    
    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        start_dir = os.path.join(self.current_project_path, "transformed_data") \
            if self.current_project_path else ""
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Transformed Data", start_dir, 
            "Data Files (*.parquet *.csv)"
        )
        if f:
            self._set_transformed_data_file(f, update_settings=True)

    def _set_transformed_data_file(self, path, update_settings=False):
        if not path or not os.path.isfile(path):
            return
        if self.data is not None and os.path.abspath(path) == os.path.abspath(getattr(self, '_data_path', '')):
            return
        loaded = self._load_data_file(path)
        if loaded is False:
            return
        if update_settings:
            self._set_global_transformed_file(path)
    
    def _load_data_file(self, path):
        if not self._allow_review_discard():
            return False
        try:
            from core.template_manager import TemplateManager
            from core.template_manager import dataset_manifest
            self._template = TemplateManager(self.current_project_path).for_dataset(path)
            self._contour_margin = dataset_manifest(path).get('contour_margin') or 0.0
            self.avg_face_bgr = cv2.cvtColor(self._template.image,cv2.COLOR_BGRA2BGR)
            self.avg_face_resized = None
            if path.endswith('.parquet'):
                self.data = pd.read_parquet(path)
            else:
                self.data = pd.read_csv(path)
        except Exception as e:
            self.data = None
            self._template = None
            self._stop_overlap_worker()
            self._clear_overlap_review()
            self.lbl_left.clear()
            self.lbl_right.clear()
            self.lbl_dist_image.clear()
            self._dist_fig_png = None
            self.btn_export_dist.setEnabled(False)
            show_error(self, "Data Load Error", f"Could not load the data:\n{e}")
            return False
        
        self.data = self.data.reset_index(drop=True)
        self._review_dirty = False
        self._clear_overlap_review()
        self._data_path = path
        self.lbl_data.setText(os.path.basename(path))
        self._set_notice(False)
        
        # Determine stimulus column
        self._stim_col = None
        for col in ['stimulusName', 'videoName', 'STIMULI_NAME']:
            if col in self.data.columns:
                self._stim_col = col
                break
        
        # Populate stimulus dropdown
        self.combo_stimulus.blockSignals(True)
        self.combo_stimulus.clear()
        if self._stim_col:
            unique_stims = sorted(self.data[self._stim_col].dropna().unique().tolist())
            self.combo_stimulus.addItems(unique_stims)
        self.combo_stimulus.blockSignals(False)

        # Populate grouping column dropdown for distribution analysis
        self._populate_group_columns()

        # Detect multi-face columns
        self._detect_face_columns()

        has_region = 'gaze_region' in self.data.columns
        self._gaze_regions = set(self.data['gaze_region'].dropna().unique()) if has_region else set()
        self.chk_highlight_off_face.setEnabled(has_region)
        if not has_region:
            self.chk_highlight_off_face.setChecked(False)
        
        # Trigger first selection
        if self.combo_stimulus.count() > 0:
            self.on_stimulus_changed(self.combo_stimulus.currentText())

    def _detect_face_columns(self):
        """Scan loaded data for transformed_x_face* columns to detect multi-face data."""
        if self.data is None:
            return

        self._transform_face_ids = []

        for col in self.data.columns:
            m = re.match(r'transformed_x_face(\d+)', col)
            if m:
                fid = int(m.group(1))
                if fid not in self._transform_face_ids:
                    self._transform_face_ids.append(fid)
        self._transform_face_ids.sort()

        # Populate face selector
        if len(self._transform_face_ids) > 1:
            self.combo_face.blockSignals(True)
            self.combo_face.clear()
            self.combo_face.addItem("All Faces", "all")
            for fid in self._transform_face_ids:
                hex_c = face_color_hex(fid)
                self.combo_face.addItem(f"Face {fid}", fid)
                self.combo_face.setItemData(self.combo_face.count() - 1, QColor(hex_c), Qt.ItemDataRole.ForegroundRole)
            self.combo_face.blockSignals(False)
            self.combo_face.setCurrentIndex(0)
            self.combo_face.show()
            self.lbl_face_selector.show()
            self._selected_vis_face = "all"
        else:
            self.combo_face.hide()
            self.lbl_face_selector.hide()
            self._selected_vis_face = "all"

        self._update_right_title()

    def _get_transform_columns(self):
        """Return (x_col, y_col) pairs for the currently selected face(s)."""
        if not self._transform_face_ids:
            return [('transformed_x', 'transformed_y')]
        if self._transform_face_ids and len(self._transform_face_ids) == 1:
            return [('transformed_x', 'transformed_y')]

        selected = self.combo_face.currentData() if self.combo_face.isVisible() else "all"
        pairs = []
        if selected == "all":
            for fid in self._transform_face_ids:
                pairs.append((f'transformed_x_face{fid}', f'transformed_y_face{fid}'))
        else:
            pairs.append((f'transformed_x_face{selected}', f'transformed_y_face{selected}'))
        return pairs
    
    def _on_face_selector_changed(self):
        """Called when the face selector dropdown changes."""
        self._update_right_title()
        self.show_frame(self.current_frame_idx)

    def _update_right_title(self):
        """Update the Transformed Gaze panel title to show active face."""
        delay_text = getattr(self, '_right_title_delay', '')
        if delay_text:
            base = f"Transformed Gaze (Delay: {delay_text})"
        else:
            base = "Transformed Gaze"

        if self.combo_face.isVisible():
            selected = self.combo_face.currentData()
            if selected == "all":
                self.lbl_right_title.setText(f"{base} — All Faces")
            else:
                self.lbl_right_title.setText(f"{base} — Face {selected}")
        else:
            self.lbl_right_title.setText(base)

    # ------------------------------------------------------------------
    # Stimulus selection
    # ------------------------------------------------------------------
    def on_stimulus_changed(self, stim_name):
        if not stim_name or self.data is None:
            return
        
        self._clear_overlap_review()
        self._set_review_face_ids(stim_name)
        self._group_stimulus_frames(stim_name)
        
        # Try to auto-resolve the video/image file
        if self._resolve_file and self.stimuli_folder:
            media_path = self._resolve_file(stim_name, self.stimuli_folder)
            if media_path:
                self._open_media(media_path)
            else:
                self.lbl_left_title.setText(f"Original Gaze — {status_icon_warning} media not found")
                # Clear display and prevent reviewing against the previous stimulus.
                if self.cap:
                    self.cap.release()
                    self.cap = None
                if self.is_playing:
                    self.toggle_play()
                self.static_image = None
                self.total_frames = 0
                self.lbl_left.clear()
                self.lbl_left.setObjectName("image-panel")
    
    def load_video_manual(self):
        start_dir = self.stimuli_folder if self.stimuli_folder else self.current_project_path
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Media", start_dir, 
            "Media Files (*.mp4 *.avi *.webm *.mov *.jpg *.jpeg *.png *.bmp)"
        )
        if f:
            self._open_media(f)
    
    def _open_media(self, path):
        """Open a video or image file for visualization."""
        # Release previous
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.is_playing:
            self.toggle_play()
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            # Image mode
            self.is_video = False
            self.static_image = cv2.imread(path)
            if self.static_image is None:
                show_error(self, "Image Error", f"Could not read the image file:\n{path}")
                return
            self.total_frames = 1
            self.fps = 0
            h = self.static_image.shape[0]
        else:
            # Video mode
            self.is_video = True
            self.static_image = None
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                show_error(self, "Video Error", f"Could not open the video file:\n{path}")
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update slider
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.slider.setValue(0)
        self.current_frame_idx = 0
        
        # Resize average face to match
        scale = h / self.avg_face_bgr.shape[0]
        dim = (int(self.avg_face_bgr.shape[1] * scale), h)
        self.avg_face_resized = cv2.resize(self.avg_face_bgr, dim, interpolation=cv2.INTER_AREA)

        # Refresh controls and immediately render the first frame/image so the
        # preview loads as soon as a stimulus is resolved.
        is_vid = self.is_video and self.total_frames > 1
        self.btn_play.setEnabled(is_vid)
        self.btn_step_back.setEnabled(is_vid)
        self.btn_step_fwd.setEnabled(is_vid)
        self.slider.setEnabled(self.total_frames > 0)
        self.btn_export.setEnabled(is_vid)
        self.show_frame(0)
        self._scan_overlaps(automatic=True)

    def _load_average_face_bgr(self):
        if average_face is not None:
            # The H5 template is stored in BGRA order.
            return cv2.cvtColor(average_face, cv2.COLOR_BGRA2BGR)
        if avg_face_image is not None:
            # Pre-converted RGBA template.
            return cv2.cvtColor(avg_face_image, cv2.COLOR_RGBA2BGR)
        try:
            import h5py
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            h5_path = os.path.join(base, 'average_face_data.h5')
            if os.path.isfile(h5_path):
                with h5py.File(h5_path, 'r') as hf:
                    avg = hf['average_face'][:]
                return cv2.cvtColor(avg, cv2.COLOR_BGRA2BGR)
        except Exception:
            pass
        return np.zeros((500, 500, 3), dtype=np.uint8)

    def _default_resolve_file_in_folder(self, name, folder, suffix='', extensions=None):
        if extensions is None:
            extensions = ['.webm', '.mp4', '.avi', '.mov', '.mkv',
                          '.jpg', '.jpeg', '.png', '.bmp', '.gif']
        base = os.path.splitext(name)[0]
        if suffix:
            candidate = os.path.join(folder, base + suffix)
        else:
            candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
        for ext in extensions:
            if suffix:
                candidate = os.path.join(folder, base + suffix + ext)
            else:
                candidate = os.path.join(folder, base + ext)
            if os.path.isfile(candidate):
                return candidate
        if os.path.isdir(folder):
            target_prefix = (base + suffix).lower()
            for f in os.listdir(folder):
                if f.lower().startswith(target_prefix):
                    return os.path.join(folder, f)
        return None

    def _set_notice(self, show):
        if show:
            self.lbl_notice.setText(
                "No transformed data found. Run Step 3 (Transformation) to generate "
                "files in transformed_data/."
            )
            self.lbl_notice.show()
        else:
            self.lbl_notice.hide()

    def _set_global_transformed_file(self, path, only_if_empty=False):
        if not self._project_settings or self._syncing_selection:
            return
        if only_if_empty:
            existing = self._project_settings.get_transformed_data_file("")
            if existing:
                return
        self._project_settings.set("global", "selected_transformed_file", path)
        
        self.lbl_left_title.setText(f"Original Gaze — {os.path.basename(path)}")
        
        # Enable/disable video controls
        is_vid = self.is_video and self.total_frames > 1
        self.btn_play.setEnabled(is_vid)
        self.btn_step_back.setEnabled(is_vid)
        self.btn_step_fwd.setEnabled(is_vid)
        self.slider.setEnabled(is_vid)
        self.btn_export.setEnabled(is_vid)
        
        self.show_frame(0)
    
    def _group_stimulus_frames(self, stim_name):
        """Cache this stimulus's samples grouped by integer frame index."""
        if self._stim_col:
            filtered = self.data[self.data[self._stim_col] == stim_name].copy()
        else:
            filtered = self.data.copy()
        if '_video_frame_index' in filtered.columns:
            filtered['_video_frame_index'] = pd.to_numeric(
                filtered['_video_frame_index'], errors='coerce'
            ).fillna(-1).astype(int)
            self.data_grouped = filtered.groupby('_video_frame_index')
        else:
            self.data_grouped = None

    def _toggle_overlap_review(self, enabled):
        # Give the original video enough space for inspecting occlusion.
        self._distribution_group.setVisible(not enabled)
        self.show_frame(self.current_frame_idx)

    def _allow_review_discard(self):
        if self._review_dirty:
            return ask_confirm(self, "Unsaved overlap decisions",
                               "Discard unsaved overlap decisions and load another dataset? Save reviewed data first to retain them.")
        return True

    def _clear_overlap_review(self):
        self._overlap_generation += 1
        if self._overlap_worker is not None and self._overlap_worker.isRunning():
            self._overlap_worker.requestInterruption()
        self._overlap_candidates = {}
        self._overlap_outlines = {}
        self._overlap_rows = []
        self._review_sample = None
        self.overlap_table.setRowCount(0)
        self.overlap_group.setChecked(False)
        self.overlap_group.setEnabled(False)
        self.btn_save_review.setEnabled(self._review_dirty)
        self.lbl_overlap.setText("Overlap review is available only for stimuli with overlapping facial meshes. Check extracted landmarks to enable it.")

    def _set_review_face_ids(self, stimulus):
        values = []
        if self.data is not None and 'selected_face_ids' in self.data:
            subset = self.data[self.data[self._stim_col] == stimulus] if self._stim_col else self.data
            values = subset.selected_face_ids.dropna().astype(str).unique().tolist()
        self.review_faces.setText(values[0] if len(values) == 1 else '')
        self.review_faces.setReadOnly(len(values) == 1)

    def _scan_overlaps(self, automatic=False):
        if self._overlap_worker is not None and self._overlap_worker.isRunning():
            if automatic:
                self._overlap_worker.requestInterruption()
                self._pending_overlap_scan = True
            return
        if self.data is None or self.total_frames <= 0:
            if automatic:
                return
            show_warning(self, "Load data and media", "Load transformed data and its original stimulus before reviewing overlaps.")
            return
        if any(col.startswith('transformed_x_face') for col in self.data):
            if automatic:
                return
            show_warning(self, "Legacy transformation", "Re-run transformation to create the current single-assignment output before review.")
            return
        if os.path.splitext(getattr(self, '_data_path', ''))[0].endswith('_AOI'):
            if automatic:
                return
            show_warning(self, 'Load transformed data', 'Review the transformed dataset before AOI analysis, then re-run AOI classification on the reviewed file.')
            return
        required = {'stimuli_x', 'stimuli_y', 'face_id', 'transformed_x', 'transformed_y', 'gaze_region'}
        if not required.issubset(self.data):
            if automatic:
                return
            show_warning(self, "Transformation required", "Re-run transformation to generate face IDs and gaze_region before review.")
            return
        try:
            face_ids = list(dict.fromkeys(int(v.strip()) for v in self.review_faces.text().split(',')))
            if len(face_ids) < 2 or any(fid <= 0 for fid in face_ids):
                raise ValueError()
        except ValueError:
            if automatic:
                self.lbl_overlap.setText("Overlap review disabled. Fewer than two candidate faces, or older data needs explicit face IDs before checking.")
                return
            show_warning(self, "Candidate faces", "Enter at least two positive face IDs selected for transformation, separated by commas. Older files require this explicit selection.")
            return
        stim = self.combo_stimulus.currentText()
        folder = os.path.join(self.current_project_path, 'processed_stimuli_data')
        path = self._resolve_file(stim, folder, suffix='_landmarks', extensions=['.parquet', '.csv'])
        if not path and automatic:
            self.lbl_overlap.setText("Overlap review disabled until landmarks are checked. Use Detect overlaps to select the extracted landmark file.")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select extracted landmarks for this stimulus", folder,
                                                  "Landmarks (*.parquet *.csv)")
        if not path:
            return
        subset = self.data[self.data[self._stim_col] == stim].copy() if self._stim_col else self.data.copy()
        self._clear_overlap_review()
        generation = self._overlap_generation
        self.btn_scan_overlap.setEnabled(False)
        self.lbl_overlap.setText("Checking facial meshes for overlapping gaze samples…")
        worker = OverlapWorker(subset, path, face_ids, self.is_video, self.total_frames, template=self._template,
                               contour_margin=self._contour_margin)
        self._overlap_worker = worker
        worker.result.connect(lambda result: self._overlaps_detected(result, generation))
        worker.error.connect(lambda message: self._overlap_error(message, generation))
        worker.finished.connect(self._overlap_scan_finished)
        worker.start()

    def _overlap_scan_finished(self):
        self.btn_scan_overlap.setEnabled(True)
        if self._pending_overlap_scan:
            self._pending_overlap_scan = False
            self._scan_overlaps(automatic=True)

    def _overlap_error(self, message, generation):
        if generation == self._overlap_generation:
            self.lbl_overlap.setText("Overlap detection failed.")
            show_error(self, "Overlap detection", message)

    def _overlaps_detected(self, result, generation):
        if generation != self._overlap_generation:
            return
        self._overlap_candidates = result['section_samples']
        self._overlap_outlines = result['outlines']
        self._overlap_rows = sorted(self._overlap_candidates,
                                   key=lambda key: (self.data.at[key, '_video_frame_index'] if self.is_video else 1, key))
        self.overlap_group.setEnabled(bool(result['intervals']))
        self.overlap_table.setEnabled(bool(result['intervals']))
        self.overlap_table.setRowCount(len(self._overlap_rows))
        for table_row, sample_id in enumerate(self._overlap_rows):
            if 'automatic_face_id' not in self.data:
                self.data['automatic_face_id'] = pd.array([pd.NA] * len(self.data), dtype='Int64')
            if pd.isna(self.data.at[sample_id, 'automatic_face_id']):
                self.data.at[sample_id, 'automatic_face_id'] = self.data.at[sample_id, 'face_id']
            sample = self.data.loc[sample_id]
            frame = int(sample['_video_frame_index']) if self.is_video else 1
            seconds = (frame - 1) / self.fps if self.is_video and self.fps > 0 else 0
            candidates = self._overlap_candidates[sample_id]
            if 'assignment_review_status' not in self.data:
                self.data['assignment_review_status'] = 'not_required'
            if 'overlap_face_ids' not in self.data:
                self.data['overlap_face_ids'] = ''
            self.data.at[sample_id, 'overlap_face_ids'] = ','.join(map(str, candidates))
            if self.data.at[sample_id, 'assignment_review_status'] == 'not_required':
                self.data.at[sample_id, 'assignment_review_status'] = 'unreviewed' if len(candidates) > 1 else 'context'
            cells = [sample_id + 1, sample.get('Participant_ID', ''), f"{seconds:.3f}", frame,
                     sample.get('TIMESTAMP', ''), ', '.join(map(str, candidates))]
            for col, value in enumerate(cells):
                self.overlap_table.setItem(table_row, col, QTableWidgetItem(str(value)))
            combo = QComboBox()
            automatic = sample.get('automatic_face_id', sample.face_id)
            if pd.isna(automatic):
                automatic = sample.face_id
            baseline_region = sample.get('automatic_gaze_region', sample.gaze_region)
            if pd.isna(baseline_region):
                baseline_region = sample.gaze_region
            label = f"Face {int(automatic)}" if pd.notna(automatic) else baseline_region
            suffix = 'unreviewed' if len(candidates) > 1 else 'context'
            combo.addItem(f"Keep original ({label}) — {suffix}", 'automatic')
            for fid in candidates:
                combo.addItem(f"Assign Face {fid}", fid)
            combo.addItem("Exclude from facial analysis", 'exclude')
            status = sample.get('assignment_review_status', 'unreviewed')
            choice = 'exclude' if status == 'excluded' else int(sample.face_id) if status == 'assigned' else 'automatic'
            combo.setCurrentIndex(max(0, combo.findData(choice)))
            combo.currentIndexChanged.connect(lambda _, key=sample_id, box=combo: self._decide_overlap(key, box.currentData()))
            self.overlap_table.setCellWidget(table_row, 6, combo)
            section = next((f"{start}–{end}" for start, end in result['intervals'] if start <= frame <= end), '')
            if 'overlap_section' not in self.data:
                self.data['overlap_section'] = ''
            self.data.at[sample_id, 'overlap_section'] = section
            self.overlap_table.setItem(table_row, 7, QTableWidgetItem(section))
            valid_gaze = np.isfinite(sample[['stimuli_x', 'stimuli_y']].to_numpy(dtype=float)).all()
            match = 'Invalid gaze' if not valid_gaze else 'Ambiguous' if len(candidates) > 1 else f"Face {next(iter(candidates))}" if candidates else str(baseline_region)
            self.overlap_table.setItem(table_row, 8, QTableWidgetItem(match))
        self.btn_save_review.setEnabled(bool(self._overlap_rows) or self._review_dirty)
        if result['intervals']:
            self.lbl_overlap.setText(f"{len(result['intervals'])} overlap sections; {len(self._overlap_rows)} gaze samples, "
                                     f"{len(result['overlaps'])} with ambiguous face matches. "
                                     f"{result['unchecked']} samples could not be fully checked. "
                                     "Expand Review overlapping faces to inspect all samples in these sections. "
                                     "Video time is frame start; Source TIMESTAMP keeps its input unit.")
        else:
            self.lbl_overlap.setText(f"No facial overlap detected in the available landmarks; review disabled. "
                                     f"{result['unchecked']} samples could not be fully checked.")
        self.show_frame(self.current_frame_idx)

    def _select_overlap_row(self):
        row = self.overlap_table.currentRow()
        if row < 0 or row >= len(self._overlap_rows):
            return
        key = self._overlap_rows[row]
        self._review_sample = key
        frame = int(self.data.at[key, '_video_frame_index']) - 1 if self.is_video else 0
        if not 0 <= frame < self.total_frames:
            self.lbl_overlap.setText("This sample's frame is outside the loaded video. Check the stimulus and frame alignment.")
            return
        if self.is_playing:
            self.toggle_play()
        self.slider.setValue(frame)
        self.current_frame_idx = frame
        self.show_frame(frame)

    def _decide_overlap(self, key, decision):
        from core.overlap_review import apply_decision
        try:
            apply_decision(self.data, key, self._overlap_candidates[key], decision)
        except Exception as exc:
            sample = self.data.loc[key]
            status = sample.get('assignment_review_status', 'unreviewed')
            previous = 'exclude' if status == 'excluded' else int(sample.face_id) if status == 'assigned' else 'automatic'
            box = self.overlap_table.cellWidget(self._overlap_rows.index(key), 6)
            box.blockSignals(True)
            box.setCurrentIndex(max(0, box.findData(previous)))
            box.blockSignals(False)
            show_error(self, "Assignment error", str(exc))
            return
        self._review_dirty = True
        self.btn_save_review.setEnabled(True)
        # Cached groups must reflect the new coordinates and assignment.
        self._group_stimulus_frames(self.combo_stimulus.currentText())
        self.overlap_table.selectRow(self._overlap_rows.index(key))
        self._select_overlap_row()
        self._dist_fig_png = None
        self._dist_full_pixmap = None
        self.lbl_dist_image.clear()
        self.btn_export_dist.setEnabled(False)

    def _save_overlap_review(self):
        from core.io_utils import atomic_save_parquet
        source = getattr(self, '_data_path', '')
        default = os.path.splitext(source)[0] + '_reviewed.parquet'
        path, _ = QFileDialog.getSaveFileName(self, "Save reviewed transformed data", default, "Parquet (*.parquet)")
        if not path:
            return
        if not path.lower().endswith('.parquet'):
            path += '.parquet'
        if os.path.realpath(path) == os.path.realpath(source):
            show_warning(self, "Choose a new file", "Save the reviewed dataset under a different name to preserve the source.")
            return
        try:
            from core.template_manager import save_dataset, dataset_manifest, save_review_decisions
            manifest = dataset_manifest(source)
            if self._template:
                manifest.update(template_id=self._template.template_id,template_hash=self._template.content_hash)
            save_dataset(self.data,path,manifest)
            save_review_decisions(self.current_project_path,self.data,manifest)
        except Exception as exc:
            show_error(self, "Save error", str(exc))
            return
        self._review_dirty = False
        self.btn_save_review.setEnabled(False)
        self._data_path = path
        self.lbl_data.setText(os.path.basename(path))
        self._set_global_transformed_file(path)
        show_success(self, "Reviewed data saved", "Saved assignments and audit columns. Use this file to re-run AOI analysis; existing AOI results are not updated.")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def toggle_play(self):
        if self.fps <= 0 or not self.is_video:
            return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶  Play")
        else:
            self.timer.start(int(1000 / self.fps))
            self.btn_play.setText("⏸ Pause")
        self.is_playing = not self.is_playing
    
    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self.show_frame(self.current_frame_idx)
        else:
            self.toggle_play()  # Stop at end
    
    def step_forward(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self.show_frame(self.current_frame_idx)
    
    def step_backward(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self.show_frame(self.current_frame_idx)
    
    def _slider_pressed(self):
        self._was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()
    
    def _slider_released(self):
        if hasattr(self, '_was_playing') and self._was_playing:
            self.toggle_play()
    
    def _seek_frame(self, idx):
        self.current_frame_idx = idx
        self.show_frame(idx)
    
    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def show_frame(self, idx):
        """Render frame idx with gaze overlay in both panels."""
        # Get the source frame
        frame = None
        if not self.is_video and self.static_image is not None:
            frame = self.static_image.copy()
        elif self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, read_frame = self.cap.read()
            if ret:
                frame = read_frame
        
        if frame is None:
            return
        
        display_frame = frame.copy()
        display_avg = self.avg_face_resized.copy() if self.avg_face_resized is not None \
            else np.zeros_like(frame)
        
        h, w = display_frame.shape[:2]
        avg_h, avg_w = display_avg.shape[:2]
        
        # Data index (1-based, matching VIDEO_FRAME_INDEX)
        data_idx = idx + 1
        
        # Get gaze points for this frame
        # For images: show ALL gaze points for the stimulus (no frame filtering)
        # For videos: filter by VIDEO_FRAME_INDEX
        points = None
        if not self.is_video:
            # Image — show all points from the filtered stimulus data
            if self.data_grouped is not None:
                # data_grouped is grouped by frame, so concatenate all groups
                points = pd.concat([g for _, g in self.data_grouped])
            elif self.data is not None and self._stim_col:
                stim = self.combo_stimulus.currentText()
                points = self.data[self.data[self._stim_col] == stim]
        else:
            # Video — show only points matching this frame
            # Skip lookup if requested frame exceeds video length; this avoids
            # showing gaze points whose _video_frame_index was assigned beyond
            # the video's actual frame count.
            if data_idx <= self.total_frames \
               and self.data_grouped is not None \
               and data_idx in self.data_grouped.groups:
                points = self.data_grouped.get_group(data_idx)
        
        if points is not None and len(points) > 0:
            use_unique_colors = self.chk_unique_colors.isChecked()
            transform_cols = self._get_transform_columns()

            for _, row in points.iterrows():
                region_color = self._region_color(row.get('gaze_region'))
                # Determine original-gaze color
                color_orig = (0, 255, 0)  # Default Green
                if use_unique_colors:
                    pid = row.get('Participant_ID', 'unknown')
                    color_orig = self._get_color_for_id(str(pid))
                color_orig = region_color or color_orig

                # Original gaze
                sx = row.get('stimuli_x', np.nan)
                sy = row.get('stimuli_y', np.nan)
                if pd.notna(sx) and pd.notna(sy) and np.isfinite([sx, sy]).all():
                    self._draw_gaze_marker(display_frame, (int(sx * w), int(sy * h)), color_orig)

                # Transformed gaze — iterate column pairs per face
                for x_col, y_col in transform_cols:
                    tx = row.get(x_col, np.nan)
                    ty = row.get(y_col, np.nan)
                    if pd.notna(tx) and pd.notna(ty) and np.isfinite([tx, ty]).all():
                        # Determine color: face-based or participant-based
                        if use_unique_colors:
                            pid = row.get('Participant_ID', 'unknown')
                            color_trans = self._get_color_for_id(str(pid))
                        else:
                            # Extract face ID from column name for face-based color
                            m = re.search(r'face(\d+)', x_col)
                            if m:
                                color_trans = face_color_bgr(int(m.group(1)))
                            else:
                                color_trans = (0, 0, 255)  # Default Red
                        color_trans = region_color or color_trans
                        self._draw_gaze_marker(display_avg, (int(tx * avg_w), int(ty * avg_h)),
                                               color_trans)
        
        if self.chk_highlight_off_face.isChecked():
            self._draw_legend(display_frame, self._legend_entries(transformed=False))
            self._draw_legend(display_avg, self._legend_entries(transformed=True))

        # Candidate outlines are guides only; detection uses the full facial mesh.
        if self.overlap_group.isChecked():
            for legend_row, (fid, hull) in enumerate(self._overlap_outlines.get(data_idx, {}).items()):
                pixels = np.round(hull * [w, h]).astype(np.int32)
                color = face_color_bgr(fid)
                cv2.polylines(display_frame, [pixels], True, color, 2)
                cv2.putText(display_frame, f"Face {fid}", (10, 54 + 24 * legend_row),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            if self._review_sample is not None and self.data is not None:
                sample = self.data.loc[self._review_sample]
                sample_frame = sample.get('_video_frame_index', 1)
                if not self.is_video or sample_frame == data_idx:
                    x, y = sample[['stimuli_x', 'stimuli_y']].to_numpy(dtype=float)
                    if np.isfinite([x, y]).all():
                        point = (int(x * w), int(y * h))
                        cv2.circle(display_frame, point, 14, (0, 215, 255), 3)
                        cv2.drawMarker(display_frame, point, (0, 215, 255),
                                       cv2.MARKER_CROSS, 22, 2)

        # Frame counter overlay
        cv2.putText(display_frame, f"Frame {idx+1}/{self.total_frames}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write to video if exporting
        if self.video_writer is not None:
            if display_avg.shape[0] != display_frame.shape[0]:
                display_avg_export = cv2.resize(
                    display_avg, 
                    (int(avg_w * h / avg_h), h)
                )
            else:
                display_avg_export = display_avg
            combined = np.hstack((display_frame, display_avg_export))
            self.video_writer.write(combined)
        
        # Convert to QPixmap for display (store full-res for responsive rescale)
        self._display_cv2(display_frame, self.lbl_left, '_left_full_pixmap')
        self._display_cv2(display_avg, self.lbl_right, '_right_full_pixmap')
        
        self.lbl_frame.setText(f"Frame: {idx+1} / {self.total_frames}")
    
    @staticmethod
    def _draw_gaze_marker(img, point, color):
        """Draw a ring-and-dot gaze marker."""
        cv2.circle(img, point, 8, color, 2)
        cv2.circle(img, point, 2, color, -1)

    def _region_color(self, region):
        """Override color for off-face/unmapped samples, or None to keep the default."""
        if not self.chk_highlight_off_face.isChecked():
            return None
        if region == 'off_face':
            return OFF_FACE_BGR
        if region == 'unmapped':
            return UNMAPPED_BGR
        return None

    def _legend_entries(self, transformed):
        """(label, BGR color or None) rows describing the current color scheme."""
        if self.chk_unique_colors.isChecked():
            entries = [("On-face: participant colors", None)]
        elif not transformed:
            entries = [("On-face", (0, 255, 0))]
        else:
            entries = []
            for x_col, _ in self._get_transform_columns():
                m = re.search(r'face(\d+)', x_col)
                if m:
                    fid = int(m.group(1))
                    entries.append((f"On-face: Face {fid}", face_color_bgr(fid)))
            entries = entries or [("On-face", (0, 0, 255))]
        entries.append(("Off-face", OFF_FACE_BGR))
        # Unmapped samples have no transformed coordinates, so only the stimulus shows them.
        if not transformed and 'unmapped' in self._gaze_regions:
            entries.append(("Unmapped", UNMAPPED_BGR))
        return entries

    @staticmethod
    def _draw_legend(img, entries):
        """Draw a semi-transparent legend box in the bottom-left corner."""
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        line_h, pad, swatch = 22, 10, 20
        text_w = max(cv2.getTextSize(label, font, scale, thick)[0][0] for label, _ in entries)
        box_w = pad * 2 + swatch + text_w
        box_h = pad * 2 + line_h * len(entries) - 6
        h = img.shape[0]
        x0, y0 = 10, h - 10 - box_h
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
        for i, (label, color) in enumerate(entries):
            cy = y0 + pad + 8 + i * line_h
            cx = x0 + pad + 7
            if color is not None:
                cv2.circle(img, (cx, cy), 7, color, 2)
                cv2.circle(img, (cx, cy), 2, color, -1)
            text_x = x0 + pad + (swatch if color is not None else 0)
            cv2.putText(img, label, (text_x, cy + 5), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)

    def _display_cv2(self, cv_img, label, store_attr=None):
        """Convert an OpenCV BGR image to QPixmap and display on a QLabel.
        If store_attr is given, store the full-res pixmap for responsive rescaling."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        full_pixmap = QPixmap.fromImage(q_img)
        if store_attr:
            setattr(self, store_attr, full_pixmap)
        target = label.contentsRect().size()
        if target.isEmpty():
            target = label.size()
        scaled = full_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)
    
    def _get_color_for_id(self, identifier):
        """Generate a deterministic consistent BGR color from a string ID."""
        import hashlib
        hash_object = hashlib.md5(identifier.encode())
        hex_dig = hash_object.hexdigest()
        # Take first 6 chars for RGB
        r = int(hex_dig[0:2], 16)
        g = int(hex_dig[2:4], 16)
        b = int(hex_dig[4:6], 16)
        return (b, g, r)  # OpenCV uses BGR

    # ------------------------------------------------------------------
    # Video export
    # ------------------------------------------------------------------
    def export_video(self):
        if not self.cap or self.total_frames <= 0:
            show_warning(self, "No Video Loaded", "Load a video first to record a playback preview.")
            return
        
        # Ask for save path
        default_name = "visualization_export.mp4"
        if self.current_project_path:
            default_path = os.path.join(self.current_project_path, default_name)
        else:
            default_path = default_name
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_path, "Video Files (*.mp4)"
        )
        if not save_path:
            return
        
        # Disable UI
        self.setEnabled(False)
    
        # Stop playback if running
        if self.is_playing:
            self.toggle_play()
        
        # Setup writer
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        avg_w = self.avg_face_resized.shape[1] if self.avg_face_resized is not None else w
        combined_w = w + avg_w
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            save_path, fourcc, self.fps, (combined_w, h)
        )
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Exporting video...", "Cancel", 0, self.total_frames, self)
        self.progress_dialog.canceled.connect(self._export_video_cancel)
        self.progress_dialog.show()
        self._overlay.set_status("Exporting Video")
        self._overlay.hide_progress()
        self._overlay.show()
        
        # Render all frames
        for i in range(self.total_frames):
            self.show_frame(i)
            # Update progress periodically
            if i % 30 == 0:
                self.lbl_frame.setText(f"Exporting: {i+1} / {self.total_frames}")
        
        self.video_writer.release()
        self.video_writer = None
        self._overlay.hide()
        
        # Enable UI
        self.setEnabled(True)
        self.progress_dialog.close()
                
        # Reset to frame 0
        self.show_frame(0)
        self.slider.setValue(0)
        
        if self.progress_dialog.wasCanceled():
            show_warning(self, "Export Canceled", "The video export was canceled.")
        else:
            show_success(self, "Export Complete", f"Video preview saved to:\n{save_path}")

    def _export_video_cancel(self):
        self._overlay.hide()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _stop_overlap_worker(self):
        self._pending_overlap_scan = False
        if self._overlap_worker is not None and self._overlap_worker.isRunning():
            self._overlap_worker.requestInterruption()
            self._overlap_worker.wait()

    def closeEvent(self, event):
        if not self._allow_review_discard():
            event.ignore()
            return
        self._stop_overlap_worker()
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if hasattr(self, '_heatmap_worker') and self._heatmap_worker.isRunning():
            self._heatmap_worker.wait()

    # ------------------------------------------------------------------
    # Responsive resize — rescale stored heatmap pixmap
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale top gaze panels
        if hasattr(self, '_left_full_pixmap') and self._left_full_pixmap and not self._left_full_pixmap.isNull():
            target = self.lbl_left.contentsRect().size()
            if not target.isEmpty():
                scaled = self._left_full_pixmap.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lbl_left.setPixmap(scaled)
        if hasattr(self, '_right_full_pixmap') and self._right_full_pixmap and not self._right_full_pixmap.isNull():
            target = self.lbl_right.contentsRect().size()
            if not target.isEmpty():
                scaled = self._right_full_pixmap.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lbl_right.setPixmap(scaled)
        # Rescale distribution heatmap
        if hasattr(self, '_dist_full_pixmap') and self._dist_full_pixmap and not self._dist_full_pixmap.isNull():
            self._rescale_dist_pixmap()

    def _rescale_dist_pixmap(self):
        if self._dist_zoom == 0:
            # Fit-to-window: scale to scroll area viewport
            vp = self.dist_scroll.viewport()
            scaled = self._dist_full_pixmap.scaled(
                vp.width(), vp.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # Fixed zoom: scale to zoom factor of original pixmap
            w = int(self._dist_full_pixmap.width() * self._dist_zoom)
            h = int(self._dist_full_pixmap.height() * self._dist_zoom)
            scaled = self._dist_full_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        self.lbl_dist_image.setPixmap(scaled)
        self.lbl_dist_image.resize(scaled.size())

    def _dist_zoom_by(self, factor):
        """Zoom the heatmap by a multiplicative factor."""
        if self._dist_full_pixmap is None or self._dist_full_pixmap.isNull():
            return
        if self._dist_zoom == 0:
            # Switch from fit-to-window to explicit zoom
            vp = self.dist_scroll.viewport()
            current = self._dist_full_pixmap.scaled(
                vp.width(), vp.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._dist_zoom = current.width() / self._dist_full_pixmap.width()
        self._dist_zoom = max(0.1, min(self._dist_zoom * factor, 5.0))
        self._rescale_dist_pixmap()

    def _dist_zoom_reset(self):
        """Reset zoom to fit-to-window."""
        self._dist_zoom = 0
        if self._dist_full_pixmap and not self._dist_full_pixmap.isNull():
            self._rescale_dist_pixmap()

    # ------------------------------------------------------------------
    # Distribution Analysis
    # ------------------------------------------------------------------
    def _populate_group_columns(self):
        """Fill the grouping column list with suitable categorical columns (checkable)."""
        self.list_group_cols.clear()
        if self.data is not None:
            for col in self.data.columns:
                n_unique = self.data[col].dropna().nunique()
                # Only include columns with 2–20 unique values (sensible grouping)
                if 2 <= n_unique <= 20:
                    item = QListWidgetItem(col)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.list_group_cols.addItem(item)

    def _get_selected_group_cols(self):
        """Return list of checked column names from the grouping list."""
        selected = []
        for i in range(self.list_group_cols.count()):
            item = self.list_group_cols.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def _generate_distribution(self):
        """Gather parameters and launch background worker for heatmap generation."""
        if self.data is None:
            show_warning(self, "No Data Loaded", "Load transformed data first to generate distribution analysis.")
            return

        self.btn_generate_dist.setEnabled(False)
        self.btn_export_dist.setEnabled(False)
        self._overlay.set_status("Generating Heatmaps")
        self._overlay.hide_progress()
        self._overlay.show()

        try:
            df = self.data.copy()

            # Prepare columns — handle both single-face and multi-face schemas
            transform_cols = self._get_transform_columns()
            fix_x_parts = []
            fix_y_parts = []
            for x_col, y_col in transform_cols:
                if x_col in df.columns and y_col in df.columns:
                    fix_x_parts.append(pd.to_numeric(df[x_col], errors='coerce'))
                    fix_y_parts.append(pd.to_numeric(df[y_col], errors='coerce'))
            if fix_x_parts:
                # Stack multiple face columns: take first non-NaN per row
                df['fix_x'] = pd.concat(fix_x_parts, axis=1).bfill(axis=1).iloc[:, 0]
                df['fix_y'] = pd.concat(fix_y_parts, axis=1).bfill(axis=1).iloc[:, 0]
            elif 'transformed_x' in df.columns:
                df['fix_x'] = pd.to_numeric(df['transformed_x'], errors='coerce')
                df['fix_y'] = pd.to_numeric(df['transformed_y'], errors='coerce')

            if 'fix_x' not in df.columns or 'fix_y' not in df.columns:
                show_warning(self, "Required Columns Missing",
                                    "Data must have transformed_x/transformed_y or fix_x/fix_y.")
                self.btn_generate_dist.setEnabled(True)
                return

            df = df.dropna(subset=['fix_x', 'fix_y'])

            # Scale to pixel coords
            face_w = self._template.width if self._template else width
            face_h = self._template.height if self._template else height
            df['fix_x'] = df['fix_x'] * face_w
            df['fix_y'] = df['fix_y'] * face_h

            # Read histogram parameters and clamp to safe ranges
            bins_text = str(int(self.spin_bins.value()))
            sigma_text = str(self.spin_sigma.value())
            n_bins = int(bins_text) if bins_text else 100
            sigma = float(sigma_text) if sigma_text else 2.0
            n_bins = max(10, min(n_bins, 500))   # prevent huge allocations
            sigma = max(0.1, min(sigma, 50.0))

            group_cols = self._get_selected_group_cols()
            normalize = self.chk_normalize.isChecked()
            show_scatter = self.chk_scatter.isChecked()
            highlight_off_face = self.chk_highlight_off_face.isChecked()

            # Start loading animation
            self._loading_dots = 0
            self._loading_timer = QTimer(self)
            self._loading_timer.timeout.connect(self._on_loading_tick)
            self._loading_timer.start(350)
            self.lbl_dist_image.setText("Generating figure\u2026")
            self.lbl_dist_image.setStyleSheet(
                f"background-color: {THEME['bg_primary']}; color: {THEME['text_secondary']}; font-size: 18px;")

            # Launch all computation in background thread
            self._heatmap_worker = HeatmapWorker(
                df, self.avg_face_bgr, face_w, face_h,
                n_bins, sigma, normalize, group_cols, show_scatter,
                highlight_off_face
            )
            self._heatmap_worker.finished.connect(self._on_heatmap_finished)
            self._heatmap_worker.error.connect(self._on_heatmap_error)
            self._heatmap_worker.start()

        except Exception as e:
            show_error(self, "Analysis Failed", f"Distribution analysis failed:\n{e}")
            self.btn_generate_dist.setEnabled(True)
            self.btn_generate_dist.setText("Generate Heatmaps")

    def _on_loading_tick(self):
        """Animate the loading text with cycling dots."""
        self._loading_dots = (self._loading_dots + 1) % 4
        dots = '.' * self._loading_dots
        self.lbl_dist_image.setText(f"Generating figure{dots}")

    def _on_heatmap_finished(self, display_png, export_png):
        """Handle completed heatmap rendering from the worker thread."""
        self._overlay.hide()
        # Stop loading animation
        if hasattr(self, '_loading_timer') and self._loading_timer.isActive():
            self._loading_timer.stop()

        # Convert PNG bytes to QPixmap
        img = QImage()
        img.loadFromData(display_png)
        self._dist_full_pixmap = QPixmap.fromImage(img)
        self._dist_fig_png = export_png
        self._dist_zoom = 0

        # Display
        self._rescale_dist_pixmap()
        self.btn_export_dist.setEnabled(True)
        self.btn_generate_dist.setEnabled(True)
        self.btn_generate_dist.setText("Generate Heatmaps")

    def _on_heatmap_error(self, err_msg):
        """Handle errors from the heatmap worker thread."""
        self._overlay.hide()
        if hasattr(self, '_loading_timer') and self._loading_timer.isActive():
            self._loading_timer.stop()
        self.lbl_dist_image.setText("")
        show_error(self, "Heatmap Failed", f"Heatmap rendering failed:\n{err_msg}")
        self.btn_generate_dist.setEnabled(True)
        self.btn_generate_dist.setText("Generate Heatmaps")

    def _export_heatmap_image(self):
        """Save the current heatmap figure to a file."""
        if not hasattr(self, '_dist_fig_png') or self._dist_fig_png is None:
            show_warning(self, "No Heatmap", "Generate the heatmaps first before exporting.")
            return

        default = os.path.join(self.current_project_path, 'kde_heatmap.png') \
            if self.current_project_path else 'kde_heatmap.png'
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Heatmap", default,
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, 'wb') as f:
                f.write(self._dist_fig_png)
            show_success(self, "Exported", f"Heatmap saved to:\n{os.path.basename(path)}")
        except Exception as e:
            show_error(self, "Export Failed", f"Could not export heatmap:\n{e}")

    @staticmethod
    def _fig_to_pixmap(fig, facecolor='white'):
        """Convert a matplotlib Figure to QPixmap."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor=facecolor)
        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())
        return QPixmap.fromImage(img)

    @staticmethod
    def _fig_to_png_bytes(fig, facecolor='white'):
        """Convert a matplotlib Figure to PNG bytes for export."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=facecolor)
        buf.seek(0)
        return buf.read()
