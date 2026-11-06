from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTabWidget, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from .button_styles import button_style
from .processing_utils import ProcessingOverlay
from .theme import (DESCRIPTION_LABEL_STYLE, THEME,
                     status_icon_success, status_icon_error,
                     status_icon_warning, status_icon_info)
from .dialog_helpers import show_success, show_warning, show_error
import os
import logging
import pandas as pd
from core.io_utils import atomic_save_parquet
from core.data_resolver import is_imported_data_file

logger = logging.getLogger('iTemplate2')


class DataWorkspace(QWidget):
    def __init__(self):
        super().__init__()
        
        self.stimuli_dir = ""
        self.gaze_data_path = ""
        self.current_project_path = ""
        self._project_settings = None
        self._syncing_selection = False
        self._overlay = ProcessingOverlay(self)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        lbl_title = QLabel("Step 2: Match Data & Stimuli")
        lbl_title.setObjectName("title-label")
        layout.addWidget(lbl_title)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        layout.addWidget(self.lbl_notice)

        # ── Mode Tabs ─────────────────────────────────────────────────
        self.mode_tabs = QTabWidget()
        self.mode_tabs.currentChanged.connect(lambda _: self._update_onboarding_state())
        self.mode_tabs.setStyleSheet("")

        # ── Tab 0: Screen-based Eye Tracker ───────────────────────────
        screen_tab = QWidget()
        screen_tab.setStyleSheet("")
        screen_layout = QVBoxLayout()

        lbl_screen_desc = QLabel(
            "Use this tab when your data comes from a <b>screen-based eye tracker</b> "
            "(e.g., EyeLink, Tobii). Match each <code>stimulusName</code> in your gaze data "
            "to its corresponding processed landmark file. "
            "This ensures each stimulus is linked to geometry data for coordinate transformation."
        )
        lbl_screen_desc.setWordWrap(True)
        lbl_screen_desc.setStyleSheet(DESCRIPTION_LABEL_STYLE + " line-height: 1.4;")
        screen_layout.addWidget(lbl_screen_desc)

        controls_layout = QHBoxLayout()
        self.btn_stimuli = QPushButton("Select Processed Stimuli Directory")
        self.btn_stimuli.clicked.connect(self.select_stimuli_dir)
        self.btn_gaze_file = QPushButton("Select Gaze Data File")
        self.btn_gaze_file.clicked.connect(self.select_gaze_file)
        controls_layout.addWidget(self.btn_stimuli)
        controls_layout.addWidget(self.btn_gaze_file)
        screen_layout.addLayout(controls_layout)

        self.lbl_paths = QLabel("No paths selected.")
        screen_layout.addWidget(self.lbl_paths)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["stimulusName (in data)", "Matched Processed File", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        screen_layout.addWidget(self.table)

        self.lbl_summary = QLabel("")
        screen_layout.addWidget(self.lbl_summary)

        btn_refresh = QPushButton("Refresh && Match")
        btn_refresh.clicked.connect(self.run_matching)
        screen_layout.addWidget(btn_refresh)

        screen_tab.setLayout(screen_layout)
        self.mode_tabs.addTab(screen_tab, "Screen-based Eye Tracker")

        # ── Tab 1: Wearable Eye Tracker ───────────────────────────────
        wearable_tab = QWidget()
        wearable_tab.setStyleSheet("")
        wearable_layout = QVBoxLayout()

        lbl_w_desc = QLabel(
            "Use this tab when your data comes from a <b>wearable / scene-based eye tracker</b> "
            "(e.g., Pupil Labs, Tobii Glasses). Each participant's data file is matched to its "
            "corresponding scene video. You can then set <code>stimulusName</code> from the matched "
            "video and merge all files into a single dataset for downstream processing."
        )
        lbl_w_desc.setWordWrap(True)
        lbl_w_desc.setStyleSheet(DESCRIPTION_LABEL_STYLE + " line-height: 1.4;")
        wearable_layout.addWidget(lbl_w_desc)

        # Folder selection
        w_controls = QHBoxLayout()
        self.btn_w_data_dir = QPushButton("Select Data Files Folder")
        self.btn_w_data_dir.clicked.connect(self._select_wearable_data_dir)
        self.btn_w_video_dir = QPushButton("Select Scene Videos Folder")
        self.btn_w_video_dir.clicked.connect(self._select_wearable_video_dir)
        w_controls.addWidget(self.btn_w_data_dir)
        w_controls.addWidget(self.btn_w_video_dir)
        wearable_layout.addLayout(w_controls)

        self.lbl_w_paths = QLabel("No paths selected.")
        wearable_layout.addWidget(self.lbl_w_paths)

        # Matching strategy
        match_row = QHBoxLayout()
        match_row.addWidget(QLabel("Matching method:"))
        self.cmb_match_strategy = QComboBox()
        self.cmb_match_strategy.setMinimumWidth(150)
        self.cmb_match_strategy.setToolTip("How data files are matched to scene videos")
        self.cmb_match_strategy.addItems([
            "Exact basename",
            "First N characters",
            "Contains",
            "Manual",
        ])
        self.cmb_match_strategy.currentIndexChanged.connect(self._on_match_strategy_changed)
        match_row.addWidget(self.cmb_match_strategy)

        self.lbl_n_chars = QLabel("N:")
        self.spn_n_chars = QSpinBox()
        self.spn_n_chars.setRange(1, 50)
        self.spn_n_chars.setValue(3)
        self.spn_n_chars.valueChanged.connect(self._run_wearable_matching)
        self.lbl_n_chars.hide()
        self.spn_n_chars.hide()
        match_row.addWidget(self.lbl_n_chars)
        match_row.addWidget(self.spn_n_chars)
        match_row.addStretch()
        wearable_layout.addLayout(match_row)

        # Matching table
        self.w_table = QTableWidget()
        self.w_table.setColumnCount(3)
        self.w_table.setHorizontalHeaderLabels(["Data File", "Matched Video", "Status"])
        self.w_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        wearable_layout.addWidget(self.w_table)

        self.lbl_w_summary = QLabel("")
        wearable_layout.addWidget(self.lbl_w_summary)

        # Action buttons
        w_btn_row = QHBoxLayout()
        btn_w_refresh = QPushButton("Refresh && Match")
        btn_w_refresh.clicked.connect(self._run_wearable_matching)
        w_btn_row.addWidget(btn_w_refresh)

        self.btn_w_set_stim = QPushButton("Set stimulusName from Video")
        self.btn_w_set_stim.setStyleSheet(button_style("primary", padding="8px"))
        self.btn_w_set_stim.clicked.connect(self._set_stimulus_name_wearable)
        self.btn_w_set_stim.setEnabled(False)
        w_btn_row.addWidget(self.btn_w_set_stim)

        self.btn_w_merge = QPushButton("Merge into Single File")
        self.btn_w_merge.setStyleSheet(button_style("secondary", padding="8px"))
        self.btn_w_merge.clicked.connect(self._merge_wearable_files)
        self.btn_w_merge.setEnabled(False)
        w_btn_row.addWidget(self.btn_w_merge)
        wearable_layout.addLayout(w_btn_row)

        self.lbl_w_status = QLabel("")
        wearable_layout.addWidget(self.lbl_w_status)

        wearable_tab.setLayout(wearable_layout)
        self.mode_tabs.addTab(wearable_tab, "Wearable / Scene-based Eye Tracker")

        layout.addWidget(self.mode_tabs)
        layout.addStretch()

        # Internal state for wearable tab
        self._w_data_dir = ""
        self._w_video_dir = ""
        self._w_matches = []  # list of (data_file, video_file_or_None)
        self._w_video_files = []  # all video filenames (for manual mode)
        self._update_onboarding_state()

    # ==================================================================
    # Common helpers
    # ==================================================================
    def set_project_path(self, path):
        self.current_project_path = path
        # Auto-populate from project folders if they exist
        processed = os.path.join(path, "processed_stimuli_data")
        imported = os.path.join(path, "imported_parquet")
        stimuli = os.path.join(path, "stimuli")
        
        if os.path.isdir(processed):
            self.stimuli_dir = processed
        if os.path.isdir(imported):
            # Find the latest .parquet file
            parquet_files = [os.path.join(imported, f) for f in os.listdir(imported) 
                             if f.lower().endswith('.parquet')]
            if parquet_files:
                latest = max(parquet_files, key=os.path.getmtime)
                self.gaze_data_path = latest

        # Auto-populate wearable dirs
        if os.path.isdir(imported):
            self._w_data_dir = imported
        if os.path.isdir(stimuli):
            self._w_video_dir = stimuli
            
        self.update_labels()
        if self.stimuli_dir and self.gaze_data_path:
            self.run_matching()

        # Wearable auto-match if both dirs set
        self._update_wearable_labels()
        if self._w_data_dir and self._w_video_dir:
            self._run_wearable_matching()
        self._update_onboarding_state()

    def load_settings(self, settings):
        self._project_settings = settings
        selected = settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != self.gaze_data_path:
            self._syncing_selection = True
            try:
                self.gaze_data_path = selected
                self.update_labels()
                if self.stimuli_dir and self.gaze_data_path:
                    self.run_matching()
            finally:
                self._syncing_selection = False
        self._update_onboarding_state()

    # ==================================================================
    # Screen-based tab (unchanged logic)
    # ==================================================================
    def select_stimuli_dir(self):
        start_dir = self.current_project_path if self.current_project_path else ""
        d = QFileDialog.getExistingDirectory(self, "Select Processed Stimuli Directory", start_dir)
        if d:
            self.stimuli_dir = d
            self.update_labels()
            self.run_matching()
            
    def select_gaze_file(self):
        start_dir = self.current_project_path if self.current_project_path else ""
        f, _ = QFileDialog.getOpenFileName(self, "Select Gaze Data File", start_dir, "Data Files (*.csv *.parquet)")
        if f:
            self.gaze_data_path = f
            self.update_labels()
            self.run_matching()
            self._set_global_selected_file(f)

    def _set_global_selected_file(self, path):
        if not self._project_settings or self._syncing_selection:
            return
        if self.current_project_path and is_imported_data_file(self.current_project_path, path):
            self._project_settings.set("global", "source_data_file", path)
        self._project_settings.set("global", "selected_data_file", path)
        
    def update_labels(self):
        stim_label = os.path.basename(self.stimuli_dir) if self.stimuli_dir else 'None'
        gaze_label = os.path.basename(self.gaze_data_path) if self.gaze_data_path else 'None'
        self.lbl_paths.setText(f"Stimuli: {stim_label}    |    Gaze Data: {gaze_label}")
        self._update_onboarding_state()

    def _extract_stimulus_base(self, filename):
        """
        Extract the base stimulus name from a processed file.
        e.g.  'AF_010F_landmarks.parquet' -> 'AF_010F'
              'AF_010F_landmarks.csv'     -> 'AF_010F'
              'AF_010F_landmarks.npy'     -> 'AF_010F'
              'AF_010F.parquet'           -> 'AF_010F'
        Strips known suffixes like _landmarks, _features, etc.
        """
        base = os.path.splitext(filename)[0]  # Remove extension
        # Strip known processing suffixes
        for suffix in ['_landmarks', '_features', '_processed', '_output', '_blendshapes', '_transformation']:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        return base

    def _collect_stimulus_names_from_data(self):
        """
        Read all gaze data files and collect unique stimulusName values.
        Returns: set of unique stimulusName strings
        """
        gaze_files = []
        if os.path.isfile(self.gaze_data_path):
            gaze_files = [self.gaze_data_path]
        elif os.path.isdir(self.gaze_data_path):
            for f in os.listdir(self.gaze_data_path):
                if f.lower().endswith(('.csv', '.parquet')):
                    gaze_files.append(os.path.join(self.gaze_data_path, f))
        
        all_stim_names = set()
        for gf in gaze_files:
            try:
                if gf.lower().endswith('.parquet'):
                    df = pd.read_parquet(gf, columns=['stimulusName'])
                else:
                    # Try to read just the stimulusName column
                    df = pd.read_csv(gf, usecols=['stimulusName'])
                
                unique_vals = df['stimulusName'].dropna().unique()
                all_stim_names.update(str(v) for v in unique_vals)
            except (KeyError, ValueError):
                # stimulusName column doesn't exist in this file
                pass
            except Exception as e:
                logger.error("Error reading %s: %s", gf, e)
                
        return all_stim_names
        
    def run_matching(self):
        if not self.stimuli_dir or not self.gaze_data_path:
            return
            
        self.table.setRowCount(0)
        self.lbl_summary.setText("Scanning data files...")
        
        # 1. Collect unique stimulusName values from gaze data
        stim_names_in_data = self._collect_stimulus_names_from_data()
        
        if not stim_names_in_data:
            self.lbl_summary.setText("No stimulusName values found in the selected gaze data.")
            self._update_onboarding_state()
            return
        
        # 2. Scan processed stimuli directory and build lookup
        # Map: base_name -> full_filename
        processed_lookup = {}
        if os.path.exists(self.stimuli_dir):
            for f in os.listdir(self.stimuli_dir):
                base = self._extract_stimulus_base(f)
                processed_lookup[base] = f
        
        # 3. Match each stimulusName to a processed file
        matched_count = 0
        missing_count = 0
        
        for stim_name in sorted(stim_names_in_data):
            # The stimulusName in data might still have path/extension
            # Try exact match first, then basename, then without extension
            match_key = stim_name
            matched_file = None
            
            # Try direct match
            if match_key in processed_lookup:
                matched_file = processed_lookup[match_key]
            else:
                # Try stripping any remaining path
                match_key = match_key.replace('\\', '/').rsplit('/', 1)[-1]
                if match_key in processed_lookup:
                    matched_file = processed_lookup[match_key]
                else:
                    # Try without extension
                    match_key_no_ext = os.path.splitext(match_key)[0]
                    if match_key_no_ext in processed_lookup:
                        matched_file = processed_lookup[match_key_no_ext]
            
            # Add to table
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(stim_name))
            
            if matched_file:
                self.table.setItem(row, 1, QTableWidgetItem(matched_file))
                status_item = QTableWidgetItem(f"{status_icon_success} Matched")
                status_item.setForeground(QColor(THEME["status_success"]))
                self.table.setItem(row, 2, status_item)
                matched_count += 1
            else:
                self.table.setItem(row, 1, QTableWidgetItem("—"))
                status_item = QTableWidgetItem(f"{status_icon_error} Missing")
                status_item.setForeground(QColor(THEME["status_error"]))
                self.table.setItem(row, 2, status_item)
                missing_count += 1
        
        total = matched_count + missing_count
        self.lbl_summary.setText(
            f"Total: {total} stimuli  |  {status_icon_success} Matched: {matched_count}  |  {status_icon_error} Missing: {missing_count}"
        )
        self._update_onboarding_state()

    # ==================================================================
    # Wearable Eye Tracker tab
    # ==================================================================
    def _select_wearable_data_dir(self):
        start = self._w_data_dir or self.current_project_path or ""
        d = QFileDialog.getExistingDirectory(self, "Select Data Files Folder", start)
        if d:
            self._w_data_dir = d
            self._update_wearable_labels()
            self._run_wearable_matching()

    def _select_wearable_video_dir(self):
        start = self._w_video_dir or self.current_project_path or ""
        d = QFileDialog.getExistingDirectory(self, "Select Scene Videos Folder", start)
        if d:
            self._w_video_dir = d
            self._update_wearable_labels()
            self._run_wearable_matching()

    def _update_wearable_labels(self):
        data_lbl = os.path.basename(self._w_data_dir) if self._w_data_dir else "None"
        video_lbl = os.path.basename(self._w_video_dir) if self._w_video_dir else "None"
        self.lbl_w_paths.setText(f"Data folder: {data_lbl}    |    Videos folder: {video_lbl}")
        self._update_onboarding_state()

    def _on_match_strategy_changed(self):
        """Show/hide the N spin box based on selected strategy."""
        strategy = self.cmb_match_strategy.currentText()
        show_n = (strategy == "First N characters")
        self.lbl_n_chars.setVisible(show_n)
        self.spn_n_chars.setVisible(show_n)
        self._run_wearable_matching()

    def _list_wearable_data_files(self):
        """
        Return canonical wearable data files (one file per participant basename).

        If both CSV and Parquet exist for the same basename, prefer Parquet to
        avoid duplicate participant rows during merge.

        Also returns a dict with discovery statistics.
        """
        if not self._w_data_dir:
            return [], {"raw_candidates": 0, "canonical_files": 0, "resolved_duplicates": 0}

        candidates = [
            f for f in os.listdir(self._w_data_dir)
            if f.lower().endswith((".parquet", ".csv"))
        ]

        by_base = {}
        duplicate_bases = set()
        for filename in sorted(candidates):
            base, ext = os.path.splitext(filename)
            base_lower = base.lower()
            ext_lower = ext.lower()

            # Never re-ingest merged outputs.
            if base_lower == "merged" or base_lower.startswith("merged_"):
                continue

            chosen = by_base.get(base)
            if chosen is None:
                by_base[base] = filename
                continue

            duplicate_bases.add(base)

            # Prefer parquet over csv when both exist for the same participant.
            chosen_ext = os.path.splitext(chosen)[1].lower()
            if chosen_ext != ".parquet" and ext_lower == ".parquet":
                by_base[base] = filename

        canonical = sorted(by_base.values())
        stats = {
            "raw_candidates": len(candidates),
            "canonical_files": len(canonical),
            "resolved_duplicates": len(duplicate_bases),
        }
        return canonical, stats

    def _run_wearable_matching(self):
        """Match data files to scene videos using the selected strategy."""
        if not self._w_data_dir or not self._w_video_dir:
            self._update_onboarding_state()
            return

        # Remove cell widgets (QComboBox) before clearing to prevent memory leak
        for row in range(self.w_table.rowCount()):
            if self.w_table.columnCount() > 1:
                self.w_table.removeCellWidget(row, 1)
        self.w_table.setRowCount(0)
        self._w_matches = []

        strategy = self.cmb_match_strategy.currentText()
        n_chars = self.spn_n_chars.value()

        # Collect data files
        data_files, _ = self._list_wearable_data_files()
        # Collect video files
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        self._w_video_files = sorted(
            f for f in os.listdir(self._w_video_dir)
            if f.lower().endswith(video_exts)
        )
        video_bases = {os.path.splitext(f)[0]: f for f in self._w_video_files}

        matched_count = 0
        missing_count = 0

        for data_file in data_files:
            data_base = os.path.splitext(data_file)[0]
            video_file = None

            if strategy == "Exact basename":
                video_file = video_bases.get(data_base)

            elif strategy == "First N characters":
                data_prefix = data_base[:n_chars].lower()
                for vbase, vfile in video_bases.items():
                    if vbase[:n_chars].lower() == data_prefix:
                        video_file = vfile
                        break

            elif strategy == "Contains":
                data_lower = data_base.lower()
                for vbase, vfile in video_bases.items():
                    vbase_lower = vbase.lower()
                    if vbase_lower in data_lower or data_lower in vbase_lower:
                        video_file = vfile
                        break

            # For Manual mode and auto strategies, populate the row
            row = self.w_table.rowCount()
            self.w_table.insertRow(row)
            self.w_table.setItem(row, 0, QTableWidgetItem(data_file))

            if strategy == "Manual":
                # Combo box with all videos + "(none)" option
                cmb = QComboBox()
                cmb.setMinimumWidth(200)
                cmb.setToolTip("Select the matching scene video for this data file")
                cmb.addItem("(none)")
                cmb.addItems(self._w_video_files)
                # Pre-select exact match if found
                exact = video_bases.get(data_base)
                if exact and exact in self._w_video_files:
                    cmb.setCurrentText(exact)
                cmb.currentTextChanged.connect(self._on_manual_combo_changed)
                self.w_table.setCellWidget(row, 1, cmb)
                status_item = QTableWidgetItem(f"{status_icon_info} Manual")
                status_item.setForeground(QColor(THEME["status_info"]))
                self.w_table.setItem(row, 2, status_item)
                self._w_matches.append((data_file, None))  # resolved at action time
            elif video_file:
                self.w_table.setItem(row, 1, QTableWidgetItem(video_file))
                status_item = QTableWidgetItem(f"{status_icon_success} Matched")
                status_item.setForeground(QColor(THEME["status_success"]))
                self.w_table.setItem(row, 2, status_item)
                matched_count += 1
                self._w_matches.append((data_file, video_file))
            else:
                self.w_table.setItem(row, 1, QTableWidgetItem("—"))
                status_item = QTableWidgetItem(f"{status_icon_error} No video")
                status_item.setForeground(QColor(THEME["status_error"]))
                self.w_table.setItem(row, 2, status_item)
                missing_count += 1
                self._w_matches.append((data_file, None))

        total = len(data_files)
        if strategy == "Manual":
            self.lbl_w_summary.setText(f"Total: {total} files  |  {status_icon_info} Manual matching")
        else:
            self.lbl_w_summary.setText(
                f"Total: {total} files  |  {status_icon_success} Matched: {matched_count}  |  {status_icon_error} Unmatched: {missing_count}"
            )
        self._update_onboarding_state()

    def _update_onboarding_state(self):
        tab = self.mode_tabs.currentIndex() if hasattr(self, "mode_tabs") else 0
        if tab == 0:
            if not self.current_project_path:
                self.lbl_notice.setText("Open a project in Project Manager to begin matching.")
                self.lbl_notice.show()
                return
            if not self.gaze_data_path:
                self.lbl_notice.setText(
                    "Screen-based workflow: select a gaze data file and processed stimuli directory, then click 'Refresh & Match'."
                )
                self.lbl_notice.show()
                return
            if not self.stimuli_dir:
                self.lbl_notice.setText(
                    "Select the processed stimuli directory to match each stimulusName to landmark files."
                )
                self.lbl_notice.show()
                return
            if self.table.rowCount() == 0:
                self.lbl_notice.setText("Click 'Refresh & Match' to generate the match table.")
                self.lbl_notice.show()
                return
            self.lbl_notice.hide()
            return

        # Wearable tab
        ready_dirs = bool(self._w_data_dir and self._w_video_dir)
        has_rows = self.w_table.rowCount() > 0
        # In Manual mode, check combo boxes directly for selections
        if self.cmb_match_strategy.currentText() == "Manual":
            has_match = self._any_manual_video_selected()
        else:
            has_match = any(v is not None for _, v in self._w_matches)
        self.btn_w_set_stim.setEnabled(ready_dirs and has_rows and has_match)
        self.btn_w_merge.setEnabled(bool(self._w_data_dir) and has_rows)
        if not self.current_project_path:
            self.lbl_notice.setText("Open a project in Project Manager to begin matching.")
            self.lbl_notice.show()
        elif not ready_dirs:
            self.lbl_notice.setText(
                "Wearable workflow: select both data and scene-video folders, then run 'Refresh & Match'."
            )
            self.lbl_notice.show()
        elif not has_rows:
            self.lbl_notice.setText("No files detected yet. Check folder contents and click 'Refresh & Match'.")
            self.lbl_notice.show()
        elif not has_match:
            self.lbl_notice.setText(
                "No matches found yet. Adjust matching method or assign manually before setting stimulusName."
            )
            self.lbl_notice.show()
        else:
            self.lbl_notice.hide()

    def _any_manual_video_selected(self):
        """Return True if any combo box in the table has a video selected."""
        for row in range(self.w_table.rowCount()):
            cmb = self.w_table.cellWidget(row, 1)
            if cmb is not None and cmb.currentText() != "(none)":
                return True
        return False

    def _on_manual_combo_changed(self):
        """Re-evaluate button state when a manual combo box selection changes."""
        self._update_onboarding_state()

    def _resolve_wearable_matches(self):
        """Build the final match list, reading combo boxes in Manual mode."""
        resolved = []
        for row in range(self.w_table.rowCount()):
            data_item = self.w_table.item(row, 0)
            if not data_item:
                continue
            data_file = data_item.text()
            cmb = self.w_table.cellWidget(row, 1)
            if cmb is not None:  # Manual mode — read combo box
                selected = cmb.currentText()
                video = selected if selected != "(none)" else None
            else:
                video = self._w_matches[row][1] if row < len(self._w_matches) else None
            resolved.append((data_file, video))
        return resolved

    def _set_stimulus_name_wearable(self):
        """Set the stimulusName column in each data file to its matched video basename."""
        resolved = self._resolve_wearable_matches()
        matched_pairs = [(d, v) for d, v in resolved if v is not None]
        if not matched_pairs:
            show_warning(self, "No Matches Found", "No matched data-video pairs were found. Run the matching process first to pair data files with scene videos.")
            return

        self._overlay.set_status("Updating Stimulus Names")
        self._overlay.hide_progress()
        self._overlay.show()

        count = 0
        errors = []
        for data_file, video_file in matched_pairs:
            data_path = os.path.join(self._w_data_dir, data_file)
            video_base = os.path.splitext(video_file)[0]
            try:
                if data_file.lower().endswith('.parquet'):
                    df = pd.read_parquet(data_path)
                else:
                    df = pd.read_csv(data_path)
                df['stimulusName'] = video_base
                # Always save as parquet
                out_path = data_path
                if not out_path.lower().endswith('.parquet'):
                    out_path = os.path.splitext(out_path)[0] + '.parquet'
                atomic_save_parquet(df, out_path)
                count += 1
            except Exception as e:
                errors.append(f"{data_file}: {e}")

        msg = f"Updated stimulusName in {count} file(s)."
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
        self._overlay.hide()
        self.lbl_w_status.setText(msg)

    def _merge_wearable_files(self):
        """Merge all data files into a single parquet file with Participant_ID column."""
        if not self._w_data_dir:
            show_warning(self, "No Folder Selected", "Please select a data files folder before merging.")
            return

        data_files, file_stats = self._list_wearable_data_files()
        if not data_files:
            show_warning(self, "No Files Found", "No wearable data files were found in the selected folder.")
            return

        self._overlay.set_status("Merging Wearable Files")
        self._overlay.hide_progress()
        self._overlay.show()

        frames = []
        errors = []
        for data_file in data_files:
            data_path = os.path.join(self._w_data_dir, data_file)
            participant_id = os.path.splitext(data_file)[0]
            try:
                if data_file.lower().endswith('.parquet'):
                    df = pd.read_parquet(data_path)
                else:
                    df = pd.read_csv(data_path)
                df['Participant_ID'] = participant_id
                frames.append(df)
            except Exception as e:
                errors.append(f"{data_file}: {e}")

        if not frames:
            show_warning(self, "Read Error", "Could not read any of the data files. Check the file formats and try again.")
            return

        merged = pd.concat(frames, ignore_index=True)

        # Save to imported_parquet folder
        out_dir = os.path.join(self.current_project_path, "imported_parquet")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "merged.parquet")
        atomic_save_parquet(merged, out_path)

        msg = f"Merged {len(frames)} file(s) → {os.path.basename(out_path)} ({len(merged)} rows)"
        if file_stats["resolved_duplicates"] > 0:
            msg += (
                f"\nAuto-resolved duplicate sources: {file_stats['resolved_duplicates']} participant(s) "
                f"had both CSV and Parquet; used Parquet only."
            )
        if errors:
            msg += f"\n\nSkipped ({len(errors)}):\n" + "\n".join(errors[:5])
        self._overlay.hide()
        self.lbl_w_status.setText(msg)

        # Set as global selected file
        self._set_global_selected_file(out_path)
