from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QHBoxLayout, QGroupBox, QComboBox, QLineEdit,
                             QRadioButton, QButtonGroup, QFileDialog,
                             QApplication, QListWidget,
                             QAbstractItemView, QGridLayout, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIntValidator, QDoubleValidator
from .processing_utils import ProcessingButton, ProcessingWorker, ProcessingOverlay
from .button_styles import button_style
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm
from .theme import (status_success, status_error, status_info,
                     status_icon_success, status_icon_error,
                     status_icon_warning, status_icon_info)
import os
import logging
import numpy as np
import pandas as pd
from core.io_utils import atomic_save_parquet
from core.temporal_prep import apply_keyword_filter, sort_data, apply_markers
from core.data_resolver import is_imported_data_file

logger = logging.getLogger('iTemplate2')


class TemporalPreprocessing(QWidget):
    data_changed = pyqtSignal()  # Signal for when data is modified (e.g. sorted/markers applied)

    def __init__(self):
        super().__init__()
        
        self.current_project_path = ""
        self.gaze_data_path = ""
        self.columns = []  # Column names from the loaded data
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        lbl_title = QLabel("Step 2.5a: Temporal Preprocessing")
        lbl_title.setObjectName("title-label")
        layout.addWidget(lbl_title)

        # === Permanent info banner ===
        lbl_info = QLabel(
            "This step is primarily important for <b>video stimuli</b>, "
            "as it assigns specific frame numbers to each gaze sample. "
            "For studies using <b>images as stimuli</b>, this step can be skipped — "
            "the transformation can simply rely on stimulusName to match gaze data "
            "to the correct landmark file."
        )
        lbl_info.setObjectName("info-banner")
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)
        # === Top-of-page notice banner ===
        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.hide()
        layout.addWidget(self.lbl_notice)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)
        
        # === Section 1: Data File ===
        data_group = QGroupBox("Gaze Data File")
        data_layout = QHBoxLayout()
        
        self.btn_select_data = QPushButton("Select Parquet File")
        self.btn_select_data.clicked.connect(self.select_data_file)
        data_layout.addWidget(self.btn_select_data)
        
        self.lbl_data_path = QLabel("No file selected")
        data_layout.addWidget(self.lbl_data_path, 1)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # === Section 2: Temporal Column Setup ===
        temporal_group = QGroupBox("Temporal Column Setup")
        temporal_layout = QVBoxLayout()
        
        temporal_layout.addWidget(QLabel(
            "Select which column(s) represent temporal order in your data.\n"
            "For gaze data: use TIMESTAMP (in ms).\n"
            "For fixation data: use Trial_Index + Fixation_Index."
        ))
        
        temp_grid = QGridLayout()
        
        temp_grid.addWidget(QLabel("Primary temporal column:"), 0, 0)
        self.combo_temporal_primary = QComboBox()
        self.combo_temporal_primary.setMinimumWidth(200)
        self.combo_temporal_primary.setToolTip("Column containing timestamp or trial data")
        self.combo_temporal_primary.addItem("-- Select Column --")
        temp_grid.addWidget(self.combo_temporal_primary, 0, 1)
        
        temp_grid.addWidget(QLabel("Secondary temporal column (optional):"), 1, 0)
        self.combo_temporal_secondary = QComboBox()
        self.combo_temporal_secondary.setMinimumWidth(200)
        self.combo_temporal_secondary.setToolTip("Optional secondary temporal column")
        self.combo_temporal_secondary.addItem("-- None --")
        temp_grid.addWidget(self.combo_temporal_secondary, 1, 1)
        
        temporal_layout.addLayout(temp_grid)
        
        self.btn_sort = ProcessingButton("Sort Data by Participant_ID + Temporal Order")
        self.btn_sort.clicked.connect(self.sort_data)
        self.btn_sort.setStyleSheet(button_style("primary", padding="8px"))
        temporal_layout.addWidget(self.btn_sort)
        
        self.lbl_sort_status = QLabel("")
        temporal_layout.addWidget(self.lbl_sort_status)
        
        temporal_group.setLayout(temporal_layout)
        layout.addWidget(temporal_group)

        # === Tab widget for approach selection (no outer QGroupBox) ===
        self.approach_tabs = QTabWidget()
        self.approach_tabs.tabBar().setExpanding(True)
        self.approach_tabs.setStyleSheet("")

        # ── Tab 0: Onset-Offset Approach ─────────────────────────────
        onset_offset_tab = QWidget()
        onset_offset_tab.setStyleSheet("")
        oo_layout = QVBoxLayout()

        # === Section 3: Video Onset ===
        onset_group = QGroupBox("Video Onset")
        onset_layout = QVBoxLayout()

        lbl_onset_desc = QLabel(
            "Define <b>when each video begins</b>. Select the column in your gaze data that "
            "contains event markers (e.g., MESSAGE), then pick the specific marker value(s) that "
            "signal the start of video presentation. Each matching row will be tagged as "
            "<code>STIM_ONSET</code> in the output. If there is a delay between the marker and the "
            "actual video playback, use the onset offset to shift the onset forward."
        )
        lbl_onset_desc.setWordWrap(True)
        lbl_onset_desc.setObjectName("description-label")
        onset_layout.addWidget(lbl_onset_desc)
        
        onset_col_layout = QHBoxLayout()
        onset_col_layout.addWidget(QLabel("Marker column:"))
        self.combo_onset_col = QComboBox()
        self.combo_onset_col.setMinimumWidth(200)
        self.combo_onset_col.setToolTip("Column containing trial onset marker values")
        self.combo_onset_col.addItem("-- Select Column --")
        self.combo_onset_col.currentTextChanged.connect(self.on_onset_col_changed)
        onset_col_layout.addWidget(self.combo_onset_col, 1)
        onset_layout.addLayout(onset_col_layout)
        
        onset_layout.addWidget(QLabel("Select onset marker value(s):"))
        self.list_onset_markers = QListWidget()
        self.list_onset_markers.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_onset_markers.setMaximumHeight(96)
        onset_layout.addWidget(self.list_onset_markers)

        onset_offset_layout = QHBoxLayout()
        onset_offset_layout.addWidget(QLabel("Onset offset (ms):"))
        self.txt_onset_offset = QLineEdit()
        self.txt_onset_offset.setPlaceholderText("0")
        self.txt_onset_offset.setToolTip(
            "Milliseconds after the onset marker when video playback actually begins.\n"
            "The STIM_ONSET timestamp will be shifted forward by this amount."
        )
        self.txt_onset_offset.setValidator(QIntValidator(-9999, 9999))
        self.txt_onset_offset.setMaximumWidth(120)
        onset_offset_layout.addWidget(self.txt_onset_offset)
        lbl_onset_offset_desc = QLabel(
            "Delay between the onset marker and the start of video playback"
        )
        lbl_onset_offset_desc.setObjectName("hint-label")
        onset_offset_layout.addWidget(lbl_onset_offset_desc, 1)
        onset_layout.addLayout(onset_offset_layout)

        onset_group.setLayout(onset_layout)
        oo_layout.addWidget(onset_group)
        
        # === Section 4: Video Offset ===
        offset_group = QGroupBox("Video Offset")
        offset_layout = QVBoxLayout()

        lbl_offset_desc = QLabel(
            "Define <b>when each video ends</b>. You have two options:<br>"
            "• <b>Use marker</b> — select a column and marker value(s) that signal the end of "
            "video duration (e.g., a STIM_OFF event). Each matching row is tagged as "
            "<code>STIM_OFFSET</code>.<br>"
            "• <b>Use duration from onset</b> — specify a fixed video length (e.g., 5 s). "
            "The offset is placed at the first gaze sample whose timestamp exceeds "
            "onset + duration. This is useful when your data does not contain explicit offset markers."
        )
        lbl_offset_desc.setWordWrap(True)
        lbl_offset_desc.setObjectName("description-label")
        offset_layout.addWidget(lbl_offset_desc)
        
        # Radio toggle
        radio_layout = QHBoxLayout()
        self.radio_offset_marker = QRadioButton("Use marker")
        self.radio_offset_duration = QRadioButton("Use duration from onset")
        self.radio_offset_marker.setChecked(True)
        
        self.offset_mode_group = QButtonGroup()
        self.offset_mode_group.addButton(self.radio_offset_marker)
        self.offset_mode_group.addButton(self.radio_offset_duration)
        self.offset_mode_group.buttonClicked.connect(self.toggle_offset_mode)
        
        radio_layout.addWidget(self.radio_offset_marker)
        radio_layout.addWidget(self.radio_offset_duration)
        offset_layout.addLayout(radio_layout)
        
        # Marker mode widgets
        self.offset_marker_widget = QWidget()
        offset_marker_layout = QVBoxLayout()
        offset_marker_layout.setContentsMargins(0, 0, 0, 0)
        
        offset_col_layout = QHBoxLayout()
        offset_col_layout.addWidget(QLabel("Marker column:"))
        self.combo_offset_col = QComboBox()
        self.combo_offset_col.setMinimumWidth(200)
        self.combo_offset_col.setToolTip("Column containing trial offset marker values")
        self.combo_offset_col.addItem("-- Select Column --")
        self.combo_offset_col.currentTextChanged.connect(self.on_offset_col_changed)
        offset_col_layout.addWidget(self.combo_offset_col, 1)
        offset_marker_layout.addLayout(offset_col_layout)
        
        offset_marker_layout.addWidget(QLabel("Select offset marker value(s):"))
        self.list_offset_markers = QListWidget()
        self.list_offset_markers.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_offset_markers.setMaximumHeight(96)
        offset_marker_layout.addWidget(self.list_offset_markers)
        
        self.offset_marker_widget.setLayout(offset_marker_layout)
        offset_layout.addWidget(self.offset_marker_widget)
        
        # Duration mode widgets
        self.offset_duration_widget = QWidget()
        duration_outer = QVBoxLayout()
        duration_outer.setContentsMargins(0, 0, 0, 0)

        duration_inner = QHBoxLayout()
        duration_inner.addWidget(QLabel("Duration:"))
        self.txt_duration = QLineEdit()
        self.txt_duration.setPlaceholderText("e.g. 5.0")
        self.txt_duration.setToolTip(
            "Video duration. Enter a positive number (e.g., 5.0 for 5 seconds, 5000 for 5000 ms).\n"
            "Use the unit dropdown to specify whether the value is in seconds or milliseconds."
        )
        self.txt_duration.setValidator(QDoubleValidator(0.1, 99999, 3))
        duration_inner.addWidget(self.txt_duration)

        self.combo_duration_unit = QComboBox()
        self.combo_duration_unit.setMinimumWidth(180)
        self.combo_duration_unit.setToolTip("Unit for the trial duration value")
        self.combo_duration_unit.addItems(["seconds (s)", "milliseconds (ms)"])
        self.combo_duration_unit.setObjectName("warning-combo")
        duration_inner.addWidget(self.combo_duration_unit)

        duration_outer.addLayout(duration_inner)

        self.lbl_unit_warning = QLabel(f"{status_icon_warning} Make sure the unit matches your TIMESTAMP column")
        self.lbl_unit_warning.setObjectName("status-warning-label")
        duration_outer.addWidget(self.lbl_unit_warning)

        self.offset_duration_widget.setLayout(duration_outer)
        self.offset_duration_widget.hide()
        offset_layout.addWidget(self.offset_duration_widget)
        
        offset_group.setLayout(offset_layout)
        oo_layout.addWidget(offset_group)

        # === Apply Button (onset-offset) ===
        self.btn_apply = ProcessingButton("Apply Video Markers")
        self.btn_apply.setStyleSheet(button_style("primary", padding="10px"))
        self.btn_apply.clicked.connect(self.apply_markers)
        oo_layout.addWidget(self.btn_apply)
        
        self.lbl_status = QLabel("")
        oo_layout.addWidget(self.lbl_status)

        oo_layout.addStretch()
        onset_offset_tab.setLayout(oo_layout)
        self.approach_tabs.addTab(onset_offset_tab, "Onset-Offset Approach")

        # ── Tab 1: Keyword Approach ──────────────────────────────────
        keyword_tab = QWidget()
        keyword_tab.setStyleSheet("")
        keyword_layout = QVBoxLayout()

        lbl_kw_desc = QLabel(
            "Filter rows by <b>column values</b>. Pick one or more keywords to "
            "<b>include</b> and/or <b>exclude</b> (each can use a different column). "
            "Matching rows will be tagged as <code>STIM_ONSET</code> at the start of "
            "each contiguous block, <b>per stimulusName</b>."
        )
        lbl_kw_desc.setWordWrap(True)
        lbl_kw_desc.setObjectName("description-label")
        keyword_layout.addWidget(lbl_kw_desc)

        # --- Include section (equals) ---
        inc_col_layout = QHBoxLayout()
        inc_col_layout.addWidget(QLabel("Include column:"))
        self.cmb_kw_inc_col = QComboBox()
        self.cmb_kw_inc_col.setMinimumWidth(200)
        self.cmb_kw_inc_col.setEditable(True)
        self.cmb_kw_inc_col.addItem("-- None --")
        self.cmb_kw_inc_col.setToolTip(
            "Column to match for INCLUSION. Leave at '-- None --' to skip."
        )
        self.cmb_kw_inc_col.currentTextChanged.connect(
            lambda col: self._populate_kw_list(col, self.list_kw_include)
        )
        inc_col_layout.addWidget(self.cmb_kw_inc_col, 1)
        keyword_layout.addLayout(inc_col_layout)

        keyword_layout.addWidget(QLabel("Include — rows where column EQUALS these values:"))
        self.list_kw_include = QListWidget()
        self.list_kw_include.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_kw_include.setMaximumHeight(96)
        self.list_kw_include.setToolTip(
            "Select one or more values. Only rows matching these values will be kept.\n"
            "Select <empty> to match rows where the cell is blank / NaN."
        )
        keyword_layout.addWidget(self.list_kw_include)

        # --- Exclude section (NOT equals) ---
        exc_col_layout = QHBoxLayout()
        lbl_not = QLabel(
            f'<span style="color: {status_error}; font-weight:bold; font-size:13px;">{status_icon_error} NOT</span>'
            ' — Exclude column:'
        )
        exc_col_layout.addWidget(lbl_not)
        self.cmb_kw_exc_col = QComboBox()
        self.cmb_kw_exc_col.setMinimumWidth(200)
        self.cmb_kw_exc_col.setEditable(True)
        self.cmb_kw_exc_col.addItem("-- None --")
        self.cmb_kw_exc_col.setToolTip(
            "Column to match for EXCLUSION. Leave at '-- None --' to skip."
        )
        self.cmb_kw_exc_col.currentTextChanged.connect(
            lambda col: self._populate_kw_list(col, self.list_kw_exclude)
        )
        exc_col_layout.addWidget(self.cmb_kw_exc_col, 1)
        keyword_layout.addLayout(exc_col_layout)

        keyword_layout.addWidget(QLabel("Exclude — rows where column equals these values:"))
        self.list_kw_exclude = QListWidget()
        self.list_kw_exclude.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_kw_exclude.setMaximumHeight(96)
        self.list_kw_exclude.setToolTip(
            "Select one or more values. Rows matching these values will be REMOVED.\n"
            "Select <empty> to match rows where the cell is blank / NaN."
        )
        self.list_kw_exclude.setObjectName("exclude-list")
        keyword_layout.addWidget(self.list_kw_exclude)

        self.btn_apply_keyword = ProcessingButton("Apply Keyword Filter")
        self.btn_apply_keyword.setStyleSheet(button_style("primary", padding="10px"))
        self.btn_apply_keyword.clicked.connect(self._apply_keyword_filter)
        keyword_layout.addWidget(self.btn_apply_keyword)

        self.lbl_kw_status = QLabel("")
        keyword_layout.addWidget(self.lbl_kw_status)

        keyword_layout.addStretch()
        keyword_tab.setLayout(keyword_layout)
        self.approach_tabs.addTab(keyword_tab, "Keyword Approach")

        layout.addWidget(self.approach_tabs)
        
        layout.addStretch()
        
        self._project_settings = None
        self._syncing_selection = False
        self._overlay = ProcessingOverlay(self)

        # Connect signals for auto-save
        self.combo_temporal_primary.currentIndexChanged.connect(self._on_settings_edited)
        self.combo_temporal_secondary.currentIndexChanged.connect(self._on_settings_edited)
        self.combo_onset_col.currentIndexChanged.connect(self._on_settings_edited)
        self.list_onset_markers.itemSelectionChanged.connect(self._on_settings_edited)
        self.offset_mode_group.buttonToggled.connect(self._on_settings_edited)
        self.combo_offset_col.currentIndexChanged.connect(self._on_settings_edited)
        self.list_offset_markers.itemSelectionChanged.connect(self._on_settings_edited)
        self.txt_duration.editingFinished.connect(self._on_settings_edited)
        self.combo_duration_unit.currentIndexChanged.connect(self._on_settings_edited)
        self.txt_onset_offset.editingFinished.connect(self._on_settings_edited)
        self.approach_tabs.currentChanged.connect(self._on_settings_edited)
        self.cmb_kw_inc_col.currentTextChanged.connect(self._on_settings_edited)
        self.cmb_kw_exc_col.currentTextChanged.connect(self._on_settings_edited)
        self.list_kw_include.itemSelectionChanged.connect(self._on_settings_edited)
        self.list_kw_exclude.itemSelectionChanged.connect(self._on_settings_edited)
    
    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings

        selected = settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != self.gaze_data_path:
            self._syncing_selection = True
            try:
                self._load_data_file(selected)
            finally:
                self._syncing_selection = False
        
        # Block signals to prevent auto-save during loading
        widgets_to_block = [
            self, self.list_onset_markers, self.list_offset_markers,
            self.combo_temporal_primary, self.combo_temporal_secondary,
            self.combo_onset_col, self.combo_offset_col,
            self.combo_duration_unit, self.txt_duration, self.txt_onset_offset,
            self.offset_mode_group, self.approach_tabs,
            self.cmb_kw_inc_col, self.cmb_kw_exc_col,
            self.list_kw_include, self.list_kw_exclude,
        ]
        for w in widgets_to_block:
            w.blockSignals(True)
        
        try:
            # Temporal column selections (only set if columns are populated)
            primary = settings.get("temporal_preprocessing", "primary_temporal_col", "")
            if primary and self.combo_temporal_primary.findText(primary) >= 0:
                self.combo_temporal_primary.setCurrentText(primary)
            secondary = settings.get("temporal_preprocessing", "secondary_temporal_col", "")
            if secondary and self.combo_temporal_secondary.findText(secondary) >= 0:
                self.combo_temporal_secondary.setCurrentText(secondary)

            # Onset
            onset_col = settings.get("temporal_preprocessing", "onset_marker_col", "")
            if onset_col and self.combo_onset_col.findText(onset_col) >= 0:
                self.combo_onset_col.setCurrentText(onset_col)
                # Manually populate onset markers since signal is blocked
                self._populate_marker_list(self.combo_onset_col, self.list_onset_markers)
            onset_markers = settings.get("temporal_preprocessing", "onset_markers", [])
            if onset_markers:
                for i in range(self.list_onset_markers.count()):
                    item = self.list_onset_markers.item(i)
                    item.setSelected(item.text() in onset_markers)

            onset_offset = settings.get("temporal_preprocessing", "onset_offset_ms", "")
            if onset_offset:
                self.txt_onset_offset.setText(str(onset_offset))

            # Offset mode
            mode = settings.get("temporal_preprocessing", "offset_mode", "marker")
            if mode == "duration":
                self.radio_offset_duration.setChecked(True)
                self.toggle_offset_mode(self.radio_offset_duration)
            else:
                self.radio_offset_marker.setChecked(True)
                self.toggle_offset_mode(self.radio_offset_marker)

            offset_col = settings.get("temporal_preprocessing", "offset_marker_col", "")
            if offset_col and self.combo_offset_col.findText(offset_col) >= 0:
                self.combo_offset_col.setCurrentText(offset_col)
                # Manually populate offset markers since signal is blocked
                self._populate_marker_list(self.combo_offset_col, self.list_offset_markers)
            off_markers = settings.get("temporal_preprocessing", "offset_markers", [])
            if off_markers:
                for i in range(self.list_offset_markers.count()):
                    item = self.list_offset_markers.item(i)
                    item.setSelected(item.text() in off_markers)

            duration = settings.get("temporal_preprocessing", "offset_duration", "")
            if duration:
                self.txt_duration.setText(str(duration))
            unit = settings.get("temporal_preprocessing", "offset_duration_unit", "seconds (s)")
            idx = self.combo_duration_unit.findText(unit)
            if idx >= 0:
                self.combo_duration_unit.setCurrentIndex(idx)

            # Approach tab
            approach = settings.get("temporal_preprocessing", "approach", "onset_offset")
            self.approach_tabs.setCurrentIndex(0 if approach == "onset_offset" else 1)

            # Keyword approach
            kw_inc_col = settings.get("temporal_preprocessing", "kw_include_col", "")
            if kw_inc_col:
                self.cmb_kw_inc_col.setCurrentText(str(kw_inc_col))
                self._populate_kw_list(kw_inc_col, self.list_kw_include)
            kw_include = settings.get("temporal_preprocessing", "kw_include_values", [])
            if kw_include:
                for i in range(self.list_kw_include.count()):
                    item = self.list_kw_include.item(i)
                    item.setSelected(item.text() in kw_include)
            kw_exc_col = settings.get("temporal_preprocessing", "kw_exclude_col", "")
            if kw_exc_col:
                self.cmb_kw_exc_col.setCurrentText(str(kw_exc_col))
                self._populate_kw_list(kw_exc_col, self.list_kw_exclude)
            kw_exclude = settings.get("temporal_preprocessing", "kw_exclude_values", [])
            if kw_exclude:
                for i in range(self.list_kw_exclude.count()):
                    item = self.list_kw_exclude.item(i)
                    item.setSelected(item.text() in kw_exclude)
        finally:
            for w in widgets_to_block:
                w.blockSignals(False)

    def _on_settings_edited(self, *args):
        """Auto-save settings when UI changes."""
        if self._project_settings:
            # Reusing save_settings logic which writes to ProjectSettings
            # ProjectSettings.set() now auto-saves to disk
            self.save_settings(self._project_settings)

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        primary = self.combo_temporal_primary.currentText()
        settings.set("temporal_preprocessing", "primary_temporal_col",
                     "" if primary.startswith("--") else primary)
        secondary = self.combo_temporal_secondary.currentText()
        settings.set("temporal_preprocessing", "secondary_temporal_col",
                     "" if secondary.startswith("--") else secondary)

        onset_col = self.combo_onset_col.currentText()
        settings.set("temporal_preprocessing", "onset_marker_col",
                     "" if onset_col.startswith("--") else onset_col)
        settings.set("temporal_preprocessing", "onset_markers",
                     [it.text() for it in self.list_onset_markers.selectedItems()])
        settings.set("temporal_preprocessing", "onset_offset_ms",
                     self.txt_onset_offset.text().strip())

        settings.set("temporal_preprocessing", "offset_mode",
                     "duration" if self.radio_offset_duration.isChecked() else "marker")

        offset_col = self.combo_offset_col.currentText()
        settings.set("temporal_preprocessing", "offset_marker_col",
                     "" if offset_col.startswith("--") else offset_col)
        settings.set("temporal_preprocessing", "offset_markers",
                     [it.text() for it in self.list_offset_markers.selectedItems()])

        settings.set("temporal_preprocessing", "offset_duration",
                     self.txt_duration.text().strip())
        settings.set("temporal_preprocessing", "offset_duration_unit",
                     self.combo_duration_unit.currentText())

        # Approach tab
        settings.set("temporal_preprocessing", "approach",
                     "onset_offset" if self.approach_tabs.currentIndex() == 0 else "keyword")

        # Keyword approach
        inc_col = self.cmb_kw_inc_col.currentText().strip()
        settings.set("temporal_preprocessing", "kw_include_col",
                     "" if inc_col.startswith("--") else inc_col)
        settings.set("temporal_preprocessing", "kw_include_values",
                     [it.text() for it in self.list_kw_include.selectedItems()])
        exc_col = self.cmb_kw_exc_col.currentText().strip()
        settings.set("temporal_preprocessing", "kw_exclude_col",
                     "" if exc_col.startswith("--") else exc_col)
        settings.set("temporal_preprocessing", "kw_exclude_values",
                     [it.text() for it in self.list_kw_exclude.selectedItems()])

    def set_project_path(self, path):
        self.current_project_path = path
        if self._project_settings:
            selected = self._project_settings.resolve_working_data_file("")
            if selected and os.path.isfile(selected):
                self._syncing_selection = True
                try:
                    self._load_data_file(selected)
                finally:
                    self._syncing_selection = False

    def refresh_data(self):
        """Reload the last used project path to refresh file lists."""
        logger.debug("TemporalPreprocessing.refresh_data called. Path=%s", self.current_project_path)
        if self.current_project_path:
            self.set_project_path(self.current_project_path)
    
    def select_data_file(self):
        start = self.current_project_path if self.current_project_path else ""
        f, _ = QFileDialog.getOpenFileName(self, "Select Gaze Data File", start, "Parquet Files (*.parquet)")
        if f:
            self._load_data_file(f)
            self._set_global_selected_file(f)

    def _set_gaze_data_file(self, path, update_settings=False):
        """Lightweight: just update the internal path and label (no file read)."""
        if not path or not os.path.isfile(path):
            return
        self.gaze_data_path = path
        self.lbl_data_path.setText(os.path.basename(path))
        if update_settings:
            self._set_global_selected_file(path)

    def _set_global_selected_file(self, path):
        if not self._project_settings or self._syncing_selection:
            return
        if self.current_project_path and is_imported_data_file(self.current_project_path, path):
            self._project_settings.set("global", "source_data_file", path)
        self._project_settings.set("global", "selected_data_file", path)
    
    def _load_data_file(self, path):
        """Load data file and populate all column dropdowns."""
        self.gaze_data_path = path
        self.lbl_data_path.setText(os.path.basename(path))
        
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            self.columns = list(dict.fromkeys(table.column_names))  # unique, preserving order
            
            # Save current selections before clearing
            prev_primary = self.combo_temporal_primary.currentText()
            prev_secondary = self.combo_temporal_secondary.currentText()
            prev_onset_col = self.combo_onset_col.currentText()
            prev_offset_col = self.combo_offset_col.currentText()
            prev_onset_markers = [item.text() for item in self.list_onset_markers.selectedItems()]
            prev_offset_markers = [item.text() for item in self.list_offset_markers.selectedItems()]
            has_previous = not prev_primary.startswith("--") or not prev_onset_col.startswith("--")
            
            # Block signals during entire repopulation to prevent _on_settings_edited
            # from saving intermediate (empty/reset) values to project_settings.json
            widgets_to_block = [
                self.combo_temporal_primary, self.combo_temporal_secondary,
                self.combo_onset_col, self.combo_offset_col,
                self.list_onset_markers, self.list_offset_markers,
                self.cmb_kw_inc_col, self.cmb_kw_exc_col,
                self.list_kw_include, self.list_kw_exclude,
            ]
            for w in widgets_to_block:
                w.blockSignals(True)
            
            try:
                # Clear and repopulate all combo boxes
                for combo in [self.combo_temporal_primary, self.combo_temporal_secondary,
                              self.combo_onset_col, self.combo_offset_col]:
                    combo.clear()
                
                self.combo_temporal_primary.addItem("-- Select Column --")
                self.combo_temporal_secondary.addItem("-- None --")
                self.combo_onset_col.addItem("-- Select Column --")
                self.combo_offset_col.addItem("-- Select Column --")
                
                for col in self.columns:
                    self.combo_temporal_primary.addItem(col)
                    self.combo_temporal_secondary.addItem(col)
                    self.combo_onset_col.addItem(col)
                    self.combo_offset_col.addItem(col)
                
                if has_previous:
                    # Restore previous selections if they still exist in the new columns
                    if not prev_primary.startswith("--") and self.combo_temporal_primary.findText(prev_primary) >= 0:
                        self.combo_temporal_primary.setCurrentText(prev_primary)
                    if not prev_secondary.startswith("--") and self.combo_temporal_secondary.findText(prev_secondary) >= 0:
                        self.combo_temporal_secondary.setCurrentText(prev_secondary)
                    if not prev_onset_col.startswith("--") and self.combo_onset_col.findText(prev_onset_col) >= 0:
                        self.combo_onset_col.setCurrentText(prev_onset_col)
                        self._populate_marker_list(self.combo_onset_col, self.list_onset_markers)
                        # Re-select previously selected onset markers
                        for i in range(self.list_onset_markers.count()):
                            if i % 10 == 0: QApplication.processEvents()
                            item = self.list_onset_markers.item(i)
                            item.setSelected(item.text() in prev_onset_markers)
                    if not prev_offset_col.startswith("--") and self.combo_offset_col.findText(prev_offset_col) >= 0:
                        self.combo_offset_col.setCurrentText(prev_offset_col)
                        self._populate_marker_list(self.combo_offset_col, self.list_offset_markers)
                        # Re-select previously selected offset markers
                        for i in range(self.list_offset_markers.count()):
                            if i % 10 == 0: QApplication.processEvents()
                            item = self.list_offset_markers.item(i)
                            item.setSelected(item.text() in prev_offset_markers)
                else:
                    # Auto-select likely temporal columns (first load only)
                    for col in self.columns:
                        cl = col.lower()
                        if cl == 'timestamp':
                            self.combo_temporal_primary.setCurrentText(col)
                        elif cl == 'trial_index':
                            if self.combo_temporal_primary.currentText().lower() == 'timestamp':
                                self.combo_temporal_secondary.setCurrentText(col)
                            else:
                                self.combo_temporal_primary.setCurrentText(col)
                        elif cl == 'fixation_index':
                            self.combo_temporal_secondary.setCurrentText(col)
                    
                    for col in self.columns:
                        cl = col.lower()
                        if 'message' in cl or 'marker' in cl or 'event' in cl:
                            self.combo_onset_col.setCurrentText(col)
                            self.combo_offset_col.setCurrentText(col)
                            break

                # Also populate keyword approach combos
                old_kw_inc_col = self.cmb_kw_inc_col.currentText()
                old_kw_exc_col = self.cmb_kw_exc_col.currentText()
                old_kw_include = [self.list_kw_include.item(i).text()
                                  for i in range(self.list_kw_include.count())
                                  if self.list_kw_include.item(i).isSelected()]
                old_kw_exclude = [self.list_kw_exclude.item(i).text()
                                  for i in range(self.list_kw_exclude.count())
                                  if self.list_kw_exclude.item(i).isSelected()]
                for cmb in [self.cmb_kw_inc_col, self.cmb_kw_exc_col]:
                    cmb.clear()
                    cmb.addItem("-- None --")
                    cmb.addItems(self.columns)
                if old_kw_inc_col and old_kw_inc_col in self.columns:
                    self.cmb_kw_inc_col.setCurrentText(old_kw_inc_col)
                    self._populate_kw_list(old_kw_inc_col, self.list_kw_include)
                    for i in range(self.list_kw_include.count()):
                        it = self.list_kw_include.item(i)
                        it.setSelected(it.text() in old_kw_include)
                if old_kw_exc_col and old_kw_exc_col in self.columns:
                    self.cmb_kw_exc_col.setCurrentText(old_kw_exc_col)
                    self._populate_kw_list(old_kw_exc_col, self.list_kw_exclude)
                    for i in range(self.list_kw_exclude.count()):
                        it = self.list_kw_exclude.item(i)
                        it.setSelected(it.text() in old_kw_exclude)
            finally:
                for w in widgets_to_block:
                    w.blockSignals(False)
            
            # Check if temporal preprocessing was already done
            notices = []
            if '_temporal_order' in self.columns:
                notices.append(f"{status_icon_success} Data has been sorted (_temporal_order column exists).")
            if 'trial_event' in self.columns:
                # Count existing markers
                df_events = pd.read_parquet(path, columns=['trial_event'])
                n_onset = (df_events['trial_event'] == 'STIM_ONSET').sum()
                n_offset = (df_events['trial_event'] == 'STIM_OFFSET').sum()
                notices.append(f"{status_icon_success} Trial markers already applied: {n_onset} onsets, {n_offset} offsets.")
            
            if notices:
                self.lbl_notice.setText(f"{status_icon_info} Previously processed:\n" + "\n".join(notices) +
                                        "\n\nYou can skip this step or re-apply to overwrite.")
                self.lbl_notice.show()
            else:
                self.lbl_notice.hide()
                    
        except Exception as e:
            show_error(self, "File Error", f"Could not read the data file:\n{e}")
    
    def _populate_marker_list(self, combo_col, list_widget):
        """Read unique values from the selected column and populate the list widget."""
        list_widget.clear()
        
        col_name = combo_col.currentText()
        if col_name.startswith("--") or not self.gaze_data_path:
            return
        
        try:
            df = pd.read_parquet(self.gaze_data_path, columns=[col_name])
            unique_vals = df[col_name].dropna().unique()
            # Sort and convert to strings
            str_vals = sorted([str(v) for v in unique_vals])
            list_widget.addItems(str_vals)
        except Exception as e:
            logger.error("Error reading column %s: %s", col_name, e)
    
    def on_onset_col_changed(self, text):
        self._populate_marker_list(self.combo_onset_col, self.list_onset_markers)
    
    def on_offset_col_changed(self, text):
        self._populate_marker_list(self.combo_offset_col, self.list_offset_markers)

    # ------------------------------------------------------------------
    # Keyword Approach helpers
    # ------------------------------------------------------------------
    def _on_kw_col_changed(self, col_name):
        """When the keyword filter column changes, populate include/exclude lists."""
        self._populate_kw_lists(col_name)

    def _populate_kw_list(self, col_name, list_widget):
        """Fill a single QListWidget with unique values from col_name, plus <empty>."""
        list_widget.blockSignals(True)
        list_widget.clear()
        if col_name and not col_name.startswith("--") and self.gaze_data_path and os.path.isfile(self.gaze_data_path):
            try:
                df = pd.read_parquet(self.gaze_data_path, columns=[col_name])
                has_empty = df[col_name].isna().any() or (df[col_name].astype(str).str.strip() == '').any()
                unique_vals = sorted(df[col_name].dropna().astype(str).unique())
                if has_empty:
                    list_widget.addItem('<empty>')
                list_widget.addItems(unique_vals)
            except Exception:
                pass
        list_widget.blockSignals(False)

    def _populate_kw_lists(self, col_name):
        """Fill both include and exclude QListWidgets (legacy helper)."""
        self._populate_kw_list(col_name, self.list_kw_include)
        self._populate_kw_list(col_name, self.list_kw_exclude)

    def _apply_keyword_filter(self):
        """Tag the start of each contiguous block of keyword-matching rows as STIM_ONSET."""
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file first.")
            return

        inc_col = self.cmb_kw_inc_col.currentText().strip()
        exc_col = self.cmb_kw_exc_col.currentText().strip()
        if inc_col.startswith("--"):
            inc_col = ""
        if exc_col.startswith("--"):
            exc_col = ""

        include_kws = [it.text() for it in self.list_kw_include.selectedItems()]
        exclude_kws = [it.text() for it in self.list_kw_exclude.selectedItems()]

        if (not inc_col or not include_kws) and (not exc_col or not exclude_kws):
            show_warning(self, "Missing Filter Config",
                                "Configure at least one include or exclude filter (choose a column and add keywords).")
            return

        self.btn_apply_keyword.start_processing("Applying filter")
        self.lbl_kw_status.setText("Applying keyword filter...")
        self._overlay.set_status("Applying Keyword Filter")
        self._overlay.hide_progress()
        self._overlay.show()

        self._kw_worker = ProcessingWorker(
            self._do_apply_keyword_filter,
            self.gaze_data_path,
            inc_col, include_kws,
            exc_col, exclude_kws
        )
        self.btn_apply_keyword.connect_worker(self._kw_worker)
        self._kw_worker.finished.connect(self._on_kw_filter_done)
        self._kw_worker.error.connect(self._on_kw_filter_error)
        self._kw_worker.start()

    @staticmethod
    def _do_apply_keyword_filter(gaze_path,
                                  inc_col, include_kws,
                                  exc_col, exclude_kws,
                                  progress_callback=None):
        """Tag STIM_ONSET — delegates to core.temporal_prep."""
        return apply_keyword_filter(gaze_path, inc_col, include_kws,
                                    exc_col, exclude_kws, progress_callback)

    def _on_kw_filter_done(self, result):
        self._overlay.hide()
        self.btn_apply_keyword.stop_processing()
        n = result['n_blocks']
        total = result['n_matching_rows']
        parts = []
        if result.get('inc_col') and result.get('include_kws'):
            parts.append(f"{result['inc_col']} IN {result['include_kws']}")
        if result.get('exc_col') and result.get('exclude_kws'):
            parts.append(f"{result['exc_col']} NOT IN {result['exclude_kws']}")
        criteria = ", ".join(parts)
        self.lbl_kw_status.setText(
            f"{status_icon_success} Tagged {n} STIM_ONSET(s) ({total} matching rows) — {criteria}"
        )
        self.lbl_kw_status.setObjectName("status-success-label")
        self.data_changed.emit()

    def _on_kw_filter_error(self, err_msg):
        self._overlay.hide()
        self.btn_apply_keyword.stop_processing()
        show_error(self, "Filter Failed", f"Keyword filter failed:\n{err_msg}")
        self.lbl_kw_status.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_kw_status.setObjectName("status-error-label")

    def toggle_offset_mode(self, btn):
        if self.radio_offset_marker.isChecked():
            self.offset_marker_widget.show()
            self.offset_duration_widget.hide()
            self.lbl_unit_warning.hide()
        else:
            self.offset_marker_widget.hide()
            self.offset_duration_widget.show()
            self.lbl_unit_warning.show()
    
    def sort_data(self):
        """Sort data by Participant_ID + temporal columns and save."""
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file before sorting.")
            return

        primary = self.combo_temporal_primary.currentText()
        if primary.startswith("--"):
            show_warning(self, "Column Not Selected", "Please select a primary temporal column before sorting.")
            return
        
        secondary = self.combo_temporal_secondary.currentText()
        
        self.btn_sort.start_processing("Sorting data")
        self._overlay.set_status("Sorting Data")
        self._overlay.hide_progress()
        self._overlay.show()
        # self.lbl_sort_status.setText("Sorting...")
        
        self._sort_worker = ProcessingWorker(
            self._do_sort,
            self.gaze_data_path, primary, secondary
        )
        self.btn_sort.connect_worker(self._sort_worker)
        self._sort_worker.finished.connect(self._on_sort_done)
        self._sort_worker.error.connect(self._on_sort_error)
        self._sort_worker.start()
    
    @staticmethod
    def _do_sort(gaze_path, primary, secondary, progress_callback=None):
        """Sort data — delegates to core.temporal_prep."""
        return sort_data(gaze_path, primary, secondary, progress_callback)
    
    def _on_sort_done(self, result):
        self._overlay.hide()
        self.btn_sort.stop_processing()
        sort_cols = result['sort_cols']
        self.lbl_sort_status.setText(f"{status_icon_success} Sorted by: {' → '.join(sort_cols)}")
        self.lbl_sort_status.setObjectName("status-success-label")
        self.data_changed.emit()
    
    def _on_sort_error(self, err_msg):
        self._overlay.hide()
        self.btn_sort.stop_processing()
        show_error(self, "Sort Failed", f"Sorting failed:\n{err_msg}")
        self.lbl_sort_status.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_sort_status.setObjectName("status-error-label")
    
    def apply_markers(self):
        """Apply trial onset/offset markers to the data."""
        if not self.gaze_data_path or not os.path.isfile(self.gaze_data_path):
            show_warning(self, "No Data Selected", "Please select a gaze data file before applying markers.")
            return

        onset_col = self.combo_onset_col.currentText()
        if onset_col.startswith("--"):
            show_warning(self, "Column Not Selected", "Please select an onset marker column to define where trials begin.")
            return

        onset_markers = [item.text() for item in self.list_onset_markers.selectedItems()]
        if not onset_markers:
            show_warning(self, "No Markers Selected", "Please select at least one onset marker value.")
            return
        
        use_duration = self.radio_offset_duration.isChecked()
        
        if use_duration:
            try:
                duration_val = float(self.txt_duration.text().strip())
            except ValueError:
                show_warning(self, "Invalid Duration", "Please enter a valid number for the offset duration.")
                return
            
            unit = self.combo_duration_unit.currentText()
            duration_ms = duration_val * 1000 if "seconds" in unit else duration_val
            
            temporal_col = self.combo_temporal_primary.currentText()
            if temporal_col.startswith("--"):
                show_warning(self, "Invalid Configuration",
                    "Duration mode requires a primary temporal column (e.g., TIMESTAMP).\n"
                    "Please set it in the Temporal Column Setup section.")
                return
            
            offset_config = {'mode': 'duration', 'duration_ms': duration_ms, 'temporal_col': temporal_col}
        else:
            offset_col = self.combo_offset_col.currentText()
            if offset_col.startswith("--"):
                show_warning(self, "Column Not Selected", "Please select an offset marker column to define where trials end.")
                return
            
            offset_markers = [item.text() for item in self.list_offset_markers.selectedItems()]
            if not offset_markers:
                show_warning(self, "No Markers Selected", "Please select at least one offset marker value.")
                return
            
            offset_config = {'mode': 'marker', 'col': offset_col, 'markers': offset_markers}
        
        # Parse onset offset
        onset_offset_text = self.txt_onset_offset.text().strip()
        try:
            onset_offset_ms = float(onset_offset_text) if onset_offset_text else 0.0
        except ValueError:
            show_warning(self, "Invalid Configuration",
                                "Onset offset must be a number (milliseconds).")
            return

        self.btn_apply.start_processing("Applying markers")
        self.lbl_status.setText("Applying trial markers...")
        self._overlay.set_status("Applying Trial Markers")
        self._overlay.hide_progress()
        self._overlay.show()
        
        self._marker_worker = ProcessingWorker(
            self._do_apply_markers,
            self.gaze_data_path, onset_col, onset_markers, offset_config, onset_offset_ms
        )
        self.btn_apply.connect_worker(self._marker_worker)
        self._marker_worker.finished.connect(self._on_markers_done)
        self._marker_worker.error.connect(self._on_markers_error)
        self._marker_worker.start()
    
    @staticmethod
    def _do_apply_markers(gaze_path, onset_col, onset_markers, offset_config,
                          onset_offset_ms=0.0, progress_callback=None):
        """Apply trial markers — delegates to core.temporal_prep."""
        return apply_markers(gaze_path, onset_col, onset_markers, offset_config,
                             onset_offset_ms, progress_callback)
    
    def _on_markers_done(self, result):
        self._overlay.hide()
        self.btn_apply.stop_processing()
        n_onset = result['n_onset']
        n_offset = result['n_offset']
        
        msg = f"{status_icon_success} Trial markers applied!\n\nOnset markers: {n_onset}\nOffset markers: {n_offset}"
        self.lbl_status.setText(f"{status_icon_success} Applied: {n_onset} onsets, {n_offset} offsets")
        self.lbl_status.setObjectName("status-success-label")
        show_success(self, "Markers Applied", msg)
        self.data_changed.emit()
    
    def _on_markers_error(self, err_msg):
        self._overlay.hide()
        self.btn_apply.stop_processing()
        show_error(self, "Markers Failed", f"Applying markers failed:\n{err_msg}")
        self.lbl_status.setText(f"{status_icon_error} Error: {err_msg}")
        self.lbl_status.setObjectName("status-error-label")

    def closeEvent(self, event: QCloseEvent):
        for attr in ('_kw_worker', '_sort_worker', '_marker_worker'):
            if hasattr(self, attr):
                w = getattr(self, attr)
                if w.isRunning():
                    w.wait()
