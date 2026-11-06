"""
Export View — Export data files to CSV format with column selection.
"""
import os
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QListWidget, QListWidgetItem, QFileDialog,
    QAbstractItemView, QSplitter, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from .button_styles import button_style
from .processing_utils import ProcessingOverlay
from .dialog_helpers import show_success, show_warning, show_error


class CsvExportWorker(QThread):
    """Background CSV export worker to keep the UI responsive."""
    finished = pyqtSignal(str, int, int)  # path, rows, cols
    error = pyqtSignal(str)

    def __init__(self, df, selected_cols, save_path, manifest=None):
        super().__init__()
        self._manifest = manifest or {}
        self._df = df
        self._selected_cols = selected_cols
        self._save_path = save_path

    def run(self):
        try:
            export_df = self._df[self._selected_cols]
            export_df.to_csv(self._save_path, index=False)
            if self._manifest:
                import json
                from pathlib import Path
                Path(self._save_path).with_suffix('.manifest.json').write_text(json.dumps(self._manifest,indent=2))
            self.finished.emit(self._save_path, len(export_df), len(self._selected_cols))
        except Exception as e:
            self.error.emit(str(e))


class ExportView(QWidget):
    def __init__(self):
        super().__init__()
        self.current_project_path = ''
        self._data_files = []
        self._selected_file = None
        self._df = None
        self._project_settings = None
        self._syncing_selection = False
        self._export_worker = None
        self._overlay = ProcessingOverlay(self)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Export Data")
        title.setObjectName("title-label")
        layout.addWidget(title)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        layout.addWidget(self.lbl_notice)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left Panel: File Selection ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)

        file_group = QGroupBox("Data Files")
        file_layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.itemClicked.connect(self._on_file_selected)
        file_layout.addWidget(self.file_list)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setToolTip("Rescan project folders for exportable parquet files")
        btn_refresh.clicked.connect(self._refresh_files)
        file_layout.addWidget(btn_refresh)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        splitter.addWidget(left_panel)

        # ── Right Panel: Column Selection ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)

        col_group = QGroupBox("Select Columns to Export")
        col_layout = QVBoxLayout()

        # Select All / Deselect All buttons
        select_btns = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setToolTip("Select all available columns for export")
        btn_select_all.clicked.connect(self._select_all_columns)
        select_btns.addWidget(btn_select_all)
        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.setToolTip("Deselect all columns")
        btn_deselect_all.clicked.connect(self._deselect_all_columns)
        select_btns.addWidget(btn_deselect_all)
        select_btns.addStretch()
        col_layout.addLayout(select_btns)

        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.column_list.setSpacing(4)
        col_layout.addWidget(self.column_list)

        # Preview label
        self.lbl_preview = QLabel("Select a file to see available columns")
        self.lbl_preview.setObjectName("preview-text")
        col_layout.addWidget(self.lbl_preview)

        col_group.setLayout(col_layout)
        right_layout.addWidget(col_group)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # ── Export Section ──
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()

        self.lbl_status = QLabel("No file selected")
        export_layout.addWidget(self.lbl_status, 1)

        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.setStyleSheet(button_style("success", padding="10px 20px"))
        self.btn_export.setToolTip("Export selected columns to a CSV file")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        export_layout.addWidget(self.btn_export)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        self._update_onboarding_state()

    def showEvent(self, event):
        """Refresh file list when the view is shown."""
        super().showEvent(event)
        if hasattr(self, 'file_list'):
            self._refresh_files()

    def set_project_path(self, path):
        """Called when the project path changes."""
        self.current_project_path = path
        if hasattr(self, 'file_list'):
            self._refresh_files()
        self._update_onboarding_state()

    def refresh_data(self):
        """Reload project file list."""
        self._refresh_files()

    def load_settings(self, settings):
        self._project_settings = settings
        selected = settings.resolve_transformed_data_file("") or settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if self.current_project_path:
            self._refresh_files()
        if selected:
            self._set_transformed_data_file(selected, update_settings=False)
        self._update_onboarding_state()

    def _refresh_files(self):
        """Scan project folders for exportable data files."""
        self.file_list.clear()
        self._data_files = []
        self._selected_file = None
        self._df = None
        self.column_list.clear()
        self.btn_export.setEnabled(False)
        self.lbl_status.setText("No file selected")
        self.lbl_preview.setText("Select a file to see available columns")

        if not self.current_project_path:
            return

        # Scan multiple folders for parquet files
        folders = [
            'imported_parquet',
            'transformed_data',
            'results'
        ]

        for folder in folders:
            folder_path = os.path.join(self.current_project_path, folder)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.endswith('.parquet'):
                        full_path = os.path.join(folder_path, fname)
                        display_name = f"{folder}/{fname}"
                        self._data_files.append((display_name, full_path))
                        item = QListWidgetItem(display_name)
                        self.file_list.addItem(item)
        if self._project_settings:
            selected = (
                self._project_settings.resolve_transformed_data_file("")
                or self._project_settings.resolve_working_data_file("")
            )
            if selected:
                self._select_file_by_path(selected)
        self._update_onboarding_state()

    def _on_file_selected(self, item):
        """Load the selected file and populate column list."""
        display_name = item.text()
        
        # Find the full path
        full_path = None
        for name, path in self._data_files:
            if name == display_name:
                full_path = path
                break

        if not full_path or not os.path.exists(full_path):
            return

        self._selected_file = full_path
        
        try:
            self._df = pd.read_parquet(full_path)
            self._populate_columns()
            self.lbl_status.setText(f"Selected: {display_name} ({len(self._df):,} rows)")
            self.btn_export.setEnabled(True)
            self._update_onboarding_state()
        except Exception as e:
            show_warning(self, "Failed to Load File", f"Could not load the selected file:\n{e}")
            self._df = None
            self.btn_export.setEnabled(False)
            self._update_onboarding_state()

    def _set_transformed_data_file(self, path, update_settings=False):
        if not path or not os.path.isfile(path):
            return
        if not self._data_files and self.current_project_path:
            self._refresh_files()
        self._select_file_by_path(path)

    def _select_file_by_path(self, path):
        if not path or not os.path.isfile(path):
            return
        display_name = None
        for name, full in self._data_files:
            if os.path.abspath(full) == os.path.abspath(path):
                display_name = name
                break
        if not display_name:
            return
        items = self.file_list.findItems(display_name, Qt.MatchFlag.MatchExactly)
        if not items:
            return
        self._syncing_selection = True
        try:
            self.file_list.setCurrentItem(items[0])
            self._on_file_selected(items[0])
        finally:
            self._syncing_selection = False

    def _populate_columns(self):
        """Populate the column list with checkboxes."""
        self.column_list.clear()
        
        if self._df is None:
            return

        for col in self._df.columns:
            item = QListWidgetItem()
            checkbox = QCheckBox(col)
            checkbox.setChecked(True)  # Default: all selected
            checkbox.setStyleSheet("padding-top: 2px; padding-bottom: 2px;")
            hint = checkbox.sizeHint()
            item.setSizeHint(QSize(hint.width(), max(hint.height() + 6, 30)))
            self.column_list.addItem(item)
            self.column_list.setItemWidget(item, checkbox)

        self.lbl_preview.setText(f"{len(self._df.columns)} columns available")
        self._update_onboarding_state()

    def _update_onboarding_state(self):
        if not self.current_project_path:
            self.lbl_notice.setText("Open a project first in Project Manager.")
            self.lbl_notice.show()
            return
        if not self._data_files:
            self.lbl_notice.setText(
                "No exportable parquet files found. Import data or run transformation first."
            )
            self.lbl_notice.show()
            return
        if self._df is None:
            self.lbl_notice.setText("Select a file on the left, choose columns, then export to CSV.")
            self.lbl_notice.show()
            return
        self.lbl_notice.hide()

    def _select_all_columns(self):
        """Select all columns."""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            checkbox = self.column_list.itemWidget(item)
            if checkbox:
                checkbox.setChecked(True)

    def _deselect_all_columns(self):
        """Deselect all columns."""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            checkbox = self.column_list.itemWidget(item)
            if checkbox:
                checkbox.setChecked(False)

    def _get_selected_columns(self):
        """Return list of selected column names."""
        selected = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            checkbox = self.column_list.itemWidget(item)
            if checkbox and checkbox.isChecked():
                selected.append(checkbox.text())
        return selected

    def _export_csv(self):
        """Export selected columns to CSV."""
        if self._export_worker and self._export_worker.isRunning():
            show_warning(self, "Export Already Running", "A CSV export is already running. Please wait for it to finish.")
            return

        if self._df is None:
            show_warning(self, "No Data Loaded", "No data is loaded. Select a file first.")
            return

        selected_cols = self._get_selected_columns()
        if not selected_cols:
            show_warning(self, "No Columns Selected", "Please select at least one column to export.")
            return

        if not self._selected_file:
            show_warning(self, "No File Selected", "Select a source file before exporting.")
            return

        # Suggest filename based on source
        default_name = os.path.splitext(os.path.basename(self._selected_file))[0] + ".csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", 
            os.path.join(self.current_project_path, default_name),
            "CSV Files (*.csv)"
        )

        if not save_path:
            return

        self._overlay.set_status("Exporting CSV")
        self._overlay.hide_progress()
        self._overlay.show()
        self.btn_export.setEnabled(False)
        self.btn_export.setText("Exporting…")

        from core.template_manager import dataset_manifest
        manifest = dataset_manifest(self._selected_file) if self._selected_file else {}
        self._export_worker = CsvExportWorker(self._df, selected_cols, save_path,manifest=manifest)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_worker.error.connect(self._export_worker.deleteLater)
        self._export_worker.start()

    def _on_export_finished(self, save_path, n_rows, n_cols):
        self._overlay.hide()
        self.btn_export.setEnabled(True)
        self.btn_export.setText("Export to CSV")
        self._export_worker = None
        show_success(
            self,
            "Export Complete",
            f"Successfully exported {n_rows:,} rows × {n_cols} columns to:\n{save_path}"
        )

    def _on_export_error(self, err_msg):
        self._overlay.hide()
        self.btn_export.setEnabled(True)
        self.btn_export.setText("Export to CSV")
        self._export_worker = None
        show_error(self, "Export Failed", f"Could not export the data:\n{err_msg}")
