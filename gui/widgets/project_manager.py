from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QTreeView, QHeaderView, QHBoxLayout, 
                             QProgressBar, QMessageBox, QListWidget, QInputDialog,
                             QDialog, QComboBox, QFormLayout, QDialogButtonBox,
                             QSplitter, QTableView, QApplication, QMenu, QCheckBox,
                             QSpinBox)
from PyQt6.QtCore import Qt, QDir, QThread, pyqtSignal, QSettings, QTimer, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QCloseEvent, QBrush, QColor
from PyQt6.QtGui import QFileSystemModel, QStandardItemModel, QStandardItem
from .button_styles import button_style
from .processing_utils import ProcessingOverlay
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm
from .theme import _THEME
import os
import logging
import shutil
import sys

logger = logging.getLogger('iTemplate2')

# Add parent dir to path to find convert_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.io_utils import convert_csv_to_parquet, convert_multiple_to_parquet, clean_stimulus_in_parquet, remap_columns_in_parquet, convert_timestamp_in_parquet
except ImportError:
    convert_csv_to_parquet = None
    convert_multiple_to_parquet = None
    clean_stimulus_in_parquet = None
    remap_columns_in_parquet = None
    convert_timestamp_in_parquet = None

class ColumnMappingDialog(QDialog):
    def __init__(self, actual_columns, required_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Data Columns")
        self.mapping = {}
        self.actual_columns = actual_columns
        self.required_columns = required_columns
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        layout.addWidget(QLabel("Some required columns are missing.\nPlease map the existing columns to the required names:"))
        
        self.combos = {}
        
        form_layout = QFormLayout()
        
        for req in self.required_columns:
            # Check if likely match exists (e.g. case insensitive)
            default_txt = ""
            for act in actual_columns:
                if act.lower() == req.lower():
                    default_txt = act
                    break
        
            if req in actual_columns:
                # Exact match found, no need to ask? 
                # Or show it as disabled/fixed?
                # User wants to Confirm logic, so show all.
                pass
            
            combo = QComboBox()
            combo.setMinimumWidth(200)
            combo.addItem("-- Select Column --")
            combo.addItems(actual_columns)
            
            if default_txt:
                combo.setCurrentText(default_txt)
            
            self.combos[req] = combo
            label_text = f"{req}:"
            if req == 'Fixation Duration':
                label_text = f"{req} (optional for gaze/sample data):"
            elif req == 'TIMESTAMP':
                label_text = f"{req} (required for video; optional for images; creates internal copy):"
            form_layout.addRow(label_text, combo)
            
        layout.addLayout(form_layout)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
    def get_mapping(self):
        # Return {old_name: new_name}
        # But we built UI as Key=NewName, Value=OldName
        # So we inverse: map[OldName] = NewName if OldName != NewName
        final_map = {}
        for req, combo in self.combos.items():
            selected = combo.currentText()
            if selected and selected != "-- Select Column --":
                if selected != req:
                    final_map[selected] = req
        return final_map

class ConversionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    data_changed = pyqtSignal()  # New signal


    def __init__(self, csv_files, destination_folder, rename_map=None, merge=False, merge_name="merged",
                 strip_stim_path=False, strip_stim_ext=False, copy_map=None):
        super().__init__()
        self.csv_files = csv_files
        self.destination_folder = destination_folder
        self.rename_map = rename_map
        self.copy_map = copy_map
        self.merge = merge
        self.merge_name = merge_name
        self.strip_stim_path = strip_stim_path
        self.strip_stim_ext = strip_stim_ext

    def run(self):
        try:
            if self.merge:
                # Merge Mode
                parquet_name = self.merge_name + ".parquet"
                target_path = os.path.join(self.destination_folder, parquet_name)
                
                self.progress.emit(f"Merging {len(self.csv_files)} files into {parquet_name}...")
                
                if convert_multiple_to_parquet:
                    convert_multiple_to_parquet(self.csv_files, target_path, rename_map=self.rename_map,
                                               strip_stim_path=self.strip_stim_path, strip_stim_ext=self.strip_stim_ext,
                                               copy_map=self.copy_map)
                else:
                    self.finished.emit(False, "Conversion module not found.")
                    return
            else:
                # Single File Mode
                for csv_path in self.csv_files:
                    filename = os.path.basename(csv_path)
                    basename = os.path.splitext(filename)[0]
                    parquet_name = basename + ".parquet"
                    target_path = os.path.join(self.destination_folder, parquet_name)
                    
                    self.progress.emit(f"Converting {filename}...")
                    
                    if convert_csv_to_parquet:
                        convert_csv_to_parquet(csv_path, target_path, rename_map=self.rename_map,
                                              strip_stim_path=self.strip_stim_path, strip_stim_ext=self.strip_stim_ext,
                                              copy_map=self.copy_map)
                    else:
                        self.finished.emit(False, "Conversion module not found.")
                        return
            
            self.finished.emit(True, "Processing complete.")
            self.data_changed.emit()
        except Exception as e:
            self.finished.emit(False, str(e))

class PandasTableModel(QAbstractTableModel):
    """A high-performance model for displaying large pandas DataFrames in a QTableView."""
    def __init__(self, data, highlight_keys=None, aoi_col_indices=None):
        super().__init__()
        self._data = data
        self.highlight_keys = highlight_keys or {}
        self.aoi_col_indices = aoi_col_indices or {}

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        from .theme import THEME

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._data.iloc[row, col]
            return str(val) if val is not None else ""

        if role == Qt.ItemDataRole.BackgroundRole:
            col_name = self._data.columns[col]
            # Priority highlights (lookup dynamic color from key)
            if col_name in self.highlight_keys:
                key = self.highlight_keys[col_name]
                return QBrush(QColor(THEME.get(key, "#000000")))
            # AOI specific highlights
            if col in self.aoi_col_indices:
                val = str(self._data.iloc[row, col])
                color_map = self.aoi_col_indices[col]
                if val in color_map:
                    # For AOIs, we generate colors based on HSL + theme lightness
                    hue = color_map[val]
                    return QBrush(QColor.fromHsl(hue, 80, THEME["aoi_lightness"]))
        
        if role == Qt.ItemDataRole.ForegroundRole:
            # If a background is set, ensure text color is theme-appropriate
            col_name = self._data.columns[col]
            if col_name in self.highlight_keys or col in self.aoi_col_indices:
                return QColor(THEME["text_primary"])

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        from .theme import THEME
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.columns[section])
        if orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole:
            return str(section + 1)
        
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.BackgroundRole:
            col_name = self._data.columns[section]
            if col_name in self.highlight_keys:
                key = self.highlight_keys[col_name]
                return QBrush(QColor(THEME.get(key, "#000000")))
        
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ForegroundRole:
            col_name = self._data.columns[section]
            if col_name in self.highlight_keys:
                return QColor(THEME["text_primary"])

        return None

class PreviewLoadWorker(QThread):
    """Background worker for loading data preview without blocking the UI."""
    finished = pyqtSignal(object, str, bool)  # (dataframe_or_None, path, load_all)
    error = pyqtSignal(str)

    def __init__(self, path, nrows, load_all):
        super().__init__()
        self.path = path
        self.nrows = nrows
        self.load_all = load_all

    def run(self):
        try:
            import pandas as pd
            if self.path.lower().endswith('.parquet'):
                import pyarrow.parquet as pq
                if self.load_all:
                    df = pd.read_parquet(self.path)
                else:
                    pf = pq.ParquetFile(self.path)
                    batches = []
                    remaining = self.nrows
                    for batch in pf.iter_batches(batch_size=min(self.nrows, 10000)):
                        batches.append(batch.to_pandas())
                        remaining -= len(batch)
                        if remaining <= 0:
                            break
                    df = pd.concat(batches, ignore_index=True).head(self.nrows)
            elif self.path.lower().endswith('.txt'):
                df = pd.read_csv(self.path, sep='\t', nrows=None if self.load_all else self.nrows)
            else:
                df = pd.read_csv(self.path, nrows=None if self.load_all else self.nrows)
            self.finished.emit(df, self.path, self.load_all)
        except Exception as e:
            self.error.emit(str(e))

class ProjectManager(QWidget):
    # Signal to notify other widgets about the project path change
    project_changed = pyqtSignal(str)
    data_import_finished = pyqtSignal() # Existing signal
    data_changed = pyqtSignal()         # New signal for ANY data change (import or remap)
    
    def __init__(self):
        super().__init__()
        
        self.project_path = ""
        self.settings = QSettings("iTemplate2", "ProjectManager")
        self._project_settings = None
        self._syncing_selection = False
        self._overlay = ProcessingOverlay(self)
        from .theme import on_theme_changed
        on_theme_changed(self._on_global_theme_changed)

        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header_layout = QHBoxLayout()
        self.lbl_project = QLabel("No Project Selected")
        self.lbl_project.setObjectName("title-label")
        
        btn_open = QPushButton("Open Project")
        btn_open.clicked.connect(self.select_project_folder)
        
        btn_new = QPushButton("New Project")
        btn_new.clicked.connect(self.create_new_project)
        
        header_layout.addWidget(self.lbl_project)
        header_layout.addStretch()
        header_layout.addWidget(btn_new)
        header_layout.addWidget(btn_open)
        
        layout.addLayout(header_layout)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        layout.addWidget(self.lbl_notice)
        
        # Three-panel vertical splitter: Recent Projects / Project Files / Data Preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(8)
        
        # Panel 1: Recent Projects
        recent_widget = QWidget()
        recent_layout = QVBoxLayout()
        recent_layout.setContentsMargins(0,0,0,0)
        recent_layout.addWidget(QLabel("Recent Projects:"))
        self.list_recent = QListWidget()
        self.list_recent.itemClicked.connect(self.recent_project_clicked)
        recent_layout.addWidget(self.list_recent)
        recent_widget.setLayout(recent_layout)
        self.load_recent_projects()
        
        # Panel 2: Project Files Tree
        tree_widget = QWidget()
        tree_layout = QVBoxLayout()
        tree_layout.setContentsMargins(0,0,0,0)
        tree_layout.addWidget(QLabel("Project Files:"))
        self.model = QFileSystemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.rootPath()))
        self.tree.setColumnWidth(0, 300)
        self.tree.header().setStretchLastSection(True)
        
        # Connect selection change
        self.tree.selectionModel().selectionChanged.connect(self.on_file_selected)
        
        # Right-click context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        
        tree_layout.addWidget(self.tree)
        tree_widget.setLayout(tree_layout)
        
        # Panel 3: Data Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0,0,0,0)
        
        preview_header = QHBoxLayout()
        self.lbl_preview_prefix = QLabel("Data Preview:")
        preview_header.addWidget(self.lbl_preview_prefix)
        
        # Inline row-count: "(First [spinbox] rows)" or "(All N rows)"
        self.lbl_paren_open = QLabel("(First")
        preview_header.addWidget(self.lbl_paren_open)
        self.spin_nrows = QSpinBox()
        self.spin_nrows.setRange(100, 999999)
        self.spin_nrows.setValue(1000)
        self.spin_nrows.setSingleStep(100)
        self.spin_nrows.setToolTip("Number of rows to load in the preview")
        self.spin_nrows.editingFinished.connect(self.reload_preview)
        preview_header.addWidget(self.spin_nrows)
        self.lbl_paren_close = QLabel("rows)")
        preview_header.addWidget(self.lbl_paren_close)
        
        preview_header.addStretch()
        self.btn_load_all = QPushButton("Load All")
        self.btn_load_all.setToolTip("Load all rows from the file (may be slow for large files)")
        self.btn_load_all.setEnabled(False)
        self.btn_load_all.clicked.connect(self.load_all_rows)
        preview_header.addWidget(self.btn_load_all)
        
        self.btn_clean_stim = QPushButton("Clean stimulusName")
        self.btn_clean_stim.setToolTip("Strip path prefixes and file extensions from the stimulusName column")
        self.btn_clean_stim.setEnabled(False)
        self.btn_clean_stim.clicked.connect(self.clean_stimulus_post_import)
        preview_header.addWidget(self.btn_clean_stim)
        
        self.btn_remap_cols = QPushButton("Remap Columns")
        self.btn_remap_cols.setToolTip("Rename columns in the currently selected parquet file")
        self.btn_remap_cols.setEnabled(False)
        self.btn_remap_cols.clicked.connect(self.remap_columns_post_import)
        preview_header.addWidget(self.btn_remap_cols)

        self.btn_convert_ts = QPushButton("Convert TIMESTAMP")
        self.btn_convert_ts.setToolTip(
            "Convert the TIMESTAMP column from its current unit (e.g. nanoseconds) into milliseconds (ms)"
        )
        self.btn_convert_ts.setEnabled(False)
        self.btn_convert_ts.clicked.connect(self.convert_timestamp_post_import)
        preview_header.addWidget(self.btn_convert_ts)
        
        preview_layout.addLayout(preview_header)
        
        self.preview_table = QTableView()
        self.preview_table.horizontalHeader().sectionDoubleClicked.connect(self.rename_column_header)
        preview_layout.addWidget(self.preview_table)
        preview_widget.setLayout(preview_layout)
        
        self.current_preview_path = None
        
        splitter.addWidget(recent_widget)
        splitter.addWidget(tree_widget)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 4)
        
        # Add grip dots to each splitter handle
        for i in range(1, splitter.count()):
            handle = splitter.handle(i)
            if handle:
                handle.setCursor(Qt.CursorShape.OpenHandCursor)
                grip = QLabel("⋯")
                grip.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grip.setObjectName("grip-label")
                grip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                grip_layout = QHBoxLayout(handle)
                grip_layout.setContentsMargins(0, 0, 0, 0)
                grip_layout.addWidget(grip)
        
        layout.addWidget(splitter)
        
        # Import Area
        import_layout = QHBoxLayout()
        self.btn_import = QPushButton("Import and Convert Eye-movement Data")
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self.import_data)
        self.btn_import.setStyleSheet(button_style("primary", padding="10px")) 
        
        import_layout.addWidget(self.btn_import)
        layout.addLayout(import_layout)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)
        self._update_onboarding_state()

    def _on_global_theme_changed(self):
        """Called when the application theme changes."""
        # Force the table to repaint with new theme colors
        if hasattr(self, 'preview_table'):
            model = self.preview_table.model()
            if model:
                model.layoutChanged.emit()
            self.preview_table.viewport().update()

    def set_settings(self, settings):
        self._project_settings = settings
    
    def load_settings(self, settings):
        """Apply global selection from ProjectSettings."""
        self._project_settings = settings
        selected = settings.resolve_source_data_file("")
        
        # If no saved selection, fall back to latest parquet in imported_parquet/
        if selected and os.path.isfile(selected):
            if not settings.get_source_data_file(""):
                settings.set("global", "source_data_file", selected)
            if not settings.get_working_data_file(""):
                settings.set("global", "selected_data_file", selected)
        
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        
        if selected and os.path.isfile(selected):
            self._select_file_in_tree(selected)
            # Load preview if not already showing this file
            if self.current_preview_path != selected:
                self.current_preview_path = selected
                self._async_load_preview(selected)
                self.btn_clean_stim.setEnabled(selected.lower().endswith('.parquet'))
                self.btn_remap_cols.setEnabled(selected.lower().endswith('.parquet'))
                self.btn_convert_ts.setEnabled(selected.lower().endswith('.parquet'))
                self.btn_load_all.setEnabled(True)
        self._update_onboarding_state()

    def load_recent_projects(self):
        self.list_recent.clear()
        recent = self.settings.value("recent_projects", [], type=list)
        # Ensure unique and existing
        valid_recent = []
        for path in recent:
            if os.path.exists(path) and path not in valid_recent:
                 valid_recent.append(path)
        
        for path in valid_recent:
            self.list_recent.addItem(path)
            
        # Update settings if cleaned
        if len(valid_recent) != len(recent):
             self.settings.setValue("recent_projects", valid_recent)

    def add_to_recent(self, path):
        current_list = self.settings.value("recent_projects", [], type=list)
        if path in current_list:
            current_list.remove(path)
        current_list.insert(0, path)
        current_list = current_list[:5] # Keep last 5
        self.settings.setValue("recent_projects", current_list)
        self.load_recent_projects()

    def recent_project_clicked(self, item):
        path = item.text()
        if os.path.exists(path):
            self.set_active_project(path)
        else:
            show_warning(self, "Project Not Found", "The project path no longer exists. Please select a different folder.")
            self.load_recent_projects()

    def create_new_project(self):
        # 1. Ask for location
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory")
        if not parent_dir:
            return
            
        # 2. Ask for name
        name, ok = QInputDialog.getText(self, "New Project", "Project Name:")
        if ok and name:
            new_path = os.path.join(parent_dir, name)
            if os.path.exists(new_path):
                show_warning(self, "Name Conflict", "A folder with this name already exists. Please choose a different project name.")
                return
                
            try:
                os.makedirs(new_path)
                # Create standard subfolders
                os.makedirs(os.path.join(new_path, "stimuli"), exist_ok=True)
                os.makedirs(os.path.join(new_path, "imported_parquet"), exist_ok=True)
                os.makedirs(os.path.join(new_path, "processed_stimuli_data"), exist_ok=True)
                
                self.set_active_project(new_path)
                show_success(self, "Project Created", f"Project '{name}' has been created successfully.")
            except Exception as e:
                show_error(self, "Project Creation Failed", f"Could not create the project:\n{e}")

    def show_tree_context_menu(self, position):
        index = self.tree.indexAt(position)
        if not index.isValid():
            return
        
        file_path = self.model.filePath(index)
        menu = QMenu(self)
        
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.tree.viewport().mapToGlobal(position))
        
        if action == delete_action:
            self.delete_selected_file(file_path)
    
    def delete_selected_file(self, file_path):
        name = os.path.basename(file_path)
        is_dir = os.path.isdir(file_path)
        kind = "folder" if is_dir else "file"
        
        reply = ask_confirm(self, f"Delete this {kind}?",
            f"Are you sure you want to permanently delete this {kind}?\n\n{name}\n\nThis action cannot be undone.")

        if reply:
            try:
                if is_dir:
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                show_error(self, "Delete Failed", f"Could not delete the {kind}:\n{e}")

    def select_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if folder:
            self.set_active_project(folder)

    def set_active_project(self, folder):
        """Show loading overlay and switch project in background."""
        self._overlay.set_status("Loading Project")
        self._overlay.show()
        
        # Defer the actual loading so the overlay can paint and finish its fade-in animation
        # (The fade-in takes 150ms, so we wait 200ms)
        QTimer.singleShot(200, lambda: self._do_set_active_project(folder))

    def _do_set_active_project(self, folder):
        try:
            self.project_path = folder
            self.lbl_project.setText(f"Project: {os.path.basename(folder)}")
            self.add_to_recent(folder)
            self.SETUP_PROJECT_VIEW(folder)
            
            # Emit signal for other widgets (heavy loading happens here)
            self.project_changed.emit(folder)
        finally:
            self._overlay.hide()
            
    def SETUP_PROJECT_VIEW(self, folder):
        self.model.setRootPath(folder)
        self.tree.show()
        self.tree.setRootIndex(self.model.index(folder))
        self.btn_import.setEnabled(True)
        self._update_onboarding_state()
        
    def on_file_selected(self, selected, deselected):
        indexes = self.tree.selectionModel().selectedIndexes()
        if not indexes:
            return
            
        # File System Model has multiple columns, just get the first one for path
        index = indexes[0]
        file_path = self.model.filePath(index)
        
        if os.path.isdir(file_path):
            return
            
        # Check extension
        lower_path = file_path.lower()
        if lower_path.endswith('.csv') or lower_path.endswith('.txt') or lower_path.endswith('.parquet'):
            self.lbl_preview_prefix.setText(f"Loading preview for: {os.path.basename(file_path)}...")
            self.current_preview_path = file_path
            self._async_load_preview(file_path)
            # Enable post-import buttons only for parquet files
            is_parquet = file_path.lower().endswith('.parquet')
            self.btn_clean_stim.setEnabled(is_parquet)
            self.btn_remap_cols.setEnabled(is_parquet)
            self.btn_convert_ts.setEnabled(is_parquet)
            self.btn_load_all.setEnabled(True)
            
            # Defer the global selection sync. This is critical because it triggers 
            # heavy synchronous I/O in other widgets (e.g. TemporalPreprocessing 
            # reading the file). Deferring it allows the ProcessingOverlay 
            # animation to start and the UI to remain responsive for the loading start.
            if is_parquet:
                QTimer.singleShot(100, lambda: self._sync_preview_with_global_selection(file_path))
        
        self._update_onboarding_state()

    def _update_onboarding_state(self):
        if not self.project_path:
            self.lbl_notice.setText(
                "Start here: create a new project or open an existing project folder."
            )
            self.lbl_notice.show()
            self.btn_import.setEnabled(False)
            return

        imported_dir = os.path.join(self.project_path, "imported_parquet")
        has_parquet = (
            os.path.isdir(imported_dir)
            and any(f.lower().endswith(".parquet") for f in os.listdir(imported_dir))
        )
        if not has_parquet:
            self.lbl_notice.setText(
                "Next step: import gaze data using 'Import and Convert Eye-movement Data'. "
                "Then choose a parquet file in the tree to preview and set as active data."
            )
            self.lbl_notice.show()
            return

        self.lbl_notice.hide()

    def _set_global_selected_file(self, path, only_if_empty=False):
        if not self._project_settings or self._syncing_selection:
            return
        if only_if_empty:
            existing = self._project_settings.get("global", "selected_data_file", "")
            if existing:
                return
        if self.project_path and os.path.isfile(path):
            imported_dir = os.path.join(self.project_path, "imported_parquet")
            try:
                if os.path.commonpath([os.path.abspath(path), os.path.abspath(imported_dir)]) == os.path.abspath(imported_dir):
                    self._project_settings.set("global", "source_data_file", path)
            except ValueError:
                pass
        self._project_settings.set("global", "selected_data_file", path)

    def _sync_preview_with_global_selection(self, path):
        """Previewing a file should not replace an existing workflow input."""
        if not self._project_settings or self._syncing_selection:
            return
        existing = self._project_settings.get_working_data_file("")
        if existing:
            return
        self._set_global_selected_file(path)

    def _async_load_preview(self, path):
        """Load a data preview in the background with a ProcessingOverlay."""
        self.lbl_status.setText("Loading data preview...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self._overlay.set_status("Loading Preview")
        self._overlay.hide_progress()
        self._overlay.show()

        # Defer worker start so the overlay paints before data loading begins
        QTimer.singleShot(0, lambda: self._start_preview_worker(path))

    def _start_preview_worker(self, path):
        """Create and start the background preview worker (called after overlay shows)."""
        self._preview_worker = PreviewLoadWorker(
            path, self.spin_nrows.value(), False
        )
        self._preview_worker.finished.connect(self._on_preview_loaded)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _select_file_in_tree(self, path):
        if not path or not os.path.isfile(path):
            return
        if self.project_path and not os.path.abspath(path).startswith(os.path.abspath(self.project_path) + os.sep):
            return
        index = self.model.index(path)
        if not index.isValid():
            return
        self._syncing_selection = True
        try:
            self.tree.setCurrentIndex(index)
            self.tree.scrollTo(index)
        finally:
            self._syncing_selection = False
            
    def load_preview(self, path, nrows=None):
        """Load a preview of the data file.
        
        Args:
            path: Path to the data file.
            nrows: Number of rows to load.  None = use spinbox value,
                   0 = load all rows.
        """
        if nrows is None:
            nrows = self.spin_nrows.value()
        load_all = (nrows == 0)
        
        try:
            import pandas as pd
            if path.lower().endswith('.parquet'):
                import pyarrow.parquet as pq
                if load_all:
                    df = pd.read_parquet(path)
                else:
                    pf = pq.ParquetFile(path)
                    batches = []
                    remaining = nrows
                    for batch in pf.iter_batches(batch_size=min(nrows, 10000)):
                        batches.append(batch.to_pandas())
                        remaining -= len(batch)
                        if remaining <= 0:
                            break
                    df = pd.concat(batches, ignore_index=True).head(nrows)
            elif path.lower().endswith('.txt'):
                df = pd.read_csv(path, sep='\t', nrows=None if load_all else nrows)
            else:
                df = pd.read_csv(path, nrows=None if load_all else nrows)
                
            self.populate_table(df)
            basename = os.path.basename(path)
            if load_all:
                self.lbl_preview_prefix.setText(f"Data Preview: {basename}")
                self.lbl_paren_open.setText(f"(All {len(df)}")
                self.spin_nrows.hide()
                self.lbl_paren_close.setText("rows)")
            else:
                self.lbl_preview_prefix.setText(f"Data Preview: {basename}")
                self.lbl_paren_open.setText("(First")
                self.spin_nrows.show()
                self.lbl_paren_close.setText("rows)")
        except Exception as e:
            self.lbl_preview_prefix.setText(f"Error loading preview: {e}")

    def reload_preview(self):
        """Reload the current preview with the row count from the spinbox."""
        if self.current_preview_path and os.path.isfile(self.current_preview_path):
            self.load_preview(self.current_preview_path)

    def load_all_rows(self):
        """Load all rows from the currently previewed file in a background thread."""
        if not self.current_preview_path or not os.path.isfile(self.current_preview_path):
            return
        self.lbl_preview_prefix.setText(
            f"Loading all rows for: {os.path.basename(self.current_preview_path)}...")
        self.btn_load_all.setEnabled(False)
        self.btn_load_all.setText("Loading...")
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.show()
        self.lbl_status.setText("Loading all data rows...")
        self._overlay.set_status("Loading All Rows")
        self._overlay.hide_progress()
        self._overlay.show()
        
        self._preview_worker = PreviewLoadWorker(self.current_preview_path, 0, True)
        self._preview_worker.finished.connect(self._on_preview_loaded)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_loaded(self, df, path, load_all):
        """Handle data loaded from background thread."""
        self.progress_bar.hide()
        self.lbl_status.setText("")
        self.populate_table(df)
        basename = os.path.basename(path)
        if load_all:
            self.lbl_preview_prefix.setText(f"Data Preview: {basename}")
            self.lbl_paren_open.setText(f"(All {len(df)}")
            self.spin_nrows.hide()
            self.lbl_paren_close.setText("rows)")
        else:
            self.lbl_preview_prefix.setText(f"Data Preview: {basename}")
            self.lbl_paren_open.setText("(First")
            self.spin_nrows.show()
            self.lbl_paren_close.setText("rows)")
        self.progress_bar.hide()
        self.btn_load_all.setText("Load All")
        self.btn_load_all.setEnabled(True)
        self.lbl_status.setText("")

    def _on_preview_error(self, error_msg):
        """Handle error from background preview loading."""
        self._overlay.hide()
        self.lbl_preview_prefix.setText(f"Error loading preview: {error_msg}")
        self.progress_bar.hide()
        self.btn_load_all.setText("Load All")
        self.btn_load_all.setEnabled(True)
        self.lbl_status.setText("")

    def populate_table(self, df):
        from PyQt6.QtGui import QBrush, QColor
        
        # Define priority columns to bring to the front
        priority_cols = [
            'Participant_ID', 'x', 'y', 'stimulusName', 'trial_event',
            'Duration', 'stimuli_x', 'stimuli_y', 'transformed_x', 'transformed_y',
            'VIDEO_FRAME_INDEX', 'AOI', 'AOI_stimulus'
        ]
        
        # Identify which priority columns exist and which don't
        existing_cols = list(df.columns)
        ordered_cols = [c for c in priority_cols if c in existing_cols]
        ordered_cols += [c for c in existing_cols if c not in ordered_cols]
        
        # Re-order the dataframe for display
        df = df[ordered_cols]
        col_names = [str(c) for c in df.columns]
        
        # Define highlight color keys for key columns
        highlight_keys = {
            'Participant_ID': 'hl_participant',
            'x': 'hl_x',
            'y': 'hl_y',
            'stimulusName': 'hl_stimulus',
            'trial_event': 'hl_trial',
            'stimuli_x': 'hl_stim_x',
            'stimuli_y': 'hl_stim_y',
            'transformed_x': 'hl_trans_x',
            'transformed_y': 'hl_trans_y',
            'VIDEO_FRAME_INDEX': 'hl_frame',
            '_video_frame_index': 'hl_frame',
            'Duration': 'hl_duration',
        }
        
        # Build per-value color maps for AOI and AOI_stimulus columns
        aoi_value_cols = ['AOI', 'AOI_stimulus']
        aoi_col_indices = {}   # col_idx -> {value_str: hue}
        for col_name in aoi_value_cols:
            if col_name in col_names:
                col_idx = col_names.index(col_name)
                unique_vals = sorted(df[col_name].dropna().astype(str).unique())
                n = max(len(unique_vals), 1)
                val_hue_map = {}
                for vi, val in enumerate(unique_vals):
                    hue = int((vi / n) * 300)
                    val_hue_map[val] = hue
                aoi_col_indices[col_idx] = val_hue_map
        
        try:
            # Use the high-performance PandasTableModel instead of QStandardItemModel
            model = PandasTableModel(df, highlight_keys, aoi_col_indices)
            self.preview_table.setModel(model)
            
            # Only resize columns for reasonably small previews to avoid UI hang
            if len(df) < 1000:
                self.preview_table.resizeColumnsToContents()
            else:
                # Use a default width for large tables and allow manual resizing
                self.preview_table.horizontalHeader().setDefaultSectionSize(120)
                self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
                
        except Exception as e:
            logger.error("Error populating table: %s", e)
            self.lbl_status.setText(f"Error displaying data: {e}")
        finally:
            self._overlay.hide()
        
    def rename_column_header(self, col_index):
        """Double-click on a column header to rename it in the parquet file."""
        if not self.current_preview_path or not self.current_preview_path.lower().endswith('.parquet'):
            return
        
        model = self.preview_table.model()
        if not model:
            return
        
        old_name = model.headerData(col_index, Qt.Orientation.Horizontal)
        if old_name is None:
            return
        
        new_name, ok = QInputDialog.getText(self, "Rename Column", 
                                             f"Rename '{old_name}' to:", 
                                             text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        
        new_name = new_name.strip()
        
        try:
            if remap_columns_in_parquet:
                remap_columns_in_parquet(self.current_preview_path, {old_name: new_name})
                self.load_preview(self.current_preview_path)
        except Exception as e:
            show_error(self, "Rename Failed", f"Failed to rename column:\n{e}")

    def import_data(self):
        # Ask User: Import Files (Individual) or Import Directory (Merge)?
        # Custom Dialog or just QMessageBox with buttons?
        # Let's use QMenu on the button technically, or just a simple Dialog now.
        # But button is already clicked.
        
        # Simple Message Box with Options
        msg = QMessageBox(self)
        msg.setWindowTitle("Import Data")
        msg.setText("How would you like to import data?")
        msg.setInformativeText(
            "<b>Select Files</b> — Each selected file becomes its own .parquet file. "
            "Use this when you have one data file per participant or session.<br><br>"
            "<b>Select Directory</b> — All data files in the chosen folder are "
            "merged into a single .parquet file. Use this when participant data is "
            "split across multiple files (e.g., one file per trial or block) and "
            "needs to be combined."
        )
        btn_files = msg.addButton("Select Files (Convert Individually)", QMessageBox.ButtonRole.ActionRole)
        btn_dir = msg.addButton("Select Directory (Merge contents)", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        
        clicked_button = msg.clickedButton()
        if clicked_button == btn_files:
            self.import_files_mode()
        elif clicked_button == btn_dir:
            self.import_dir_mode()
            
    def import_files_mode(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Data Files", "", "Data Files (*.csv *.txt *.parquet)")
        if not files:
            return
        self.start_conversion(files, merge=False)

    def import_dir_mode(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if not folder:
            return
            
        # Find all valid files
        files = []
        valid_exts = ['.csv', '.txt', '.parquet']
        for f in os.listdir(folder):
            if any(f.lower().endswith(ext) for ext in valid_exts):
                files.append(os.path.join(folder, f))
        
        if not files:
            show_warning(self, "No Files Found", "No valid data files (.csv, .txt, .parquet) were found in this directory.")
            return
            
        # Ask for merged output name
        folder_name = os.path.basename(folder)
        name, ok = QInputDialog.getText(self, "Merge Name", "Name for merged file:", text=folder_name)
        if not ok or not name:
            return
            
        self.start_conversion(files, merge=True, merge_name=name)

    def start_conversion(self, files, merge=False, merge_name="merged"):
        # Target folder
        target_dir = os.path.join(self.project_path, "imported_parquet")
        os.makedirs(target_dir, exist_ok=True)
        
        # Check Columns
        required_cols = ['Participant_ID', 'TRIAL_INDEX', 'x', 'y', 'stimulusName', 'Duration', 'TIMESTAMP']
        rename_map = None
        copy_map = None
        
        # Read header of first file to check columns
        try:
            import pandas as pd
            f_path = files[0]
            if f_path.lower().endswith('.txt'):
                df_head = pd.read_csv(f_path, sep='\t', nrows=0)
            elif f_path.lower().endswith('.parquet'):
                df_head = pd.read_parquet(f_path) 
            else:
                df_head = pd.read_csv(f_path, nrows=0)
                
            actual_cols = list(df_head.columns)
            
            missing = [c for c in required_cols if c not in actual_cols]
            
            if missing:
                dlg = ColumnMappingDialog(actual_cols, required_cols, self)
                if dlg.exec():
                    rename_map = dlg.get_mapping()
                    # Extract TIMESTAMP mapping as copy (preserve original column)
                    if rename_map:
                        copy_map = {}
                        for old, new in list(rename_map.items()):
                            if new == 'TIMESTAMP':
                                copy_map[old] = new
                                del rename_map[old]
                        if not copy_map:
                            copy_map = None
                        if not rename_map:
                            rename_map = None
                else:
                    return # Cancelled
                
        except Exception as e:
            show_warning(self, "Column Check Failed", f"Could not read the file header to check columns:\n{e}")
            return
        
        # --- StimulusName cleanup options dialog ---
        stim_dlg = QDialog(self)
        stim_dlg.setWindowTitle("stimulusName Options")
        stim_layout = QVBoxLayout(stim_dlg)
        stim_layout.addWidget(QLabel("Optional: clean the stimulusName column"))
        
        chk_strip_path = QCheckBox("Strip path prefixes  (e.g. expt2/AF_010F.png → AF_010F.png)")
        chk_strip_ext  = QCheckBox("Strip file extensions  (e.g. AF_010F.png → AF_010F)")
        stim_layout.addWidget(chk_strip_path)
        stim_layout.addWidget(chk_strip_ext)
        
        stim_btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        stim_btns.accepted.connect(stim_dlg.accept)
        stim_btns.rejected.connect(stim_dlg.reject)
        stim_layout.addWidget(stim_btns)
        
        if not stim_dlg.exec():
            return  # Cancelled
            
        strip_path = chk_strip_path.isChecked()
        strip_ext  = chk_strip_ext.isChecked()
        
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.lbl_status.setText("Starting process...")
        self.btn_import.setEnabled(False)
        self._overlay.set_status("Importing Data")
        self._overlay.hide_progress()
        self._overlay.show()
        
        self.worker = ConversionWorker(files, target_dir, rename_map, merge, merge_name,
                                       strip_stim_path=strip_path, strip_stim_ext=strip_ext,
                                       copy_map=copy_map)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.data_changed.connect(self.data_changed) # Forward signal
        self.worker.start()
        
    def update_status(self, msg):
        self.lbl_status.setText(msg)
        
    def conversion_finished(self, success, msg):
        self._overlay.hide()
        self.progress_bar.hide()
        self.btn_import.setEnabled(True)
        self.lbl_status.setText(msg)
        if success:
            show_success(self, "Import Complete", "Your data has been imported successfully.")
            self.data_import_finished.emit()
            logger.debug("Emitting data_changed from ProjectManager")
            self.data_changed.emit()  # Emit generic change signal
            self.SETUP_PROJECT_VIEW(self.project_path)
        else:
            show_error(self, "Import Failed", msg)

    def clean_stimulus_post_import(self):
        """Post-import: clean stimulusName in currently previewed parquet file."""
        if not self.current_preview_path or not self.current_preview_path.lower().endswith('.parquet'):
            show_warning(self, "No Parquet File", "Please select a .parquet file in the project tree to continue.")
            return
            
        from PyQt6.QtWidgets import QLineEdit, QGroupBox, QGridLayout
        
        # Options dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Clean stimulusName")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"File: {os.path.basename(self.current_preview_path)}"))
        
        # Strip options
        strip_group = QGroupBox("Strip")
        strip_layout = QVBoxLayout()
        chk_path = QCheckBox("Strip path prefixes  (e.g. expt2/AF_010F.png → AF_010F.png)")
        chk_ext  = QCheckBox("Strip file extensions  (e.g. AF_010F.png → AF_010F)")
        chk_path.setChecked(True)
        chk_ext.setChecked(True)
        strip_layout.addWidget(chk_path)
        strip_layout.addWidget(chk_ext)
        strip_group.setLayout(strip_layout)
        layout.addWidget(strip_group)
        
        # Remove prefix/suffix
        remove_group = QGroupBox("Remove text")
        remove_layout = QGridLayout()
        remove_layout.addWidget(QLabel("Remove prefix:"), 0, 0)
        txt_remove_prefix = QLineEdit()
        txt_remove_prefix.setPlaceholderText("e.g. expt2_")
        remove_layout.addWidget(txt_remove_prefix, 0, 1)
        remove_layout.addWidget(QLabel("Remove suffix:"), 1, 0)
        txt_remove_suffix = QLineEdit()
        txt_remove_suffix.setPlaceholderText("e.g. _old")
        remove_layout.addWidget(txt_remove_suffix, 1, 1)
        remove_group.setLayout(remove_layout)
        layout.addWidget(remove_group)
        
        # Add prefix/suffix
        add_group = QGroupBox("Add text")
        add_layout = QGridLayout()
        add_layout.addWidget(QLabel("Add prefix:"), 0, 0)
        txt_add_prefix = QLineEdit()
        txt_add_prefix.setPlaceholderText("e.g. stim_")
        add_layout.addWidget(txt_add_prefix, 0, 1)
        add_layout.addWidget(QLabel("Add suffix:"), 1, 0)
        txt_add_suffix = QLineEdit()
        txt_add_suffix.setPlaceholderText("e.g. _v2")
        add_layout.addWidget(txt_add_suffix, 1, 1)
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        
        if not dlg.exec():
            return
        
        strip_path = chk_path.isChecked()
        strip_ext = chk_ext.isChecked()
        remove_prefix = txt_remove_prefix.text().strip()
        remove_suffix = txt_remove_suffix.text().strip()
        add_prefix = txt_add_prefix.text().strip()
        add_suffix = txt_add_suffix.text().strip()
        
        if not any([strip_path, strip_ext, remove_prefix, remove_suffix, add_prefix, add_suffix]):
            return
            
        try:
            if clean_stimulus_in_parquet:
                ok = clean_stimulus_in_parquet(self.current_preview_path,
                                              strip_path=strip_path,
                                              strip_ext=strip_ext,
                                              remove_prefix=remove_prefix,
                                              remove_suffix=remove_suffix,
                                              add_prefix=add_prefix,
                                              add_suffix=add_suffix)
                if ok:
                    self.load_preview(self.current_preview_path)
                    show_success(self, "Cleanup Complete", "The stimulusName column has been cleaned successfully.")
                else:
                    show_warning(self, "Column Not Found", "No stimulusName column found in this file.")
            else:
                show_error(self, "Module Missing", "The required conversion module was not found.")
        except Exception as e:
            show_error(self, "Cleanup Failed", f"Failed to clean stimulusName:\n{e}")

    def remap_columns_post_import(self):
        """Post-import: rename columns in currently previewed parquet file."""
        if not self.current_preview_path or not self.current_preview_path.lower().endswith('.parquet'):
            show_warning(self, "No Parquet File", "Please select a .parquet file in the project tree first.")
            return
            
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(self.current_preview_path)
            actual_cols = table.column_names
            # Deduplicate for the dialog (show unique names only)
            seen = set()
            unique_cols = []
            for c in actual_cols:
                if c not in seen:
                    unique_cols.append(c)
                    seen.add(c)
            actual_cols = unique_cols
        except Exception as e:
            show_error(self, "File Read Error", f"Could not read the file:\n{e}")
            return
            
        required_cols = ['Participant_ID', 'TRIAL_INDEX', 'x', 'y', 'stimulusName', 'Duration', 'TIMESTAMP']  # Duration is not required if data is gaze/sample data
        dlg = ColumnMappingDialog(actual_cols, required_cols, self)
        dlg.setWindowTitle("Remap Columns (Post-Import)")
        
        if not dlg.exec():
            return
            
        rename_map = dlg.get_mapping()
        # Extract TIMESTAMP mapping as copy (preserve original column)
        copy_map = None
        if rename_map:
            copy_map = {}
            for old, new in list(rename_map.items()):
                if new == 'TIMESTAMP':
                    copy_map[old] = new
                    del rename_map[old]
            if not copy_map:
                copy_map = None
            if not rename_map:
                rename_map = None
        if not rename_map and not copy_map:
            show_warning(self, "No Changes", "No column mappings were changed.")
            return
            
        try:
            if remap_columns_in_parquet:
                remap_columns_in_parquet(self.current_preview_path, rename_map, copy_map=copy_map)
                self.load_preview(self.current_preview_path)
                parts = []
                if rename_map:
                    parts.append(f"Renamed: {rename_map}")
                if copy_map:
                    parts.append(f"Copied: {copy_map}")
                show_success(self, "Done", f"Columns remapped successfully.\n{chr(10).join(parts)}")
                self.data_changed.emit() # Emit signal
            else:
                show_error(self, "Module Missing", "The required conversion module was not found.")
        except Exception as e:
            show_error(self, "Remap Failed", f"Failed to remap columns:\n{e}")

    def convert_timestamp_post_import(self):
        """Post-import: convert TIMESTAMP column to milliseconds (ms) in the
        currently previewed parquet file."""
        if not self.current_preview_path or not self.current_preview_path.lower().endswith('.parquet'):
            show_warning(self, "No Parquet File", "Please select a .parquet file in the project tree first.")
            return

        from PyQt6.QtWidgets import QLineEdit, QGroupBox, QRadioButton, QButtonGroup

        dlg = QDialog(self)
        dlg.setWindowTitle("Convert TIMESTAMP Unit")
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel(
            f"<b>File:</b> {os.path.basename(self.current_preview_path)}<br>"
            "Select the <b>current</b> unit of the TIMESTAMP column.<br>"
            "Values will be converted to <b>milliseconds (ms)</b>."
        ))

        unit_group = QGroupBox("Source unit")
        unit_layout = QVBoxLayout()

        units = [
            ("ns",  "Nanoseconds  (÷ 1,000,000)"),
            ("us",  "Microseconds (÷ 1,000)"),
            ("ms",  "Milliseconds (÷ 1)"),
            ("s",   "Seconds      (× 1,000)"),
        ]

        btn_group = QButtonGroup(dlg)
        radio_buttons = {}
        for code, label in units:
            rb = QRadioButton(label)
            btn_group.addButton(rb)
            unit_layout.addWidget(rb)
            radio_buttons[code] = rb
        radio_buttons["ms"].setChecked(True)  # default

        # Custom divisor row
        rb_custom = QRadioButton("Custom divisor:")
        btn_group.addButton(rb_custom)
        custom_row = QHBoxLayout()
        custom_row.addWidget(rb_custom)
        txt_custom = QLineEdit()
        txt_custom.setPlaceholderText("e.g. 1,000")
        txt_custom.setEnabled(False)
        custom_row.addWidget(txt_custom)
        unit_layout.addLayout(custom_row)

        rb_custom.toggled.connect(txt_custom.setEnabled)

        unit_group.setLayout(unit_layout)
        layout.addWidget(unit_group)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if not dlg.exec():
            return

        # Determine chosen unit / divisor
        source_unit = None
        for code, rb in radio_buttons.items():
            if rb.isChecked():
                source_unit = code
                break
        if rb_custom.isChecked():
            custom_val = txt_custom.text().strip()
            if not custom_val:
                show_warning(self, "Input Required", "Please enter a custom divisor number.")
                return
            source_unit = custom_val

        if not source_unit:
            return

        try:
            if convert_timestamp_in_parquet:
                ok = convert_timestamp_in_parquet(self.current_preview_path, source_unit)
                if ok:
                    self.load_preview(self.current_preview_path)
                    self.data_changed.emit()
                    show_success(self, "Conversion Complete", f"TIMESTAMP column has been converted from '{source_unit}' to milliseconds successfully.")
                else:
                    show_warning(self, "Column Not Found", "No TIMESTAMP column found in this file.")
            else:
                show_error(self, "Module Missing", "The required conversion module was not found.")
        except Exception as e:
            show_error(self, "Conversion Failed", f"Failed to convert TIMESTAMP:\n{e}")

    def closeEvent(self, event: QCloseEvent):
        for attr in ('_preview_worker', 'worker'):
            if hasattr(self, attr):
                w = getattr(self, attr)
                if w.isRunning():
                    w.wait()
