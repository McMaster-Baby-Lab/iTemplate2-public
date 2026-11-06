import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QStackedWidget, QListWidgetItem,
                             QScrollArea, QSplitter, QLabel, QToolButton,
                             QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QAction

# Initialize centralized logging (console handler set up on import)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import setup_project_file_logging

# Import our custom widgets
from gui.widgets import (ProjectManager, VideoPreprocessor, DataWorkspace, 
                     TemporalPreprocessing, CoordinateSetup, ProcessingView, VisualizationView,
                     AOIAnalysis, ExportView, AboutView, ProjectSettings, SettingsView)
from gui.widgets.button_styles import button_style
from gui.widgets.theme import (GLOBAL_QSS, SIDEBAR_STYLE, SPLITTER_STYLE,
                               set_theme, get_theme)
from gui.widgets.dialog_helpers import show_warning, show_success

# Shared sidebar stylesheet — imported from theme module

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_project_path = ""
        self._pushing_settings = False  # reentrancy guard
        self._processing_count = 0     # tracks active overlays across widgets

        self.setWindowTitle("iTemplate2 - Eye Tracking Analysis")

        # Adapt to screen size — 90% of available geometry, no hard cap
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = int(avail.width() * 0.9)
            h = int(avail.height() * 0.9)
        else:
            w, h = 1400, 900
        self.setMinimumSize(960, 600)  # scrollbars kick in below this
        self.resize(w, h)

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar — a container with main nav list + stretch + settings list
        sidebar_container = QWidget()
        # Scale sidebar width with screen size
        sidebar_w = max(180, min(280, int(w * 0.15)))
        sidebar_container.setMinimumWidth(sidebar_w)
        sidebar_container.setMaximumWidth(sidebar_w + 80)
        sidebar_container.setObjectName("sidebar-container")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Main navigation list
        self.sidebar = QListWidget()
        self.sidebar.setStyleSheet(SIDEBAR_STYLE)

        nav_items = [
            "Project Manager",
            "Face Extraction",
            "Data Review",
            "Temporal Preprocessing",
            "Coordinate Setup",
            "Transformation",
            "Visualization",
            "AOI Analysis",
            "Export",
            "About",
        ]
        
        for item_name in nav_items:
            item = QListWidgetItem(item_name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.sidebar.addItem(item)

        sidebar_layout.addWidget(self.sidebar, 1)  # stretch=1 fills available space

        # Theme toggle (pinned above Settings)
        self.theme_toggle = QToolButton()
        self.theme_toggle.setObjectName("theme-toggle")
        self.theme_toggle.setText("◑  Light" if get_theme() == "light" else "◐  Dark")
        self.theme_toggle.setToolTip("Switch between dark and light theme")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(get_theme() == "light")
        self.theme_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.theme_toggle.toggled.connect(self._on_theme_toggle)
        sidebar_layout.addWidget(self.theme_toggle)

        # Settings list (single item, pinned at bottom)
        self.sidebar_settings = QListWidget()
        self.sidebar_settings.setStyleSheet(SIDEBAR_STYLE)
        settings_item = QListWidgetItem("Settings")
        settings_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.sidebar_settings.addItem(settings_item)
        one_row_h = self.sidebar_settings.sizeHintForRow(0)
        if one_row_h > 0:
            frame_h = self.sidebar_settings.frameWidth() * 2
            self.sidebar_settings.setMaximumHeight(one_row_h + frame_h + 8)

        sidebar_layout.addWidget(self.sidebar_settings)

        # 2. Stacked Widget (Pages)
        self.pages = QStackedWidget()
        
        # Shared settings model
        self.project_settings = ProjectSettings()
        self.project_settings.settingChanged.connect(self._on_global_setting_changed)


        # Instantiate Pages
        self.project_manager = ProjectManager()
        self.video_preprocessor = VideoPreprocessor()
        self.data_workspace = DataWorkspace()
        self.temporal_preprocessing = TemporalPreprocessing()
        self.coordinate_setup = CoordinateSetup()
        self.processing_view = ProcessingView()
        self.visualization_view = VisualizationView()
        self.aoi_analysis = AOIAnalysis()
        self.export_view = ExportView()
        self.about_view = AboutView()
        self.settings_view = SettingsView()
        self.settings_view.set_settings(self.project_settings)
        self.project_manager.set_settings(self.project_settings)
        
        # Add Pages in Order — each wrapped in a scroll area so content
        # is scrollable on smaller screens.
        def _scrollable(widget):
            sa = QScrollArea()
            sa.setWidget(widget)
            sa.setWidgetResizable(True)
            sa.setFrameShape(QScrollArea.Shape.NoFrame)
            return sa

        self.pages.addWidget(_scrollable(self.project_manager))          # Index 0
        self.pages.addWidget(_scrollable(self.video_preprocessor))       # Index 1
        self.pages.addWidget(_scrollable(self.data_workspace))           # Index 2
        self.pages.addWidget(_scrollable(self.temporal_preprocessing))   # Index 3
        self.pages.addWidget(_scrollable(self.coordinate_setup))         # Index 4
        self.pages.addWidget(_scrollable(self.processing_view))          # Index 5
        self.pages.addWidget(_scrollable(self.visualization_view))       # Index 6
        self.pages.addWidget(_scrollable(self.aoi_analysis))             # Index 7
        self.pages.addWidget(_scrollable(self.export_view))              # Index 8
        self.pages.addWidget(_scrollable(self.about_view))               # Index 9
        self.pages.addWidget(_scrollable(self.settings_view))            # Index 10

        # Layout Assembly — horizontal splitter so the sidebar is resizable
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(sidebar_container)
        self.main_splitter.addWidget(self.pages)
        self.main_splitter.setStretchFactor(0, 0)  # sidebar stays compact
        self.main_splitter.setStretchFactor(1, 1)  # pages take remaining space
        self.main_splitter.setSizes([sidebar_w, w - sidebar_w])
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet(SPLITTER_STYLE)
        # Grip dots on the splitter handle
        handle = self.main_splitter.handle(1)
        self._grip_label = None
        if handle:
            handle.setCursor(Qt.CursorShape.OpenHandCursor)
            grip = QLabel("⋮")
            grip.setObjectName("grip-label")
            grip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            grip_layout = QVBoxLayout(handle)
            grip_layout.setContentsMargins(0, 0, 0, 0)
            grip_layout.addWidget(grip)
            self._grip_label = grip
        
        main_layout.addWidget(self.main_splitter)

        # Connect main sidebar to pages (indices 0-7)
        self.sidebar.itemClicked.connect(self._on_main_nav_clicked)
        
        # Connect Settings sidebar
        self.sidebar_settings.itemClicked.connect(self._on_settings_nav_clicked)
        
        # Connect Project Manager signals
        self.project_manager.project_changed.connect(self.on_project_changed)

        # Connect Settings page
        self.settings_view.settings_changed.connect(self._on_settings_changed)

        # Wire up processing overlay tracking across all widgets
        self._init_processing_tracking()
        
        # ── Cross-widget data updates ──
        self.processing_view.template_change_guard = self.visualization_view._allow_review_discard
        self.processing_view.template_selection_requested.connect(self._choose_stimulus_template)
        self.video_preprocessor.template_candidate_selected.connect(self._use_stimulus_template)
        self.video_preprocessor.template_selection_cancelled.connect(lambda: self._navigate_template_page(5))
        # Step 1 and Step 3 both edit the face contour margin; keep the two boxes in sync.
        self.video_preprocessor.spin_contour_margin.valueChanged.connect(
            self.processing_view.spin_contour_margin.setValue)
        self.processing_view.spin_contour_margin.valueChanged.connect(
            self.video_preprocessor.spin_contour_margin.setValue)

        # Connect data_changed signals to broadcast refresh
        self.project_manager.data_changed.connect(self.broadcast_data_update)
        self.temporal_preprocessing.data_changed.connect(self.broadcast_data_update)
        self.coordinate_setup.data_changed.connect(self.broadcast_data_update)
        self.processing_view.data_changed.connect(self.broadcast_data_update)
        self.aoi_analysis.data_changed.connect(self.broadcast_data_update)

        # Default Selection
        self.sidebar.setCurrentRow(0)

    def center(self):
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
        else:
            avail = self.geometry()
        frame = self.frameGeometry()
        frame.moveCenter(QPoint(avail.x() + avail.width() // 2,
                                avail.y() + avail.height() // 2))
        self.move(frame.x(), frame.y())

    def showEvent(self, event):
        """Center window on first show."""
        super().showEvent(event)
        if event.spontaneous() or not hasattr(self, '_shown'):
            self.center()
            self._shown = True

    def _on_theme_toggle(self, checked):
        """Toggle between dark and light theme."""
        new_theme = "light" if checked else "dark"
        set_theme(new_theme)
        self.project_settings.set("global", "theme", new_theme)
        self.project_settings.save()

        # Reapply app-level stylesheet (GLOBAL_QSS regenerated by set_theme)
        from gui.widgets.theme import GLOBAL_QSS, SIDEBAR_STYLE
        from gui.widgets.button_styles import button_style
        QApplication.instance().setStyleSheet(GLOBAL_QSS + button_style("neutral", padding="6px 12px"))

        # Refresh sidebar (SIDEBAR_STYLE regenerated by set_theme)
        self.sidebar.setStyleSheet(SIDEBAR_STYLE)
        self.sidebar_settings.setStyleSheet(SIDEBAR_STYLE)

        # Update toggle button label
        self.theme_toggle.setText("◑  Light" if checked else "◐  Dark")

    # ── Sidebar navigation ───────────────────────────────────────

    def _on_main_nav_clicked(self, item):
        """Main nav item clicked (pages 0-7)."""
        index = self.sidebar.row(item)
        if index < 0:
            return
        # Deselect Settings when a main nav item is selected
        self.sidebar_settings.clearSelection()
        self._switch_to_page(index)

    def _on_settings_nav_clicked(self, item):
        """Settings item clicked (page 8)."""
        index = self.sidebar_settings.row(item)
        if index < 0:
            return
        # Deselect main nav when Settings is selected
        self.sidebar.clearSelection()
        self._switch_to_page(10)  # Settings is now page 10

    def _switch_to_page(self, page_index):
        # Guard: prevent navigation while any widget is processing
        if self._processing_count > 0 and page_index != self.pages.currentIndex():
            show_warning(
                self, "Process Running",
                "A background process is still running. Please wait for it to complete before navigating."
            )
            return

        # When navigating TO Settings page, collect current widget values FIRST
        if page_index == 10:  # Settings is now page 10
            self._collect_settings_from_widgets()
            self.settings_view.refresh_from_settings()

        # Now switch the page index
        self.pages.setCurrentIndex(page_index)

    def increment_processing(self):
        """Called by a widget when it starts a long-running operation."""
        self._processing_count += 1

    def decrement_processing(self):
        """Called by a widget when a long-running operation finishes."""
        self._processing_count = max(0, self._processing_count - 1)

    def _init_processing_tracking(self):
        """Connect all widget ProcessingOverlay instances to the nav guard counter.

        Looks for `_overlay` attribute on each widget. If the overlay has
        `processing_started` and `processing_finished` signals, connects them
        to increment/decrement the processing counter.
        """
        for widget in [self.video_preprocessor, self.data_workspace,
                       self.temporal_preprocessing, self.coordinate_setup,
                       self.processing_view, self.visualization_view,
                       self.aoi_analysis, self.export_view, self.project_manager]:
            overlay = getattr(widget, '_overlay', None)
            if overlay is None:
                continue
            if hasattr(overlay, 'processing_started'):
                overlay.processing_started.connect(self.increment_processing)
            if hasattr(overlay, 'processing_finished'):
                overlay.processing_finished.connect(self.decrement_processing)

    # ── Project switching ─────────────────────────────────────────

    def on_project_changed(self, path):
        """Propagate the active project path to other widgets."""
        self.setWindowTitle(f"iTemplate2 - {path}")
        self.current_project_path = path

        # Redirect file logging to the new project directory
        setup_project_file_logging(path)

        # Load settings from project folder
        self.project_settings.load(path)

        # Restore theme preference
        saved_theme = self.project_settings.get("global", "theme", get_theme())
        if saved_theme != get_theme():
            set_theme(saved_theme)
            self.theme_toggle.setChecked(saved_theme == "light")
            from gui.widgets.theme import GLOBAL_QSS, SIDEBAR_STYLE
            from gui.widgets.button_styles import button_style
            QApplication.instance().setStyleSheet(GLOBAL_QSS + button_style("neutral", padding="6px 12px"))
            self.sidebar.setStyleSheet(SIDEBAR_STYLE)
            self.sidebar_settings.setStyleSheet(SIDEBAR_STYLE)

        # Disable auto-save while initializing widgets to prevent
        # _on_settings_edited from writing empty/default values to disk
        was_auto = self.project_settings.auto_save
        self.project_settings.auto_save = False
        
        try:
            # Set project path on ALL widgets
            widgets = [self.video_preprocessor, self.data_workspace,
                       self.temporal_preprocessing, self.coordinate_setup,
                       self.processing_view, self.visualization_view,
                       self.aoi_analysis, self.export_view, self.settings_view]
            for widget in widgets:
                if hasattr(widget, 'set_project_path'):
                    widget.set_project_path(path)
                    QApplication.processEvents()
        finally:
            self.project_settings.auto_save = was_auto
        
        # Now push saved settings to all widgets (restores user selections)
        self._push_settings_to_widgets()

    def _navigate_template_page(self,index):
        self.sidebar.setCurrentRow(index)
        self._switch_to_page(index)

    def _choose_stimulus_template(self):
        self.video_preprocessor.begin_template_selection()
        self._navigate_template_page(1)

    def _use_stimulus_template(self, image, faces, provenance):
        if self.processing_view.accept_template_candidate(image,faces,provenance):
            self.video_preprocessor.end_template_selection()
            self._navigate_template_page(5)

    def broadcast_data_update(self):
        """Called when any widget reports a data change.
        
        Lightweight: just persists current widget settings to disk so they
        stay consistent. Heavy re-initialisation (re-reading parquet files,
        re-scanning stimuli, etc.) is only done in on_project_changed().
        """
        self._collect_settings_from_widgets()

    # ── Settings helpers ─────────────────────────────────────────

    def _push_settings_to_widgets(self):
        """Push ProjectSettings values into individual widget controls."""
        if self._pushing_settings:
            return  # prevent reentrancy
        self._pushing_settings = True
        
        # Disable auto-save during push so that load_settings changing
        # widget values doesn't trigger _on_settings_edited → save_settings
        was_auto = self.project_settings.auto_save
        self.project_settings.auto_save = False
        
        try:
            s = self.project_settings
            widgets = [self.project_manager, self.video_preprocessor,
                       self.temporal_preprocessing, self.coordinate_setup,
                       self.processing_view, self.visualization_view,
                       self.aoi_analysis, self.data_workspace, self.export_view]
            for widget in widgets:
                if hasattr(widget, 'load_settings'):
                    widget.load_settings(s)
                    QApplication.processEvents()
        finally:
            self.project_settings.auto_save = was_auto
            self._pushing_settings = False

    def _collect_settings_from_widgets(self):
        """Read current widget control values back into ProjectSettings."""
        s = self.project_settings
        
        # Block signals to prevent cascading updates during batch save
        s.blockSignals(True)
        was_auto = s.auto_save
        s.auto_save = False
        try:
            for widget in [self.video_preprocessor, self.temporal_preprocessing,
                           self.coordinate_setup, self.processing_view,
                           self.visualization_view, self.aoi_analysis]:
                if hasattr(widget, 'save_settings'):
                    widget.save_settings(s)
        finally:
            s.auto_save = was_auto
            s.blockSignals(False)
            # Force a single save if auto-save was enabled
            if was_auto:
                s.save()

    def _on_settings_changed(self):
        """Called when the user saves via the Settings page."""
        self._push_settings_to_widgets()

    def _on_global_setting_changed(self, section, key, value):
        """Handle cross-widget setting changes. Only propagate settings that
        affect other widgets (e.g. selected data file). Avoid full re-push
        on every change to prevent UI freezes from heavy I/O."""
        if self._pushing_settings:
            return  # avoid cascading during push
        
        # Only propagate cross-widget settings that other widgets care about
        if section == "global" and key == "selected_data_file":
            # Sync selected data file across all widgets
            for widget in [self.temporal_preprocessing, self.coordinate_setup,
                           self.processing_view, self.data_workspace]:
                if hasattr(widget, '_set_gaze_data_file'):
                    widget._set_gaze_data_file(value, update_settings=False)
                # Update the label showing which file is selected
                if hasattr(widget, 'lbl_global_data'):
                    name = os.path.basename(value) if value else "None"
                    widget.lbl_global_data.setText(f"Selected data file: {name}")
        
        if section == "global" and key == "selected_transformed_file":
            for widget in [self.visualization_view, self.aoi_analysis, self.export_view]:
                if hasattr(widget, '_set_transformed_data_file'):
                    widget._set_transformed_data_file(value, update_settings=False)
                if hasattr(widget, 'lbl_global_data'):
                    name = os.path.basename(value) if value else "None"
                    widget.lbl_global_data.setText(f"Selected data file: {name}")
        
        # Notify user if video delay changed, reminding them to apply it
        elif section == "coordinate_setup" and key == "gaze_response_lag":
            show_success(
                self,
                "Gaze Response Lag Changed",
                "You have modified the Gaze Response Lag.\n\n"
                "Please go to the Coordinate Setup page and run 'Assign Frame Index' again "
                "to apply this new value to your data."
            )

    def closeEvent(self, event):
        # The visualization page is embedded, so it does not receive a close
        # event when the main window closes.
        if not self.visualization_view._allow_review_discard():
            event.ignore()
            return
        self.visualization_view._stop_overlap_worker()
        # Embedded widgets do not receive their own close events.
        for worker in (getattr(self.processing_view,'_worker',None),
                       getattr(self.processing_view,'_template_detection',None)):
            if worker is not None and worker.isRunning(): worker.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    # --- macOS dock icon & app name fix ---
    _gui_dir = os.path.dirname(os.path.abspath(__file__))
    _logo_path = os.path.join(_gui_dir, "resources", "logo.png")
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = "iTemplate2"
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("iTemplate2")
    app.setApplicationDisplayName("iTemplate2")
    
    # Set window icon (title bar) and macOS dock icon
    if os.path.isfile(_logo_path):
        app.setWindowIcon(QIcon(_logo_path))
        try:
            from AppKit import NSApplication, NSImage
            ns_app = NSApplication.sharedApplication()
            ns_image = NSImage.alloc().initWithContentsOfFile_(_logo_path)
            if ns_image:
                ns_app.setApplicationIconImage_(ns_image)
        except Exception:
            pass
    
    # Global Style (Dark Mode)
    app.setStyle("Fusion")
    app.setStyleSheet(
        GLOBAL_QSS
        + button_style("neutral", padding="6px 12px")
    )
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
