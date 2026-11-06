"""
Centralized project settings manager.
Reads / writes a ``project_settings.json`` file in each project folder so
that every widget can persist its configuration between sessions.
"""
import copy
import json
import logging
import os
import multiprocessing
from PyQt6.QtCore import QObject, pyqtSignal
from core.data_resolver import (
    build_expected_transformed_path,
    is_imported_data_file,
    is_transformed_data_file,
    resolve_existing_transformed_file,
    resolve_latest_imported_parquet,
)

logger = logging.getLogger('iTemplate2')

# ── Default schema ────────────────────────────────────────────────────
_DEFAULTS = {
    "global": {
        "source_data_file": "",
        "selected_data_file": "",
        "selected_transformed_file": "",
    },
    "video_preprocessor": {
        "parallel_workers": multiprocessing.cpu_count(),
        "selected_face_ids": None,
        "face_labels": {},
    },
    "temporal_preprocessing": {
        "primary_temporal_col": "",
        "secondary_temporal_col": "",
        "approach": "onset_offset",
        "onset_marker_col": "",
        "onset_markers": [],
        "offset_mode": "marker",          # "marker" | "duration"
        "offset_marker_col": "",
        "offset_markers": [],
        "offset_duration": "",
        "offset_duration_unit": "seconds (s)",
        "kw_include_col": "",
        "kw_include_values": [],
        "kw_exclude_col": "",
        "kw_exclude_values": [],
    },
    "coordinate_setup": {
        "screen_width": "",
        "screen_height": "",
        "stimuli_centered": True,
        "center_x": "",
        "center_y": "",
        "frame_rate_fps": "",
        "gaze_response_lag": "0",
        "stimuli_dir": "",
    },
    "processing": {
        "stimuli_folder": "",
        "landmark_folder": "",
        "active_template_id": "default",
        "last_successful_template_id": "default",
        "template_processing_status": "ready",
        "stimuli_type": "video",           # "video" | "image"
        "parallel_workers": multiprocessing.cpu_count(),
        "selected_face_ids": None,
        "max_gap_seconds": 2.0,
        "contour_margin_pct": 0,
        "default_fps": 30.0,
    },
    "visualization": {
        "bins": "",
        "sigma": "",
        "show_scatter": False,
        "normalize_by_participant": True,
    },
}

SETTINGS_FILENAME = "project_settings.json"


class ProjectSettings(QObject):
    """Read / write project-level settings stored as JSON."""
    
    settingChanged = pyqtSignal(str, str, object)  # section, key, value

    def __init__(self, auto_save=True):
        super().__init__()
        self._data: dict = {}
        self._project_path: str = ""
        self.auto_save = auto_save
        self._reset_to_defaults()

    # ── public API ─────────────────────────────────────────────────────

    def load(self, project_path: str):
        """Load settings from *project_path*/project_settings.json.
        
        Missing keys are filled with defaults so new settings are always
        available even for older project folders.
        """
        self._project_path = project_path
        self._reset_to_defaults()

        path = os.path.join(project_path, SETTINGS_FILENAME)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    on_disk = json.load(f)
                # Merge on-disk values into defaults (respecting new keys)
                for section, defaults in _DEFAULTS.items():
                    if section in on_disk and isinstance(on_disk[section], dict):
                        for key, default_val in defaults.items():
                            if key in on_disk[section]:
                                self._data[section][key] = on_disk[section][key]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load project settings from %s: %s (using defaults)", path, e)

    def save(self, project_path: str = None):
        """Write current settings to *project_path*/project_settings.json."""
        path = project_path or self._project_path
        if not path:
            return
        self._project_path = path
        dest = os.path.join(path, SETTINGS_FILENAME)
        try:
            with open(dest, "w", encoding="utf-8") as f:
                stored = copy.deepcopy(self._data)
                for key in ('source_data_file','selected_data_file','selected_transformed_file'):
                    value = stored.get('global',{}).get(key)
                    if value and os.path.isabs(value):
                        try:
                            if os.path.commonpath([os.path.abspath(value),os.path.abspath(path)]) == os.path.abspath(path):
                                stored['global'][key] = os.path.relpath(value,path)
                        except ValueError: pass
                json.dump(stored, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("Failed to save project settings to %s: %s", dest, e)

    def get(self, section: str, key: str, default=None):
        """Return a single value (as a deepcopy to prevent in-place mutation side effects)."""
        val = self._data.get(section, {}).get(key, default)
        try:
            return copy.deepcopy(val)
        except (TypeError, ValueError):
            return val

    def get_existing_path(self, section: str, key: str, default="") -> str:
        """Return a stored path only if it exists on disk."""
        path = self.get(section, key, default)
        if isinstance(path,str) and path and not os.path.isabs(path):
            path = os.path.join(self._project_path,path)
        if isinstance(path, str) and path and os.path.isfile(path):
            return path
        return default

    def get_source_data_file(self, default="") -> str:
        """Return the dataset anchor file, if available."""
        path = self.get_existing_path("global", "source_data_file", default)
        if path and self._project_path and not is_imported_data_file(self._project_path, path):
            return default
        return path

    def get_working_data_file(self, default="") -> str:
        """Return the current preprocessing input file, if available."""
        return self.get_existing_path("global", "selected_data_file", default)

    def get_transformed_data_file(self, default="") -> str:
        """Return the current downstream transformed file, if available."""
        return self.get_existing_path("global", "selected_transformed_file", default)

    def resolve_source_data_file(self, default="") -> str:
        """Resolve the dataset anchor file, falling back to latest imported parquet."""
        source = self.get_source_data_file("")
        if source:
            return source
        latest = resolve_latest_imported_parquet(self._project_path)
        return latest or default

    def resolve_working_data_file(self, default="") -> str:
        """Resolve the default file for preprocessing steps."""
        working = self.get_working_data_file("")
        if working and self._project_path and not is_transformed_data_file(self._project_path, working):
            return working
        return self.resolve_source_data_file(default)

    def resolve_transformed_data_file(self, default="") -> str:
        """Resolve the default file for downstream transformed-data steps."""
        transformed = self.get_transformed_data_file("")
        if transformed:
            return transformed
        source = self.resolve_source_data_file("")
        if not source:
            return default
        resolved = resolve_existing_transformed_file(self._project_path, source)
        return resolved or default

    def expected_transformed_data_file(self, default="") -> str:
        """Return the exact transformed path expected from the current source file."""
        source = self.resolve_source_data_file("")
        if not source:
            return default
        expected = build_expected_transformed_path(self._project_path, source)
        return expected or default

    def set(self, section: str, key: str, value):
        """Set a single value and auto-save if enabled."""
        if section not in self._data:
            self._data[section] = {}

        # Deep-copy compare so in-place mutations (caller gets ref from get(),
        # mutates it, then passes it back) are still detected as changes.
        current = self._data[section].get(key)
        try:
            if copy.deepcopy(current) == copy.deepcopy(value):
                return
        except (TypeError, ValueError):
            pass  # un-copyable types — just proceed to set

        self._data[section][key] = value
        
        # Emit signal so other widgets update immediately
        self.settingChanged.emit(section, key, value)
        
        if self.auto_save:
            self.save()

    @property
    def data(self) -> dict:
        return self._data

    @property
    def project_path(self) -> str:
        return self._project_path

    # ── internal ───────────────────────────────────────────────────────

    def _reset_to_defaults(self):
        import copy
        self._data = copy.deepcopy(_DEFAULTS)
