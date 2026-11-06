"""
Dual-theme (dark/light) palette and QSS for the iTemplate2 GUI.

All widget styles should reference this module rather than hard-coding colors.
"""

import os as _os
from typing import Callable

# ── Dark theme (warm, slightly blue-tinted darks) ──────────────────────────

_THEME_DARK = {
    "bg_primary": "#1e2024",
    "bg_secondary": "#252830",
    "bg_tertiary": "#2c2f38",
    "text_primary": "#e0e0e0",
    "text_secondary": "#a0a4aa",
    "text_accent": "#3d7e9a",
    "border": "#3d3d3d",
    "border_light": "#4d4d4d",
    "notice_bg": "#1a2a3a",
    "notice_text": "#5a9aba",
    "notice_border": "#2a4a6a",
    "scrollbar_bg": "#2c2f38",
    "scrollbar_thumb": "#555555",
    "log_bg": "#16181c",
    "log_text": "#8f9f8f",
    "data_file_text": "#4a9a6a",
    "gaze_green": "#4aaa6a",
    "gaze_red": "#cc6666",
    "status_success": "#5a9a7a",
    "status_error": "#d4787e",
    "status_warning": "#c4a544",
    "status_info": "#5a9aba",
    "status_icon_success": "✓",
    "status_icon_error": "✗",
    "status_icon_warning": "⚠",
    "status_icon_info": "ℹ",
    "sidebar_bg": "#252830",
    "sidebar_selected": "#3d7e9a",
    "sidebar_hover": "#3d3d3d",
    "splitter_handle": "#3d3d3d",
    "splitter_hover": "#5a5a5a",
    "grip_text": "#888888",
    "hl_participant": "#3a2d5f",
    "hl_x": "#2d3d5f",
    "hl_y": "#2d5f3d",
    "hl_stimulus": "#5f4a2d",
    "hl_trial": "#5f5f2d",
    "hl_stim_x": "#2a4a6f",
    "hl_stim_y": "#2a6f4a",
    "hl_trans_x": "#5f2d2d",
    "hl_trans_y": "#5f2d4a",
    "hl_frame": "#2d5f5f",
    "hl_duration": "#5f4a20",
    "aoi_lightness": 30,
    "checkbox_hover_bg": "#3a3d45",
    "combo_arrow_color": "#cccccc",
    "combo_arrow_hover_color": "#ffffff",
    "checkmark_color": "#ffffff",
    "radio_center": "#e5e7eb",
}

# ── Light theme (modern, clean, professional) ──────────────────────────────

_THEME_LIGHT = {
    "bg_primary": "#f8f9fc",
    "bg_secondary": "#ffffff",
    "bg_tertiary": "#e8ecf1",
    "text_primary": "#1a1d23",
    "text_secondary": "#6b7280",
    "text_accent": "#2563eb",
    "border": "#d1d5db",
    "border_light": "#e2e5ea",
    "notice_bg": "#eff6ff",
    "notice_text": "#1d4ed8",
    "notice_border": "#bfdbfe",
    "scrollbar_bg": "#e8ecf1",
    "scrollbar_thumb": "#9ca3af",
    "log_bg": "#f3f4f6",
    "log_text": "#374151",
    "data_file_text": "#059669",
    "gaze_green": "#059669",
    "gaze_red": "#dc2626",
    "status_success": "#059669",
    "status_error": "#dc2626",
    "status_warning": "#d97706",
    "status_info": "#2563eb",
    "status_icon_success": "✓",
    "status_icon_error": "✗",
    "status_icon_warning": "⚠",
    "status_icon_info": "ℹ",
    "sidebar_bg": "#ffffff",
    "sidebar_selected": "#2563eb",
    "sidebar_hover": "#e5e7eb",
    "splitter_handle": "#d1d5db",
    "splitter_hover": "#9ca3af",
    "grip_text": "#6b7280",
    "hl_participant": "#e9d5ff",
    "hl_x": "#dbeafe",
    "hl_y": "#d1fae5",
    "hl_stimulus": "#fef3c7",
    "hl_trial": "#fde68a",
    "hl_stim_x": "#bfdbfe",
    "hl_stim_y": "#a7f3d0",
    "hl_trans_x": "#fecaca",
    "hl_trans_y": "#fbcfe8",
    "hl_frame": "#ccfbf1",
    "hl_duration": "#fde68a",
    "aoi_lightness": 80,
    "checkbox_hover_bg": "#d1d5db",
    "combo_arrow_color": "#374151",
    "combo_arrow_hover_color": "#ffffff",
    "checkmark_color": "#ffffff",
    "radio_center": "#ffffff",
}

# ── Active theme ───────────────────────────────────────────────────────────

_current_theme = "dark"
_theme_map = {"dark": _THEME_DARK, "light": _THEME_LIGHT}
THEME = _THEME_LIGHT.copy()
_THEME = THEME  # backward-compat alias (tracks active theme)
_theme_callbacks: list[Callable[[], None]] = []

# ── Icon asset paths ───────────────────────────────────────────────────────

_gui_assets = _os.path.join(_os.path.dirname(__file__), "..", "assets")

_ICON_PATHS_DARK = {
    "down_arrow": _os.path.join(_gui_assets, "down-arrow.svg").replace("\\", "/"),
    "up_arrow": _os.path.join(_gui_assets, "up-arrow.svg").replace("\\", "/"),
    "checkmark": _os.path.join(_gui_assets, "checkmark.svg").replace("\\", "/"),
}

_ICON_PATHS_LIGHT = {
    "down_arrow": _os.path.join(_gui_assets, "down-arrow-light.svg").replace("\\", "/"),
    "up_arrow": _os.path.join(_gui_assets, "up-arrow-light.svg").replace("\\", "/"),
    "checkmark": _os.path.join(_gui_assets, "checkmark-light.svg").replace("\\", "/"),
}

_icon_paths = _ICON_PATHS_LIGHT


# ── Re-exported constants (mirror active theme) ────────────────────────────

bg_primary = THEME["bg_primary"]
bg_secondary = THEME["bg_secondary"]
bg_tertiary = THEME["bg_tertiary"]
text_primary = THEME["text_primary"]
text_secondary = THEME["text_secondary"]
text_accent = THEME["text_accent"]
border = THEME["border"]
border_light = THEME["border_light"]
notice_bg = THEME["notice_bg"]
notice_text = THEME["notice_text"]
notice_border = THEME["notice_border"]
log_bg = THEME["log_bg"]
log_text = THEME["log_text"]
data_file_text = THEME["data_file_text"]
gaze_green = THEME["gaze_green"]
gaze_red = THEME["gaze_red"]
status_success = THEME["status_success"]
status_error = THEME["status_error"]
status_warning = THEME["status_warning"]
status_info = THEME["status_info"]
status_icon_success = THEME["status_icon_success"]
status_icon_error = THEME["status_icon_error"]
status_icon_warning = THEME["status_icon_warning"]
status_icon_info = THEME["status_icon_info"]
sidebar_bg = THEME["sidebar_bg"]
sidebar_selected = THEME["sidebar_selected"]
sidebar_hover = THEME["sidebar_hover"]
splitter_handle = THEME["splitter_handle"]
splitter_hover = THEME["splitter_hover"]
grip_text = THEME["grip_text"]


# ── Face color palette (BGR tuples, matches VideoPreprocessor) ─────────────

FACE_COLORS = {
    1: (75, 180, 60),
    2: (60, 120, 220),
    3: (220, 60, 220),
    4: (220, 180, 0),
    5: (60, 220, 220),
}


def face_color_hex(face_id: int) -> str:
    bgr = FACE_COLORS.get(face_id, (200, 200, 200))
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def face_color_bgr(face_id: int) -> tuple:
    return FACE_COLORS.get(face_id, (200, 200, 200))


# Gaze-region colors used when off-face gaze is highlighted (BGR, like FACE_COLORS).
GAZE_OFF_FACE_BGR = (255, 0, 255)    # magenta
GAZE_UNMAPPED_BGR = (160, 160, 160)  # gray


def bgr_to_hex(bgr: tuple) -> str:
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


# ── Style generation ───────────────────────────────────────────────────────

def _t():
    return THEME


def _ic():
    return _icon_paths


def generate_global_qss() -> str:
    t = _t()
    ic = _ic()
    return f"""
QWidget {{
    background-color: {t['bg_primary']};
    color: {t['text_primary']};
    font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

QScrollArea {{
    border: none;
    background-color: {t['bg_primary']};
}}

QScrollArea > QWidget {{
    background-color: {t['bg_primary']};
}}

QGroupBox {{
    font-size: 14px;
    font-weight: bold;
    border: 1px solid {t['border']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
}}

QGroupBox::title {{
    color: {t['text_primary']};
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}

QLabel {{
    color: {t['text_primary']};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
    padding: 3px 6px;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {t['text_accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
    background-color: {t['border_light']};
}}

QComboBox::drop-down:hover {{
    background-color: {t['text_accent']};
}}

QComboBox::down-arrow {{
    width: 16px;
    height: 16px;
    image: url({ic['down_arrow']});
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {t['border_light']};
    border-bottom: 0.5px solid {t['border_light']};
    background-color: {t['bg_secondary']};
    border-top-right-radius: 3px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background-color: {t['text_accent']};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({ic['up_arrow']});
    width: 10px;
    height: 10px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {t['border_light']};
    border-top: 0.5px solid {t['border_light']};
    background-color: {t['bg_secondary']};
    border-bottom-right-radius: 3px;
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {t['text_accent']};
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({ic['down_arrow']});
    width: 10px;
    height: 10px;
}}

QTabWidget::pane {{
    border: 1px solid {t['border']};
    border-radius: 4px;
    top: -1px;
    background-color: {t['bg_primary']};
}}

QTabBar::tab {{
    background: {t['bg_tertiary']};
    color: {t['text_secondary']};
    border: 1px solid {t['border_light']};
    padding: 4px 8px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background: {t['bg_primary']};
    color: {t['text_primary']};
    border-bottom-color: {t['bg_primary']};
    font-weight: bold;
}}

QTabBar::tab:hover:!selected {{
    background: {t['bg_secondary']};
}}

QHeaderView::section {{
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border']};
    padding: 4px 8px;
    font-weight: bold;
}}

QTableWidget, QTableView {{
    gridline-color: {t['border']};
    alternate-background-color: {t['bg_secondary']};
}}

QTableWidget::item, QTableView::item {{
    color: {t['text_primary']};
}}

QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {t['text_accent']};
    color: white;
}}

QListWidget {{
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {t['text_accent']};
    color: white;
}}

QListWidget::indicator {{
    width: 16px;
    height: 16px;
}}

QListWidget::indicator:unchecked {{
    border: 1.5px solid {t['grip_text']};
    background: {t['bg_tertiary']};
    border-radius: 3px;
}}

QListWidget::indicator:unchecked:hover {{
    border: 1.5px solid {t['text_secondary']};
    background: {t['checkbox_hover_bg']};
}}

QListWidget::indicator:checked {{
    border: 1.5px solid {t['text_accent']};
    background: {t['text_accent']};
    border-radius: 3px;
    image: url({ic['checkmark']});
}}

QTableWidget::indicator:unchecked, QTreeView::indicator:unchecked, QGroupBox::indicator:unchecked {{
    border: 1.5px solid {t['grip_text']};
    background: {t['bg_tertiary']};
    border-radius: 3px;
    width: 16px;
    height: 16px;
}}

QTableWidget::indicator:checked, QTreeView::indicator:checked, QGroupBox::indicator:checked {{
    border: 1.5px solid {t['text_accent']};
    background: {t['text_accent']};
    border-radius: 3px;
    image: url({ic['checkmark']});
    width: 16px;
    height: 16px;
}}

QScrollBar:vertical {{
    background-color: {t['scrollbar_bg']};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {t['scrollbar_thumb']};
    border-radius: 5px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {t['splitter_hover']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: {t['scrollbar_bg']};
    height: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {t['scrollbar_thumb']};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {t['splitter_hover']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

QSlider::groove:horizontal {{
    background: {t['bg_tertiary']};
    border: 1px solid {t['border']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::add-page:horizontal {{
    background: {t['border']};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {t['text_accent']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {t['text_primary']};
    border: 1px solid {t['border_light']};
    width: 12px;
    height: 12px;
    margin: -3px 0;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background: white;
    border: 1px solid {t['text_accent']};
}}

QSlider::groove:vertical {{
    background: {t['bg_tertiary']};
    border: 1px solid {t['border']};
    width: 6px;
    border-radius: 3px;
}}

QSlider::add-page:vertical {{
    background: {t['border']};
    border-radius: 3px;
}}

QSlider::sub-page:vertical {{
    background: {t['text_accent']};
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background: {t['text_primary']};
    border: 1px solid {t['border_light']};
    width: 12px;
    height: 12px;
    margin: 0 -3px;
    border-radius: 6px;
}}

QSlider::handle:vertical:hover {{
    background: white;
    border: 1px solid {t['text_accent']};
}}

QRadioButton, QCheckBox {{
    color: {t['text_primary']};
    spacing: 8px;
}}

QRadioButton::indicator, QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}

QCheckBox::indicator:unchecked {{
    border: 1.5px solid {t['grip_text']};
    background: {t['bg_tertiary']};
    border-radius: 3px;
}}

QCheckBox::indicator:unchecked:hover {{
    border: 1.5px solid {t['text_secondary']};
    background: {t['checkbox_hover_bg']};
}}

QCheckBox::indicator:checked {{
    border: 1.5px solid {t['text_accent']};
    background: {t['text_accent']};
    border-radius: 3px;
    image: url({ic['checkmark']});
}}

QRadioButton::indicator:unchecked {{
    width: 18px;
    height: 18px;
    border: 2px solid {t['grip_text']};
    background: {t['bg_tertiary']};
    border-radius: 9px;
}}

QRadioButton::indicator:unchecked:hover {{
    border: 2px solid {t['text_secondary']};
    background: {t['checkbox_hover_bg']};
}}

QRadioButton::indicator:checked {{
    width: 18px;
    height: 18px;
    border: 2px solid {t['text_accent']};
    background: qradialgradient(
        cx: 0.5, cy: 0.5, radius: 0.5,
        fx: 0.5, fy: 0.5,
        stop: 0 {t['text_accent']},
        stop: 0.34 {t['text_accent']},
        stop: 0.36 {t['radio_center']},
        stop: 1 {t['radio_center']}
    );
    border-radius: 9px;
}}

QMenu {{
    background-color: {t['bg_secondary']};
    border: 1px solid {t['border_light']};
}}

QMenu::item {{
    padding: 6px 24px 6px 12px;
}}

QMenu::item:selected {{
    background-color: {t['text_accent']};
    color: white;
}}

QMenu::separator {{
    background-color: {t['border']};
    height: 1px;
    margin: 4px 0;
}}

QDialog {{
    background-color: {t['bg_secondary']};
    min-width: 320px;
}}

QMessageBox {{
    background-color: {t['bg_secondary']};
    min-width: 320px;
}}

QMessageBox QLabel {{
    color: {t['text_primary']};
}}

QProgressBar {{
    background-color: {t['bg_tertiary']};
    border: 1px solid {t['border_light']};
    border-radius: 4px;
    text-align: center;
    color: {t['text_primary']};
    height: 16px;
}}

QProgressBar::chunk {{
    background-color: {t['text_accent']};
    border-radius: 3px;
}}

QSplitter::handle {{
    background-color: {t['border']};
}}

QSplitter::handle:hover {{
    background-color: {t['border_light']};
}}

QTextEdit {{
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
    padding: 2px;
}}

QToolButton {{
    background-color: {t['bg_tertiary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
    padding: 2px;
}}

/* ── Named widget styles (object-name-based) ───────────────────── */

#title-label {{
    font-size: 14px;
    font-weight: bold;
    margin-top: 4px;
    margin-bottom: 2px;
}}

#notice-label {{
    background-color: {t['notice_bg']};
    color: {t['notice_text']};
    padding: 8px;
    border: 1px solid {t['notice_border']};
    border-radius: 4px;
    font-weight: bold;
}}

#data-file-label {{
    color: {t['data_file_text']};
    font-style: italic;
}}

#description-label {{
    color: {t['text_secondary']};
    font-size: 11px;
    padding: 4px 0px;
}}

#hint-label {{
    color: {t['text_secondary']};
    font-size: 11px;
}}

#fine-label {{
    color: {t['text_secondary']};
    font-size: 10px;
}}

#fine-label-error {{
    color: {t['status_error']};
    font-size: 10px;
    margin-bottom: 4px;
}}

#status-success-label {{
    color: {t['status_success']};
    font-weight: bold;
}}

#status-error-label {{
    color: {t['status_error']};
    font-weight: bold;
}}

#status-warning-label {{
    color: {t['status_warning']};
    font-style: italic;
}}

#info-banner {{
    background-color: {t['bg_secondary']};
    color: {t['text_secondary']};
    padding: 8px;
    border: 1px solid {t['border']};
    border-radius: 4px;
}}

#panel-title-blue {{
    color: {t['status_info']};
    font-weight: bold;
    font-size: 13px;
}}

#panel-title-warning {{
    color: {t['status_warning']};
    font-weight: bold;
    font-size: 13px;
}}

#section-label {{
    color: {t['text_accent']};
    font-weight: bold;
}}

#face-info-label {{
    color: {t['text_accent']};
    font-weight: bold;
}}

#face-info-error-label {{
    color: {t['gaze_red']};
    font-weight: bold;
}}

#log-console {{
    background-color: {t['log_bg']};
    color: {t['log_text']};
    font-family: Menlo, Consolas, monospace;
    font-size: 12px;
}}

#preview-stage {{
    background-color: {t['log_bg']};
    border: 1px solid {t['border_light']};
    border-radius: 4px;
}}

#image-panel {{
    background-color: {t['log_bg']};
    border: 1px solid {t['border_light']};
}}

#bg-primary-panel {{
    background-color: {t['bg_primary']};
}}

#toggle-panel-btn {{
    text-align: left;
    padding: 5px 10px;
    font-weight: bold;
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border-radius: 4px;
    border: 1px solid {t['border_light']};
}}

#warning-combo {{
    font-weight: bold;
    color: {t['status_error']};
}}

QTableWidget#section-table {{
    font-size: 11px;
    gridline-color: {t['border']};
}}

QListWidget#exclude-list {{
    border: 2px solid {t['status_error']};
}}

#grip-label {{
    color: {t['grip_text']};
    font-weight: bold;
    background: transparent;
    padding: 0;
    margin: 0;
}}

#preview-text {{
    color: {t['text_secondary']};
    font-style: italic;
}}

#about-role-label {{
    color: {t['text_secondary']};
    font-size: 11px;
    font-weight: bold;
}}

#about-separator {{
    background-color: {t['border']};
}}

#theme-toggle {{
    background-color: transparent;
    border: none;
    padding: 8px 12px;
    font-size: 12px;
    text-align: left;
    color: {t['grip_text']};
}}

#theme-toggle:hover {{
    background-color: {t['sidebar_hover']};
}}

#theme-toggle:checked {{
    color: {t['text_accent']};
}}

#sidebar-container {{
    background-color: {t['sidebar_bg']};
}}

QLineEdit#face-label-combo {{
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-right: none;
    border-radius: 3px 0 0 3px;
    padding: 4px 6px;
    font-size: 12px;
    selection-background-color: {t['text_accent']};
}}

QPushButton#face-label-dropdown {{
    background-color: {t['bg_tertiary']};
    color: {t['text_secondary']};
    border: 1px solid {t['border_light']};
    border-left: none;
    border-radius: 0 3px 3px 0;
    font-size: 11px;
    padding-bottom: 1px;
}}

QPushButton#face-label-dropdown:hover {{
    background-color: {t['bg_secondary']};
    color: {t['text_accent']};
    border-color: {t['border']};
}}

QPushButton#face-label-dropdown:pressed {{
    background-color: {t['bg_tertiary']};
    color: {t['text_accent']};
}}

QPushButton#face-label-dropdown::menu-indicator {{
    image: none;
}}

QPushButton#face-label-ok-btn {{
    padding: 4px 12px;
    background-color: {t['text_accent']};
    color: #ffffff;
    border: 1px solid {t['notice_border']};
    border-radius: 3px;
}}

QPushButton#face-label-ok-btn:hover {{
    background-color: {t['notice_text']};
}}

QPushButton#face-label-clear-btn {{
    padding: 4px 12px;
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
}}

QPushButton#face-label-clear-btn:hover {{
    background-color: {t['bg_secondary']};
}}

QPushButton#label-picker-btn {{
    font-size: 11px;
    padding: 2px 4px;
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
}}

QPushButton#label-picker-btn:hover {{
    background-color: {t['border']};
}}
"""


def generate_sidebar_style() -> str:
    t = _t()
    return f"""
QListWidget {{
    background-color: {t['sidebar_bg']};
    color: {t['text_primary']};
    border: none;
    outline: none;
    font-size: 14px;
}}
QListWidget::item {{
    padding: 15px;
    border-bottom: 1px solid {t['border']};
}}
QListWidget::item:selected {{
    background-color: {t['sidebar_selected']};
    color: white;
    font-weight: bold;
}}
QListWidget::item:hover {{
    background-color: {t['sidebar_hover']};
    color: {t['text_primary']};
}}
QListWidget::item:selected:hover {{
    background-color: {t['sidebar_selected']};
    color: white;
}}
"""


def generate_splitter_style() -> str:
    t = _t()
    return f"""
QSplitter::handle:horizontal {{
    background-color: {t['splitter_handle']};
    width: 6px;
}}
QSplitter::handle:horizontal:hover {{
    background-color: {t['splitter_hover']};
}}
"""


def generate_combobox_style() -> str:
    t = _t()
    ic = _ic()
    return f"""
QComboBox {{
    combobox-popup: 0;
    background-color: {t['bg_tertiary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 3px;
    padding: 3px 6px;
    font-size: 11px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    width: 16px;
    height: 16px;
    image: url({ic['down_arrow']});
}}
"""


def generate_combobox_popup_style() -> str:
    t = _t()
    ic = _ic()
    return f"""
QListView {{
    background-color: {t['bg_secondary']};
    color: {t['text_primary']};
    border: 1px solid {t['border_light']};
    border-radius: 4px;
    outline: none;
}}
QListView::item:selected {{
    background-color: {t['text_accent']};
    color: white;
}}
QListView::item:hover {{
    background-color: {t['bg_tertiary']};
}}
QComboBox QAbstractItemView {{
    background-color: {t['bg_tertiary']};
    border: 1px solid {t['border_light']};
    border-radius: 4px;
    padding: 4px;
    outline: 0;
}}
QComboBox QAbstractItemView::item {{
    min-height: 28px;
    padding-left: 8px;
    border-radius: 2px;
    margin: 1px 2px;
    color: {t['text_primary']};
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {t['text_accent']};
    color: white;
}}
QComboBox::down-arrow {{
    width: 16px;
    height: 16px;
    image: url({ic['down_arrow']});
}}
"""


def generate_notice_style() -> str:
    t = _t()
    return (
        f"background-color: {t['notice_bg']}; color: {t['notice_text']}; padding: 8px; "
        f"border: 1px solid {t['notice_border']}; border-radius: 4px; font-weight: bold;"
    )


def generate_data_file_label_style() -> str:
    return f"color: {_t()['data_file_text']}; font-style: italic;"


def generate_hint_style() -> str:
    return f"color: {_t()['text_secondary']}; font-size: 11px;"


def generate_description_label_style() -> str:
    return f"color: {_t()['text_secondary']}; font-size: 11px; padding: 4px 0px;"


# ── Pre-built style strings (generated from active theme) ──────────────────

GLOBAL_QSS = generate_global_qss()
SIDEBAR_STYLE = generate_sidebar_style()
SPLITTER_STYLE = generate_splitter_style()
COMBOBOX_STYLE = generate_combobox_style()
COMBOBOX_POPUP_STYLE = generate_combobox_popup_style()
NOTICE_STYLE = generate_notice_style()
DATA_FILE_LABEL_STYLE = generate_data_file_label_style()
HINT_STYLE = generate_hint_style()
DESCRIPTION_LABEL_STYLE = generate_description_label_style()
TITLE_STYLE = "font-size: 14px; font-weight: bold; margin-top: 4px; margin-bottom: 2px;"
TAB_CONTENT_STYLE = "background-color: transparent;"


# ── Theme switching ────────────────────────────────────────────────────────

def set_theme(name: str) -> None:
    global _current_theme, THEME, _THEME, _icon_paths
    global GLOBAL_QSS, SIDEBAR_STYLE, SPLITTER_STYLE
    global COMBOBOX_STYLE, COMBOBOX_POPUP_STYLE
    global NOTICE_STYLE, DATA_FILE_LABEL_STYLE, HINT_STYLE, DESCRIPTION_LABEL_STYLE

    # Update module-level state
    if name in _theme_map:
        _current_theme = name
        THEME.clear()
        THEME.update(_theme_map[name])
        _THEME.clear()
        _THEME.update(_theme_map[name])
        _icon_paths = (
            _ICON_PATHS_DARK if name == "dark" else _ICON_PATHS_LIGHT
        )

    # Regenerate all pre-built strings
    GLOBAL_QSS = generate_global_qss()
    SIDEBAR_STYLE = generate_sidebar_style()
    SPLITTER_STYLE = generate_splitter_style()
    COMBOBOX_STYLE = generate_combobox_style()
    COMBOBOX_POPUP_STYLE = generate_combobox_popup_style()
    NOTICE_STYLE = generate_notice_style()
    DATA_FILE_LABEL_STYLE = generate_data_file_label_style()
    HINT_STYLE = generate_hint_style()
    DESCRIPTION_LABEL_STYLE = generate_description_label_style()

    # Fire callbacks so widgets can refresh inline styles
    for cb in _theme_callbacks:
        try:
            cb()
        except Exception:
            pass


def get_theme() -> str:
    return _current_theme


def on_theme_changed(callback: Callable[[], None]) -> None:
    _theme_callbacks.append(callback)
