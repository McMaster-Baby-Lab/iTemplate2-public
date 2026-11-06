"""
Friendly dialog wrappers around QMessageBox.

All functions speak to the user, not the codebase.
Titles say what happened; messages say what to do next.
"""

from PyQt6.QtWidgets import QMessageBox, QDialog, QDialogButtonBox
from PyQt6.QtCore import Qt
from .theme import status_icon_success, status_icon_error, status_icon_warning


def show_success(parent, title, message):
    """Show a success dialog with checkmark icon."""
    QMessageBox.information(
        parent, f"{status_icon_success} {title}", message
    )


def show_warning(parent, title, message):
    """Show a warning dialog with warning icon."""
    QMessageBox.warning(
        parent, f"{status_icon_warning} {title}", message
    )


def show_error(parent, title="Operation Failed", message="An unexpected error occurred."):
    """Show an error dialog with cross icon."""
    QMessageBox.critical(
        parent, f"{status_icon_error} {title}", message
    )


def ask_confirm(parent, title, message, yes_text="Yes", no_text="No"):
    """Ask a yes/no confirmation question.

    Returns True if the user clicks Yes.
    """
    reply = QMessageBox.question(
        parent, title, message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def ask_choice(parent, title, message, choice_a, choice_b, default=1):
    """Present a two-choice decision dialog.

    Creates custom buttons with choice_a and choice_b labels.
    Returns 'a' if choice_a was clicked, 'b' if choice_b.

    Args:
        default: Which button is default (0 = choice_a, 1 = choice_b).
    """
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(message)
    btn_a = msg.addButton(choice_a, QMessageBox.ButtonRole.ActionRole)
    btn_b = msg.addButton(choice_b, QMessageBox.ButtonRole.ActionRole)
    msg.addButton(QMessageBox.StandardButton.Cancel)
    if default == 0:
        msg.setDefaultButton(btn_a)
    else:
        msg.setDefaultButton(btn_b)
    msg.exec()

    clicked = msg.clickedButton()
    if clicked == btn_a:
        return "a"
    elif clicked == btn_b:
        return "b"
    return None  # Cancel


# ── Common title mappings ──────────────────────────────────────────────────
# These provide friendly defaults. Use by key name:
#   show_warning(self, DIALOG_TITLES["no_data_file"], "Please select ...")
# or just use the inline title directly (recommended for readability).

DIALOG_TITLES = {
    "no_data_file": "No Data File Selected",
    "no_data_loaded": "No Data Loaded",
    "no_video": "No Video Available",
    "no_videos_added": "No Videos Added",
    "no_files_found": "No Files Found",
    "no_folder_selected": "No Folder Selected",
    "no_stimuli": "No Stimuli Found",
    "no_aois": "No Areas of Interest",
    "no_faces": "No Faces Detected",
    "no_selection": "Nothing Selected",
    "missing_config": "Settings Not Configured",
    "invalid_input": "Invalid Input",
    "already_running": "Process Already Running",
    "no_merge_needed": "No Merge Needed",
    "export_no_data": "No Data to Export",
    "heatmap_missing": "Heatmap Not Generated",
    "column_missing": "Required Columns Missing",
    "no_landmark_folder": "No Landmark Folder",
    "no_stimuli_folder": "No Stimuli Folder",
}
