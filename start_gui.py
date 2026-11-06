import sys
import os

# Ensure we can find the gui package
sys.path.append(os.path.join(os.getcwd(), 'gui'))


def _fix_qt_xcb_dependency():
    """
    Auto-fix: Symlink libxcb-cursor.so.0 from conda env to PyQt6 libs
    if missing. This fixes 'qt.qpa.plugin: Could not load... xcb' on Linux
    when installed via pip in a conda env.
    """
    if sys.platform != 'linux':
        return

    try:
        import PyQt6
        qt_lib_dir = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "lib")
        conda_lib_dir = os.path.join(sys.prefix, "lib")
        
        lib_name = "libxcb-cursor.so.0"
        src = os.path.join(conda_lib_dir, lib_name)
        dst = os.path.join(qt_lib_dir, lib_name)
        
        if os.path.exists(src) and os.path.isdir(qt_lib_dir) and not os.path.exists(dst):
            print(f"Fixing Qt XCB dependency: Linking {lib_name} to {qt_lib_dir}")
            os.symlink(src, dst)
    except Exception as e:
        print(f"Warning: Failed to apply Qt XCB fix: {e}")

_fix_qt_xcb_dependency()


def _fix_macos_dock():
    """Set app name and dock icon on macOS (must run BEFORE QApplication)."""
    if sys.platform != 'darwin':
        return
    
    # Override the process name shown in the dock
    import ctypes
    import ctypes.util
    libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('c'))
    # setprogname changes what macOS displays in the dock
    libc.setprogname(b'iTemplate 2')
    
    try:
        from Foundation import NSBundle, NSProcessInfo
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = "iTemplate 2"
        # Also override the process info name
        NSProcessInfo.processInfo().setValue_forKey_("iTemplate 2", "processName")
    except Exception:
        pass

_fix_macos_dock()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from gui.main import MainWindow
from gui.widgets.theme import GLOBAL_QSS
from gui.widgets.button_styles import button_style

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("iTemplate 2")
    app.setApplicationDisplayName("iTemplate 2")
    
    # Set window icon and macOS dock icon
    _gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui")
    _logo_path = os.path.join(_gui_dir, "resources", "logo.png")
    if os.path.isfile(_logo_path):
        app.setWindowIcon(QIcon(_logo_path))
        if sys.platform == 'darwin':
            try:
                from AppKit import NSApplication, NSImage
                ns_app = NSApplication.sharedApplication()
                ns_image = NSImage.alloc().initByReferencingFile_(_logo_path)
                if ns_image:
                    ns_app.setApplicationIconImage_(ns_image)
            except Exception:
                pass
    
    # Optional: Apply fusion style for better look across platforms
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_QSS + button_style("neutral", padding="6px 12px"))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
