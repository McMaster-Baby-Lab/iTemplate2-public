"""
Reusable utilities for running heavy processing in background threads with visual feedback.
"""
import inspect

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QRect, Qt, QPropertyAnimation, QEasingCurve, QPoint, QPointF, QElapsedTimer
from PyQt6.QtWidgets import QPushButton, QLabel, QVBoxLayout, QFrame, QProgressBar, QGraphicsOpacityEffect, QApplication
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QFont, QPen, QBrush
import math


class ProcessingWorker(QThread):
    """Generic worker thread that runs a callable in the background."""
    finished = pyqtSignal(object)  # emits result
    error = pyqtSignal(str)        # emits error message
    progress = pyqtSignal(str)     # emits status text
    progress_int = pyqtSignal(int) # emits 0-100 integer progress

    def __init__(self, func, *args, accepts_callback=True, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._accepts_callback = accepts_callback

    def run(self):
        try:
            kwargs = self.kwargs.copy()
            if self._accepts_callback:
                kwargs["progress_callback"] = self._emit_progress
            result = self.func(
                *self.args,
                **kwargs,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, value):
        """Accept either a string status or an int (0-100) percentage."""
        if isinstance(value, (int, float)):
            self.progress_int.emit(int(value))
        else:
            self.progress.emit(str(value))


class ProcessingButton(QPushButton):
    """A button that draws a progress bar overlay when running a background task.

    Uses custom paintEvent instead of child widgets so it works reliably
    on macOS where native button rendering covers child widgets.
    """

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._original_text = text
        self._original_style = ""
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._dot_count = 0
        self._processing = False
        self._progress_value = 0   # 0-100
        self._status_text = "Processing"

    def mousePressEvent(self, event):
        """Block clicks while processing."""
        if self._processing:
            return
        super().mousePressEvent(event)

    # ── public API ──────────────────────────────────────────────

    def start_processing(self, status_text=None):
        """Switch to processing visual state."""
        self._processing = True
        self._original_text = self.text()
        self._original_style = self.styleSheet()
        self._status_text = status_text or "Processing"
        self._progress_value = 0
        self._dot_count = 0
        self._animation_timer.start(400)
        self.update()

    def stop_processing(self, success=True):
        """Restore button to normal state."""
        self._animation_timer.stop()
        self._processing = False
        self._progress_value = 0
        self.setText(self._original_text)
        self.setStyleSheet(self._original_style)
        self.update()

    def set_progress(self, value):
        """Set progress to an integer 0-100."""
        if self._processing:
            self._progress_value = max(0, min(100, int(value)))
            self.update()

    def connect_worker(self, worker):
        """Convenience: wire a ProcessingWorker's progress_int signal to this button."""
        worker.progress_int.connect(self.set_progress)

    # ── private ──────────────────────────────────────────────

    def _animate(self):
        """Cycle through dot animation."""
        self._dot_count = (self._dot_count % 3) + 1
        self.update()  # trigger repaint

    def paintEvent(self, event):
        """Custom paint: full progress bar when processing, normal button otherwise."""
        if not self._processing:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # ── Draw dark background (entire button) ──
        painter.setBrush(QColor("#546e7a"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRect(0, 0, w, h), 4, 4)

        # ── Draw green progress fill ──
        if self._progress_value > 0:
            fill_w = int(w * self._progress_value / 100)
            gradient = QLinearGradient(0, 0, w, 0)
            gradient.setColorAt(0.0, QColor("#2a7a8a"))
            gradient.setColorAt(1.0, QColor("#5a9aba"))
            painter.setBrush(gradient)
            painter.drawRoundedRect(QRect(0, 0, fill_w, h), 4, 4)

        # ── Draw text overlay ──
        dots = "." * self._dot_count
        if self._progress_value > 0:
            label = f"{self._status_text}  {self._progress_value}%"
        else:
            label = f"{self._status_text}{dots}"

        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(QRect(0, 0, w, h),
                         Qt.AlignmentFlag.AlignCenter, label)

        painter.end()


class _AnimatedSpinner(QLabel):
    """A rose-curve particle-trail spinner drawn via custom paintEvent.

    Ported from the "Rose Curve" loader (r = a cos(kθ)) — particles trace
    a five-petal rose that breathes and slowly rotates.
    """

    # Curve parameters — ROSE_A in 0–100 space (must fill 96×96 canvas)
    ROSE_K = 5
    ROSE_A = 45.0
    ROSE_BREATH_BASE = 0.72
    ROSE_BREATH_BOOST = 0.28
    PARTICLE_COUNT = 56
    TRAIL_SPAN = 0.32
    DURATION_MS = 5400
    ROTATION_DURATION_MS = 28000
    PULSE_DURATION_MS = 4600

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(96, 96)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._running = False
        self._elapsed_timer = QElapsedTimer()

    def start(self):
        if not self._running:
            self._running = True
            self._elapsed_timer.restart()
            self._timer.start(30)

    def stop(self):
        if self._running:
            self._running = False
            self._timer.stop()
            self._elapsed_timer.invalidate()
            self.update()

    def _tick(self):
        self.update()

    # ── curve maths ────────────────────────────────────────────

    @classmethod
    def _rose_point(cls, t, detail_scale):
        """Return (x, y) in 0–100 space for the rose curve at phase t∈[0,1]."""
        angle = t * math.pi * 2
        a = cls.ROSE_A
        breath = cls.ROSE_BREATH_BASE + detail_scale * cls.ROSE_BREATH_BOOST
        r = a * breath * math.cos(cls.ROSE_K * angle)
        x = 50 + math.cos(angle) * r
        y = 50 + math.sin(angle) * r
        return x, y

    @classmethod
    def _detail_scale(cls, elapsed_ms):
        """Smooth 0.52–1.0 pulse over PULSE_DURATION_MS."""
        pulse_progress = (elapsed_ms % cls.PULSE_DURATION_MS) / cls.PULSE_DURATION_MS
        angle = pulse_progress * math.pi * 2
        return 0.52 + ((math.sin(angle + 0.55) + 1) / 2) * 0.48

    @classmethod
    def _rotation_deg(cls, elapsed_ms):
        return -((elapsed_ms % cls.ROTATION_DURATION_MS) / cls.ROTATION_DURATION_MS) * 360

    # ── painting ───────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Scale: curve is in 0–100 space; map to widget pixels
        scale = min(w, h) / 100.0

        if not self._running:
            self._elapsed_timer.invalidate()
            painter.end()
            return

        elapsed = self._elapsed_timer.elapsed()
        detail_scale = self._detail_scale(elapsed)
        rotation = self._rotation_deg(elapsed)
        progress = (elapsed % self.DURATION_MS) / self.DURATION_MS

        # Move origin to widget center, then rotate around center
        painter.translate(w / 2, h / 2)
        painter.rotate(rotation)

        for i in range(self.PARTICLE_COUNT):
            tail_offset = i / (self.PARTICLE_COUNT - 1)
            pt = (progress - tail_offset * self.TRAIL_SPAN) % 1.0
            cx, cy = self._rose_point(pt, detail_scale)

            # Shift curve center (50,50) to painter origin, then scale
            px = (cx - 50) * scale
            py = (cy - 50) * scale

            fade = (1 - tail_offset) ** 0.56
            radius = scale * (0.9 + fade * 2.7)
            opacity = 0.04 + fade * 0.96

            c = QColor("#3d7e9a")
            c.setAlphaF(opacity)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px, py), radius, radius)

        painter.end()

    def closeEvent(self, event):
        self.stop()


class ProcessingOverlay(QFrame):
    """Semi-transparent overlay that locks a widget during long-running operations.

    Shows an animated spinner, status text, optional progress bar, and optional cancel button.
    """

    cancel_clicked = pyqtSignal()
    processing_started = pyqtSignal()  # emitted when overlay is shown
    processing_finished = pyqtSignal()  # emitted when overlay is hidden

    def __init__(self, parent_widget, cancelable=False):
        super().__init__(parent_widget)
        self._cancelable = cancelable
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Opacity effect for fade animations
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(0)
        
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(150)  # Snappier duration
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._fade_out_initiated = False
        
        self.init_ui()
        self.hide()

    def init_ui(self):
        # Slightly darker, more premium background with subtle gradient
        self.setStyleSheet(
            "QFrame { background-color: rgba(30, 32, 36, 220); }"
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(24)
        layout.setContentsMargins(48, 48, 48, 48)

        # Spinner
        self._spinner = _AnimatedSpinner(self)
        layout.addWidget(self._spinner, 0, Qt.AlignmentFlag.AlignHCenter)

        # Status label
        self._status_label = QLabel(self)
        self._status_label.setStyleSheet(
            "color: #ffffff; font-size: 14px; font-weight: bold;"
            "background: transparent; border: none;"
        )
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # Sub-status label (optional detail)
        self._sub_status_label = QLabel(self)
        self._sub_status_label.setStyleSheet(
            "color: #7b8798; font-size: 12px; font-style: italic;"
            "background: transparent; border: none;"
        )
        self._sub_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_status_label.hide()
        layout.addWidget(self._sub_status_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setMinimumWidth(250)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            "QProgressBar {"
            "  background-color: #2c2f38;"
            "  border: 1px solid #444444;"
            "  border-radius: 4px;"
            "  text-align: center;"
            "  color: white;"
            "  height: 16px;"
            "}"
            "QProgressBar::chunk {"
            "  background-color: #3d7e9a;"
            "  border-radius: 3px;"
            "}"
        )
        layout.addWidget(self._progress_bar, 0, Qt.AlignmentFlag.AlignHCenter)

        # Cancel button (optional)
        if self._cancelable:
            self._cancel_btn = QPushButton("Cancel", self)
            self._cancel_btn.setStyleSheet(button_style("neutral", padding="8px 24px"))
            self._cancel_btn.clicked.connect(self.cancel_clicked)
            layout.addWidget(self._cancel_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        else:
            self._cancel_btn = None

    def showEvent(self, event):
        """Resize overlay to fill parent widget."""
        if self.parent():
            self.setGeometry(self.parent().rect())
            self.raise_()
        super().showEvent(event)

    def show(self):
        """Show overlay with fade-in animation.

        Only emits ``processing_started`` once — subsequent calls while the
        overlay is visible are no-ops for the signal.  This prevents callers
        that ``show()`` twice (e.g. auto-merge) from leaving the navigation
        block counter elevated.
        """
        if self.parent():
            self.setGeometry(self.parent().rect())
            self.raise_()

        was_hidden = not self.isVisible()

        self._spinner.start()

        # Ensure it's visible immediately even if the main thread blocks
        # before the animation can progress.
        self._opacity_effect.setOpacity(0.4)
        super().show()

        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_finished)
        except TypeError:
            pass
        self._fade_anim.setStartValue(0.4)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        
        # Force immediate first paint and state update
        self._spinner._tick()
        self._spinner.repaint()
        
        # Reset fade-out guard so a subsequent hide() can fire the signal
        self._fade_out_initiated = False

        # Process events multiple times to ensure the animation is fully primed
        for _ in range(3):
            QApplication.processEvents()

        if was_hidden:
            self.processing_started.emit()

    def hide(self):
        """Hide overlay with fade-out animation.

        Only emits ``processing_finished`` once per show/hide cycle.
        """
        if getattr(self, '_fade_out_initiated', False):
            return
        self._fade_out_initiated = True

        self.processing_finished.emit()

        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_finished)
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.finished.connect(self._on_fade_out_finished)
        self._fade_anim.start()

    def _on_fade_out_finished(self):
        """Actually hide the widget after fade-out."""
        try:
            self._fade_anim.finished.disconnect(self._on_fade_out_finished)
        except:
            pass
        if self._opacity_effect.opacity() == 0:
            self._spinner.stop()
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(False)
            self._sub_status_label.hide()
            super().hide()

    def set_status(self, text):
        """Update main status text."""
        self._status_label.setText(text)

    def set_sub_status(self, text):
        """Update sub-status detail text."""
        self._sub_status_label.setText(text)
        if text:
            self._sub_status_label.show()
        else:
            self._sub_status_label.hide()

    def set_progress(self, value):
        """Set progress (0-100) and show the progress bar."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(int(value))

    def hide_progress(self):
        """Hide the progress bar (indeterminate mode)."""
        self._progress_bar.setVisible(False)

    def set_cancel_enabled(self, enabled):
        """Enable or disable the cancel button."""
        if self._cancel_btn:
            self._cancel_btn.setEnabled(enabled)

    def closeEvent(self, event):
        self._spinner.stop()
        super().closeEvent(event)
