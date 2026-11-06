from pathlib import Path
import os
import sys
import json
import math
import logging
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger('iTemplate2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QAbstractItemView, QSplitter, QInputDialog, QCheckBox,
    QSpinBox
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QThread
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QImage, QPixmap, QPolygonF, QFont,
    QMouseEvent, QPainterPath, QWheelEvent
)
from .button_styles import button_style
from .processing_utils import ProcessingOverlay
from .theme import THEME, on_theme_changed, status_icon_warning
from .theme import GAZE_OFF_FACE_BGR, GAZE_UNMAPPED_BGR, bgr_to_hex
from .dialog_helpers import show_success, show_warning, show_error

# Shared template assets from the core layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.config import average_face, avg_face_image, avg_landmarks
    from core.io_utils import atomic_save_parquet
except ImportError:
    average_face = None
    avg_face_image = None
    avg_landmarks = None
    atomic_save_parquet = None

HANDLE_RADIUS = 6
MAX_GAZE_POINTS = 50000  # overlay subsamples beyond this to keep drawing fast
GAZE_DATA_COLUMNS = ['transformed_x', 'transformed_y', 'stimuli_x', 'stimuli_y', 'gaze_region',
                     'stimulusName', 'videoName', 'STIMULI_NAME', '_video_frame_index']


# ═══════════════════════════════════════════════════════════════════════
#  AOI Canvas — draws shapes on top of the average face
# ═══════════════════════════════════════════════════════════════════════
class AOICanvas(QWidget):
    """Interactive canvas for drawing and editing AOI regions on the average face."""
    """
    Interactive canvas for drawing and editing AOI regions.
    Supports both template-face and stimulus backgrounds.
    All coordinates are stored in normalised [0,1] space.
    """

    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view
        self.setMinimumSize(320, 320)
        self.setCursor(Qt.CursorShape.CrossCursor)
        on_theme_changed(self.update)

        # Background image (average face by default)
        self._bg_pixmap = None
        self._template_pixmap = None  # store template face for context switching
        avg_rgba = None
        # Prefer pre-converted RGBA if available.
        if avg_face_image is not None:
            avg_rgba = avg_face_image
        elif average_face is not None:
            # Raw dataset from H5 is BGRA; convert before handing to Qt.
            avg_rgba = cv2.cvtColor(average_face, cv2.COLOR_BGRA2RGBA)

        if avg_rgba is not None:
            from PyQt6.QtGui import QImage
            h, w, ch = avg_rgba.shape
            fmt = QImage.Format.Format_RGBA8888 if ch == 4 else QImage.Format.Format_RGB888
            q_img = QImage(avg_rgba.data, w, h, ch * w, fmt)
            self._bg_pixmap = QPixmap.fromImage(q_img)
            self._template_pixmap = self._bg_pixmap

        # Drawing state
        self._mode = 'create'  # 'create' or 'edit'
        self._shape_type = 'rectangle'
        self._drag_start = None  # normalised start point
        self._drag_current = None
        self._polygon_pts = []
        self._show_landmarks = True
        self._snap_enabled = False
        self._snap_radius_px = 15  # magnetic snap radius in pixels
        self._snapped_idx = -1  # index of currently snapped landmark

        # Gaze overlay: pre-rendered RGBA layer matching the background pixmap,
        # plus legend rows of (label, QColor or None).
        self._gaze_layer = None
        self._gaze_legend = []

        # Load average face landmarks as normalised [0,1] coords.
        self._landmarks_norm = []
        if avg_landmarks is not None:
            lm = avg_landmarks  # shape (N, 2 or 3), already normalised
            for i in range(lm.shape[0]):
                nx = float(lm[i, 0])
                ny = float(lm[i, 1])
                self._landmarks_norm.append((nx, ny))

        # Edit state
        self._moving = False
        self._move_offset = None
        self._move_orig_coords = None
        self._resize_handle = None  # index or None
        self._resize_orig = None
        self._vertex_drag = None  # index or None

        # Zoom & pan state
        self._zoom = 1.0
        self._pan_x = 0.0  # pan offset in pixels
        self._pan_y = 0.0
        self._panning = False
        self._pan_start = None

    # ── coordinate helpers ──
    def _image_rect(self):
        """Return (x_off, y_off, img_w, img_h) for the face image
        fitted inside the widget keeping its native aspect ratio,
        accounting for zoom & pan."""
        if not self._bg_pixmap:
            return (0, 0, self.width(), self.height())
        src_w = self._bg_pixmap.width()
        src_h = self._bg_pixmap.height()
        wgt_w = self.width()
        wgt_h = self.height()
        # Fit inside widget
        scale = min(wgt_w / src_w, wgt_h / src_h)
        img_w = src_w * scale * self._zoom
        img_h = src_h * scale * self._zoom
        # Centre when not panned
        x_off = (wgt_w - src_w * scale) / 2 * self._zoom + self._pan_x
        y_off = (wgt_h - src_h * scale) / 2 * self._zoom + self._pan_y
        return (x_off, y_off, img_w, img_h)

    def _to_norm(self, px, py):
        """Screen pixel → normalised [0,1] accounting for zoom, pan & aspect ratio."""
        x_off, y_off, img_w, img_h = self._image_rect()
        return ((px - x_off) / img_w if img_w else 0,
                (py - y_off) / img_h if img_h else 0)

    def _from_norm(self, nx, ny):
        """Normalised → screen pixel accounting for zoom, pan & aspect ratio."""
        x_off, y_off, img_w, img_h = self._image_rect()
        return (nx * img_w + x_off, ny * img_h + y_off)

    def _norm_rect(self, coords):
        """Return pixel rect from normalised (x0,y0,x1,y1)."""
        x0, y0 = self._from_norm(coords[0], coords[1])
        x1, y1 = self._from_norm(coords[2], coords[3])
        return QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    def zoom_in(self):
        self._zoom_at_center(self._zoom * 1.25)

    def zoom_out(self):
        self._zoom_at_center(max(0.5, self._zoom / 1.25))

    def zoom_reset(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def _zoom_at_center(self, new_zoom):
        """Zoom keeping the widget center steady."""
        cx, cy = self.width() / 2, self.height() / 2
        # The normalised point under cx,cy before zoom
        old_zoom = self._zoom
        # Adjust pan so the center stays at the same normalised point
        self._pan_x = cx - (cx - self._pan_x) * new_zoom / old_zoom
        self._pan_y = cy - (cy - self._pan_y) * new_zoom / old_zoom
        self._zoom = new_zoom
        self.update()

    # ── painting ──
    def set_template(self,template):
        from PyQt6.QtGui import QImage
        rgba = cv2.cvtColor(template.image,cv2.COLOR_BGRA2RGBA)
        image = QImage(rgba.data,template.width,template.height,rgba.strides[0],QImage.Format.Format_RGBA8888).copy()
        self._template_pixmap = QPixmap.fromImage(image)
        self._landmarks_norm = [tuple(map(float,xy)) for xy in template.landmarks[:,:2]]
        if self.parent_view._aoi_context == 'template':
            self._bg_pixmap = self._template_pixmap
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background (always fill full widget)
        painter.fillRect(self.rect(), QColor(THEME["bg_primary"]))

        # Draw face image with zoom & pan (preserving aspect ratio)
        if self._bg_pixmap:
            x_off, y_off, img_w, img_h = self._image_rect()
            scaled = self._bg_pixmap.scaled(
                int(img_w), int(img_h),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.drawPixmap(int(x_off), int(y_off), scaled)

        if self._gaze_layer is not None and self._bg_pixmap:
            x_off, y_off, img_w, img_h = self._image_rect()
            painter.setOpacity(0.7)
            painter.drawPixmap(QRectF(x_off, y_off, img_w, img_h), self._gaze_layer,
                               QRectF(self._gaze_layer.rect()))
            painter.setOpacity(1.0)

        # Draw landmark dots (when visible)
        if self._show_landmarks and self._landmarks_norm:
            for i, (lx, ly) in enumerate(self._landmarks_norm):
                px, py = self._from_norm(lx, ly)
                if i == self._snapped_idx:
                    # Highlighted snapped landmark
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(255, 255, 0, 200))  # bright yellow
                    painter.drawEllipse(QPointF(px, py), 6, 6)
                else:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(0, 220, 220, 120))  # cyan semi-transparent
                    painter.drawEllipse(QPointF(px, py), 3, 3)

        # Draw existing AOIs (faded in create mode or when another is selected)
        shapes = self.parent_view.active_shapes
        sel_idx = self.parent_view.selected_idx
        has_selection = (sel_idx >= 0 and self._mode == 'edit')
        for idx, aoi in enumerate(shapes):
            is_sel = (idx == sel_idx and self._mode == 'edit')
            faded = self._mode == 'create' or (has_selection and not is_sel)
            self._draw_aoi(painter, aoi, is_sel, faded)

        # Temp shape while dragging in create mode
        if self._mode == 'create' and self._drag_start and self._drag_current:
            pen = QPen(QColor('#4488ff'), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(68, 136, 255, 50))
            x0, y0 = self._from_norm(*self._drag_start)
            x1, y1 = self._from_norm(*self._drag_current)
            if self._shape_type == 'rectangle':
                painter.drawRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))
            elif self._shape_type == 'oval':
                painter.drawEllipse(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

        # Temp polygon preview
        if self._mode == 'create' and self._shape_type == 'polygon' and self._polygon_pts:
            pen = QPen(QColor('#cc6666'), 2)
            painter.setPen(pen)
            pts_px = [QPointF(*self._from_norm(x, y)) for x, y in self._polygon_pts]
            for i in range(len(pts_px) - 1):
                painter.drawLine(pts_px[i], pts_px[i + 1])
            # Draw dots
            painter.setBrush(QColor('#cc6666'))
            for pt in pts_px:
                painter.drawEllipse(pt, 4, 4)

        if self._gaze_legend:
            self._draw_gaze_legend(painter)

        painter.end()

    def _draw_gaze_legend(self, painter):
        """Draw the gaze color legend in the bottom-left corner of the widget."""
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        fm = painter.fontMetrics()
        line_h, pad, swatch = fm.height() + 4, 8, 18
        has_swatch = any(color is not None for _, color in self._gaze_legend)
        text_w = max(fm.horizontalAdvance(label) for label, _ in self._gaze_legend)
        box_w = pad * 2 + (swatch if has_swatch else 0) + text_w + 4
        box_h = pad * 2 + line_h * len(self._gaze_legend)
        box = QRectF(10, self.height() - 10 - box_h, box_w, box_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(30, 30, 30, 170))
        painter.drawRoundedRect(box, 6, 6)
        for i, (label, color) in enumerate(self._gaze_legend):
            cy = box.top() + pad + line_h * i + line_h / 2
            text_x = box.left() + pad
            if color is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(text_x + 6, cy), 5, 5)
                text_x += swatch
            painter.setPen(QColor('white'))
            painter.drawText(QRectF(text_x, cy - line_h / 2, text_w + 4, line_h),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)

    def _draw_aoi(self, painter, aoi, selected, faded=False):
        """Draw a single AOI shape."""
        if selected:
            outline = QColor('#ff9900')
            fill = QColor(255, 153, 0, 60)
            pen_w = 3
        elif faded:
            outline = QColor(100, 100, 100, 60)
            fill = QColor(100, 100, 100, 15)
            pen_w = 1
        else:
            outline = QColor('#4488ff')
            fill = QColor(68, 136, 255, 40)
            pen_w = 2

        painter.setPen(QPen(outline, pen_w))
        painter.setBrush(QBrush(fill))

        if aoi['type'] == 'rectangle':
            rect = self._norm_rect(aoi['coords'])
            painter.drawRect(rect)
            # Handles
            if selected:
                self._draw_rect_handles(painter, aoi['coords'])
            # Label
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont('Arial', 10, QFont.Weight.Bold))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, aoi['name'])

        elif aoi['type'] == 'oval':
            rect = self._norm_rect(aoi['coords'])
            painter.drawEllipse(rect)
            if selected:
                self._draw_rect_handles(painter, aoi['coords'])
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont('Arial', 10, QFont.Weight.Bold))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, aoi['name'])

        elif aoi['type'] == 'polygon':
            pts_px = [QPointF(*self._from_norm(x, y)) for x, y in aoi['coords']]
            if len(pts_px) >= 3:
                poly = QPolygonF(pts_px)
                painter.drawPolygon(poly)
                # Label at centroid
                cx = sum(p.x() for p in pts_px) / len(pts_px)
                cy = sum(p.y() for p in pts_px) / len(pts_px)
                painter.setPen(Qt.GlobalColor.white)
                painter.setFont(QFont('Arial', 10, QFont.Weight.Bold))
                painter.drawText(QPointF(cx, cy), aoi['name'])
            if selected:
                painter.setBrush(QColor('#ff9900'))
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                for pt in pts_px:
                    painter.drawEllipse(pt, HANDLE_RADIUS, HANDLE_RADIUS)

    def _draw_rect_handles(self, painter, coords):
        """Draw corner resize handles."""
        painter.setBrush(QColor('#cc6666'))
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        for hx, hy in self._get_handles_px(coords):
            painter.drawEllipse(QPointF(hx, hy), HANDLE_RADIUS, HANDLE_RADIUS)

    def _get_handles_px(self, coords):
        """Pixel positions of the 4 corner handles."""
        x0, y0 = self._from_norm(coords[0], coords[1])
        x1, y1 = self._from_norm(coords[2], coords[3])
        return [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]

    def _point_in_aoi(self, nx, ny, aoi):
        if aoi['type'] == 'rectangle':
            x0, y0, x1, y1 = aoi['coords']
            return min(x0, x1) <= nx <= max(x0, x1) and min(y0, y1) <= ny <= max(y0, y1)
        elif aoi['type'] == 'oval':
            x0, y0, x1, y1 = aoi['coords']
            rx = abs(x1 - x0) / 2
            ry = abs(y1 - y0) / 2
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if rx > 0 and ry > 0:
                return ((nx - cx) ** 2) / (rx ** 2) + ((ny - cy) ** 2) / (ry ** 2) <= 1
            return False
        elif aoi['type'] == 'polygon':
            return self._point_in_polygon(nx, ny, aoi['coords'])
        return False

    def _hit_test_aoi(self, nx, ny):
        """Return index of the topmost AOI at normalised (nx, ny), or -1."""
        for idx in reversed(range(len(self.parent_view.active_shapes))):
            if self._point_in_aoi(nx, ny, self.parent_view.active_shapes[idx]):
                return idx
        return -1

    @staticmethod
    def _point_in_polygon(x, y, poly):
        n = len(poly)
        inside = False
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_inter = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-10) + x1
                if x < x_inter:
                    inside = not inside
        return inside

    def _hit_handle(self, px, py, coords):
        """Return handle index (0-3) if near a corner, else -1."""
        for i, (hx, hy) in enumerate(self._get_handles_px(coords)):
            if abs(px - hx) <= 8 and abs(py - hy) <= 8:
                return i
        return -1

    def _hit_vertex(self, px, py, poly_coords):
        """Return vertex index if near a polygon vertex, else -1."""
        for i, (nx, ny) in enumerate(poly_coords):
            vx, vy = self._from_norm(nx, ny)
            if math.hypot(px - vx, py - vy) <= HANDLE_RADIUS + 2:
                return i
        return -1

    def _nearest_landmark(self, px, py):
        """Return (index, (nx, ny)) for nearest landmark within snap radius."""
        if not self._landmarks_norm:
            return -1, None

        best_dist = self._snap_radius_px
        best_idx = -1
        for i, (lx, ly) in enumerate(self._landmarks_norm):
            lpx, lpy = self._from_norm(lx, ly)
            d = math.hypot(px - lpx, py - lpy)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0:
            return best_idx, self._landmarks_norm[best_idx]
        return -1, None

    # ── mouse interaction ──
    def _handle_press(self, event: QMouseEvent):
        pos = event.position()
        px, py = pos.x(), pos.y()
        nx, ny = self._to_norm(px, py)

        snap_idx = -1
        snap_point = None
        if self._snap_enabled:
            snap_idx, snap_point = self._nearest_landmark(px, py)
        self._snapped_idx = snap_idx
        if snap_point is not None:
            nx, ny = snap_point

        if self._mode == 'create':
            if self._shape_type == 'polygon':
                self._polygon_pts.append((nx, ny))
                self.update()
            else:
                self._drag_start = (nx, ny)
                self._drag_current = (nx, ny)

        elif self._mode == 'edit':
            hit = self._hit_test_aoi(nx, ny)

            # 1) If we already have a selection, check its handles first
            if self.parent_view.selected_idx >= 0:
                aoi = self.parent_view.active_shapes[self.parent_view.selected_idx]
                if aoi['type'] in ('rectangle', 'oval'):
                    h = self._hit_handle(px, py, aoi['coords'])
                    if h >= 0:
                        self._resize_handle = h
                        self._resize_orig = tuple(aoi['coords'])
                        return
                elif aoi['type'] == 'polygon':
                    v = self._hit_vertex(px, py, aoi['coords'])
                    if v >= 0:
                        self._vertex_drag = v
                        return

                # If the click hits the currently selected AOI, prefer it over
                # other overlapping AOIs to allow moving it reliably.
                if self._point_in_aoi(nx, ny, aoi):
                    hit = self.parent_view.selected_idx

            # 2) If not on a handle, handle selection and moving
            if hit >= 0:
                # If clicked a different AOI, select it
                if hit != self.parent_view.selected_idx:
                    self.parent_view.selected_idx = hit
                    self.parent_view._sync_selection()
                    self.update()

                # Start moving the newly or already selected AOI
                self._moving = True
                self._move_offset = (nx, ny)
                aoi = self.parent_view.active_shapes[hit]
                if aoi['type'] in ('rectangle', 'oval'):
                    self._move_orig_coords = tuple(aoi['coords'])
                elif aoi['type'] == 'polygon':
                    self._move_orig_coords = list(aoi['coords'])
            else:
                # Clicked empty space: deselect
                if self.parent_view.selected_idx >= 0:
                    self.parent_view.selected_idx = -1
                    self.parent_view._sync_selection()
                    self.update()

    def _handle_move(self, event: QMouseEvent):
        pos = event.position()
        px, py = pos.x(), pos.y()
        nx, ny = self._to_norm(px, py)

        snap_idx = -1
        snap_point = None
        if self._snap_enabled:
            snap_idx, snap_point = self._nearest_landmark(px, py)
        if snap_idx != self._snapped_idx:
            self._snapped_idx = snap_idx

        if self._mode == 'create' and self._drag_start:
            # Apply snap for drag shapes.
            if snap_point is not None:
                nx, ny = snap_point
            self._drag_current = (nx, ny)
            self.update()
            return

        if self._mode == 'edit' and self.parent_view.selected_idx >= 0:
            aoi = self.parent_view.active_shapes[self.parent_view.selected_idx]

            # Resizing
            if self._resize_handle is not None and aoi['type'] in ('rectangle', 'oval'):
                # Apply snap when resizing corners.
                if snap_point is not None:
                    nx, ny = snap_point
                coords = list(self._resize_orig)
                h = self._resize_handle
                if h == 0:
                    coords[0], coords[1] = nx, ny
                elif h == 1:
                    coords[2], coords[1] = nx, ny
                elif h == 2:
                    coords[0], coords[3] = nx, ny
                elif h == 3:
                    coords[2], coords[3] = nx, ny
                aoi['coords'] = tuple(coords)
                self.parent_view._refresh_table()
                self.update()
                return

            # Vertex drag
            if self._vertex_drag is not None and aoi['type'] == 'polygon':
                # Apply snap when dragging polygon vertices.
                if snap_point is not None:
                    nx, ny = snap_point
                coords = list(aoi['coords'])
                coords[self._vertex_drag] = (nx, ny)
                aoi['coords'] = coords
                self.update()
                return

            # Moving
            if self._moving:
                dx = nx - self._move_offset[0]
                dy = ny - self._move_offset[1]
                if aoi['type'] in ('rectangle', 'oval'):
                    o = self._move_orig_coords
                    aoi['coords'] = (o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy)
                elif aoi['type'] == 'polygon':
                    aoi['coords'] = [(x + dx, y + dy) for x, y in self._move_orig_coords]
                self.parent_view._refresh_table()
                self.update()
                return

        # Landmark snapping feedback refresh.
        if self._snap_enabled:
            self.update()

        # Cursor feedback
        if self._mode == 'edit':
            hit = self._hit_test_aoi(nx, ny)
            if hit >= 0:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _handle_release(self, event: QMouseEvent):
        pos = event.position()
        px, py = pos.x(), pos.y()
        nx, ny = self._to_norm(px, py)
        if self._snap_enabled:
            snap_idx, snap_point = self._nearest_landmark(px, py)
            self._snapped_idx = snap_idx
            if snap_point is not None:
                nx, ny = snap_point

        if self._mode == 'create' and self._drag_start and self._shape_type != 'polygon':
            x0, y0 = self._drag_start
            x1, y1 = nx, ny
            # Only create if shape is large enough
            if abs(x1 - x0) > 0.01 and abs(y1 - y0) > 0.01:
                aoi = {
                    'name': f'AOI_{len(self.parent_view.active_shapes) + 1}',
                    'type': self._shape_type,
                    'coords': (x0, y0, x1, y1)
                }
                self.parent_view.active_shapes.append(aoi)
                self.parent_view._refresh_table()
            self._drag_start = None
            self._drag_current = None
            self.update()

        self._moving = False
        self._move_offset = None
        self._move_orig_coords = None
        self._resize_handle = None
        self._resize_orig = None
        self._vertex_drag = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click finishes polygon drawing."""
        if self._mode == 'create' and self._shape_type == 'polygon' and len(self._polygon_pts) >= 3:
            aoi = {
                'name': f'AOI_{len(self.parent_view.active_shapes) + 1}',
                'type': 'polygon',
                'coords': list(self._polygon_pts)
            }
            self.parent_view.active_shapes.append(aoi)
            self.parent_view._refresh_table()
            self._polygon_pts = []
            self.update()

    def keyPressEvent(self, event):
        """Escape cancels current polygon drawing."""
        if event.key() == Qt.Key.Key_Escape:
            self._polygon_pts = []
            self._drag_start = None
            self._drag_current = None
            self.update()

    def set_mode(self, mode):
        self._mode = mode
        self._polygon_pts = []
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_shape_type(self, shape):
        self._shape_type = shape
        self._polygon_pts = []
        self._drag_start = None
        self._drag_current = None

    def wheelEvent(self, event: QWheelEvent):
        """Scroll wheel to zoom in/out."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_at_center(self._zoom * 1.1)
        elif delta < 0:
            self._zoom_at_center(max(0.5, self._zoom / 1.1))
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        # Middle button = pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        self._handle_press(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Middle button drag = pan
        if self._panning and self._pan_start:
            dx = event.position().x() - self._pan_start[0]
            dy = event.position().y() - self._pan_start[1]
            self._pan_x += dx
            self._pan_y += dy
            self._pan_start = (event.position().x(), event.position().y())
            self.update()
            return
        self._handle_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        self._handle_release(event)


# ═══════════════════════════════════════════════════════════════════════
#  AOI Analysis Worker — runs hit-testing in background thread
# ═══════════════════════════════════════════════════════════════════════
class _AoiWorker(QThread):
    progress = pyqtSignal(str)
    progress_int = pyqtSignal(int)
    finished = pyqtSignal(object)  # emits dict with result
    error = pyqtSignal(str)

    def __init__(self, data_path, shapes, stimulus_shapes):
        super().__init__()
        self.data_path = data_path
        self.shapes = shapes
        self.stimulus_shapes = stimulus_shapes

    def run(self):
        try:
            from core.io_utils import atomic_save_parquet
            if str(self.data_path).lower().endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_csv(self.data_path)

            tx = df['transformed_x'].values if 'transformed_x' in df.columns else None
            ty = df['transformed_y'].values if 'transformed_y' in df.columns else None
            sx = df['stimuli_x'].values if 'stimuli_x' in df.columns else None
            sy = df['stimuli_y'].values if 'stimuli_y' in df.columns else None

            if not self.shapes and not self.stimulus_shapes:
                raise ValueError("No AOIs defined.")
            if self.shapes and (tx is None or ty is None):
                raise ValueError("Data must contain 'transformed_x' and 'transformed_y' for template AOIs.")
            if self.stimulus_shapes and (sx is None or sy is None):
                raise ValueError("Data must contain 'stimuli_x' and 'stimuli_y' for stimulus AOIs.")

            # hit-test: inline since _hit_test_vectorized is instance method
            for i, aoi in enumerate(self.shapes):
                self.progress.emit(f"Processing template AOI {i+1}/{len(self.shapes)}")
                mask = self._hit_test(tx, ty, aoi)
                if 'gaze_region' in df.columns:
                    mask &= df['gaze_region'].eq('on_face').fillna(False).to_numpy(dtype=bool)
                df[aoi['name']] = mask

            stimulus_columns = {}
            for i, aoi in enumerate(self.stimulus_shapes):
                self.progress.emit(f"Processing stimulus AOI {i+1}/{len(self.stimulus_shapes)}")
                mask = self._hit_test(sx, sy, aoi)
                if 'start_frame' in aoi:
                    required = {'stimulusName', '_video_frame_index'}
                    if not required.issubset(df.columns):
                        raise ValueError("Video interval AOIs require stimulusName and _video_frame_index. "
                                         "Assign video frame indices in Coordinate Setup first.")
                    frames = pd.to_numeric(df['_video_frame_index'], errors='coerce')
                    names = df['stimulusName'].fillna('').astype(str).map(
                        lambda value: os.path.splitext(value.replace('\\', '/').rsplit('/', 1)[-1])[0])
                    stimulus = os.path.splitext(aoi['stimulus'])[0]
                    mask &= (names.eq(stimulus) & frames.between(
                        aoi['start_frame'], aoi['end_frame'])).to_numpy(dtype=bool)
                column = f"stim_{aoi['name']}"
                stimulus_columns[column] = stimulus_columns.get(column, np.zeros(len(df), dtype=bool)) | mask
            for column, mask in stimulus_columns.items():
                df[column] = mask

            # save output
            base, ext = os.path.splitext(self.data_path)
            if base.endswith('_AOI'):
                base = base[:-len('_AOI')]
            # Always save results as parquet for performance
            out_path = f"{base}_AOI.parquet"
            from core.template_manager import dataset_manifest, save_dataset
            manifest = dataset_manifest(self.data_path)
            manifest['aoi_analysis'] = True
            manifest['aoi_columns'] = [s['name'] for s in self.shapes] + list(stimulus_columns)
            save_dataset(df,out_path,manifest)

            # build summary
            parts = []
            for col in [s['name'] for s in self.shapes]:
                parts.append((col, int(df[col].sum())))
            for col in stimulus_columns:
                parts.append((col, int(df[col].sum())))

            result = {
                'out_path': out_path,
                'parts': parts,
                'total': len(df),
                'aoi_names': [s['name'] for s in self.shapes] + list(stimulus_columns),
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _hit_test(x, y, aoi):
        """Vectorized point-in-AOI hit test."""
        aoi_type = aoi.get('type', 'polygon')
        if aoi_type == 'rectangle':
            x0, y0, x1, y1 = aoi['coords']
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        elif aoi_type == 'circle':
            cx, cy, r = aoi['coords']
            return ((x - cx)**2 + (y - cy)**2) <= r**2
        elif aoi_type == 'oval':
            x0, y0, x1, y1 = aoi['coords']
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx > 0 and ry > 0:
                return (((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2)) <= 1
            return np.zeros_like(x, dtype=bool)
        else:  # polygon
            from matplotlib.path import Path
            coords = np.array(aoi['coords'])
            path = Path(coords)
            points = np.column_stack([x, y])
            return path.contains_points(points)


# ═══════════════════════════════════════════════════════════════════════
#  AOI Analysis Widget — main container
# ═══════════════════════════════════════════════════════════════════════
class AOIAnalysis(QWidget):
    data_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.shapes = []  # template-face AOIs: list of dicts {name, type, coords}
        self.stimulus_shapes = []  # legacy, global stimulus AOIs
        self.video_intervals = []
        self._active_interval = None
        self._video_frames = 0
        self.selected_idx = -1
        self.current_project_path = ''
        self._project_settings = None
        self._aoi_context = 'template'  # 'template' or 'stimulus'
        self._syncing_selection = False
        self._gaze_df = None          # cached gaze columns of the selected data file
        self._gaze_df_path = None
        self._gaze_groups = []        # [(label, BGR, xs, ys)] currently drawn on the canvas

        self._build_ui()
        self._overlay = ProcessingOverlay(self)

    @property
    def active_shapes(self):
        """Return the shape list for the current AOI context."""
        if self._aoi_context == 'stimulus':
            if self._active_interval is not None:
                return self._active_interval['shapes']
            return self.stimulus_shapes
        return self.shapes

    @active_shapes.setter
    def active_shapes(self, value):
        if self._aoi_context == 'stimulus':
            if self._active_interval is not None:
                self._active_interval['shapes'] = value
            else:
                self.stimulus_shapes = value
        else:
            self.shapes = value

    # ── UI construction ──
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)

        self.video_controls = QGroupBox("Video AOI intervals (frames are 1-based, bounds inclusive)")
        video_row = QHBoxLayout(self.video_controls)
        self.combo_interval = QComboBox()
        self.combo_interval.currentIndexChanged.connect(self._select_interval)
        video_row.addWidget(self.combo_interval)
        self.spin_key_frame = QSpinBox()
        self.spin_start_frame = QSpinBox()
        self.spin_end_frame = QSpinBox()
        for label, spin in (("Key frame:", self.spin_key_frame),
                            ("Start:", self.spin_start_frame), ("End:", self.spin_end_frame)):
            spin.setRange(1, 1)
            video_row.addWidget(QLabel(label))
            video_row.addWidget(spin)
        self.spin_key_frame.valueChanged.connect(self._preview_key_frame)
        for label, callback in (("Add interval", self._add_interval),
                                ("Update interval", self._update_interval),
                                ("Delete interval", self._delete_interval)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            video_row.addWidget(button)
        self.video_controls.hide()
        layout.addWidget(self.video_controls)

        
        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.hide()
        layout.addWidget(self.lbl_notice)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        # AOI Context
        toolbar.addWidget(QLabel("Context:"))
        self.combo_context = QComboBox()
        self.combo_context.setMinimumWidth(150)
        self.combo_context.setToolTip("Switch between template face and stimulus contexts")
        self.combo_context.addItems(['Template Face', 'Stimulus'])
        self.combo_context.currentTextChanged.connect(self._on_context_changed)
        toolbar.addWidget(self.combo_context)

        # Stimulus image selector (hidden until Stimulus context)
        self.lbl_stim_select = QLabel("Image:")
        self.combo_stim_image = QComboBox()
        self.combo_stim_image.setMinimumWidth(120)
        self.combo_stim_image.currentTextChanged.connect(self._on_stim_image_changed)
        self.lbl_stim_select.hide()
        self.combo_stim_image.hide()
        toolbar.addWidget(self.lbl_stim_select)
        toolbar.addWidget(self.combo_stim_image)

        toolbar.addSpacing(8)

        # Mode
        toolbar.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.setMinimumWidth(100)
        self.combo_mode.addItems(['Create', 'Edit'])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        self.combo_mode.setToolTip("Create mode to draw new AOIs, Edit mode to modify existing ones")
        toolbar.addWidget(self.combo_mode)

        # Shape type
        toolbar.addWidget(QLabel("Shape:"))
        self.combo_shape = QComboBox()
        self.combo_shape.setMinimumWidth(100)
        self.combo_shape.addItems(['Rectangle', 'Oval', 'Polygon'])

        self.combo_shape.currentTextChanged.connect(self._on_shape_changed)
        self.combo_shape.setToolTip("Shape type for new AOIs")
        toolbar.addWidget(self.combo_shape)

        toolbar.addSpacing(16)

        # Landmark toggle
        self.chk_landmarks = QCheckBox("Show Landmarks")
        self.chk_landmarks.setChecked(True)
        self.chk_landmarks.toggled.connect(self._on_landmarks_toggled)
        toolbar.addWidget(self.chk_landmarks)

        self.chk_snap = QCheckBox("Snap to Landmarks")
        self.chk_snap.setChecked(False)
        self.chk_snap.toggled.connect(self._on_snap_toggled)
        toolbar.addWidget(self.chk_snap)

        self.chk_show_gaze = QCheckBox("Show Gaze")
        self.chk_show_gaze.setToolTip(
            "Overlay gaze samples from the selected data file.\n"
            "Template context: transformed gaze from all stimuli.\n"
            "Stimulus context: gaze on this stimulus (video: the selected interval).")
        self.chk_show_gaze.toggled.connect(self._refresh_gaze_overlay)
        toolbar.addWidget(self.chk_show_gaze)

        self.chk_highlight_off_face = QCheckBox("Highlight Off-face")
        self.chk_highlight_off_face.setToolTip(
            "Color off-face gaze magenta and unmapped samples gray.\n"
            "Uses the gaze_region column written during transformation.")
        self.chk_highlight_off_face.setChecked(True)
        self.chk_highlight_off_face.setEnabled(False)
        self.chk_highlight_off_face.toggled.connect(self._refresh_gaze_overlay)
        toolbar.addWidget(self.chk_highlight_off_face)

        toolbar.addSpacing(16)

        # Zoom buttons
        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setMinimumWidth(24)
        btn_zoom_in.setToolTip("Zoom In")
        btn_zoom_in.setAccessibleName("Zoom in")
        btn_zoom_in.clicked.connect(lambda: self.canvas.zoom_in())
        toolbar.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("-")
        btn_zoom_out.setMinimumWidth(24)
        btn_zoom_out.setToolTip("Zoom Out")
        btn_zoom_out.setAccessibleName("Zoom out")
        btn_zoom_out.clicked.connect(lambda: self.canvas.zoom_out())
        toolbar.addWidget(btn_zoom_out)

        btn_zoom_reset = QPushButton("Reset")
        btn_zoom_reset.setMinimumWidth(40)
        btn_zoom_reset.setAccessibleName("Reset zoom")
        btn_zoom_reset.clicked.connect(lambda: self.canvas.zoom_reset())
        toolbar.addWidget(btn_zoom_reset)

        toolbar.addSpacing(16)

        # Import / Export
        btn_import = QPushButton("Import AOI")
        btn_import.setToolTip("Import AOI definitions from a JSON file")
        btn_import.clicked.connect(self.import_aoi)
        toolbar.addWidget(btn_import)

        btn_export = QPushButton("Export AOI")
        btn_export.setToolTip("Export current AOI definitions to a JSON file")
        btn_export.clicked.connect(self.export_aoi)
        toolbar.addWidget(btn_export)

        btn_export_img = QPushButton("Export Image")
        btn_export_img.setToolTip("Export a publication-quality image with AOIs drawn")
        btn_export_img.clicked.connect(self.export_image)
        toolbar.addWidget(btn_export_img)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # ── Main splitter: canvas + table ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Canvas
        self.canvas = AOICanvas(self)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setAccessibleName("AOI drawing canvas")
        splitter.addWidget(self.canvas)

        # Right panel: table + buttons
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)

        lbl_defs = QLabel("AOI Definitions")
        lbl_defs.setObjectName("title-label")
        right_layout.addWidget(lbl_defs)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Name', 'Type', 'Area %'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        self.table.cellDoubleClicked.connect(self._on_table_dbl_click)
        right_layout.addWidget(self.table)

        # ── AOI Controls Section ──
        controls_group = QGroupBox("AOI Controls")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)
        
        # Table buttons row
        tbl_btns = QHBoxLayout()
        btn_rename = QPushButton("Rename")
        btn_rename.setToolTip("Rename the selected AOI")
        btn_rename.clicked.connect(self._rename_selected)
        tbl_btns.addWidget(btn_rename)
        btn_copy = QPushButton("Copy")
        btn_copy.setToolTip("Duplicate the selected AOI")
        btn_copy.clicked.connect(self._copy_selected)
        tbl_btns.addWidget(btn_copy)
        btn_delete = QPushButton("Delete")
        btn_delete.setToolTip("Delete the selected AOI")
        btn_delete.clicked.connect(self._delete_selected)
        tbl_btns.addWidget(btn_delete)
        controls_layout.addLayout(tbl_btns)

        # Scale controls row
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale %:"))
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(1, 100)
        self.spin_scale.setValue(10)
        self.spin_scale.setToolTip("Percentage to shrink or expand the selected AOI")
        scale_row.addWidget(self.spin_scale)
        
        btn_shrink = QPushButton("- Shrink")
        btn_shrink.setToolTip("Shrink selected AOI by percentage (centered)")
        btn_shrink.clicked.connect(self._shrink_selected)
        scale_row.addWidget(btn_shrink)
        
        btn_expand = QPushButton("+ Expand")
        btn_expand.setToolTip("Expand selected AOI by percentage (centered)")
        btn_expand.clicked.connect(self._expand_selected)
        scale_row.addWidget(btn_expand)
        controls_layout.addLayout(scale_row)
        
        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # ── Analysis section ──
        analysis_group = QGroupBox("AOI Analysis")
        analysis_layout = QHBoxLayout()

        analysis_layout.addWidget(QLabel("Data:"))
        self.lbl_data_file = QLabel("(auto-loaded from transformed_data/)")
        analysis_layout.addWidget(self.lbl_data_file, 1)

        btn_browse_data = QPushButton("Browse…")
        btn_browse_data.setToolTip("Browse to select a gaze data file manually")
        btn_browse_data.clicked.connect(self._browse_data)
        analysis_layout.addWidget(btn_browse_data)

        self.btn_run = QPushButton("Run AOI Analysis")
        self.btn_run.setStyleSheet(button_style("success", padding="8px 16px"))
        self.btn_run.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.btn_run)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        self._data_path = None
        self._template = None
        self._template_error = None

        # Auto-save settings on scale change
        self.spin_scale.valueChanged.connect(self._on_settings_edited)

    # ── Project path ──
    def set_project_path(self, path):
        self.current_project_path = path
        self._data_path = None
        self._set_notice(True)
        if self._project_settings:
            selected = self._project_settings.resolve_transformed_data_file("")
            if selected:
                self._data_path = selected
                self.lbl_data_file.setText(os.path.basename(selected))
                self._set_notice(False)
            else:
                expected = self._project_settings.expected_transformed_data_file("")
                if expected:
                    self.lbl_notice.setText(
                        f"{status_icon_warning} Expected transformed data not found:\n"
                        f"{os.path.relpath(expected, self.current_project_path)}\n"
                        "Run Step 3 (Transformation) for the selected source dataset, or load a different transformed file."
                    )
                    self.lbl_notice.show()

        # Load the template associated with the selected dataset.
        self._sync_dataset_template()
        self._auto_load()

        # Populate stimulus image selector
        self._populate_stim_images()

    # ── Settings persistence ──
    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings
        selected = settings.resolve_transformed_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        if selected and os.path.isfile(selected) and selected != getattr(self, "_data_path", None):
            self._syncing_selection = True
            try:
                self._data_path = selected
                self.lbl_data_file.setText(os.path.basename(selected))
                self._set_notice(False)
            finally:
                self._syncing_selection = False
        self._sync_dataset_template()
        self.spin_scale.blockSignals(True)
        try:
            scale = settings.get("aoi_analysis", "scale_percentage", "")
            if scale:
                self.spin_scale.setValue(int(scale))
        finally:
            self.spin_scale.blockSignals(False)

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        settings.set("aoi_analysis", "scale_percentage",
                     self.spin_scale.value())

    def _on_settings_edited(self, *args):
        """Auto-save settings when UI changes."""
        if self._project_settings:
            self.save_settings(self._project_settings)

    # ── Toolbar handlers ──
    def _on_mode_changed(self, text):
        self.canvas.set_mode(text.lower())

    def refresh_data(self):
        """Reload project path to refresh file lists."""
        logger.debug("AOIAnalysis.refresh_data called. Path=%s", self.current_project_path)
        if self.current_project_path:
            self.set_project_path(self.current_project_path)

    def _on_shape_changed(self, text):
        self.canvas.set_shape_type(text.lower())
        # Auto-toggle snap: only for polygon
        is_polygon = text.lower() == 'polygon'
        self.chk_snap.setChecked(is_polygon)

    def _on_landmarks_toggled(self, checked):
        self.canvas._show_landmarks = checked
        self.canvas.update()

    def _on_snap_toggled(self, checked):
        self.canvas._snap_enabled = checked
        if not checked:
            self.canvas._snapped_idx = -1
            self.canvas.update()

    def _on_context_changed(self, text):
        """Switch between Template Face and Stimulus AOI contexts."""
        is_stimulus = (text == 'Stimulus')
        self._aoi_context = 'stimulus' if is_stimulus else 'template'

        # Show/hide stimulus image selector
        self.lbl_stim_select.setVisible(is_stimulus)
        self.combo_stim_image.setVisible(is_stimulus)

        if not is_stimulus:
            self.video_controls.hide()
            self._active_interval = None

        # Show/hide landmark controls (not relevant for stimulus)
        self.chk_landmarks.setVisible(not is_stimulus)
        self.chk_snap.setVisible(not is_stimulus)

        # Swap canvas background
        if is_stimulus:
            # Try to load selected stimulus image
            self._load_stimulus_background()
            self._refresh_intervals()
            self.canvas._show_landmarks = False
        else:
            self.canvas._bg_pixmap = self.canvas._template_pixmap
            self.canvas._show_landmarks = self.chk_landmarks.isChecked()
            self._refresh_gaze_overlay()

        # Reset selection and refresh table
        self.selected_idx = -1
        self.canvas.set_mode(self.combo_mode.currentText().lower())
        self._refresh_table()
        self.canvas.zoom_reset()

    def _on_stim_image_changed(self, text):
        """Load a different stimulus image as the canvas background."""
        if text and self._aoi_context == 'stimulus':
            self._active_interval = None
            self._load_stimulus_background()
            self._refresh_intervals()

    def _populate_stim_images(self):
        """Fill the stimulus image combo from the project's stimuli folder."""
        self.combo_stim_image.blockSignals(True)
        self.combo_stim_image.clear()
        if not self.current_project_path:
            self.combo_stim_image.blockSignals(False)
            return
        stim_dir = os.path.join(self.current_project_path, 'stimuli')
        if not os.path.isdir(stim_dir):
            self.combo_stim_image.blockSignals(False)
            return
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        for f in sorted(os.listdir(stim_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                self.combo_stim_image.addItem(f)
        self.combo_stim_image.blockSignals(False)
        if self._aoi_context == 'stimulus':
            self._on_stim_image_changed(self.combo_stim_image.currentText())

    def _load_stimulus_background(self):
        """Load the currently-selected stimulus image onto the canvas."""
        import cv2
        stim_name = self.combo_stim_image.currentText()
        if not stim_name or not self.current_project_path:
            self.canvas._bg_pixmap = None
            self._refresh_gaze_overlay()
            return
        stim_path = os.path.join(self.current_project_path, 'stimuli', stim_name)
        if not os.path.isfile(stim_path):
            self.canvas._bg_pixmap = None
            self._refresh_gaze_overlay()
            return
        ext = os.path.splitext(stim_name)[1].lower()
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(stim_path)
            self._video_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.video_controls.setVisible(self._aoi_context == 'stimulus')
            for spin in (self.spin_key_frame, self.spin_start_frame, self.spin_end_frame):
                spin.blockSignals(True)
                spin.setMaximum(self._video_frames)
                spin.blockSignals(False)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.spin_key_frame.value() - 1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.canvas._bg_pixmap = None
                self._refresh_gaze_overlay()
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.video_controls.hide()
            self._active_interval = None
            frame = cv2.imread(stim_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                self.canvas._bg_pixmap = None
                self._refresh_gaze_overlay()
                return
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        ch = frame.shape[2] if len(frame.shape) == 3 else 1
        fmt = QImage.Format.Format_RGBA8888 if ch == 4 else QImage.Format.Format_RGB888
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        q_img = QImage(frame.tobytes(), w, h, ch * w, fmt)
        pixmap = QPixmap.fromImage(q_img)
        self.canvas._bg_pixmap = pixmap
        self._refresh_gaze_overlay()

    def _preview_key_frame(self):
        if self._aoi_context == 'stimulus':
            self._load_stimulus_background()

    def _refresh_intervals(self):
        self.combo_interval.blockSignals(True)
        self.combo_interval.clear()
        self.combo_interval.addItem("Global AOIs (all frames)", None)
        for interval_index, interval in enumerate(self.video_intervals):
            if interval['stimulus'] == self.combo_stim_image.currentText():
                self.combo_interval.addItem(
                    f"Frames {interval['start_frame']}–{interval['end_frame']} (key {interval['key_frame']})",
                    interval_index)
        index = next((i for i in range(1, self.combo_interval.count())
                      if self.video_intervals[self.combo_interval.itemData(i)] is self._active_interval), 0)
        self.combo_interval.setCurrentIndex(index)
        self.combo_interval.blockSignals(False)
        self._select_interval(index)

    def _select_interval(self, index):
        interval_index = self.combo_interval.itemData(index)
        self._active_interval = (self.video_intervals[interval_index]
                                 if interval_index is not None else None)
        if self._active_interval is not None:
            for spin, field in ((self.spin_key_frame, 'key_frame'),
                                (self.spin_start_frame, 'start_frame'),
                                (self.spin_end_frame, 'end_frame')):
                spin.blockSignals(True)
                spin.setValue(self._active_interval[field])
                spin.blockSignals(False)
        self.selected_idx = -1
        self._preview_key_frame()
        self._refresh_table()

    def _interval_values(self):
        start, end, key = (self.spin_start_frame.value(), self.spin_end_frame.value(),
                           self.spin_key_frame.value())
        if not start <= key <= end:
            show_warning(self, "Invalid interval", "Choose Start ≤ Key frame ≤ End.")
            return None
        stimulus = self.combo_stim_image.currentText()
        for interval in self.video_intervals:
            if interval is not self._active_interval and interval['stimulus'] == stimulus:
                if start <= interval['end_frame'] and end >= interval['start_frame']:
                    show_warning(self, "Overlapping intervals", "Intervals for the same video must not overlap.")
                    return None
        return dict(stimulus=stimulus, start_frame=start, end_frame=end, key_frame=key)

    def _add_interval(self):
        previous = self._active_interval
        self._active_interval = None
        values = self._interval_values()
        self._active_interval = previous
        if values is not None:
            values['shapes'] = []
            self.video_intervals.append(values)
            self._active_interval = values
            self._refresh_intervals()

    def _update_interval(self):
        if self._active_interval is None:
            return
        values = self._interval_values()
        if values is not None:
            self._active_interval.update(values)
            self._refresh_intervals()

    def _delete_interval(self):
        if self._active_interval is not None:
            self.video_intervals.remove(self._active_interval)
            self._active_interval = None
            self._refresh_intervals()

    def _analysis_stimulus_shapes(self):
        shapes = [dict(shape) for shape in self.stimulus_shapes]
        for interval in self.video_intervals:
            for shape in interval['shapes']:
                shapes.append(dict(shape, stimulus=interval['stimulus'],
                                   start_frame=interval['start_frame'], end_frame=interval['end_frame']))
        return shapes

    # ── Auto-save / Auto-load ──
    def _auto_save(self):
        """Save AOI definitions to project folder automatically."""
        if not self.current_project_path:
            return
        path = os.path.join(self.current_project_path, 'aoi_definitions.json')

        def _serialize(shape_list):
            result = []
            for aoi in shape_list:
                entry = {
                    'name': aoi['name'],
                    'type': aoi['type'],
                    'coords': aoi['coords'] if aoi['type'] == 'polygon' else list(aoi['coords'])
                }
                result.append(entry)
            return result

        export_data = {
            'template_aois': _serialize(self.shapes),
            'stimulus_aois': _serialize(self.stimulus_shapes),
            'video_intervals': self.video_intervals,
        }
        try:
            existing = json.loads(Path(path).read_text()) if os.path.isfile(path) else {}
            if isinstance(existing,list): existing = {'template_aois':existing}
            if 'aois' in existing and 'template_aois' not in existing: existing['template_aois'] = existing['aois']
            sets = existing.get('template_sets',{})
            if 'template_aois' in existing and 'default' not in sets:
                sets['default'] = {'aois':existing['template_aois'],'template_hash':None}
            tid = self._template.template_id if self._template else 'default'
            sets[tid] = {'aois':export_data.pop('template_aois'), 'template_hash':self._template.content_hash if self._template else None}
            export_data['template_sets'] = sets
            temp_path = Path(path).with_suffix('.tmp')
            temp_path.write_text(json.dumps(export_data,indent=2))
            os.replace(temp_path,path)
        except Exception as exc:
            logger.error('Could not save AOI definitions: %s',exc)

    def _auto_load(self):
        """Load AOI definitions from project folder if available."""
        if not self.current_project_path:
            return
        path = os.path.join(self.current_project_path, 'aoi_definitions.json')
        self.video_intervals = []
        self._active_interval = None
        if not os.path.isfile(path):
            self.shapes = []
            self.stimulus_shapes = []
            self.selected_idx = -1
            self._refresh_table()
            self.canvas.update()
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            return

        def _deserialize(items):
            result = []
            for item in items:
                aoi = {
                    'name': item.get('name', f'AOI_{len(result)+1}'),
                    'type': item.get('type', 'rectangle'),
                    'coords': item.get('coords', (0, 0, 0.1, 0.1))
                }
                if aoi['type'] == 'polygon':
                    aoi['coords'] = [tuple(pt) for pt in aoi['coords']]
                else:
                    aoi['coords'] = tuple(aoi['coords'])
                result.append(aoi)
            return result

        if isinstance(data,list):
            data = {'template_aois':data,'stimulus_aois':[]}
        if isinstance(data,dict):
            tid = self._template.template_id if self._template else 'default'
            if 'template_sets' in data:
                definition = data['template_sets'].get(tid,{})
                expected_hash = definition.get('template_hash')
                if expected_hash and self._template and expected_hash != self._template.content_hash:
                    logger.warning('Skipping AOIs whose template hash does not match the dataset.')
                    definition = {}
                data['template_aois'] = definition.get('aois',[])
            elif tid != data.get('template_id','default'):
                data['template_aois'] = []

        self.video_intervals = data.get('video_intervals', []) if isinstance(data, dict) else []
        self._active_interval = None

        # Backward compatibility: old format has 'aois' key
        if 'template_aois' in data:
            self.shapes = _deserialize(data.get('template_aois', []))
            self.stimulus_shapes = _deserialize(data.get('stimulus_aois', []))
        else:
            loaded = data.get('aois', data if isinstance(data, list) else [])
            self.shapes = _deserialize(loaded)
            self.stimulus_shapes = []

        self.selected_idx = -1
        self._refresh_table()
        self.canvas.update()

    # ── Table management ──
    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.active_shapes))
        for i, aoi in enumerate(self.active_shapes):
            name_item = QTableWidgetItem(aoi['name'])
            type_item = QTableWidgetItem(aoi['type'])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            area_item = QTableWidgetItem(self._compute_aoi_area_pct(aoi))
            area_item.setFlags(area_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, name_item)
            self.table.setItem(i, 1, type_item)
            self.table.setItem(i, 2, area_item)
        self.table.blockSignals(False)
        self._sync_selection()
        self._auto_save()

    def _sync_selection(self):
        """Sync table selection with selected_idx."""
        self.table.blockSignals(True)
        if 0 <= self.selected_idx < len(self.active_shapes):
            self.table.selectRow(self.selected_idx)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        self.canvas.update()

    def _on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.selected_idx = rows[0].row()
            self.combo_mode.setCurrentText('Edit')
            self.canvas.set_mode('edit')
            self.canvas.update()

    def _on_table_dbl_click(self, row, col):
        if col == 0:
            self._rename_selected()

    def _rename_selected(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.active_shapes):
            return
        current_name = self.active_shapes[self.selected_idx]['name']
        new_name, ok = QInputDialog.getText(self, "Rename AOI", "Name:", text=current_name)
        if ok and new_name.strip():
            self.active_shapes[self.selected_idx]['name'] = new_name.strip()
            self._refresh_table()
            self.canvas.update()

    def _copy_selected(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.active_shapes):
            return
        import copy
        aoi = copy.deepcopy(self.active_shapes[self.selected_idx])
        aoi['name'] = aoi['name'] + '_copy'
        self.active_shapes.append(aoi)
        self.selected_idx = len(self.active_shapes) - 1
        self._refresh_table()
        self.canvas.update()

    def _delete_selected(self):
        if self.selected_idx < 0 or self.selected_idx >= len(self.active_shapes):
            return
        del self.active_shapes[self.selected_idx]
        self.selected_idx = -1
        self._refresh_table()
        self.canvas.update()

    def _shrink_selected(self):
        """Shrink the selected AOI by the percentage in spin_scale."""
        pct = self.spin_scale.value()
        factor = 1.0 - pct / 100.0
        self._scale_selected(factor)

    def _expand_selected(self):
        """Expand the selected AOI by the percentage in spin_scale."""
        pct = self.spin_scale.value()
        factor = 1.0 + pct / 100.0
        self._scale_selected(factor)

    def _scale_selected(self, factor):
        """Scale the selected AOI by factor, centered on its center/centroid."""
        if self.selected_idx < 0 or self.selected_idx >= len(self.active_shapes):
            return
        aoi = self.active_shapes[self.selected_idx]
        
        if aoi['type'] in ('rectangle', 'oval'):
            x0, y0, x1, y1 = aoi['coords']
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
            new_hw, new_hh = hw * factor, hh * factor
            aoi['coords'] = (cx - new_hw, cy - new_hh, cx + new_hw, cy + new_hh)
        
        elif aoi['type'] == 'polygon':
            coords = aoi['coords']
            if len(coords) < 3:
                return
            # Calculate centroid
            cx = sum(p[0] for p in coords) / len(coords)
            cy = sum(p[1] for p in coords) / len(coords)
            # Scale each vertex from centroid
            new_coords = []
            for px, py in coords:
                dx, dy = px - cx, py - cy
                new_coords.append((cx + dx * factor, cy + dy * factor))
            aoi['coords'] = new_coords
        
        self._refresh_table()
        self.canvas.update()
        self._auto_save()

    # ── Import / Export ──
    def import_aoi(self):
        start_dir = self.current_project_path or ''
        path, _ = QFileDialog.getOpenFileName(
            self, "Import AOI Definitions", start_dir,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            show_error(self, "Import Failed", f"Could not read the AOI file:\n{e}")
            return

        tid = self._template.template_id if self._template else 'default'
        if isinstance(data,list): data = {'template_aois':data,'template_id':'default'}
        if isinstance(data,dict) and 'template_sets' in data:
            definition = data['template_sets'].get(tid,{})
            data['template_aois'] = definition.get('aois',[])
            data['template_id'] = tid
            data['template_hash'] = definition.get('template_hash')
        if isinstance(data,dict) and data.get('template_aois',data.get('aois',[])) and (data.get('template_id','default') != tid or
                (data.get('template_hash') and self._template and data['template_hash'] != self._template.content_hash)):
            show_warning(self,'Template Mismatch','These facial AOIs belong to a different template. Recreate them for the current template.')
            return

        def _deserialize(items):
            result = []
            for item in items:
                aoi = {
                    'name': item.get('name', f'AOI_{len(result)+1}'),
                    'type': item.get('type', 'rectangle'),
                    'coords': item.get('coords', (0, 0, 0.1, 0.1))
                }
                if aoi['type'] == 'polygon':
                    aoi['coords'] = [tuple(pt) for pt in aoi['coords']]
                else:
                    aoi['coords'] = tuple(aoi['coords'])
                result.append(aoi)
            return result

        self.video_intervals = data.get('video_intervals', []) if isinstance(data, dict) else []
        self._active_interval = None

        # Handle both new format (template_aois/stimulus_aois) and legacy (aois)
        if 'template_aois' in data:
            self.shapes = _deserialize(data.get('template_aois', []))
            self.stimulus_shapes = _deserialize(data.get('stimulus_aois', []))
        else:
            loaded = data.get('aois', data if isinstance(data, list) else [])
            self.shapes = _deserialize(loaded)
            self.stimulus_shapes = []

        self.selected_idx = -1
        self._refresh_table()
        self.canvas.update()
        self._refresh_intervals()
        total = len(self.shapes) + len(self._analysis_stimulus_shapes())
        show_success(self, "Import Complete", f"Imported {total} AOIs from {os.path.basename(path)}")

    def export_aoi(self):
        total = len(self.shapes) + len(self._analysis_stimulus_shapes())
        if not total:
            show_warning(self, "No AOIs Defined", "Draw at least one Area of Interest on the canvas before exporting.")
            return
        default_dir = self.current_project_path or ''
        path, _ = QFileDialog.getSaveFileName(
            self, "Export AOI Definitions",
            os.path.join(default_dir, 'aoi_definitions.json'),
            "JSON Files (*.json)"
        )
        if not path:
            return

        export_data = dict(template_id=self._template.template_id if self._template else "default",
                           template_hash=self._template.content_hash if self._template else None,
                           template_aois=self.shapes, stimulus_aois=self.stimulus_shapes,
                           video_intervals=self.video_intervals)

        try:
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.lbl_data_file.setText(f"Exported {total} AOIs to {os.path.basename(path)}")
            show_success(self, "Export Complete", f"Exported {total} AOI definitions to {os.path.basename(path)}")
            self.data_changed.emit()
        except Exception as e:
            show_error(self, "Export Failed", f"Could not export AOI definitions:\n{e}")

    # ── Data browsing ──
    def _browse_data(self):
        start_dir = os.path.join(self.current_project_path, 'transformed_data') \
            if self.current_project_path else ''
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data", start_dir,
            "Data Files (*.parquet *.csv)"
        )
        if path:
            self._set_transformed_data_file(path, update_settings=True)

    def _sync_dataset_template(self):
        from core.template_manager import TemplateManager
        try:
            spec = TemplateManager(self.current_project_path).for_dataset(self._data_path) if self._data_path else TemplateManager(self.current_project_path).load()
            changed = self._template is None or self._template.template_id != spec.template_id
            self._template = spec
            self._template_error = None
            self.canvas.set_template(spec)
            if changed: self._auto_load()
            self._refresh_gaze_overlay()
        except Exception as exc:
            self._template_error = str(exc)
            self._template = None
            self.canvas._bg_pixmap = None
            self.canvas._template_pixmap = None
            self.canvas._landmarks_norm = []
            self.shapes = []
            self.lbl_notice.setText('Template unavailable: '+str(exc))
            self.lbl_notice.show()
            self._refresh_gaze_overlay()

    def _set_transformed_data_file(self, path, update_settings=False):
        if not path or not os.path.isfile(path):
            return
        self._data_path = path
        self._sync_dataset_template()
        self.lbl_data_file.setText(os.path.basename(path))
        self._set_notice(False)
        if update_settings:
            self._set_global_transformed_file(path)

    def _set_notice(self, show):
        if getattr(self,'_template_error',None):
            self.lbl_notice.setText('Template unavailable: '+self._template_error)
            self.lbl_notice.show()
            return
        if show:
            self.lbl_notice.setText(
                "No transformed data found. Run Step 3 (Transformation) to generate "
                "files in transformed_data/."
            )
            self.lbl_notice.show()
        else:
            self.lbl_notice.hide()

    def _set_global_transformed_file(self, path, only_if_empty=False):
        if not self._project_settings or self._syncing_selection:
            return
        if only_if_empty:
            existing = self._project_settings.get_transformed_data_file("")
            if existing:
                return
        self._project_settings.set("global", "selected_transformed_file", path)

    # ── Gaze overlay ──
    def _load_gaze_data(self):
        """Return the gaze columns of the selected data file (cached per path)."""
        path = self._data_path
        if not path or not os.path.isfile(path):
            return None
        if path != self._gaze_df_path:
            self._gaze_df, self._gaze_df_path = None, path
            try:
                if path.lower().endswith('.parquet'):
                    import pyarrow.parquet as pq
                    names = set(pq.read_schema(path).names)
                    self._gaze_df = pd.read_parquet(
                        path, columns=[c for c in GAZE_DATA_COLUMNS if c in names])
                else:
                    self._gaze_df = pd.read_csv(path, usecols=lambda c: c in GAZE_DATA_COLUMNS)
            except Exception as exc:
                logger.warning("Could not load gaze data for AOI overlay: %s", exc)
        return self._gaze_df

    def _gaze_rows_for_view(self, df):
        """Return (rows, x_col, y_col, default_bgr) for the current AOI context."""
        if self._aoi_context == 'template':
            return df, 'transformed_x', 'transformed_y', (0, 0, 255)  # red, as in Visualization
        stim_col = next((c for c in ('stimulusName', 'videoName', 'STIMULI_NAME') if c in df), None)
        stim_name = self.combo_stim_image.currentText()
        if stim_col is None or not stim_name:
            return df.iloc[0:0], 'stimuli_x', 'stimuli_y', (0, 255, 0)
        # Match the way AOI analysis compares stimulus names (basename, no extension).
        names = df[stim_col].fillna('').astype(str).map(
            lambda v: os.path.splitext(v.replace('\\', '/').rsplit('/', 1)[-1])[0])
        mask = names.eq(os.path.splitext(stim_name)[0])
        if self._active_interval is not None and '_video_frame_index' in df:
            frames = pd.to_numeric(df['_video_frame_index'], errors='coerce')
            mask &= frames.between(self._active_interval['start_frame'],
                                   self._active_interval['end_frame'])
        return df[mask], 'stimuli_x', 'stimuli_y', (0, 255, 0)  # green, as in Visualization

    def _refresh_gaze_overlay(self, *args):
        """Rebuild the canvas gaze layer and legend for the current view."""
        canvas = self.canvas
        canvas._gaze_layer, canvas._gaze_legend, self._gaze_groups = None, [], []
        df = self._load_gaze_data() if self.chk_show_gaze.isChecked() else None
        has_region = df is not None and 'gaze_region' in df
        self.chk_highlight_off_face.setEnabled(self.chk_show_gaze.isChecked() and has_region)
        bg = canvas._bg_pixmap
        if df is None or bg is None or bg.isNull():
            canvas.update()
            return

        rows, x_col, y_col, default_bgr = self._gaze_rows_for_view(df)
        if x_col not in rows or y_col not in rows:
            canvas._gaze_legend = [(f"No {x_col}/{y_col} columns in data", None)]
            canvas.update()
            return
        xs = pd.to_numeric(rows[x_col], errors='coerce').to_numpy(dtype=float)
        ys = pd.to_numeric(rows[y_col], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        regions = (rows['gaze_region'].astype('string').fillna('').to_numpy(dtype=object)
                   if has_region else np.full(len(rows), '', dtype=object))
        xs, ys, regions = xs[valid], ys[valid], regions[valid]
        if not len(xs):
            canvas._gaze_legend = [("No gaze samples for this view", None)]
            canvas.update()
            return

        n_total = len(xs)
        if n_total > MAX_GAZE_POINTS:
            keep = np.random.default_rng(0).choice(n_total, MAX_GAZE_POINTS, replace=False)
            xs, ys, regions = xs[keep], ys[keep], regions[keep]

        if has_region and self.chk_highlight_off_face.isChecked():
            off, unmapped = regions == 'off_face', regions == 'unmapped'
            groups = [("On-face", default_bgr, ~(off | unmapped)),
                      ("Unmapped", GAZE_UNMAPPED_BGR, unmapped),
                      ("Off-face", GAZE_OFF_FACE_BGR, off)]  # drawn last, on top
        else:
            groups = [("Gaze", default_bgr, np.ones(len(xs), dtype=bool))]
        self._gaze_groups = [(label, bgr, xs[m], ys[m]) for label, bgr, m in groups if m.any()]

        # Render at >=1500 px on the long side so dots stay smooth on small stimuli.
        scale = max(1.0, 1500 / max(bg.width(), bg.height()))
        w, h = round(bg.width() * scale), round(bg.height() * scale)
        radius = max(2, round(min(w, h) / 300))
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        for _, (b, g, r), gx, gy in self._gaze_groups:
            for px, py in zip((gx * w).astype(int), (gy * h).astype(int)):
                cv2.circle(layer, (int(px), int(py)), radius, (r, g, b, 255), -1, cv2.LINE_AA)
        q_img = QImage(layer.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        canvas._gaze_layer = QPixmap.fromImage(q_img.copy())

        legend = [(f"{label} ({len(gx):,})", QColor(*bgr[::-1]))
                  for label, bgr, gx, _ in self._gaze_groups]
        if n_total > MAX_GAZE_POINTS:
            legend.append((f"Showing {MAX_GAZE_POINTS:,} of {n_total:,} samples", None))
        canvas._gaze_legend = legend
        canvas.update()

    # ── Analysis ──
    def run_analysis(self):
        if self._template_error:
            show_error(self,'Template Unavailable',self._template_error)
            return
        has_template = bool(self.shapes)
        has_stimulus = bool(self._analysis_stimulus_shapes())
        if not has_template and not has_stimulus:
            show_warning(self, "No AOIs Defined", "Draw at least one Area of Interest on the canvas before running analysis.")
            return
        if not self._data_path or not os.path.isfile(self._data_path):
            show_warning(self, "No Data Loaded", "Load or select a transformed data file first.")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running…")
        self._overlay.set_status("Running AOI Analysis")
        self._overlay.hide_progress()
        self._overlay.show()

        self._aoi_worker = _AoiWorker(
            self._data_path,
            [dict(s) for s in self.shapes],
            self._analysis_stimulus_shapes(),
        )
        self._aoi_worker.finished.connect(self._on_analysis_done)
        self._aoi_worker.error.connect(self._on_analysis_error)
        self._aoi_worker.start()

    def _on_analysis_done(self, result):
        self._overlay.hide()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run AOI Analysis")
        parts = [f"{name}: {count:,}/{result['total']:,}" for name, count in result['parts']]
        summary = f"AOI analysis complete (columns: {', '.join(result['aoi_names'])}).\n"
        summary += f"Hits per AOI: {'; '.join(parts)}.\nSaved to {os.path.basename(result['out_path'])}"
        show_success(self, "AOI Analysis Complete", summary)
        self.data_changed.emit()

    def _on_analysis_error(self, err_msg):
        self._overlay.hide()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run AOI Analysis")
        show_error(self, "Analysis Failed", f"AOI analysis failed:\n{err_msg}")

    @staticmethod
    def _hit_test_vectorized(tx, ty, aoi):
        """Vectorized hit test for a single AOI against all points."""
        valid = np.isfinite(tx) & np.isfinite(ty)
        result = np.zeros(len(tx), dtype=bool)

        if aoi['type'] == 'rectangle':
            x0, y0, x1, y1 = aoi['coords']
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            result = valid & (tx >= xmin) & (tx <= xmax) & (ty >= ymin) & (ty <= ymax)

        elif aoi['type'] == 'oval':
            x0, y0, x1, y1 = aoi['coords']
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx > 0 and ry > 0:
                result = valid & (((tx - cx) ** 2) / (rx ** 2) + ((ty - cy) ** 2) / (ry ** 2) <= 1)

        elif aoi['type'] == 'polygon':
            from matplotlib.path import Path as MplPath
            poly = aoi['coords']
            n = len(poly)
            if n < 3:
                return result
            poly_path = MplPath(poly, closed=True)
            points = np.column_stack((tx, ty))
            result = valid & poly_path.contains_points(points)

        return result

    # ── Area % computation ──
    _FACE_INDICES = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
                     389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                     400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                     132, 93, 234, 127, 162]

    @staticmethod
    def _shoelace(coords):
        """Shoelace formula for the area of a simple polygon."""
        n = len(coords)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _face_area_norm(self):
        """Area of the face contour in normalised [0,1] space."""
        lm = self.canvas._landmarks_norm
        if not lm:
            return 1.0  # fallback to avoid division by zero
        face_pts = [lm[i] for i in self._FACE_INDICES if i < len(lm)]
        area = self._shoelace(face_pts)
        return area if area > 0 else 1.0

    def _compute_aoi_area_pct(self, aoi):
        """Return AOI area as a formatted percentage of the reference area."""
        # In stimulus context, reference area is full image (1.0)
        if self._aoi_context == 'stimulus':
            ref_area = 1.0
        else:
            ref_area = self._face_area_norm()
        if aoi['type'] == 'rectangle':
            x0, y0, x1, y1 = aoi['coords']
            aoi_area = abs(x1 - x0) * abs(y1 - y0)
        elif aoi['type'] == 'oval':
            x0, y0, x1, y1 = aoi['coords']
            rx = abs(x1 - x0) / 2
            ry = abs(y1 - y0) / 2
            aoi_area = math.pi * rx * ry
        elif aoi['type'] == 'polygon':
            aoi_area = self._shoelace(aoi['coords'])
        else:
            aoi_area = 0.0
        pct = (aoi_area / ref_area) * 100
        return f"{pct:.1f}%"

    # ── Export Image ──
    def export_image(self):
        """Export a publication-quality image with AOIs drawn on the current background."""
        if not self.active_shapes:
            show_warning(self, "No AOIs Defined", "Draw at least one Area of Interest on the canvas before exporting an image.")
            return

        default_dir = self.current_project_path or ''
        ctx_suffix = '_stimulus' if self._aoi_context == 'stimulus' else ''
        path, _ = QFileDialog.getSaveFileName(
            self, "Export AOI Image",
            os.path.join(default_dir, f'AOI{ctx_suffix}.png'),
            "PNG Files (*.png);;All Files (*)"
        )
        if not path:
            return

        # Distinct colors for AOIs
        color_cycle = [
            '#000000', '#0000FF', '#FF0000', '#00AA00', '#FF8800',
            '#8800CC', '#00BBBB', '#CC0066', '#668800', '#0066CC'
        ]

        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            # Draw the background image
            if self._aoi_context == 'stimulus':
                import cv2
                stim_name = self.combo_stim_image.currentText()
                stim_path = os.path.join(self.current_project_path, 'stimuli', stim_name) if stim_name else ''
                if stim_path and os.path.isfile(stim_path):
                    img = cv2.imread(stim_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img, extent=[0, 1, 1, 0])
                    else:
                        ax.set_xlim(0, 1)
                        ax.set_ylim(1, 0)
                else:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(1, 0)
            elif self._template is not None:
                avg = cv2.cvtColor(self._template.image,cv2.COLOR_BGRA2RGBA)
                ax.imshow(avg, extent=[0, 1, 1, 0])
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)

            legend_handles = []
            for i, aoi in enumerate(self.active_shapes):
                color = color_cycle[i % len(color_cycle)]
                if aoi['type'] == 'rectangle':
                    x0, y0, x1, y1 = aoi['coords']
                    rect = mpatches.Rectangle(
                        (min(x0, x1), min(y0, y1)),
                        abs(x1 - x0), abs(y1 - y0),
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                elif aoi['type'] == 'oval':
                    x0, y0, x1, y1 = aoi['coords']
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    w_e, h_e = abs(x1 - x0), abs(y1 - y0)
                    ellipse = mpatches.Ellipse(
                        (cx, cy), w_e, h_e,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(ellipse)
                elif aoi['type'] == 'polygon':
                    poly = plt.Polygon(
                        aoi['coords'], closed=True,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(poly)
                legend_handles.append(
                    mpatches.Patch(edgecolor=color, facecolor='none',
                                   linewidth=2, label=aoi['name'])
                )

            if self.chk_show_gaze.isChecked() and self._gaze_groups:
                from matplotlib.lines import Line2D
                for label, bgr, gx, gy in self._gaze_groups:
                    ax.scatter(gx, gy, s=3, c=bgr_to_hex(bgr), alpha=0.5,
                               edgecolors='none', zorder=0.5)  # under AOI outlines
                    legend_handles.append(Line2D(
                        [], [], marker='o', linestyle='none', markersize=6,
                        markerfacecolor=bgr_to_hex(bgr), markeredgecolor='none',
                        label=f"{label} gaze (n={len(gx):,})"))

            ax.legend(handles=legend_handles, loc='upper right',
                      framealpha=0.9, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect('equal')
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)

            show_success(self, "Export Complete",
                                    f"Saved AOI image to {os.path.basename(path)}")
        except Exception as e:
            show_error(self, "Export Failed", f"Could not export AOI image:\n{e}")

    # ── Cleanup ──
    def closeEvent(self, event):
        if hasattr(self, '_aoi_worker') and self._aoi_worker.isRunning():
            self._aoi_worker.wait()
