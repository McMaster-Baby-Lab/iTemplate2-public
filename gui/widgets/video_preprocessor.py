from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QListWidget, QHBoxLayout,
                             QSpinBox, QComboBox, QCheckBox, QSlider, QFrame,
                             QSplitter, QGroupBox, QGridLayout, QTabWidget,
                             QSizePolicy, QStyledItemDelegate, QStyle,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QTableView, QStyleOptionComboBox, QStyleOptionButton,
                             QLineEdit, QDialog, QMenu, QBoxLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from .button_styles import button_style
from .theme import (
    SIDEBAR_STYLE, COMBOBOX_POPUP_STYLE, THEME,
    status_icon_success, status_icon_warning, status_icon_info,
)
from .dialog_helpers import show_success, show_warning, show_error, ask_confirm, ask_choice
import os
import sys
import math
import time
import tempfile
import logging
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

logger = logging.getLogger('iTemplate2')

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core import transformation as landmark_extraction
    from core.transformation import _process_video_chunk
except ImportError:
    landmark_extraction = None
    _process_video_chunk = None

try:
    from core.config import delaunay_tri as _delaunay_tri_cfg
except ImportError:
    _delaunay_tri_cfg = None


def _process_single_file(file_path, output_dir):
    """Legacy helper (now used for images)."""
    from core import transformation as le
    le.process_image(file_path, output_dir)
    return file_path


def _process_image_file(file_path, output_dir, export_csv_compat=False, num_faces=1):
    """Worker for images."""
    from core import transformation as le
    le.process_image(file_path, output_dir, export_csv=export_csv_compat, num_faces=num_faces)
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _pool_probe_worker():
    """Return process id so caller can verify pool width."""
    return os.getpid()


def _process_video_chunk_with_meta(args):
    """Run video chunk extraction and persist outputs to temp parquet files."""
    fp, start, end, chunk_id, tmp_dir, num_faces = args
    from core.transformation import _process_video_chunk as chunk_fn
    import pandas as _pd

    l, b, t = chunk_fn((fp, start, end, num_faces))
    pid = os.getpid()

    base = os.path.splitext(os.path.basename(fp))[0]
    stem = f"{base}_chunk{chunk_id:04d}_pid{pid}_{start}_{end}"

    l_path = None
    b_path = None
    t_path = None
    if isinstance(l, _pd.DataFrame) and not l.empty:
        l_path = os.path.join(tmp_dir, f"{stem}_landmarks.parquet")
        l.to_parquet(l_path, index=False)
    if isinstance(b, _pd.DataFrame) and not b.empty:
        b_path = os.path.join(tmp_dir, f"{stem}_blendshapes.parquet")
        b.to_parquet(b_path, index=False)
    if isinstance(t, _pd.DataFrame) and not t.empty:
        t_path = os.path.join(tmp_dir, f"{stem}_transformation.parquet")
        t.to_parquet(t_path, index=False)

    return {
        'pid': pid,
        'start': start,
        'end': end,
        'chunk_id': chunk_id,
        'landmarks_path': l_path,
        'blendshapes_path': b_path,
        'transformation_path': t_path,
    }


class _AutoMergeWorker(QThread):
    """Multi-core background worker for auto-merge gap interpolation."""
    progress = pyqtSignal(int, int)  # completed, total
    finished = pyqtSignal(bool, str)

    def __init__(self, csv_path, face_gap_tasks, fps):
        super().__init__()
        self.csv_path = csv_path
        self.face_gap_tasks = face_gap_tasks  # [(face_id, [(end, start), ...]), ...]
        self.fps = fps

    def run(self):
        from core.transformation import _glue_face_gaps_worker
        total = len(self.face_gap_tasks)
        completed = 0
        all_success = True
        messages = []

        with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as executor:
            futures = {
                executor.submit(_glue_face_gaps_worker,
                                (self.csv_path, fid, gaps, self.fps)): fid
                for fid, gaps in self.face_gap_tasks
            }
            for future in as_completed(futures):
                fid = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        messages.append(result['message'])
                    else:
                        all_success = False
                        messages.append(f"Face {fid}: {result['message']}")
                except Exception as e:
                    all_success = False
                    messages.append(f"Face {fid}: {e}")
                completed += 1
                self.progress.emit(completed, total)

        self.finished.emit(all_success, "\n".join(messages))


class _ReAnalysisWorker(QThread):
    """Re-run MediaPipe face detection on a frame range and merge results.

    The newly detected faces are assigned new Face_IDs starting from
    `new_face_id_offset`.  Existing landmark data for the filtered face
    in that range is removed before the new data is inserted.
    """
    progress = pyqtSignal(int, int)  # current_frame, total_frames
    finished = pyqtSignal(bool, str)

    def __init__(self, video_path, start_frame, end_frame, num_faces,
                 existing_face_id, new_face_id_offset, landmarks_path):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.num_faces = num_faces
        self.existing_face_id = existing_face_id
        self.new_face_id_offset = new_face_id_offset
        self.landmarks_path = landmarks_path

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished.emit(False, "Could not open video for re-analysis.")
                return

            total = self.end_frame - self.start_frame + 1
            fps = cap.get(cv2.CAP_PROP_FPS)

            from core.transformation import get_detector
            local_detector = get_detector(self.num_faces)

            import mediapipe as mp_mod

            new_rows = []
            frames_processed = 0

            for frame_idx in range(self.start_frame, self.end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / max(fps, 1.0)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb_frame)

                results = local_detector.detect(mp_image)
                if results.face_landmarks:
                    for idx, ls in enumerate(results.face_landmarks):
                        face_id = self.new_face_id_offset + idx
                        row = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                        for i, lm in enumerate(ls):
                            row[f'x_{i}'] = lm.x
                            row[f'y_{i}'] = lm.y
                            row[f'z_{i}'] = lm.z
                        new_rows.append(row)
                frames_processed += 1
                self.progress.emit(frame_idx, self.end_frame)

            cap.release()

            if not new_rows:
                self.finished.emit(False,
                                   "Re-analysis detected no faces in the selected frame range. "
                                   "MediaPipe may not have detected a face in these frames.")
                return

            new_df = pd.DataFrame(new_rows)

            # Load existing landmark data, remove old face in range, insert new
            if str(self.landmarks_path).lower().endswith('.parquet'):
                existing_df = pd.read_parquet(self.landmarks_path)
            else:
                existing_df = pd.read_csv(self.landmarks_path)

            # Remove existing face data in the range
            keep_mask = (
                (existing_df['Face_ID'] != self.existing_face_id) |
                (existing_df['Frame_Index'] < self.start_frame) |
                (existing_df['Frame_Index'] > self.end_frame)
            )
            existing_df = existing_df.loc[keep_mask]

            # Combine and sort
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)

            # Save back
            from core.io_utils import atomic_save_parquet
            if str(self.landmarks_path).lower().endswith('.parquet'):
                atomic_save_parquet(combined, self.landmarks_path)
            else:
                combined.to_csv(self.landmarks_path, index=False)

            new_fids = sorted(new_df['Face_ID'].unique().tolist())
            new_fid_str = ', '.join('Face ' + str(f) for f in new_fids)
            self.finished.emit(True,
                               f"Detected {len(new_rows)} rows ({len(new_fids)} face(s): {new_fid_str}). "
                               f"Removed old Face {self.existing_face_id} data in range. "
                               f"Existing faces outside this range are unchanged.")
        except Exception as e:
            import traceback
            self.finished.emit(False, f"Re-analysis failed: {e}\n{traceback.format_exc()}")


class _LandmarkLoadWorker(QThread):
    """Background worker for loading landmarks and building spatial cache (Step 2.1)."""
    finished = pyqtSignal(bool, object, dict, list, str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            import pandas as pd
            import numpy as np
            import pyarrow.csv as pacsv

            if str(self.file_path).lower().endswith('.parquet'):
                df = pd.read_parquet(self.file_path)
            else:
                # Use PyArrow for significantly faster multi-threaded CSV reading
                df = pacsv.read_csv(self.file_path).to_pandas()
            if 'Face_ID' not in df.columns:
                df['Face_ID'] = 1

            frame_indices = df['Frame_Index'].values
            face_ids = df['Face_ID'].values
            x_cols = [c for c in df.columns if c.startswith('x_')]
            y_cols = [c for c in df.columns if c.startswith('y_')]
            all_x = df[x_cols].values
            all_y = df[y_cols].values
            all_pts = np.stack([all_x, all_y], axis=2)

            # Issue 1+2: Single-threaded vectorized dict build
            # Sort by (frame_idx, face_id) for grouping
            sort_idx = np.lexsort((face_ids, frame_indices))
            frame_indices = frame_indices[sort_idx]
            face_ids = face_ids[sort_idx]
            all_pts = all_pts[sort_idx]

            # Find boundaries where frame_index changes
            frame_changes = np.where(np.diff(frame_indices) != 0)[0] + 1
            unique_frames = frame_indices[np.concatenate(([0], frame_changes))]

            # Split into per-frame groups and build dict
            frame_face_ids = np.split(face_ids, frame_changes)
            frame_all_pts = np.split(all_pts, frame_changes)

            lbf = {}
            for f, fids, pts in zip(unique_frames, frame_face_ids, frame_all_pts):
                lbf[int(f)] = dict(zip(fids.astype(int), pts))

            all_face_ids = sorted(df['Face_ID'].unique().tolist())
            self.finished.emit(True, df, lbf, all_face_ids, "")
            
        except Exception as e:
            self.finished.emit(False, None, {}, [], str(e))


class _CsvSaveWorker(QThread):
    """Background worker using PyArrow for high-performance multi-threaded CSV writing (Step 3.2)."""
    finished = pyqtSignal(bool, str)

    def __init__(self, arrow_table, path):
        super().__init__()
        self.table = arrow_table
        self.path = path

    def run(self):
        try:
            import pyarrow.csv as pacsv
            # PyArrow handles multi-threading natively for CSV writes
            pacsv.write_csv(self.table, self.path)
            self.finished.emit(True, self.path)
        except Exception as e:
            self.finished.emit(False, str(e))


class _ParquetSaveWorker(QThread):
    """Background worker for parquet writes."""
    finished = pyqtSignal(bool, str)

    def __init__(self, df, path):
        super().__init__()
        self.df = df
        self.path = path

    def run(self):
        try:
            from core.io_utils import atomic_save_parquet
            atomic_save_parquet(self.df, self.path)
            self.finished.emit(True, self.path)
        except Exception as e:
            self.finished.emit(False, str(e))


class _SectionDeleteWorker(QThread):
    """Background worker to delete sections from all three data files."""
    finished = pyqtSignal(bool, str)

    def __init__(self, landmarks_path, delete_tasks):
        super().__init__()
        self.landmarks_path = landmarks_path
        # delete_tasks: list of (face_id, start_frame, end_frame)
        self.delete_tasks = delete_tasks

    def run(self):
        try:
            import pandas as pd
            import numpy as np
            from core.io_utils import atomic_save_parquet

            base_dir = os.path.dirname(self.landmarks_path)
            stem = os.path.splitext(os.path.basename(self.landmarks_path))[0]
            # Strip '_landmarks' suffix to get the base video name
            if stem.endswith('_landmarks'):
                stem = stem[:-len('_landmarks')]

            file_stems = ['_landmarks', '_blendshapes', '_transformation']
            rows_per_file = []

            for fs in file_stems:
                fname = f"{stem}{fs}.parquet"
                fpath = os.path.join(base_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                df = pd.read_parquet(fpath)
                keep = np.ones(len(df), dtype=bool)
                for face_id, start_f, end_f in self.delete_tasks:
                    mask = (
                        (df['Face_ID'] == face_id) &
                        (df['Frame_Index'] >= start_f) &
                        (df['Frame_Index'] <= end_f)
                    )
                    keep &= ~mask
                rows_removed = int((~keep).sum())
                if rows_removed > 0:
                    df_kept = df[keep].reset_index(drop=True)
                    atomic_save_parquet(df_kept, fpath)
                    rows_per_file.append(rows_removed)

            if rows_per_file:
                msg = f"Deleted {rows_per_file[0]} row/file ({len(rows_per_file)} files)."
            else:
                msg = "No matching data found."
            self.finished.emit(True, msg)
        except Exception as e:
            self.finished.emit(False, str(e))


class ExtractionWorker(QThread):
    progress_batch = pyqtSignal(int, int)   # completed_count, total_files
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, video_files, output_dir=None, num_workers=1, stimuli_dir=None,
                 export_csv_compat=False, num_faces=1):
        super().__init__()
        self.video_files = video_files
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.stimuli_dir = stimuli_dir
        self.export_csv_compat = export_csv_compat
        self.num_faces = num_faces

    def _log(self, msg):
        """Emit to both the UI console and the file logger."""
        logger.info(msg)
        self.log.emit(msg)

    def _resolve_output_dir(self, file_path):
        """Return a concrete output directory for a file, even without a project path."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            return self.output_dir

        fallback_dir = os.path.join(os.path.dirname(file_path), "processed_stimuli_data")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

    def run(self):
        if not landmark_extraction:
            self.finished.emit(False, "Landmark Extraction module not found.")
            return

        if self.num_workers > 1 and _process_video_chunk is None: 
            self._log("Chunk worker unavailable; falling back to single-worker extraction.")
            self.num_workers = 1

        chunk_worker = _process_video_chunk

        # Copy files to stimuli_dir inside worker thread (avoids blocking UI)
        video_files = list(self.video_files)
        if self.stimuli_dir:
            os.makedirs(self.stimuli_dir, exist_ok=True)
            copied = []
            for v in video_files:
                dest = os.path.join(self.stimuli_dir, os.path.basename(v))
                if os.path.abspath(v) != os.path.abspath(dest):
                    shutil.copy2(v, dest)
                copied.append(dest)
            video_files = copied

        total_files = len(video_files)
        all_success = True
        failures = []

        try:
            if self.num_workers <= 1:
                # Sequential mode — keeps per-file frame progress
                for i, file_path in enumerate(video_files):
                    self._log(f"Processing ({i+1}/{total_files}): {os.path.basename(file_path)}")
                    self.progress_batch.emit(i, total_files)
                    output_dir = self._resolve_output_dir(file_path)

                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        result_paths = landmark_extraction.process_image(
                            file_path,
                            output_dir,
                            export_csv=self.export_csv_compat,
                            num_faces=self.num_faces,
                        )
                    else:
                        result_paths = landmark_extraction.process_video(
                            file_path,
                            output_dir,
                            export_csv=self.export_csv_compat,
                            num_faces=self.num_faces,
                        )

                    if not any(os.path.exists(path) for path in result_paths):
                        raise RuntimeError(
                            f"No output files were generated for {os.path.basename(file_path)}. "
                            "Face detection may have failed for all frames."
                        )

                    self.progress_batch.emit(i + 1, total_files)
            else:
                # Optimized Parallel Mode: Chunk long videos to use all cores on single files
                self._log(f"Starting optimized parallel extraction with {self.num_workers} workers...")
                completed = 0
                temp_root = self.output_dir if self.output_dir else tempfile.gettempdir()
                os.makedirs(temp_root, exist_ok=True)
                tmp_chunk_dir = tempfile.mkdtemp(prefix="it2_chunks_", dir=temp_root)
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    # Fast probe: confirms how many worker processes were actually created.
                    probe_futures = [executor.submit(_pool_probe_worker) for _ in range(self.num_workers)]
                    probe_pids = sorted({f.result() for f in as_completed(probe_futures)})
                    self._log(
                        f"Process pool ready: requested={self.num_workers}, active={len(probe_pids)}, "
                        f"pids={probe_pids}"
                    )

                    futures = {}
                    chunk_plan = {}
                    file_frame_counts = {}
                    chunk_frames = {}
                    chunk_start_times = {}
                    file_start_times = {}

                    # Long videos (>= 5 minutes) are split into many chunks so one video can fully use the pool.
                    long_video_seconds = 5 * 60
                    target_chunk_seconds = 45
                    max_chunks_per_video = max(self.num_workers * 4, self.num_workers)

                    for fp in video_files:
                        ext = os.path.splitext(fp)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                            file_start_times[fp] = time.perf_counter()
                            fut = executor.submit(
                                _process_image_file,
                                fp,
                                self.output_dir,
                                self.export_csv_compat,
                                self.num_faces,
                            )
                            futures[fut] = (fp, 'image', 1)
                            chunk_plan[fp] = 1
                            file_frame_counts[fp] = 1
                            chunk_frames[(fp, 1)] = 1
                            chunk_start_times[(fp, 1)] = time.perf_counter()
                        else:
                            cap = cv2.VideoCapture(fp)
                            f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                            cap.release()
                            file_start_times[fp] = time.perf_counter()
                            file_frame_counts[fp] = f_count

                            duration_sec = f_count / max(fps, 1.0)
                            if duration_sec >= long_video_seconds:
                                n_chunks = min(
                                    max_chunks_per_video,
                                    max(self.num_workers, int(math.ceil(duration_sec / target_chunk_seconds))),
                                    max(1, f_count),
                                )
                            else:
                                # Keep short videos in one chunk to avoid seek overhead.
                                n_chunks = 1

                            chunk_plan[fp] = n_chunks
                            self._log(
                                f"  · {os.path.basename(fp)}: {duration_sec / 60.0:.1f} min, "
                                f"{f_count} frames -> {n_chunks} chunk(s)"
                            )

                            edges = np.linspace(0, f_count, n_chunks + 1, dtype=np.int64)
                            for i in range(n_chunks):
                                start = int(edges[i])
                                end = int(edges[i + 1])
                                if end <= start:
                                    continue

                                if chunk_worker is None:
                                    raise RuntimeError("Chunk worker is unavailable")
                                chunk_id = i + 1
                                fut = executor.submit(
                                    _process_video_chunk_with_meta,
                                    (fp, start, end, chunk_id, tmp_chunk_dir, self.num_faces)
                                )
                                futures[fut] = (fp, 'video_chunk', chunk_id)
                                chunk_frames[(fp, chunk_id)] = end - start
                                chunk_start_times[(fp, chunk_id)] = time.perf_counter()

                    file_data = {fp: {'l': [], 'b': [], 't': []} for fp in video_files}
                    file_failed = {fp: False for fp in video_files}
                    chunks_done = {fp: 0 for fp in video_files}

                    for future in as_completed(futures):
                        fp, task_type, chunk_id = futures[future]
                        try:
                            if task_type == 'video_chunk':
                                meta = future.result()
                                worker_pid = meta['pid']
                                start_frame = meta['start']
                                end_frame = meta['end']
                                chunk_id = meta['chunk_id']

                                l_path = meta.get('landmarks_path')
                                b_path = meta.get('blendshapes_path')
                                t_path = meta.get('transformation_path')
                                if l_path:
                                    file_data[fp]['l'].append(l_path)
                                if b_path:
                                    file_data[fp]['b'].append(b_path)
                                if t_path:
                                    file_data[fp]['t'].append(t_path)
                            else:
                                l, b, t = future.result()
                                worker_pid = None
                                start_frame, end_frame = 0, 0
                                if not l.empty:
                                    file_data[fp]['l'].append(l)
                                if not b.empty:
                                    file_data[fp]['b'].append(b)
                                if not t.empty:
                                    file_data[fp]['t'].append(t)

                            chunks_done[fp] += 1
                            if task_type == 'video_chunk':
                                c_key = (fp, chunk_id)
                                c_elapsed = max(time.perf_counter() - chunk_start_times.get(c_key, time.perf_counter()), 1e-6)
                                c_frames = max(chunk_frames.get(c_key, 0), 0)
                                c_fps = c_frames / c_elapsed if c_frames > 0 else 0.0
                                self._log(
                                    f"  .. {os.path.basename(fp)} chunk {chunk_id}/{chunk_plan.get(fp, 1)} done "
                                    f"({chunks_done[fp]}/{chunk_plan.get(fp, 1)} completed) | "
                                    f"{c_fps:.1f} frames/s | pid={worker_pid} | frames={start_frame}-{end_frame}"
                                )
                            elif task_type == 'image':
                                c_key = (fp, chunk_id)
                                c_elapsed = max(time.perf_counter() - chunk_start_times.get(c_key, time.perf_counter()), 1e-6)
                                c_frames = max(chunk_frames.get(c_key, 1), 1)
                                c_fps = c_frames / c_elapsed
                                self._log(f"  .. {os.path.basename(fp)} image task done | {c_fps:.1f} frames/s")
                        except Exception as e:
                            all_success = False
                            file_failed[fp] = True
                            failures.append(f"{os.path.basename(fp)} chunk {chunk_id}: {e}")
                            self._log(f"Chunk failed for {os.path.basename(fp)}: {e}")

                    # Final merge and save per file
                    for fp in video_files:
                        try:
                            output_dir = self._resolve_output_dir(fp)
                            base = os.path.splitext(os.path.basename(fp))[0]
                            out_l = os.path.join(output_dir, f"{base}_landmarks.parquet")
                            out_b = os.path.join(output_dir, f"{base}_blendshapes.parquet")
                            out_t = os.path.join(output_dir, f"{base}_transformation.parquet")
                            out_l_csv = os.path.join(output_dir, f"{base}_landmarks.csv")
                            out_b_csv = os.path.join(output_dir, f"{base}_blendshapes.csv")
                            out_t_csv = os.path.join(output_dir, f"{base}_transformation.csv")

                            if file_failed[fp]:
                                raise RuntimeError("one or more chunk tasks failed before merge")
                            if not file_data[fp]['l'] and not file_data[fp]['b'] and not file_data[fp]['t']:
                                raise RuntimeError(
                                    "no face features were extracted for this file; no output artifacts were produced"
                                )

                            d_l = file_data[fp]['l']
                            if d_l:
                                if isinstance(d_l[0], str):
                                    df_l = pd.concat((pd.read_parquet(p) for p in d_l), ignore_index=True)
                                else:
                                    df_l = pd.concat(d_l, ignore_index=True)
                                df_l = df_l.sort_values(['Face_ID', 'Frame_Index'])
                                xy_cols = [c for c in df_l.columns if c.startswith('x_') or c.startswith('y_')]
                                for _, group_df in df_l.groupby('Face_ID'):
                                    smoothed = group_df[xy_cols].rolling(window=5, center=True, min_periods=1).mean()
                                    df_l.loc[group_df.index, xy_cols] = smoothed
                                df_l.to_parquet(out_l, index=False)
                                if self.export_csv_compat:
                                    df_l.to_csv(out_l_csv, index=False)

                            d_b = file_data[fp]['b']
                            if d_b:
                                if isinstance(d_b[0], str):
                                    df_b = pd.concat((pd.read_parquet(p) for p in d_b), ignore_index=True)
                                else:
                                    df_b = pd.concat(d_b, ignore_index=True)
                                df_b = df_b.sort_values(['Face_ID', 'Frame_Index'])
                                df_b.to_parquet(out_b, index=False)
                                if self.export_csv_compat:
                                    df_b.to_csv(out_b_csv, index=False)

                            d_t = file_data[fp]['t']
                            if d_t:
                                if isinstance(d_t[0], str):
                                    df_t = pd.concat((pd.read_parquet(p) for p in d_t), ignore_index=True)
                                else:
                                    df_t = pd.concat(d_t, ignore_index=True)
                                df_t = df_t.sort_values(['Face_ID', 'Frame_Index'])
                                df_t.to_parquet(out_t, index=False)
                                if self.export_csv_compat:
                                    df_t.to_csv(out_t_csv, index=False)

                            f_elapsed = max(time.perf_counter() - file_start_times.get(fp, time.perf_counter()), 1e-6)
                            f_frames = max(file_frame_counts.get(fp, 0), 0)
                            f_fps = f_frames / f_elapsed if f_frames > 0 else 0.0

                            completed += 1
                            self._log(
                                f"✓ ({completed}/{total_files}) {os.path.basename(fp)} | "
                                f"{f_fps:.1f} frames/s ({f_frames} frames in {f_elapsed:.1f}s)"
                            )
                            self.progress_batch.emit(completed, total_files)
                        except Exception as e:
                            all_success = False
                            failures.append(f"{os.path.basename(fp)} merge failed: {e}")
                            completed += 1
                            self._log(f"✗ ({completed}/{total_files}) {os.path.basename(fp)} merge failed: {e}")
                            self.progress_batch.emit(completed, total_files)

                try:
                    shutil.rmtree(tmp_chunk_dir)
                except Exception:
                    pass

        except Exception as e:
            self._log(f"Error: {e}")
            self.finished.emit(False, str(e))
            return

        if self.num_workers > 1 and not all_success:
            summary = "\n".join(failures[:10])
            if len(failures) > 10:
                summary += f"\n... and {len(failures) - 10} more failure(s)"
            self.finished.emit(False, f"Extraction failed for one or more files.\n{summary}")
            return

        self.finished.emit(True, f"Batch processing complete. {total_files} file(s) processed.")


class SectionTableModel(QAbstractTableModel):
    """Lightweight data model for face sections to avoid QTableWidget overhead."""
    def __init__(self, sections=None, overlaps=None, face_colors=None, fps=30.0):
        super().__init__()
        self._sections = sections or []
        self._overlaps = overlaps or {}
        self._face_colors = face_colors or {}
        self._fps = fps
        self._headers = ["#", "ID", "Frames", "Duration", "Overlaps With", "Play"]

    def update_data(self, sections, overlaps):
        self.beginResetModel()
        self._sections = sections
        self._overlaps = overlaps
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._sections)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if row >= len(self._sections):
            return None
        sec = self._sections[row]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == 0: return str(row + 1) if role == Qt.ItemDataRole.DisplayRole else row
            if col == 1:
                fid = sec['face_id']
                if hasattr(self, '_parent_widget') and row in self._parent_widget._pending_renames:
                    fid = self._parent_widget._pending_renames[row]
                if role == Qt.ItemDataRole.EditRole:
                    return fid
                if hasattr(self, '_parent_widget'):
                    pw = self._parent_widget
                    stim = pw._current_stimulus_basename()
                    if stim:
                        return pw._get_face_label(stim, fid)
                return f"Face {fid}"
            if col == 2: return f"{sec['start']} – {sec['end']}"
            if col == 3:
                dur = (sec['end'] - sec['start'] + 1) / max(self._fps, 0.1)
                return f"{dur:.1f}s"
            if col == 4:
                ov_list = self._overlaps.get(row, [])
                if not ov_list: return "—"
                if hasattr(self, '_parent_widget'):
                    pw = self._parent_widget
                    stim = pw._current_stimulus_basename()
                    fn = lambda f: pw._get_face_label(stim, f) if stim else f"Face {f}"
                else:
                    fn = lambda f: f"Face {f}"
                ov_text = ", ".join(f"{status_icon_warning} {fn(f)} (fr. {fr})" for _, f, fr in ov_list[:3])
                if len(ov_list) > 3: ov_text += f" (+{len(ov_list)-3})"
                return ov_text
            if col == 5:
                # Handle Play/Pause text
                if hasattr(self, '_parent_widget') and self._parent_widget._playing_row == row:
                    return "⏸"
                return "▶"
        
        if role == Qt.ItemDataRole.ForegroundRole and col == 4:
            if self._overlaps.get(row):
                return QColor(200, 100, 0)
            return QColor(150, 150, 150)

        if role == Qt.ItemDataRole.BackgroundRole:
            # Check for pending renames or collisions
            # This logic was previously in _on_section_dropdown_changed
            if hasattr(self, '_parent_widget'):
                pw = self._parent_widget
                if row in pw._pending_renames:
                    return QColor(255, 248, 225) # Pending change color
                ov_list = self._overlaps.get(row, [])
                if ov_list:
                    return QColor(255, 235, 238) # Collision color
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == 0: return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        return None

    def flags(self, index):
        """Enable editing for the ID column to allow dropdown interaction."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        f = super().flags(index)
        if index.column() == 1:
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.column() != 1:
            return False
        
        row = index.row()
        if hasattr(self, '_parent_widget'):
            pw = self._parent_widget
            old_fid = self._sections[row]['face_id']
            if old_fid == value:
                pw._pending_renames.pop(row, None)
            else:
                pw._pending_renames[row] = value
            
            self.dataChanged.emit(index, index)
            return True
        return False


class SectionTableDelegate(QStyledItemDelegate):
    """Delegate to paint ComboBoxes and Buttons without instantiating widgets for every row."""
    playClicked = pyqtSignal(int)
    idChanged = pyqtSignal(int, int)  # row_idx, new_fid

    def __init__(self, parent=None, unique_fids=None, face_colors=None):
        super().__init__(parent)
        self._unique_fids = unique_fids or []
        self._face_colors = face_colors or {}

    def set_unique_fids(self, fids):
        self._unique_fids = fids

    def paint(self, painter, option, index):
        if index.column() == 1:
            # Draw ComboBox appearance
            opt = QStyleOptionComboBox()
            opt.rect = option.rect.adjusted(2, 2, -2, -2)
            opt.currentText = index.data()
            opt.editable = False
            opt.state = option.state | QStyle.StateFlag.State_Enabled
            QApplication.style().drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt, painter)
            QApplication.style().drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt, painter)
        elif index.column() == 5:
            # Draw Play Button appearance
            opt = QStyleOptionButton()
            opt.rect = option.rect.adjusted(8, 2, -8, -2)
            opt.text = index.data() # Get ▶ or ⏸ from model
            opt.state = option.state | QStyle.StateFlag.State_Enabled
            if opt.text == "⏸":
                opt.state |= QStyle.StateFlag.State_On
            QApplication.style().drawControl(QStyle.ControlElement.CE_PushButton, opt, painter)
        else:
            super().paint(painter, option, index)

    def editorEvent(self, event, model, option, index):
        if event.type() == event.Type.MouseButtonRelease:
            if index.column() == 5:
                self.playClicked.emit(index.row())
                return True
        return super().editorEvent(event, model, option, index)

    def createEditor(self, parent, option, index):
        if index.column() == 1:
            combo = QComboBox(parent)
            combo.setStyleSheet(
                f"QComboBox {{ combobox-popup: 0; font-size: 11px; color: {THEME['text_primary']}; }}" +
                COMBOBOX_POPUP_STYLE
            )
            for fid in self._unique_fids:
                bgr = self._face_colors.get(fid, (200, 200, 200))
                color = QColor(bgr[2], bgr[1], bgr[0])
                if hasattr(self, '_parent_widget'):
                    pw = self._parent_widget
                    stim = pw._current_stimulus_basename()
                    label = pw._get_face_label(stim, fid) if stim else f"Face {fid}"
                else:
                    label = f"Face {fid}"
                combo.addItem(label, fid)
                combo.setItemData(combo.count() - 1, color, Qt.ItemDataRole.ForegroundRole)
            
            # New Face option
            max_fid = max(self._unique_fids) if self._unique_fids else 1
            new_fid = max_fid + 1
            combo.addItem(f"+ New ID {new_fid}", new_fid)
            
            combo.activated.connect(lambda: self.commitData.emit(combo))
            combo.activated.connect(lambda: self.closeEditor.emit(combo))
            return combo
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            fid = index.data(Qt.ItemDataRole.EditRole)
            idx = editor.findData(fid)
            if idx >= 0:
                editor.setCurrentIndex(idx)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            new_fid = editor.currentData()
            # Push to model first
            model.setData(index, new_fid)
            # Then emit signal for other side effects
            self.idChanged.emit(index.row(), new_fid)
        else:
            super().setModelData(editor, model, index)


class FaceLabelsDialog(QDialog):
    """Dialog to view and edit face labels across all stimuli."""

    def __init__(self, parent, all_stimuli_labels, all_stimuli_face_ids):
        super().__init__(parent)
        self.setWindowTitle("Face Labels — All Stimuli")
        self.setMinimumSize(600, 400)
        self._project_settings = parent._project_settings
        self._face_colors = getattr(parent, '_face_colors', {})

        layout = QVBoxLayout(self)

        info = QLabel("Edit display names for each face per stimulus. "
                      "Leave blank to show 'Face {N}'.")
        info.setWordWrap(True)
        info.setObjectName("description-label")
        layout.addWidget(info)

        # Table: Stimulus | Face ID | Label
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Stimulus", "Face ID", "Label"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().resizeSection(1, 80)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        # Populate rows
        row = 0
        for stim_name, face_ids in sorted(all_stimuli_face_ids.items()):
            stim_labels = (all_stimuli_labels.get(stim_name, {}) or {})
            for fid in sorted(face_ids):
                # Column 0: stimulus name
                stim_item = QTableWidgetItem(stim_name if row == 0 else "")
                if row > 0:
                    prev = self._table.item(row - 1, 0)
                    if prev and prev.text() == stim_name:
                        stim_item.setText("")
                stim_item.setFlags(stim_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, 0, stim_item)

                # Column 1: face ID (colored, not editable)
                fid_item = QTableWidgetItem(f"Face {fid}")
                fid_item.setFlags(fid_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                bgr = self._face_colors.get(fid, (200, 200, 200))
                r, g, b = bgr[2], bgr[1], bgr[0]
                fid_item.setForeground(QColor(r, g, b))
                self._table.setItem(row, 1, fid_item)

                # Column 2: label (editable)
                label_item = QTableWidgetItem(stim_labels.get(str(fid), ""))
                label_item.setData(Qt.ItemDataRole.UserRole, stim_name)
                label_item.setData(Qt.ItemDataRole.UserRole + 1, fid)
                self._table.setItem(row, 2, label_item)

                row += 1

        self._table.itemChanged.connect(self._on_label_changed)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _on_label_changed(self, item):
        """Persist label change immediately."""
        if item.column() != 2:
            return
        stim_name = item.data(Qt.ItemDataRole.UserRole)
        face_id = item.data(Qt.ItemDataRole.UserRole + 1)
        new_label = item.text().strip()
        all_labels = self._project_settings.get("video_preprocessor", "face_labels") or {}
        if stim_name not in all_labels:
            all_labels[stim_name] = {}
        if new_label:
            all_labels[stim_name][str(face_id)] = new_label
        else:
            all_labels[stim_name].pop(str(face_id), None)
            if not all_labels[stim_name]:
                all_labels.pop(stim_name, None)
        self._project_settings.set("video_preprocessor", "face_labels", all_labels)

class FaceLabelPickerDialog(QDialog):
    """Popup to pick an existing face label or create a new one."""

    def __init__(self, parent, existing_labels, current_label, face_id):
        super().__init__(parent)
        self.setWindowTitle(f"Label for Face {face_id}")
        self.setMinimumWidth(280)
        self._result = None

        layout = QVBoxLayout(self)

        info = QLabel("Choose an existing label or type a new one:")
        info.setWordWrap(True)
        info.setObjectName("description-label")
        layout.addWidget(info)

        # QLineEdit + dropdown button (replaces QComboBox whose arrow
        # is invisible on macOS dark theme)
        _input_row = QHBoxLayout()
        _input_row.setContentsMargins(0, 0, 0, 0)
        _input_row.setSpacing(1)

        self._combo = QLineEdit()
        self._combo.setObjectName("face-label-combo")
        self._combo.setPlaceholderText("Type a label...")
        self._combo.setFont(QFont("", -1, QFont.Weight.Normal))
        if current_label:
            self._combo.setText(current_label)
        _input_row.addWidget(self._combo)

        # Use a more refined small triangle and improve styling for a premium feel
        self._dropdown_btn = QPushButton("▾")
        self._dropdown_btn.setObjectName("face-label-dropdown")
        self._dropdown_btn.setFixedSize(28, self._combo.sizeHint().height())
        self._dropdown_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if existing_labels:
            menu = QMenu(self)
            menu.setObjectName("face-label-menu")
            menu.setStyleSheet(
                f"QMenu {{ background-color: {THEME['bg_tertiary']}; color: {THEME['text_primary']}; "
                f"border: 1px solid {THEME['border_light']}; border-radius: 4px; padding: 4px; }}"
                f"QMenu::item {{ padding: 6px 28px 6px 12px; border-radius: 2px; margin: 1px 2px; }}"
                f"QMenu::item:selected {{ background-color: {THEME['text_accent']}; color: #ffffff; }}"
            )
            for lbl in existing_labels:
                action = menu.addAction(lbl)
                action.triggered.connect(lambda checked, l=lbl: self._combo.setText(l))
            self._dropdown_btn.setMenu(menu)
        _input_row.addWidget(self._dropdown_btn)

        wrapper = QWidget()
        wrapper.setLayout(_input_row)
        layout.addWidget(wrapper)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setObjectName("face-label-clear-btn")
        self._btn_clear.setToolTip("Reset to default 'Face {N}'")
        self._btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._btn_clear)

        self._btn_ok = QPushButton("OK")
        self._btn_ok.setObjectName("face-label-ok-btn")
        self._btn_ok.clicked.connect(self._on_ok)
        btn_layout.addWidget(self._btn_ok)

        layout.addLayout(btn_layout)

        # Allow Enter to confirm
        self._combo.returnPressed.connect(self._on_ok)

    def _on_ok(self):
        self._result = self._combo.text().strip()
        self.accept()

    def _on_clear(self):
        self._result = ""
        self.accept()

    def result(self):
        return self._result


class VideoPreprocessor(QWidget):
    template_candidate_selected = pyqtSignal(object, object, object)
    template_selection_cancelled = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        lbl_title = QLabel("Step 1: Extract Features from Videos")
        lbl_title.setObjectName("title-label")
        layout.addWidget(lbl_title)

        self._template_mode = False
        self.template_banner = QWidget()
        template_row = QHBoxLayout(self.template_banner)
        template_hint = QLabel('Choosing a template face: find a clear, frontal frame with the playback controls, '
                               'check exactly one detected face under "Detected Faces", then click '
                               '"Use Selected Face as Template" beneath the frame controls.')
        template_hint.setWordWrap(True)
        template_row.addWidget(template_hint, 1)
        cancel_template = QPushButton('Cancel')
        cancel_template.clicked.connect(self._cancel_template_selection)
        template_row.addWidget(cancel_template)
        layout.addWidget(self.template_banner)
        self.template_banner.hide()
        # Lives beside the frame controls (added to the playback group below) so it
        # is within reach while the user steps to the target frame.
        self.btn_use_template = QPushButton('Use Selected Face as Template')
        self.btn_use_template.setEnabled(False)
        self.btn_use_template.setToolTip('Enabled when the frame shown has exactly one checked face with extracted landmarks')
        self.btn_use_template.clicked.connect(self._emit_template_candidate)
        self.btn_use_template.hide()

        # === Top-of-page notice banner ===
        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("notice-label")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.hide()
        
        self.lbl_global_data = QLabel("Selected data file: None")
        self.lbl_global_data.setObjectName("data-file-label")
        layout.addWidget(self.lbl_global_data)
        layout.addWidget(self.lbl_notice)

        self.current_project_path = ""

        self.btn_toggle_top = QPushButton("▼ Facial Feature Extraction")
        self.btn_toggle_top.setObjectName("toggle-panel-btn")
        self.btn_toggle_top.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_top.clicked.connect(self._toggle_top_panel)
        layout.addWidget(self.btn_toggle_top)

        self.top_panel = QWidget()
        layout.addWidget(self.top_panel)
        top_layout = QVBoxLayout(self.top_panel)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.setSpacing(8)
        
        # File List
        self.list_videos = QListWidget()
        self.list_videos.setMinimumHeight(0)
        top_layout.addWidget(self.list_videos, 1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Stimuli")
        btn_add.clicked.connect(self.add_stimuli)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear_list)
        
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_clear)
        top_layout.addLayout(btn_layout)
        
        # Worker count
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Parallel Workers:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, multiprocessing.cpu_count())
        self.spin_workers.setValue(min(4, multiprocessing.cpu_count()))
        self.spin_workers.setToolTip(
            "Number of videos to process simultaneously.\n"
            "1 = sequential (lowest memory).\n"
            f"Max = {multiprocessing.cpu_count()} (your CPU cores).\n"
            "Each worker uses ~200 MB for the MediaPipe model."
        )
        worker_layout.addWidget(self.spin_workers)
        worker_layout.addWidget(QLabel("Expected Faces:"))
        self.spin_faces = QSpinBox()
        self.spin_faces.setRange(1, 5)
        self.spin_faces.setValue(1)
        self.spin_faces.setToolTip(
            "Number of faces expected in the video.\n"
            "Default = 1. Max = 5."
        )
        worker_layout.addWidget(self.spin_faces)
        worker_layout.addStretch()
        top_layout.addLayout(worker_layout)

        self.chk_export_csv = QCheckBox("Also export CSV compatibility files")
        self.chk_export_csv.setChecked(False)
        self.chk_export_csv.setToolTip(
            "When unchecked, extraction writes only Parquet (faster).\n"
            "Enable if downstream tools require *_landmarks.csv files."
        )
        top_layout.addWidget(self.chk_export_csv)
        
        from .processing_utils import ProcessingButton, ProcessingOverlay
        self.btn_run = ProcessingButton("Start Extraction")
        self.btn_run.clicked.connect(self.run_process)
        self.btn_run.setStyleSheet(button_style("primary", padding="10px"))
        top_layout.addWidget(self.btn_run)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)

        layout.addWidget(preview_panel, 1)

        # ──────────────────────────────────────────────────────────
        # Video Preview Section
        # ──────────────────────────────────────────────────────────

        lbl_preview_title = QLabel("Video Preview")
        lbl_preview_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 4px; padding: 2px 0 0 0;")
        lbl_preview_title.setContentsMargins(0, 0, 0, 0)
        lbl_preview_title.setMinimumHeight(0)
        lbl_preview_title.setMaximumHeight(24)
        preview_layout.addWidget(lbl_preview_title, 0, Qt.AlignmentFlag.AlignLeft)

        # Setup split layout: Left (Video) | Right (Controls) — QSplitter for user-resizable panels
        self._preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._preview_splitter.splitterMoved.connect(self._refresh_frame)
        preview_layout.addWidget(self._preview_splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setMinimumWidth(240)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_panel.setMinimumWidth(280)

        self._preview_splitter.addWidget(left_panel)
        self._preview_splitter.addWidget(right_panel)
        self._preview_splitter.setStretchFactor(0, 4)
        self._preview_splitter.setStretchFactor(1, 3)

        # Defer initial sizes until after first layout pass (7:3 split)
        QTimer.singleShot(0, self._init_splitter_sizes)

        # --- LEFT PANEL: Video Frame ---
        self.preview_stage = QFrame()
        self.preview_stage.setMinimumHeight(200)
        self.preview_stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_stage.setObjectName("preview-stage")
        stage_layout = QVBoxLayout(self.preview_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_frame = QLabel()
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_frame.setStyleSheet("background: transparent; border: none;")
        self.lbl_frame.setScaledContents(False)
        stage_layout.addWidget(self.lbl_frame, 1)
        left_layout.addWidget(self.preview_stage, 1)

        # --- RIGHT PANEL: Video Selector + Overlays + Landmark Management ---
        # Video selector
        selector_layout = QVBoxLayout()
        lbl_select_stimulus = QLabel("Select Stimulus")
        lbl_select_stimulus.setStyleSheet("font-size: 18px; font-weight: bold;")
        selector_layout.addWidget(lbl_select_stimulus)
        self.combo_video = QComboBox()
        self.combo_video.setMinimumWidth(240)
        self.combo_video.setMinimumHeight(40)
        self.combo_video.setStyleSheet(
            "QComboBox { combobox-popup: 0; font-size: 15px; font-weight: 600; "
            "padding: 6px 10px; border: 2px solid palette(highlight); border-radius: 6px; }"
            "QComboBox QAbstractItemView { font-size: 15px; }"
            + COMBOBOX_POPUP_STYLE
        )
        self.combo_video.currentIndexChanged.connect(self._on_video_selected)
        selector_layout.addWidget(self.combo_video)
        right_layout.addLayout(selector_layout)

        # Overlay checkboxes
        overlay_group = QGroupBox("View Overlays")
        overlay_layout = QVBoxLayout(overlay_group)
        self.chk_landmarks = QCheckBox("Show Landmarks")
        self.chk_landmarks.setEnabled(False)
        self.chk_landmarks.toggled.connect(self._refresh_frame)
        overlay_layout.addWidget(self.chk_landmarks)

        self.chk_delaunay = QCheckBox("Show Delaunay Triangles")
        self.chk_delaunay.setEnabled(False)
        self.chk_delaunay.toggled.connect(self._refresh_frame)
        overlay_layout.addWidget(self.chk_delaunay)

        # Face contour margin: same project setting as Step 3, applied at transformation.
        margin_row = QHBoxLayout()
        self.chk_contour_margin = QCheckBox("Show Contour Margin")
        self.chk_contour_margin.setEnabled(False)
        self.chk_contour_margin.setToolTip("Draw the face outline enlarged by the margin used during transformation")
        self.chk_contour_margin.toggled.connect(self._refresh_frame)
        margin_row.addWidget(self.chk_contour_margin)
        self.spin_contour_margin = QSpinBox()
        self.spin_contour_margin.setRange(0, 30)
        self.spin_contour_margin.setSuffix(" %")
        self.spin_contour_margin.setToolTip(
            "Enlarges each face outline by this percentage during transformation,\n"
            "so gaze just outside the face (e.g. from tracker error) is mapped\n"
            "onto the template face. Landmarks and the template are not changed."
        )
        self.spin_contour_margin.valueChanged.connect(self._on_contour_margin_changed)
        margin_row.addWidget(self.spin_contour_margin)
        margin_row.addStretch()
        overlay_layout.addLayout(margin_row)
        margin_tip = QLabel(
            "💡 Pro tip: eye trackers have some error, so looks at the face edge can land just outside it. "
            "The margin enlarges each face outline by this percentage during transformation, so those "
            "gaze points are mapped onto the template face instead of being counted as off-face. "
            "Turn on \"Show Contour Margin\" to see the area it covers (thick line). Landmarks and the "
            "template are not changed. Report the value you use in your methods."
        )
        margin_tip.setWordWrap(True)
        margin_tip.setObjectName("hint-label")
        overlay_layout.addWidget(margin_tip)
        right_layout.addWidget(overlay_group)

        # Slider and Playback Controls
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(self._on_slider_changed)
        playback_layout.addWidget(self.slider)

        # Playback controls: ⏪  ▶/⏹  ⏩
        ctrl_row = QHBoxLayout()
        ctrl_row.addStretch()

        center_vbox = QVBoxLayout()
        self.lbl_time = QLabel("Frame 0 / 0")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_vbox.addWidget(self.lbl_time)

        btn_row = QHBoxLayout()
        self.btn_backward = QPushButton("⏪")
        self.btn_backward.setMinimumWidth(40)
        self.btn_backward.clicked.connect(self._step_backward)
        self.btn_backward.setToolTip("Step one frame backward")
        self.btn_backward.setAccessibleName("Step backward")
        btn_row.addWidget(self.btn_backward)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setMinimumWidth(70)
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_play.setToolTip("Play or pause the video")
        self.btn_play.setAccessibleName("Play/Pause")
        btn_row.addWidget(self.btn_play)

        self.btn_forward = QPushButton("⏩")
        self.btn_forward.setMinimumWidth(40)
        self.btn_forward.clicked.connect(self._step_forward)
        self.btn_forward.setToolTip("Step one frame forward")
        self.btn_forward.setAccessibleName("Step forward")
        btn_row.addWidget(self.btn_forward)

        # Frame jump navigator
        btn_row.addSpacing(8)
        frame_nav_lbl = QLabel("Frame:")
        frame_nav_lbl.setObjectName("hint-label")
        frame_nav_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        btn_row.addWidget(frame_nav_lbl)
        self.spin_frame_nav = QSpinBox()
        self.spin_frame_nav.setRange(1, 1)
        self.spin_frame_nav.setEnabled(False)
        self.spin_frame_nav.setFixedWidth(85)
        self.spin_frame_nav.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_frame_nav.setKeyboardTracking(False)
        self.spin_frame_nav.setToolTip("Type a frame number and press Enter to jump")
        self.spin_frame_nav.editingFinished.connect(self._on_frame_nav_editing_finished)
        btn_row.addWidget(self.spin_frame_nav)

        center_vbox.addLayout(btn_row)
        center_vbox.addWidget(self.btn_use_template)
        ctrl_row.addLayout(center_vbox)
        ctrl_row.addStretch()
        
        playback_layout.addLayout(ctrl_row)
        right_layout.addWidget(playback_group)

        # Face selection group (populated when landmarks loaded)
        self.face_selection_group = QGroupBox("Detected Faces")
        self._face_selection_inner = QVBoxLayout()
        self.face_selection_group.setLayout(self._face_selection_inner)
        self.face_selection_group.hide()
        right_layout.addWidget(self.face_selection_group)

        # Face Management group (tabs: by Sections, by Frame indices)
        self.face_manage_group = QGroupBox("Face ID Management")
        self._face_manage_inner = QVBoxLayout()
        self.face_manage_group.setLayout(self._face_manage_inner)

        face_manage_hint = QLabel(
            "by Sections — batch operations (rename, merge, delete) across contiguous face sections. "
            "Use this to consolidate multiple detections of the same person, or split sections where faces swapped IDs.\n\n"
            "by Frame indices — precise, frame-level edits. Remove a face from a frame range, "
            "reassign a slice of frames to another face, or merge faces with duplicate-frame detection."
        )
        face_manage_hint.setWordWrap(True)
        face_manage_hint.setObjectName("hint-label")
        self._face_manage_inner.addWidget(face_manage_hint)

        # Face Labels Summary button
        self._btn_face_labels = QPushButton("Face Labels Summary")
        self._btn_face_labels.setFixedWidth(150)
        self._btn_face_labels.setStyleSheet(button_style("neutral", padding="4px 12px", radius=4))
        self._btn_face_labels.setToolTip(
            "View and edit face labels across all stimuli in the project"
        )
        self._btn_face_labels.clicked.connect(self._show_face_labels_dialog)
        self._face_manage_inner.addWidget(self._btn_face_labels)

        self.face_manage_group.hide()
        right_layout.addWidget(self.face_manage_group)

        right_layout.addStretch()

        # Internal video state
        self._cap = None
        self._total_frames = 0
        self._fps = 30.0
        self._playing = False
        self._play_end_frame = None
        self._is_playing_section = False
        self._play_section_start = 0
        self._is_image = False              # True when previewing a static image
        self._current_frame = None          # raw BGR frame (no overlay)
        self._landmarks_df = None                # DataFrame or None
        self._landmarks_by_frame = {}            # {frame_idx: {face_id: Nx2 array}}
        self._landmarks_data_path = None          # path to the loaded landmarks file (.csv/.parquet)
        self._all_face_ids = []                  # All face IDs found in loaded landmark file
        self._selected_face_ids = []             # Face IDs user wants to transform
        self._face_checkboxes = []               # QCheckBox widgets for face selection
        self._face_colors = {
            1: (75, 180, 60),    # Green (BGR)
            2: (60, 120, 220),   # Blue
            3: (220, 60, 220),   # Magenta
            4: (220, 180, 0),    # Orange
            5: (60, 220, 220),   # Cyan
        }

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        self._project_settings = None
        self._overlay = ProcessingOverlay(self)
        self._last_ui_face_ids = []           # Tracks face IDs from last full UI build
        self._original_frame_pixmap = None    # Stored original-scaled pixmap for resize

        self._section_cache = {}              # {face_id: list_of_sections}
        self._save_pending = False            # Debounce flag for CSV saves
        self._sorted_frame_indices = None     # Cached sorted frame indices (Issue 7)

    def begin_template_selection(self):
        self._template_mode = True
        self._template_previous_faces = list(self._selected_face_ids)
        # Collapse the extraction panel so the frame viewer has room while picking a face
        self._template_previous_top_visible = self.top_panel.isVisible()
        self._set_top_panel_visible(False)
        self.template_banner.show()
        self.btn_use_template.show()
        self._update_template_button()

    def end_template_selection(self):
        self._template_mode = False
        self.template_banner.hide()
        self.btn_use_template.hide()
        self._set_top_panel_visible(getattr(self, '_template_previous_top_visible', True))
        previous = getattr(self, '_template_previous_faces', [])
        for cb in self._face_checkboxes:
            cb.setChecked(cb.face_id in previous)

    def _cancel_template_selection(self):
        self.end_template_selection()
        self.template_selection_cancelled.emit()

    def _update_template_button(self):
        if not getattr(self,'_template_mode',False): return
        fid = self._selected_face_ids[0] if len(self._selected_face_ids) == 1 else None
        frame = self.slider.value()+1
        valid = (self._current_frame is not None and self._landmarks_df is not None and fid is not None
                 and getattr(self,'_displayed_frame_index',self.slider.value()) == self.slider.value())
        if valid:
            df = self._landmarks_df
            matches = df[df.Face_ID == fid]
            if 'Frame_Index' in df: matches = matches[matches.Frame_Index == frame]
            valid = len(matches) == 1 and all(f'{a}_{i}' in matches for a in 'xyz' for i in range(478))
        self.btn_use_template.setEnabled(bool(valid))

    def _emit_template_candidate(self):
        self._stop_playback()
        self._update_template_button()
        if not self.btn_use_template.isEnabled(): return
        # Pending Face ID edits must be applied before snapshotting an extracted record.
        if getattr(self,'_pending_renames',None) or getattr(self,'_save_pending',False):
            show_warning(self,'Apply Face Edits','Apply or discard pending face-ID changes before selecting a template.')
            return
        from core.template_manager import file_hash, array_hash
        fid = self._selected_face_ids[0]
        frame = self.slider.value()+1
        rows = self._landmarks_df[self._landmarks_df.Face_ID == fid]
        if 'Frame_Index' in rows: rows = rows[rows.Frame_Index == frame]
        row = rows.iloc[0]
        xyz = np.column_stack([row[[f'{a}_{i}' for i in range(478)]].to_numpy(dtype=np.float32) for a in 'xyz'])
        source = self.combo_video.currentData()
        provenance = {'source_kind':'stimulus_frame','name':f'{os.path.basename(source)} — frame {frame}, face {fid}',
                      'source_name':os.path.basename(source),'source_hash':array_hash(self._current_frame),
                      'source_hash_kind':'decoded_frame',
                      'source_path':os.path.relpath(source,self.current_project_path),
                      'frame_index':frame,'video_time_seconds':(frame-1)/max(self._fps,1),
                      'face_id':int(fid),'landmark_provenance':'extracted',
                      'landmark_asset_hash':file_hash(self._landmarks_data_path),
                      'detector_provenance':'unavailable in existing extraction asset'}
        self.template_candidate_selected.emit(self._current_frame.copy(),[xyz],provenance)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Defer rescale to after child layouts settle (splitter + stage)
        if self._original_frame_pixmap is not None:
            QTimer.singleShot(0, lambda: self._scale_and_set_pixmap())
        self._update_section_action_layout()

    def _update_section_action_layout(self):
        """Stack the section action buttons on narrower widths."""
        layout = getattr(self, "_section_action_layout", None)
        if layout is None:
            return

        available_width = self.width()
        if available_width <= 0:
            return

        target_direction = (
            QBoxLayout.Direction.TopToBottom
            if available_width < 760
            else QBoxLayout.Direction.LeftToRight
        )

        if layout.direction() == target_direction:
            return

        layout.setDirection(target_direction)
        if target_direction == QBoxLayout.Direction.TopToBottom:
            layout.setSpacing(6)
            self._btn_discard.setMinimumWidth(0)
            self._btn_apply.setMinimumWidth(0)
            self._btn_merge.setMinimumWidth(0)
            self._btn_delete_section.setMinimumWidth(0)
            for button in (
                self._btn_discard,
                self._btn_apply,
                self._btn_merge,
                self._btn_delete_section,
            ):
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            layout.setSpacing(8)
            self._btn_discard.setMinimumWidth(90)
            self._btn_apply.setMinimumWidth(130)
            self._btn_merge.setMinimumWidth(130)
            self._btn_delete_section.setMinimumWidth(110)

    def eventFilter(self, obj, event):
        if event.type() == event.Type.FocusIn:
            if obj is self._adv_start_spin:
                self._adv_active_range_spin = "start"
            elif obj is self._adv_end_spin:
                self._adv_active_range_spin = "end"
        return super().eventFilter(obj, event)

    def _init_splitter_sizes(self):
        """Set initial 7:3 split (left:video, right:controls) after first layout."""
        total = self._preview_splitter.width()
        if total > 0:
            left = int(total * 0.7)
            right = total - left
            self._preview_splitter.setSizes([left, right])

    def load_settings(self, settings):
        """Apply values from ProjectSettings to this widget."""
        self._project_settings = settings
        selected = settings.resolve_working_data_file("")
        name = os.path.basename(selected) if selected else "None"
        self.lbl_global_data.setText(f"Selected data file: {name}")
        val = settings.get("video_preprocessor", "parallel_workers", None)
        if val is not None:
            self.spin_workers.setValue(int(val))

        export_csv = settings.get("video_preprocessor", "export_csv_compat", None)
        if export_csv is not None:
            self.chk_export_csv.setChecked(bool(export_csv))

        saved_faces = settings.get("video_preprocessor", "selected_face_ids", None)
        if saved_faces is not None:
            self._selected_face_ids = saved_faces

        self.spin_contour_margin.blockSignals(True)
        self.spin_contour_margin.setValue(int(settings.get("processing", "contour_margin_pct", 0)))
        self.spin_contour_margin.blockSignals(False)

        # Connect signal after loading to avoid initial trigger
        try:
            self.spin_workers.valueChanged.disconnect(self._on_workers_changed)
        except TypeError:
            pass  # not connected yet
        self.spin_workers.valueChanged.connect(self._on_workers_changed)
        try:
            self.chk_export_csv.toggled.disconnect(self._on_export_csv_toggled)
        except TypeError:
            pass
        self.chk_export_csv.toggled.connect(self._on_export_csv_toggled)

    def save_settings(self, settings):
        """Write current widget values back into ProjectSettings."""
        settings.set("video_preprocessor", "parallel_workers",
                     self.spin_workers.value())
        settings.set("video_preprocessor", "export_csv_compat",
                     self.chk_export_csv.isChecked())
        if self._selected_face_ids:
            settings.set("video_preprocessor", "selected_face_ids",
                         self._selected_face_ids)

    def _on_workers_changed(self, val):
        if self._project_settings:
            # Auto-saves because ProjectSettings.set() now calls save() if auto_save=True
            self._project_settings.set("video_preprocessor", "parallel_workers", val)

    def _on_contour_margin_changed(self, val):
        if self._project_settings:
            self._project_settings.set("processing", "contour_margin_pct", val)
        if self.chk_contour_margin.isChecked():
            self._refresh_frame()

    def _on_export_csv_toggled(self, checked):
        if self._project_settings:
            self._project_settings.set("video_preprocessor", "export_csv_compat", bool(checked))

    def _toggle_top_panel(self):
        self._set_top_panel_visible(not self.top_panel.isVisible())

    def _set_top_panel_visible(self, visible):
        self.top_panel.setVisible(visible)
        if visible:
            self.btn_toggle_top.setText("▼ Step 1: Facial Feature Extraction")
        else:
            self.btn_toggle_top.setText("▶ Step 1: Facial Feature Extraction")

    def set_project_path(self, path):
        self.current_project_path = path
        self.list_videos.clear()
        self._stop_playback()
        self._release_capture()
        self._check_existing_processed(path)
        self._populate_video_combo()

    def _check_existing_processed(self, path):
        processed_dir = os.path.join(path, "processed_stimuli_data")
        if os.path.isdir(processed_dir):
            files = [
                f for f in os.listdir(processed_dir)
                if f.endswith(('_landmarks.parquet'))
            ]
            if files:
                self.lbl_notice.setText(
                    f"{status_icon_info} {len(files)} processed stimuli already found in processed_stimuli_data/. "
                    f"You may skip this step and proceed to Step 2."
                )
                self.lbl_notice.show()
                self._set_top_panel_visible(False)
                return
        self.lbl_notice.hide()
        self._set_top_panel_visible(True)

    def add_stimuli(self):
        start_dir = self.current_project_path if self.current_project_path else ""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", start_dir, "Media Files (*.mp4 *.avi *.mov *.webm *.jpg *.jpeg *.png *.bmp)")
        if files:
            self._overlay.set_status("Importing files")
            self._overlay.show()
            QApplication.processEvents()
            self.list_videos.addItems(files)
            self._overlay.hide()
            self._set_top_panel_visible(True)

    def _on_clear_list(self):
        self.list_videos.clear()
        self._set_top_panel_visible(False)
            
    def run_process(self):
        count = self.list_videos.count()
        if count == 0:
            show_warning(self, "No Videos Added", "Please add videos to process.")
            return
            
        videos = [self.list_videos.item(i).text() for i in range(count)]
        
        # Determine directories (file copy happens inside worker thread)
        stimuli_dir = None
        output_dir = None
        if self.current_project_path:
            stimuli_dir = os.path.join(self.current_project_path, "stimuli")
            output_dir = os.path.join(self.current_project_path, "processed_stimuli_data")
            os.makedirs(output_dir, exist_ok=True)

        num_workers = self.spin_workers.value()
        export_csv_compat = self.chk_export_csv.isChecked()
        num_faces = self.spin_faces.value()

        self.btn_run.start_processing("Extracting")
        self.spin_workers.setEnabled(False)
        self.spin_faces.setEnabled(False)
        self.chk_export_csv.setEnabled(False)
        # Remember the currently previewed video to reload after extraction
        self._preview_filename_before_extract = os.path.basename(
            self.combo_video.itemData(self.combo_video.currentIndex())
        ) if self.combo_video.currentIndex() > 0 else ""
        self._worker_video_files = videos
        self._overlay.set_status("Extracting Facial Features")
        self._overlay.hide_progress()
        self._overlay.show()

        self.worker = ExtractionWorker(
            videos,
            output_dir,
            num_workers,
            stimuli_dir,
            export_csv_compat=export_csv_compat,
            num_faces=num_faces,
        )
        self.worker.progress_batch.connect(self._update_batch_progress)
        self.worker.log.connect(self._update_log)
        self.worker.finished.connect(self._process_finished)
        self.worker.start()
        
    def _update_batch_progress(self, curr, total):
        pct = int((curr / max(total, 1)) * 100)
        self.btn_run.set_progress(pct)

    def _update_log(self, msg):
        if hasattr(self, '_overlay') and self._overlay is not None:
            self._overlay.set_sub_status(msg)
        
    def _process_finished(self, success, msg):
        self._overlay.hide()
        self.btn_run.set_progress(100)
        self._finish_success = success
        self._finish_msg = msg
        # Show 100% for 1.5s before resetting the button
        QTimer.singleShot(1500, self._finish_cleanup)

    def _finish_cleanup(self):
        self.btn_run.stop_processing()
        self.spin_workers.setEnabled(True)
        self.spin_faces.setEnabled(True)
        self.chk_export_csv.setEnabled(True)
        if self._finish_success:
            show_success(self, "Extraction Complete", self._finish_msg)
            self._populate_video_combo()
            # Reload the video that was previewed before extraction
            if hasattr(self, '_preview_filename_before_extract') and self._preview_filename_before_extract:
                self._reload_previewed_video(self._preview_filename_before_extract)
            self._set_top_panel_visible(False)
        else:
            show_error(self, "Processing Failed", self._finish_msg)

    # ──────────────────────────────────────────────────────────
    # Video Preview Methods
    # ──────────────────────────────────────────────────────────

    def _populate_video_combo(self):
        """Fill the combo box with video/image files from the stimuli/ folder."""
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        self.combo_video.addItem("— select a stimulus —")
        if self.current_project_path:
            stimuli_dir = os.path.join(self.current_project_path, "stimuli")
            if os.path.isdir(stimuli_dir):
                video_exts = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
                image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
                exts = video_exts + image_exts
                files = sorted(f for f in os.listdir(stimuli_dir)
                               if f.lower().endswith(exts))
                for f in files:
                    self.combo_video.addItem(f, os.path.join(stimuli_dir, f))
        self.combo_video.blockSignals(False)

    def _reload_previewed_video(self, filename):
        """Re-select a video by filename in the combo box to reload preview and landmarks."""
        for i in range(self.combo_video.count()):
            if self.combo_video.itemText(i) == filename:
                self.combo_video.setCurrentIndex(i)
                return
        # File not found in stimuli/ — try loading landmarks by original path
        for path in getattr(self, '_worker_video_files', []):
            if os.path.basename(path) == filename:
                self._load_landmarks(path)
                return

    def _on_video_selected(self, index):
        """Open a video or image when the user picks one from the combo box."""
        self._stop_playback()

        if index == 0:
            self._release_capture()
            return

        path = self.combo_video.itemData(index)
        if not path or not os.path.isfile(path):
            self._release_capture()
            return

        self._release_capture()
        ext = os.path.splitext(path)[1].lower()
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        if ext in image_exts:
            # --- Image mode (fast, do in UI thread) ---
            img = cv2.imread(path)
            if img is None:
                show_warning(self, "Cannot Open Image", f"Cannot open image:\n{path}")
                return
            self._is_image = True
            self._current_frame = img
            self._total_frames = 1
            self._fps = 1.0
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
            self.slider.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_forward.setEnabled(False)
            self.btn_backward.setEnabled(False)
            self.spin_frame_nav.setEnabled(False)
            self.spin_frame_nav.setRange(1, 1)
            self._load_landmarks(path)
            self._display_frame(img, 0)
        else:
            # --- Video mode: show loading overlay, then load in background ---
            self._release_capture()
            self._show_loading_indicator()

            # Defer thread start to next event cycle so label can paint
            QTimer.singleShot(0, lambda: self._start_video_loader(path))

    def _show_loading_indicator(self):
        """Show a loading message during video file open and first frame read."""
        loading_msg = "Loading video..."
        self.lbl_frame.clear()
        self.lbl_frame.setStyleSheet(f"background: transparent; border: none; color: {THEME['text_accent']}; font-size: 16px; font-weight: bold;")
        self.lbl_frame.setText(loading_msg)
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _start_video_loader(self, path):
        """Start the background video loader from the main event thread."""
        self._video_loader = _VideoLoaderThread(path)
        self._video_loader.finished.connect(self._on_video_loaded)
        self._video_loader.start()

    def _hide_loading_indicator(self):
        """Hide the loading message and restore normal frame display."""
        self.lbl_frame.clear()
        self.lbl_frame.setStyleSheet("background: transparent; border: none;")
        self.lbl_frame.setText("")

    def _on_video_loaded(self, success, cap, total_frames, fps, first_frame):
        """Called when video background loading completes."""
        # Discard stale callbacks when user switched videos
        loader = self.sender()
        if loader != self._video_loader:
            if cap is not None:
                cap.release()
            return

        self._video_loader = None
        if not success or cap is None:
            show_warning(self, "Cannot Open Video", "The video file could not be opened. Check that the file exists and is not corrupted.")
            return

        self._is_image = False
        self._cap = cap
        self._total_frames = max(1, int(total_frames))
        self._fps = max(fps, 0.1)  # Avoid division by zero

        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_forward.setEnabled(True)
        self.btn_backward.setEnabled(True)
        self.spin_frame_nav.setEnabled(True)
        self.spin_frame_nav.setRange(1, self._total_frames)
        self.slider.setRange(0, self._total_frames - 1)
        self.slider.setValue(0)

        # Get video path from combo for landmark loading
        video_path = self.combo_video.itemData(self.combo_video.currentIndex())

        # Display the first frame that was pre-loaded immediately
        self._current_frame = first_frame
        self._display_frame(first_frame, 0)
        QApplication.processEvents()  # Paint frame before heavy CSV parsing

        # Defer landmark (heavy) loading to the next event cycle to let the first frame appear
        if video_path and os.path.isfile(video_path):
            QTimer.singleShot(0, lambda vp=video_path: self._load_landmarks(vp))

    def _release_capture(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_image = False
        self._total_frames = 0
        self._current_frame = None
        self._landmarks_df = None
        self._landmarks_by_frame = {}
        self._all_face_ids = []
        self._selected_face_ids = []
        self._face_checkboxes = []
        self._face_label_buttons = {}
        self._landmarks_data_path = None
        self.lbl_frame.clear()
        self._original_frame_pixmap = None
        self._sorted_frame_indices = None
        self.lbl_time.setText("Frame 0 / 0")
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_forward.setEnabled(True)
        self.btn_backward.setEnabled(True)
        self.spin_frame_nav.setEnabled(False)
        self.spin_frame_nav.setRange(1, 1)
        self.spin_frame_nav.setValue(1)
        self.chk_landmarks.setEnabled(False)
        self.chk_delaunay.setEnabled(False)
        self.chk_contour_margin.setEnabled(False)
        # Disconnect timeline link to avoid dangling callback
        try:
            self.slider.valueChanged.disconnect(self._update_timeline_position)
        except TypeError:
            pass
        self.face_selection_group.hide()
        self.face_manage_group.hide()

    def _load_landmarks(self, video_path):
        """Asynchronously load landmarks matching the video (.parquet preferred, .csv fallback)."""
        self._landmarks_df = None
        self._landmarks_by_frame = {}
        self._all_face_ids = []
        base = os.path.splitext(os.path.basename(video_path))[0]
        processed_dir = os.path.join(self.current_project_path, "processed_stimuli_data")
        lm_path = os.path.join(processed_dir, f"{base}_landmarks.parquet")
        if not os.path.isfile(lm_path):
            lm_path = os.path.join(processed_dir, f"{base}_landmarks.csv")

        if not os.path.isfile(lm_path):
            self.chk_landmarks.setEnabled(False)
            self.chk_delaunay.setEnabled(False)
            self.chk_contour_margin.setEnabled(False)
            self.face_selection_group.hide()
            self.face_manage_group.hide()
            return

        self._landmarks_data_path = lm_path
        self._overlay.set_status("Rebuilding spatial cache...")
        self._overlay.show()
        
        self._load_worker = _LandmarkLoadWorker(lm_path)
        self._load_worker.finished.connect(self._on_landmarks_loaded)
        self._load_worker.start()

    def _on_landmarks_loaded(self, success, df, lbf, face_ids, err):
        """Callback when background landmark loading finishes."""
        self._overlay.hide()
        if success:
            self._landmarks_df = df
            self._landmarks_by_frame = lbf
            self._all_face_ids = face_ids
            # Issue 7: Cache sorted frame indices once on load
            self._sorted_frame_indices = sorted(lbf.keys())

            self._section_cache.clear()
            self.chk_landmarks.setEnabled(True)
            self.chk_delaunay.setEnabled(True)
            self.chk_contour_margin.setEnabled(True)
            
            # Update face selection UI
            self._update_face_selection_ui()
        else:
            logger.error(f"Failed to load landmarks: {err}")
            show_error(self, "Failed to Load Landmarks", f"Failed to load landmarks:\n{err}")
            self._landmarks_data_path = None
            self.chk_landmarks.setEnabled(False)
            self.chk_delaunay.setEnabled(False)
            self.chk_contour_margin.setEnabled(False)
            self.face_selection_group.hide()
            self.face_manage_group.hide()

    def _refresh_face_selection_data(self):
        """Incremental data refresh — updates section counts, table without rebuilding widgets."""
        if not self._face_checkboxes:
            return

        # Update section info labels next to each checkbox
        face_sections = {}
        for fid in self._all_face_ids:
            face_sections[fid] = self._compute_face_sections(fid)

        for cb in self._face_checkboxes:
            fid = cb.face_id
            secs = face_sections.get(fid, [])
            label = self._face_info_labels.get(fid)
            if label:
                if secs:
                    section_info = f"{len(secs)} section{'s' if len(secs) != 1 else ''}"
                    min_f, max_f = secs[0][0], secs[-1][1]
                    section_info += f" ({min_f}–{max_f})"
                else:
                    section_info = "no data"
                label.setText(section_info)
        if hasattr(self, '_face_timeline'):
            self._face_timeline._face_sections = face_sections
            self._face_timeline.update()

        # Update per-face timeline bars with new section data
        for fid, ft_bar in zip(self._all_face_ids, self._face_timelines):
            ft_bar.set_face_sections({fid: face_sections.get(fid, [])})

        # Rebuild section table for the filtered face
        if hasattr(self, '_nav_face_combo'):
            nav_fid = self._nav_face_combo.currentData()
            if nav_fid is not None:
                self._build_section_table()

    def _clear_layout(self, layout):
        """Recursively clear all widgets and nested layouts from a layout."""
        if layout is None: return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self._clear_layout(sub_layout)

    def _update_face_selection_ui(self):
        """Smart rebuild face selection UI — partial refresh when face IDs unchanged."""
        # SMART REFRESH: If face IDs haven't changed, only update tables/counts
        if (self._face_checkboxes and
                self._all_face_ids and
                sorted(self._all_face_ids) == sorted(self._last_ui_face_ids)):
            self._refresh_face_selection_data()
            return

        # Thoroughly clear existing widgets and sub-layouts
        self._clear_layout(self._face_selection_inner)

        if not self._all_face_ids:
            self.face_selection_group.hide()
            self._face_checkboxes = []
            return

        self.face_selection_group.show()
        self._face_checkboxes = []
        self._face_label_buttons = {}
        self._face_info_labels = {}
        self._face_timelines = []

        # Compute per-face contiguous sections from CSV
        face_sections = {}
        for fid in self._all_face_ids:
            face_sections[fid] = self._compute_face_sections(fid)

        # Header label
        label = QLabel("Select faces to transform:")
        label.setObjectName("description-label")
        self._face_selection_inner.addWidget(label)

        # Explanation text
        expl_lbl = QLabel("Only selected faces will be mapped to the template and included in later analysis.")
        expl_lbl.setObjectName("fine-label")
        self._face_selection_inner.addWidget(expl_lbl)

        # Interleaved Face IDs and Timelines
        selected = set(self._selected_face_ids) if self._selected_face_ids else set(self._all_face_ids)
        for face_id in self._all_face_ids:
            stim = self._current_stimulus_basename()
            display_name = self._get_face_label(stim, face_id) if stim else f"Face {face_id}"
            # Container for the label row
            info_layout = QHBoxLayout()
            cb = QCheckBox()
            bgr_color = self._face_colors.get(face_id, (200, 200, 200))
            r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
            color_hex = f"#{r:02x}{g:02x}{b:02x}"
            cb.setText(f"  {display_name}")
            cb.setStyleSheet(f"color: {color_hex}; font-weight: bold;")
            cb.setChecked(face_id in selected)
            cb.face_id = face_id
            cb.toggled.connect(self._on_face_selection_changed)
            self._face_checkboxes.append(cb)
            info_layout.addWidget(cb)

            # Label picker button
            label_btn = QPushButton("Assign Label")
            label_btn.setObjectName("label-picker-btn")
            label_btn.setFixedWidth(90)
            label_btn.setToolTip("Assign a label to this face to easily categorize and group participants during the analysis stage.")
            _label_fid = face_id
            _label_btn = label_btn
            label_btn.clicked.connect(
                lambda checked=False, fid=_label_fid, btn=_label_btn:
                self._show_label_picker(fid, btn)
            )
            self._face_label_buttons[face_id] = label_btn
            info_layout.addWidget(label_btn)

            secs = face_sections.get(face_id, [])
            if secs:
                section_info = f"{len(secs)} section{'s' if len(secs) != 1 else ''} ({secs[0][0]}–{secs[-1][1]})"
            else:
                section_info = "no data"
            range_lbl = QLabel(section_info)
            range_lbl.setObjectName("fine-label")
            self._face_info_labels[face_id] = range_lbl
            info_layout.addStretch()
            info_layout.addWidget(range_lbl)
            self._face_selection_inner.addLayout(info_layout)

            # Timeline bar for this specific face
            ft = _FaceTimelineBar([face_id], {face_id: secs},
                                 self._total_frames, self._face_colors,
                                 self.slider.value() + 1)
            ft.setMinimumHeight(14)
            ft.clicked.connect(self._on_timeline_clicked)
            self._face_timelines.append(ft)
            self._face_selection_inner.addWidget(ft)

            # Spacing before next face
            spacer = QFrame()
            spacer.setFixedHeight(6)
            self._face_selection_inner.addWidget(spacer)

        # Ensure playhead stays in sync
        try:
            self.slider.valueChanged.disconnect(self._update_timeline_position)
        except TypeError:
            pass
        self.slider.valueChanged.connect(self._update_timeline_position)

        # Timeline legend — explains the color coding
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(0, 4, 0, 2)
        legend_layout.setSpacing(10)

        def _make_swatch(color_hex, text):
            """Create a small color swatch with label."""
            container = QHBoxLayout()
            container.setSpacing(3)
            swatch = QFrame()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(f"background-color: {color_hex}; border-radius: 2px;")
            label = QLabel(text)
            label.setObjectName("fine-label")
            container.addWidget(swatch)
            container.addWidget(label)
            return container

        legend_layout.addLayout(_make_swatch("#8264be", "Detection"))
        legend_layout.addLayout(_make_swatch("#50a050", "Gap"))
        legend_layout.addLayout(_make_swatch("#82d2ff", "Selected section"))
        line_sw = QFrame()
        line_sw.setFixedSize(2, 10)
        line_sw.setStyleSheet("background-color: #ffffff;")
        line_layout = QHBoxLayout()
        line_layout.setSpacing(3)
        line_layout.addWidget(line_sw)
        line_label = QLabel("Current frame")
        line_label.setObjectName("fine-label")
        line_layout.addWidget(line_label)
        legend_layout.addLayout(line_layout)
        legend_layout.addStretch()

        legend_widget = QWidget()
        legend_widget.setLayout(legend_layout)
        self._face_selection_inner.addWidget(legend_widget)

        # Build Face Management tabs
        self._clear_layout(self._face_manage_inner)
        self._tabs_manage = QTabWidget()
        self._tabs_manage.setStyleSheet("")

        # Tab 1: Manipulate Face IDs
        tab_manipulate = QWidget()
        tab_manipulate.setStyleSheet("")
        manipulate_layout = QVBoxLayout(tab_manipulate)
        manipulate_layout.setContentsMargins(8, 8, 8, 8)

        # Face filter dropdown
        filter_layout = QHBoxLayout()
        filter_lbl = QLabel("Filter:")
        filter_lbl.setObjectName("hint-label")
        filter_layout.addWidget(filter_lbl)
        self._nav_face_combo = QComboBox()
        self._nav_face_combo.setMinimumWidth(110)
        self._nav_face_combo.setStyleSheet(
            "QComboBox { combobox-popup: 0; }" + COMBOBOX_POPUP_STYLE
        )
        for fid in self._all_face_ids:
            bgr_color = self._face_colors.get(fid, (200, 200, 200))
            r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
            self._nav_face_combo.addItem(self._face_label(fid), fid)
        self._nav_face_combo.currentIndexChanged.connect(self._on_nav_face_changed)
        filter_layout.addWidget(self._nav_face_combo)
        filter_layout.addStretch()
        manipulate_layout.addLayout(filter_layout)

        # Section table (Virtualize with QTableView)
        self._section_table = QTableView()
        self._section_model = SectionTableModel(face_colors=self._face_colors, fps=self._fps)
        self._section_model._parent_widget = self
        self._section_table.setModel(self._section_model)
        
        # Apply delegate for buttons and combos
        self._section_delegate = SectionTableDelegate(
            self._section_table, 
            unique_fids=self._all_face_ids,
            face_colors=self._face_colors
        )
        self._section_delegate.playClicked.connect(self._play_section_from_table)
        self._section_delegate.idChanged.connect(self._on_table_id_changed)
        self._section_table.setItemDelegate(self._section_delegate)

        self._update_section_table_widths()
        self._section_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._section_table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self._section_table.setEditTriggers(QTableView.EditTrigger.AllEditTriggers)
        self._section_table.verticalHeader().setVisible(False)
        self._section_table.setObjectName("section-table")
        self._section_table.doubleClicked.connect(lambda idx: self._play_section_from_table(idx.row()))
        self._section_table.clicked.connect(
            lambda idx: self._navigate_to_section(idx.row()) if idx.column() != 5 else None
        )
        
        # Ensure merge button updates on selection
        self._section_table.selectionModel().selectionChanged.connect(self._update_merge_button)
        manipulate_layout.addWidget(self._section_table, 1)

        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        self._section_action_layout = action_layout
        self._btn_discard = QPushButton("Discard Changes")
        self._btn_discard.setStyleSheet(button_style("neutral", padding="4px 12px", radius=4))
        self._btn_discard.setMinimumWidth(90)
        self._btn_discard.setToolTip("Clear all pending ID changes in the table. This does not affect the saved data file.")
        self._btn_discard.clicked.connect(self._discard_section_changes)
        action_layout.addWidget(self._btn_discard)

        self._btn_apply = QPushButton("Apply Reassignments")
        self._btn_apply.setStyleSheet(button_style("success", padding="4px 12px", radius=4))
        self._btn_apply.setMinimumWidth(130)
        self._btn_apply.setToolTip("Commit all pending ID reassignments to the landmark data file.")
        self._btn_apply.clicked.connect(self._apply_section_reassignments)
        action_layout.addWidget(self._btn_apply)

        self._btn_merge = QPushButton("Auto-Merge Same IDs")
        self._btn_merge.setStyleSheet(button_style("neutral", padding="4px 12px", radius=4))
        self._btn_merge.setMinimumWidth(130)
        self._btn_merge.setToolTip("Connect contiguous sections of the same Face ID if the gap between them is below the 'Max Gap Fill' threshold.")
        self._btn_merge.clicked.connect(self._auto_merge_sections)
        action_layout.addWidget(self._btn_merge)

        action_layout.addStretch()
        self._btn_delete_section = QPushButton("Delete Selection")
        self._btn_delete_section.setStyleSheet(
            button_style("danger", padding="4px 12px", radius=4)
        )
        self._btn_delete_section.setMinimumWidth(110)
        self._btn_delete_section.setToolTip(
            "Delete all data for the selected section(s)\n"
            "(removes face from landmarks, blendshapes, transformation)"
        )
        self._btn_delete_section.clicked.connect(self._delete_selected_sections)
        self._btn_delete_section.setEnabled(False)
        action_layout.addWidget(self._btn_delete_section)

        manipulate_layout.addLayout(action_layout)
        self._update_section_action_layout()

        # Instruction label
        instr_lbl = QLabel(
            "Double-click a row (or ▶) to play its video preview. "
            "Change dropdowns to reassign IDs. Overlaps flagged on apply.\n"
            "Highlighted rows indicate pending ID changes. Click 'Apply Reassignments' to save them."
        )
        instr_lbl.setObjectName("fine-label")
        instr_lbl.setWordWrap(True)
        manipulate_layout.addWidget(instr_lbl)

        self._tabs_manage.addTab(tab_manipulate, "by Sections")

        # Initialize pending changes tracking
        self._pending_renames = {}
        self._all_sections_data = []
        self._playing_row = -1  # Track row index for ▶/⏸ toggle
        
        # Build the section table after adding to layout
        self._build_section_table()

        # Tab 2: Advanced Operations
        tab_advanced = QWidget()
        tab_advanced.setStyleSheet("")
        advanced_layout = QVBoxLayout(tab_advanced)
        advanced_layout.setContentsMargins(8, 8, 8, 8)
        advanced_layout.setSpacing(8)

        # Tab Guidance
        adv_guide = QLabel(
            "Use this tab for precise, frame-level adjustments. Ideal for removing specific frames of noise, "
            "splitting a section exactly at a certain frame, or resolving complex overlap issues that 'by Sections' cannot handle."
        )
        adv_guide.setWordWrap(True)
        adv_guide.setObjectName("hint-label")
        advanced_layout.addWidget(adv_guide)

        # Face filter for advanced tab
        adv_filter_layout = QHBoxLayout()
        adv_filter_lbl = QLabel("Filter:")
        adv_filter_lbl.setObjectName("hint-label")
        adv_filter_layout.addWidget(adv_filter_lbl)
        self._adv_face_combo = QComboBox()
        self._adv_face_combo.setMinimumWidth(110)
        self._adv_face_combo.setStyleSheet(
            f"QComboBox {{ combobox-popup: 0; font-size: 11px; color: {THEME['text_primary']}; }}" +
            COMBOBOX_POPUP_STYLE
        )
        for fid in self._all_face_ids:
            bgr_color = self._face_colors.get(fid, (200, 200, 200))
            r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
            self._adv_face_combo.addItem(self._face_label(fid), fid)
        adv_filter_layout.addWidget(self._adv_face_combo)
        adv_filter_layout.addStretch()
        advanced_layout.addLayout(adv_filter_layout)

        # Frame range input
        frame_group = QGroupBox("Frame Range")
        frame_group_layout = QHBoxLayout(frame_group)
        frame_group_layout.setContentsMargins(8, 8, 8, 8)
        frame_group_layout.setSpacing(12)

        start_pair = QHBoxLayout()
        start_pair.setSpacing(4)
        start_pair.addWidget(QLabel("Start:"))
        self._adv_start_spin = QSpinBox()
        self._adv_start_spin.setRange(1, max(self._total_frames, 1))
        self._adv_start_spin.setValue(1)
        self._adv_start_spin.setFixedWidth(80)
        start_pair.addWidget(self._adv_start_spin)
        frame_group_layout.addLayout(start_pair)

        end_pair = QHBoxLayout()
        end_pair.setSpacing(4)
        end_pair.addWidget(QLabel("End:"))
        self._adv_end_spin = QSpinBox()
        self._adv_end_spin.setRange(1, max(self._total_frames, 1))
        self._adv_end_spin.setValue(1)
        self._adv_end_spin.setFixedWidth(80)
        end_pair.addWidget(self._adv_end_spin)
        frame_group_layout.addLayout(end_pair)

        # Quick nav button: jump to current slider frame
        self._adv_set_current_btn = QPushButton("← Current Frame")
        self._adv_set_current_btn.setFixedWidth(95)
        self._adv_set_current_btn.clicked.connect(self._adv_set_current_frame)
        frame_group_layout.addWidget(self._adv_set_current_btn)

        frame_group_layout.addStretch()

        # Track focus to know which spin box "Current Frame" should update
        self._adv_active_range_spin = "start"  # default to start
        self._adv_start_spin.installEventFilter(self)
        self._adv_end_spin.installEventFilter(self)

        # Wire spin box changes to update info label and enforce range
        self._adv_start_spin.valueChanged.connect(self._adv_on_start_changed)
        self._adv_end_spin.valueChanged.connect(self._adv_update_info)
        self._adv_face_combo.currentIndexChanged.connect(self._adv_update_info)
        self._adv_face_combo.currentIndexChanged.connect(self._adv_update_action_labels)

        advanced_layout.addWidget(frame_group)

        # Section info label (shows what data exists in the range)
        self._adv_info_lbl = QLabel("Set a range above to see section info.")
        self._adv_info_lbl.setObjectName("hint-label")
        self._adv_info_lbl.setWordWrap(True)
        advanced_layout.addWidget(self._adv_info_lbl)

        # Action buttons row 1: Reassign Face ID
        swap_layout = QHBoxLayout()
        swap_layout.setSpacing(6)
        swap_layout.addWidget(QLabel("Reassign filtered face to:"))
        self._adv_target_combo = QComboBox()
        self._adv_target_combo.setMinimumWidth(110)
        self._adv_target_combo.setStyleSheet(
            f"QComboBox {{ combobox-popup: 0; font-size: 11px; color: {THEME['text_primary']}; }}" +
            COMBOBOX_POPUP_STYLE
        )
        for fid in self._all_face_ids:
            self._adv_target_combo.addItem(self._face_label(fid), fid)
        new_fid = (max(self._all_face_ids) if self._all_face_ids else 1) + 1
        self._adv_target_combo.addItem(f"+ New ID {new_fid}", new_fid)
        swap_layout.addWidget(self._adv_target_combo)
        self._adv_swap_btn = QPushButton("Reassign Face ID")
        self._adv_swap_btn.setStyleSheet(button_style("primary", padding="4px 12px", radius=4))
        self._adv_swap_btn.setMinimumWidth(140)
        self._adv_swap_btn.setToolTip(
            "Reassign only the currently filtered face to the target ID within the selected frame range. "
            "Other faces in this range remain unchanged."
        )
        self._adv_swap_btn.clicked.connect(self._adv_swap_face_id)
        swap_layout.addWidget(self._adv_swap_btn)
        swap_layout.addStretch()
        advanced_layout.addLayout(swap_layout)

        self._adv_swap_desc = QLabel("")
        self._adv_swap_desc.setWordWrap(True)
        self._adv_swap_desc.setObjectName("fine-label")
        advanced_layout.addWidget(self._adv_swap_desc)

        # Action row 2: Merge All Faces
        merge_layout = QHBoxLayout()
        merge_layout.setSpacing(6)
        merge_layout.addWidget(QLabel("Merge all faces in range to:"))
        self._adv_merge_combo = QComboBox()
        self._adv_merge_combo.setMinimumWidth(110)
        self._adv_merge_combo.setStyleSheet(
            f"QComboBox {{ combobox-popup: 0; font-size: 11px; color: {THEME['text_primary']}; }}" +
            COMBOBOX_POPUP_STYLE
        )
        for fid in self._all_face_ids:
            self._adv_merge_combo.addItem(self._face_label(fid), fid)
        merge_fid = (max(self._all_face_ids) if self._all_face_ids else 1) + 1
        self._adv_merge_combo.addItem(f"+ New ID {merge_fid}", merge_fid)
        merge_layout.addWidget(self._adv_merge_combo)
        self._adv_merge_btn = QPushButton("Merge All Faces")
        self._adv_merge_btn.setStyleSheet(button_style("danger", padding="4px 12px", radius=4))
        self._adv_merge_btn.setMinimumWidth(140)
        self._adv_merge_btn.setToolTip(
            "Reassign ALL face data in the frame range to the target ID. "
            "CRITICAL: If multiple faces exist on the same frame, only the last one "
            "is kept — all other landmark data for that frame is permanently DELETED."
        )
        self._adv_merge_btn.clicked.connect(self._adv_merge_to_id)
        merge_layout.addWidget(self._adv_merge_btn)
        merge_layout.addStretch()
        advanced_layout.addLayout(merge_layout)

        self._adv_merge_desc = QLabel("")
        self._adv_merge_desc.setWordWrap(True)
        self._adv_merge_desc.setObjectName("fine-label-error")
        advanced_layout.addWidget(self._adv_merge_desc)

        # Action row 3: Remove Selected Face in Range
        gap_action_layout = QHBoxLayout()
        gap_action_layout.setSpacing(6)
        self._adv_declare_gap_btn = QPushButton("")
        self._adv_declare_gap_btn.setStyleSheet(button_style("danger", padding="4px 12px", radius=4))
        self._adv_declare_gap_btn.setMinimumWidth(140)
        self._adv_declare_gap_btn.setToolTip(
            "Remove all landmark data for the selected face "
            "in the specified frame range."
        )
        self._adv_declare_gap_btn.clicked.connect(self._adv_declare_gap)
        gap_action_layout.addWidget(self._adv_declare_gap_btn)
        gap_action_layout.addStretch()
        advanced_layout.addLayout(gap_action_layout)

        self._adv_remove_desc = QLabel("")
        self._adv_remove_desc.setWordWrap(True)
        self._adv_remove_desc.setObjectName("fine-label-error")
        advanced_layout.addWidget(self._adv_remove_desc)

        # Action row 4: Re-analyze Face Detection in Range
        reanalyze_layout = QHBoxLayout()
        reanalyze_layout.setSpacing(6)
        self._adv_reanalyze_btn = QPushButton("Re-analyze Face Detection in Range")
        self._adv_reanalyze_btn.setStyleSheet(button_style("secondary", padding="4px 12px", radius=4))
        self._adv_reanalyze_btn.setMinimumWidth(140)
        self._adv_reanalyze_btn.setToolTip(
            "Re-run MediaPipe face detection on the selected frame range. "
            "The newly detected face(s) will be assigned a new Face ID. "
            "Existing data for the filtered face in this range will be replaced."
        )
        self._adv_reanalyze_btn.clicked.connect(self._adv_reanalyze_face_range)
        reanalyze_layout.addWidget(self._adv_reanalyze_btn)
        reanalyze_layout.addStretch()
        advanced_layout.addLayout(reanalyze_layout)

        self._adv_reanalyze_desc = QLabel("")
        self._adv_reanalyze_desc.setWordWrap(True)
        self._adv_reanalyze_desc.setObjectName("fine-label")
        advanced_layout.addWidget(self._adv_reanalyze_desc)

        # Notice label: appears after re-analysis to remind user of follow-up
        self._adv_notice_lbl = QLabel("")
        self._adv_notice_lbl.setWordWrap(True)
        self._adv_notice_lbl.setObjectName("notice-label")
        advanced_layout.addWidget(self._adv_notice_lbl)

        advanced_layout.addStretch()

        self._tabs_manage.addTab(tab_advanced, "by Frame indices")
        self._face_manage_inner.addWidget(self._tabs_manage)
        self.face_manage_group.show()

        # Initialize action labels based on first face selected
        self._adv_update_action_labels()

        # Update selected face IDs
        self._selected_face_ids = [fid for fid in self._all_face_ids if fid in selected]

        # Track face IDs for smart refresh check
        self._last_ui_face_ids = sorted(self._all_face_ids)

    def _compute_face_sections_with_indices(self, face_id):
        """Compute contiguous sections with cached pandas row indices for O(1) updates.
        Returns list of dicts: {'start': int, 'end': int, 'indices': np.ndarray}.
        Vectorized gap detection (Issue 3) with caching.
        """
        if face_id in self._section_cache:
            return self._section_cache[face_id]

        mask = self._landmarks_df['Face_ID'] == face_id
        if not mask.any():
            return []
        grouped_df = self._landmarks_df.loc[mask].sort_values('Frame_Index')
        frames = grouped_df['Frame_Index'].values
        orig_indices = grouped_df.index.values

        # Vectorized gap detection
        gaps = np.where(np.diff(frames) != 1)[0]
        boundaries = np.concatenate([[0], gaps + 1, [len(frames)]])

        sections = []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            sections.append({
                'start': int(frames[start_idx]),
                'end': int(frames[end_idx - 1]),
                'indices': orig_indices[start_idx:end_idx],
            })
        
        self._section_cache[face_id] = sections
        return sections

    def _compute_face_sections(self, face_id):
        """Compute contiguous sections of frame detections for a face.
        Returns list of (start_frame, end_frame) tuples.
        Vectorized gap detection (Issue 3).
        """
        mask = self._landmarks_df['Face_ID'] == face_id
        if not mask.any():
            return []
        frames = np.sort(self._landmarks_df.loc[mask, 'Frame_Index'].values)

        # Vectorized gap detection
        gaps = np.where(np.diff(frames) != 1)[0]
        boundaries = np.concatenate([[0], gaps + 1, [len(frames)]])

        sections = []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            sections.append((int(frames[start_idx]), int(frames[end_idx - 1])))
        return sections

    def _update_section_label(self, face_id):
        """Update the section info label for the given face."""
        if not hasattr(self, '_lbl_sec_info'):
            return
        sections = self._compute_face_sections(face_id)
        if not sections:
            self._lbl_sec_info.setText("No sections")
            return
        current_csv = self.slider.value() + 1
        current_sec = 0
        for idx, (s, e) in enumerate(sections):
            if s <= current_csv <= e:
                current_sec = idx + 1
                break
        else:
            if current_csv < sections[0][0]:
                current_sec = 1
            else:
                current_sec = len(sections)
        self._lbl_sec_info.setText(f"Section {current_sec}/{len(sections)}")

    def _on_nav_face_changed(self):
        """Update section label and rebuild table when face filter changes."""
        if hasattr(self, '_section_table'):
            self._build_section_table()
        self._overlay.hide()

    def _update_merge_button(self):
        """Update merge button text, mode, and enable state based on table selection."""
        if not hasattr(self, '_btn_merge'):
            return
        sel = self._section_table.selectionModel().selectedRows()
        selected_count = len(sel)
        total_sections = len(getattr(self, '_all_sections_data', []))

        if selected_count >= 2:
            self._btn_merge.setText("Merge Selected Sections")
            self._merge_mode = "selected"
        else:
            self._btn_merge.setText("Auto-Merge Same IDs")
            self._merge_mode = "all"

        # Disable merge when only one section exists (nothing to merge)
        self._btn_merge.setEnabled(total_sections >= 2)

        # Enable delete button only when rows are selected
        if hasattr(self, '_btn_delete_section'):
            self._btn_delete_section.setEnabled(selected_count > 0)

    def _goto_prev_section(self):
        """Jump to the start of the previous section for selected face."""
        fid = self._nav_face_combo.currentData()
        sections = self._compute_face_sections(fid)
        if not sections:
            return
        current_csv = self.slider.value() + 1
        for idx in range(len(sections) - 1, -1, -1):
            s, e = sections[idx]
            if s < current_csv:
                self.slider.setValue(s - 1)
                self._update_section_label(fid)
                return
        # Circle back to the last section if already at or before the first
        self.slider.setValue(sections[-1][0] - 1)
        self._update_section_label(fid)

    def _goto_next_section(self):
        """Jump to the start of the next section for selected face."""
        fid = self._nav_face_combo.currentData()
        sections = self._compute_face_sections(fid)
        if not sections:
            return
        current_csv = self.slider.value() + 1
        for idx, (s, e) in enumerate(sections):
            if current_csv < s:
                self.slider.setValue(s - 1)
                self._update_section_label(fid)
                return
        # Circle back to the first section if already past the last
        self.slider.setValue(sections[0][0] - 1)
        self._update_section_label(fid)

    def _reload_after_glue(self):
        """Defer heavy landmark reload to avoid UI blocking after glue operations."""
        self._adv_notice_lbl.setText("")  # Clear re-analysis notice after follow-up
        video_path = self.combo_video.itemData(self.combo_video.currentIndex())
        if video_path and os.path.isfile(video_path):
            self._load_landmarks(video_path)

    def _select_all_faces(self):
        """Select all face checkboxes."""
        for cb in self._face_checkboxes:
            cb.setChecked(True)
        self._selected_face_ids = list(self._all_face_ids)
        if self._project_settings:
            self._project_settings.set("video_preprocessor", "selected_face_ids",
                                       self._selected_face_ids)

    def _rename_current_section(self):
        fid = self._nav_face_combo.currentData()
        if fid is None: return
        new_fid = self._rename_face_spin.value()
        if fid == new_fid: return
        
        sections = self._compute_face_sections(fid)
        if not sections: return
        
        current_csv = self.slider.value() + 1
        target_sec = None
        for s, e in sections:
            if s <= current_csv <= e:
                target_sec = (s, e)
                break
        
        if not target_sec:
            show_warning(self, "No Valid Section", "Slider is not currently on a valid section for this face.")
            return
        
        s, e = target_sec
        if not ask_confirm(self, "Rename Face in Section?",
            f"Rename Face {fid} to Face {new_fid} for frames {s}–{e}?"):
            return
            mask = (self._landmarks_df['Face_ID'] == fid) & (self._landmarks_df['Frame_Index'] >= s) & (self._landmarks_df['Frame_Index'] <= e)
            
            # Check for collisions
            collision_mask = (self._landmarks_df['Face_ID'] == new_fid) & (self._landmarks_df['Frame_Index'] >= s) & (self._landmarks_df['Frame_Index'] <= e)
            if collision_mask.any():
                col_frames = self._landmarks_df.loc[collision_mask, 'Frame_Index'].tolist()
                show_warning(self, "Collision Detected", f"Face {new_fid} already exists on frames {col_frames[0]}...{col_frames[-1]}.\nPlease delete them or choose a different Face ID.")
                return
                
            self._landmarks_df.loc[mask, 'Face_ID'] = new_fid

            # Incremental cache update
            for f_idx in range(s, e + 1):
                if f_idx in self._landmarks_by_frame:
                    pts = self._landmarks_by_frame[f_idx].pop(fid, None)
                    if pts is not None:
                        self._landmarks_by_frame[f_idx][new_fid] = pts

            self._landmarks_df = self._landmarks_df.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)
            self._section_cache.clear()
            self._save_landmarks_to_file()
            self._all_face_ids = sorted(self._landmarks_df['Face_ID'].unique().tolist())
            self._update_face_selection_ui()
            self._refresh_frame()

    def _merge_faces(self):
        src_fid = self._merge_source_combo.currentData()
        tgt_fid = self._merge_target_combo.currentData()
        if src_fid is None or tgt_fid is None or src_fid == tgt_fid: return
        
        if not ask_confirm(self, "Merge Selected Sections?",
            f"Merge all frames of Face {src_fid} into Face {tgt_fid}?"):
            return
            mask_src = self._landmarks_df['Face_ID'] == src_fid
            src_frames = self._landmarks_df.loc[mask_src, 'Frame_Index'].values
            
            mask_tgt = self._landmarks_df['Face_ID'] == tgt_fid
            tgt_frames = self._landmarks_df.loc[mask_tgt, 'Frame_Index'].values
            
            collisions = np.intersect1d(src_frames, tgt_frames)
            if len(collisions) > 0:
                show_warning(self, "Collision Detected", f"Both faces exist on {len(collisions)} frames (e.g. frame {collisions[0]}).\nPlease remove overlapping detections before merging.")
                return
                
            self._landmarks_df.loc[mask_src, 'Face_ID'] = tgt_fid

            # Incremental cache update
            for f_idx in src_frames:
                if f_idx in self._landmarks_by_frame:
                    pts = self._landmarks_by_frame[f_idx].pop(src_fid, None)
                    if pts is not None:
                        self._landmarks_by_frame[f_idx][tgt_fid] = pts

            self._landmarks_df = self._landmarks_df.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)
            self._section_cache.clear()
            self._save_landmarks_to_file()
            self._all_face_ids = sorted(self._landmarks_df['Face_ID'].unique().tolist())
            self._update_face_selection_ui()
            self._refresh_frame()

    # ──────────────────────────────────────────────────────────
    # Section table backend (Phase 1)
    # ──────────────────────────────────────────────────────────

    def _compute_all_sections(self, face_id=None):
        """Build a flat list of section dicts for all faces, sorted by start frame.
        If face_id is provided, only return sections for that face.
        Each dict: {'face_id': int, 'start': int, 'end': int, 'indices': ndarray}"""
        all_sections = []
        fids = [face_id] if face_id is not None else self._all_face_ids
        for fid in fids:
            for sec in self._compute_face_sections_with_indices(fid):
                all_sections.append({
                    'face_id': fid,
                    'start': int(sec['start']),
                    'end': int(sec['end']),
                    'indices': sec['indices'],
                })
        all_sections.sort(key=lambda s: (s['start'], s['end']))
        return all_sections

    def _compute_section_overlaps_swepline(self, all_sections):
        """Compute overlaps between sections using sweep-line O(N log N).
        Returns dict: row_idx -> list of (overlapping_row_idx, overlapping_face_id, overlap_frames_str)"""
        # Build events: (frame, +1/-1, row_idx)
        events = []
        for i, sec in enumerate(all_sections):
            events.append((sec['start'], 1, i))
            events.append((sec['end'] + 1, -1, i))
        events.sort(key=lambda e: (e[0], -e[1]))

        active = set()
        overlaps = {i: [] for i in range(len(all_sections))}

        for frame, typ, i in events:
            if typ == 1:
                for j in active:
                    if all_sections[j]['face_id'] != all_sections[i]['face_id']:
                        ov_start = max(all_sections[i]['start'], all_sections[j]['start'])
                        ov_end = min(all_sections[i]['end'], all_sections[j]['end'])
                        overlaps[i].append((j, all_sections[j]['face_id'], f"{ov_start}–{ov_end}"))
                        overlaps[j].append((i, all_sections[i]['face_id'], f"{ov_start}–{ov_end}"))
                active.add(i)
            else:
                active.discard(i)
        return overlaps

    def _check_reassignment_collisions(self, pending_changes):
        """Validate pending reassignments. Returns (is_valid, collision_message_or_None).
        pending_changes: dict of {row_idx: new_fid}
        """
        # Issue 6: Lightweight copy of only the columns needed for validation
        df_check = self._landmarks_df[['Frame_Index', 'Face_ID']].copy()
        for row_idx, new_fid in pending_changes.items():
            sec = self._all_sections_data[row_idx]
            df_check.loc[sec['indices'], 'Face_ID'] = new_fid

        collision = df_check.duplicated(subset=['Frame_Index', 'Face_ID'], keep=False)
        if collision.any():
            bad_frames = df_check.loc[collision, 'Frame_Index'].unique()[:10]
            bad_fids = df_check.loc[collision, 'Face_ID'].unique()[:5]
            frames_str = ', '.join(str(f) for f in bad_frames)
            return False, (
                f"Collision detected: After reassignment, duplicate Face_ID entries exist "
                f"on frames: {frames_str}...\n"
                f"Conflicting Face_IDs: {', '.join(str(f) for f in bad_fids)}\n\n"
                f"Resolve by removing overlapping sections or choosing different IDs."
            )
        return True, None

    def _get_next_new_face_id(self):
        """Return the next available Face_ID not present in the dataset."""
        if not self._all_face_ids:
            return 1
        return max(self._all_face_ids) + 1

    def _auto_merge_sections(self):
        """Merge sections — either all eligible gaps for filtered face, or selected rows."""
        max_gap_secs = self._project_settings.get("processing", "max_gap_seconds", 2.0) \
            if self._project_settings else 2.0

        current_face = self._nav_face_combo.currentData() if hasattr(self, '_nav_face_combo') else None
        sections = self._all_sections_data
        merge_mode = getattr(self, '_merge_mode', 'all')

        if merge_mode == 'selected':
            selected_rows = sorted(it.row() for it in self._section_table.selectionModel().selectedRows())
            if len(selected_rows) < 2:
                show_warning(self, "Not Enough Sections Selected",
                                        "Please select at least 2 sections in the table to merge.")
                return
            selected_sections = [sections[r] for r in selected_rows]
            face_id = selected_sections[0]['face_id']
            gaps = []
            for i in range(len(selected_sections) - 1):
                s1 = selected_sections[i]
                s2 = selected_sections[i + 1]
                if s1['face_id'] != s2['face_id']:
                    continue
                end_f = s1['end']
                start_f = s2['start']
                if start_f > end_f + 1:
                    gaps.append((end_f, start_f))
        else:
            face_id = current_face
            face_secs = self._compute_face_sections(face_id)
            gaps = []
            for i in range(len(face_secs) - 1):
                end_f = face_secs[i][1]
                start_f = face_secs[i + 1][0]
                if start_f > end_f + 1:
                    gaps.append((end_f, start_f))

        if not gaps:
            show_warning(self, "No Merge Needed", "No sections are eligible for merging. Sections must be adjacent and have matching face IDs.")
            return

        self._overlay.set_status("Auto-merging sections...")
        self._overlay.show()
        QApplication.processEvents()

        # Check for long gaps
        long_gaps = []
        short_gaps = []
        for end_f, start_f in gaps:
            gap_secs = (start_f - end_f - 1) / max(self._fps, 1.0)
            if gap_secs > max_gap_secs:
                long_gaps.append((end_f, start_f, gap_secs))
            else:
                short_gaps.append((end_f, start_f))

        if long_gaps:
            gap_lines = "\n".join(f"  • Frames {e+1}→{s-1}: {g:.1f}s" for e, s, g in long_gaps)
            result = self._show_long_gap_dialog(
                long_gaps, max_gap_secs, gap_lines, short_gaps
            )
            if result is None:
                # Cancel
                self._overlay.hide()
                return
            elif result == "skip":
                gaps_to_merge = short_gaps
            elif result == "include":
                gaps_to_merge = [(e, s) for e, s, g in long_gaps] + short_gaps
            elif isinstance(result, tuple):
                # result = (new_max_gap_secs, user_confirmed)
                new_max_gap_secs, user_confirmed = result
                if not user_confirmed:
                    self._overlay.hide()
                    return
                self._project_settings.set("processing", "max_gap_seconds", new_max_gap_secs)
                max_gap_secs = new_max_gap_secs
                gaps_to_merge = []
                gaps_to_merge.extend(short_gaps)
                for e, s, g in long_gaps:
                    if g <= max_gap_secs:
                        gaps_to_merge.append((e, s))
        else:
            gaps_to_merge = list(gaps)

        if not gaps_to_merge:
            self._overlay.hide()
            show_warning(self, "No Merge Needed",
                                    "No gaps remain to merge after applying your filters.")
            return

        # Launch background worker
        tasks = [(face_id, gaps_to_merge)]
        self._btn_merge.setEnabled(False)
        self._btn_merge.setText("Merging...")
        self._overlay.set_status("Merging Sections")
        self._overlay.hide_progress()
        self._overlay.show()

        self._merge_worker = _AutoMergeWorker(
            self._landmarks_data_path, tasks, self._fps
        )
        self._merge_worker.progress.connect(self._on_merge_progress)
        self._merge_worker.finished.connect(self._on_merge_finished)
        self._merge_worker.start()

    def _show_long_gap_dialog(self, long_gaps, current_max_secs, gap_lines, short_gaps):
        """Show dialog for handling long gaps.

        Returns:
            'skip' — skip long gaps
            'include' — include long gaps
            (new_max_secs, confirmed) — redefine threshold (confirmed: bool)
            None — cancel
        """
        from PyQt6.QtWidgets import QDialog, QLineEdit
        from PyQt6.QtCore import QRegularExpression
        from PyQt6.QtGui import QRegularExpressionValidator

        dialog = QDialog(self)
        dialog.setWindowTitle("Long Gap Warning")
        dialog.setMinimumWidth(450)
        dialog._action = None
        layout = QVBoxLayout(dialog)

        info_lbl = QLabel(
            f"{len(long_gaps)} gap(s) exceed your Max Gap Fill threshold of "
            f"{current_max_secs}s:\n\n{gap_lines}\n\n"
            f"Interpolating these may produce inaccurate landmarks."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setObjectName("status-error-label")
        layout.addWidget(info_lbl)

        actions_lbl = QLabel("What would you like to do?")
        actions_lbl.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(actions_lbl)

        btn_row = QHBoxLayout()
        btn_skip = QPushButton("Skip Long Gaps")
        btn_skip.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        btn_include = QPushButton("Include All Gaps")
        btn_include.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        btn_row.addWidget(btn_skip)
        btn_row.addWidget(btn_include)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        redefine_layout = QHBoxLayout()
        redefine_layout.addWidget(QLabel("Or redefine Max Gap Fill threshold (seconds):"))
        redefine_input = QLineEdit(str(current_max_secs))
        redefine_input.setFixedWidth(80)
        validator = QRegularExpressionValidator(QRegularExpression(r"^\d+(\.\d+)?$"))
        redefine_input.setValidator(validator)
        redefine_layout.addWidget(redefine_input)
        btn_redefine = QPushButton("Apply New Threshold")
        btn_redefine.setStyleSheet("font-size: 12px; padding: 6px 16px;")
        redefine_layout.addWidget(btn_redefine)
        redefine_layout.addStretch()
        layout.addLayout(redefine_layout)

        def on_skip():
            dialog._action = "skip"
            dialog.accept()

        def on_include():
            dialog._action = "include"
            dialog.accept()

        def on_cancel():
            dialog._action = None
            dialog.reject()

        def on_redefine():
            text = redefine_input.text().strip()
            if not text:
                show_warning(self, "Invalid Input", "Please enter a positive number.")
                return
            try:
                new_val = float(text)
                if new_val <= 0:
                    raise ValueError
            except ValueError:
                show_warning(self, "Invalid Input", "Please enter a positive number.")
                return
            still_long = [(e, s, g) for e, s, g in long_gaps if g > new_val]
            if still_long:
                still_lines = "\n".join(f"  • Frames {e+1}→{s-1}: {g:.1f}s" for e, s, g in still_long)
                include_anyway = ask_confirm(self, "Still Exceeding Threshold",
                    f"{len(still_long)} gap(s) still exceed {new_val}s:\n\n"
                    f"{still_lines}\n\nInclude them anyway?")
                dialog._action = (new_val, include_anyway)
            else:
                dialog._action = (new_val, True)
            dialog.accept()

        btn_skip.clicked.connect(on_skip)
        btn_include.clicked.connect(on_include)
        btn_cancel.clicked.connect(on_cancel)
        btn_redefine.clicked.connect(on_redefine)

        dialog.exec()
        return dialog._action

    def _on_merge_progress(self, completed, total):
        self._overlay.set_status(f"Merging... {completed}/{total} face(s)")

    def _on_merge_finished(self, success, message):
        self._btn_merge.setEnabled(True)
        # Clear table selection so button resets to "Auto-Merge Same IDs"
        self._section_table.selectionModel().clearSelection()
        self._update_merge_button()
        self._merge_result_msg = message

        if success:
            self._overlay.set_status("Reloading landmarks...")
            self._overlay.hide_progress()
            QTimer.singleShot(50, self._on_reload_after_merge)
        else:
            self._overlay.hide()
            show_error(self, "Merge Failed", message)

    def _on_reload_after_merge(self):
        """Defer landmark reload to main event loop so the overlay can hide first."""
        self._overlay.hide()
        show_success(self, "Merge Complete", self._merge_result_msg)
        video_path = self.combo_video.itemData(self.combo_video.currentIndex())
        if video_path and os.path.isfile(video_path):
            self._load_landmarks(video_path)
        self._refresh_frame()

    # ──────────────────────────────────────────────────────────
    # Section table frontend (Phase 2)
    # ──────────────────────────────────────────────────────────

    def _build_section_table(self):
        """Update the section table model (Phase 1 virtualization)."""
        if not hasattr(self, '_section_model'):
            return

        current_face = self._nav_face_combo.currentData() if hasattr(self, '_nav_face_combo') else None
        all_sections = self._compute_all_sections(face_id=current_face)
        self._all_sections_data = all_sections
        overlaps = self._compute_section_overlaps_swepline(all_sections)

        # Update model data
        self._section_model._parent_widget = self
        self._section_model.update_data(all_sections, overlaps)
        
        # Update delegate's unique face list
        self._section_delegate._unique_fids = self._all_face_ids
        
        self._update_section_table_widths()
        self._update_section_action_layout()

    def _on_table_id_changed(self, row_idx, new_fid):
        """Handle ID reassignment from the table delegate."""
        sec = self._all_sections_data[row_idx]
        if sec['face_id'] == new_fid:
            self._pending_renames.pop(row_idx, None)
        else:
            self._pending_renames[row_idx] = new_fid
        
        # Notify the model to repaint this row to show pending color
        self._section_model.dataChanged.emit(
            self._section_model.index(row_idx, 0),
            self._section_model.index(row_idx, 5)
        )
        self._update_merge_button()

    def _update_section_table_widths(self):
        """Set consistent column widths for the section table."""
        header = self._section_table.horizontalHeader()
        # Column 0: # (Index)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 40)
        # Column 1: ID (Dropdown)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 100)
        # Column 2: Frames
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 90)
        # Column 3: Duration
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(3, 70)
        # Column 4: Overlaps (Stretch to fill)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        # Column 5: Play
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(5, 50)


    def _navigate_to_section(self, row_idx):
        """Single-click: jump to the start of a section without playing."""
        sec = self._all_sections_data[row_idx]
        self._stop_playback()
        self.slider.setValue(sec['start'] - 1)
        self._seek_and_show(sec['start'] - 1)

    def _play_section_from_table(self, row_idx):
        """Play a section from the virtual table (Step 1.3)."""
        sec = self._all_sections_data[row_idx]

        if self._playing and self._playing_row == row_idx:
            self._stop_playback()
            return

        # Start playing
        self._stop_playback()
        self._playing_row = row_idx
        self.slider.setValue(sec['start'] - 1)
        self._is_playing_section = True
        self._play_section_start = sec['start'] - 1
        
        # Force model update to show ⏸
        self._section_model.dataChanged.emit(
            self._section_model.index(row_idx, 5),
            self._section_model.index(row_idx, 5)
        )
        
        self._start_playback(end_frame=sec['end'] - 1)


    def _discard_section_changes(self):
        """Discard all pending dropdown changes and rebuild table."""
        if self._pending_renames:
            self._overlay.set_status("Discarding changes...")
            self._overlay.show()
            QApplication.processEvents()
            self._pending_renames = {}
            self._build_section_table()
            self._overlay.hide()
        else:
            show_warning(self, "No Changes", "There are no pending changes to discard.")

    def _apply_section_reassignments(self):
        """Validate and apply all pending reassignments."""
        if not self._pending_renames:
            show_warning(self, "No Changes", "No ID changes have been made. Change the dropdown values in the table first.")
            return

        is_valid, collision_msg = self._check_reassignment_collisions(self._pending_renames)
        if not is_valid:
            show_warning(self, "Collision Detected", collision_msg)
            return

        self._overlay.set_status("Applying reassignments...")
        self._overlay.show()
        QApplication.processEvents()

        changes_summary = []
        for row_idx, new_fid in self._pending_renames.items():
            sec = self._all_sections_data[row_idx]
            old_fid = sec['face_id']
            changes_summary.append(
                f"Section {row_idx + 1}: Face {old_fid} → Face {new_fid} (frames {sec['start']}–{sec['end']})"
            )
            self._landmarks_df.loc[sec['indices'], 'Face_ID'] = new_fid

            # Update cache incrementally (avoids O(N) reload)
            for f_idx in range(sec['start'], sec['end'] + 1):
                if f_idx in self._landmarks_by_frame:
                    pts = self._landmarks_by_frame[f_idx].pop(old_fid, None)
                    if pts is not None:
                        self._landmarks_by_frame[f_idx][new_fid] = pts

        self._landmarks_df = self._landmarks_df.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)
        # Background save to avoid UI freeze
        self._save_landmarks_to_file()

        # Update all_face_ids for UI consistency
        self._all_face_ids = sorted(self._landmarks_df['Face_ID'].unique().tolist())

        show_success(
            self, "Changes Applied",
            f"Successfully applied {len(self._pending_renames)} change(s):\n\n" + "\n".join(changes_summary)
        )

        self._pending_renames = {}
        # Clear caches since IDs and row indices have changed
        self._section_cache.clear()

        self._overlay.hide()
        # Defer heavy UI rebuild to the next event cycle so the overlay can hide
        # and the spinner animation isn't frozen during the rebuild.
        QTimer.singleShot(0, self._update_face_selection_ui)
        QTimer.singleShot(0, self._refresh_frame)

    def _deselect_all_faces(self):
        """Deselect all face checkboxes."""
        for cb in self._face_checkboxes:
            cb.setChecked(False)
        self._selected_face_ids = []
        if self._project_settings:
            self._project_settings.set("video_preprocessor", "selected_face_ids",
                                       self._selected_face_ids)

    def _save_landmarks_to_file(self):
        """Non-blocking save with debounce queue (Issue 8).

        If a save is in progress, sets _save_pending flag instead of
        blocking the UI thread.  When the worker finishes it checks the
        flag and triggers the next save automatically.
        """
        if self._landmarks_df is None or self._landmarks_data_path is None:
            return

        if hasattr(self, '_data_save_worker') and self._data_save_worker and self._data_save_worker.isRunning():
            self._save_pending = True
            return

        if str(self._landmarks_data_path).lower().endswith('.parquet'):
            self._data_save_worker = _ParquetSaveWorker(self._landmarks_df, self._landmarks_data_path)
        else:
            import pyarrow as pa
            arrow_table = pa.Table.from_pandas(self._landmarks_df)
            self._data_save_worker = _CsvSaveWorker(arrow_table, self._landmarks_data_path)
        self._save_pending = False
        self._data_save_worker.finished.connect(self._on_data_save_finished)
        self._data_save_worker.start()

    def _on_data_save_finished(self, success, message):
        """Handle data save completion; trigger pending save if any."""
        if self._save_pending and success:
            self._save_pending = False
            self._save_landmarks_to_file()

    def _on_face_selection_changed(self):
        """Update selected face IDs when checkboxes change."""
        self._selected_face_ids = sorted([cb.face_id for cb in self._face_checkboxes if cb.isChecked()])
        self._update_template_button()
        if self._project_settings:
            self._project_settings.set("video_preprocessor", "selected_face_ids",
                                       self._selected_face_ids)
        self._refresh_frame()

    # ------------------------------------------------------------------
    # Face label helpers
    # ------------------------------------------------------------------
    def _face_label(self, face_id):
        """Shorthand: get face label for current stimulus."""
        stim = self._current_stimulus_basename()
        return self._get_face_label(stim, face_id) if stim else f"Face {face_id}"

    def _current_stimulus_basename(self):
        """Return combo_video display text for current selection, or None."""
        idx = self.combo_video.currentIndex()
        if idx <= 0:
            return None
        return self.combo_video.itemText(idx) or None

    def _get_face_label(self, stimulus_basename, face_id):
        """Return display label for a face, falling back to 'Face {fid}'."""
        if not stimulus_basename or not self._project_settings:
            return f"Face {face_id}"
        all_labels = self._project_settings.get("video_preprocessor", "face_labels") or {}
        label = (all_labels.get(stimulus_basename, {}) or {}).get(str(face_id), "")
        return label if label else f"Face {face_id}"

    def _set_face_label(self, stimulus_basename, face_id, label):
        """Persist a face label for the given stimulus immediately."""
        if not self._project_settings:
            return
        all_labels = self._project_settings.get("video_preprocessor", "face_labels") or {}
        if stimulus_basename not in all_labels:
            all_labels[stimulus_basename] = {}
        if label and label.strip():
            all_labels[stimulus_basename][str(face_id)] = label.strip()
        else:
            all_labels[stimulus_basename].pop(str(face_id), None)
            if not all_labels[stimulus_basename]:
                all_labels.pop(stimulus_basename, None)
        self._project_settings.set("video_preprocessor", "face_labels", all_labels)

    def _show_label_picker(self, face_id, button_widget):
        """Open a popup dialog to pick/create a label for a face."""
        stim = self._current_stimulus_basename()
        current_label = self._get_face_label(stim, face_id) if stim else f"Face {face_id}"

        # Collect all existing labels across all stimuli for reuse
        all_labels = self._project_settings.get("video_preprocessor", "face_labels") or {} if self._project_settings else {}
        existing = sorted({
            lbl for d in all_labels.values() for lbl in d.values() if lbl and lbl.strip()
        })

        dialog = FaceLabelPickerDialog(self, existing, current_label, face_id)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_label = dialog.result() or ""
        if stim:
            self._set_face_label(stim, face_id, new_label)
        display = new_label if new_label else f"Face {face_id}"
        # Also update the checkbox text
        for cb in self._face_checkboxes:
            if cb.face_id == face_id:
                cb.setText(f"  {display}")
                break
        self._refresh_face_display_names()

    def _refresh_face_display_names(self):
        """Update display names in combos and section table without full rebuild."""
        stim = self._current_stimulus_basename()
        if not stim:
            return
        get = lambda fid: self._get_face_label(stim, fid)

        for combo in (getattr(self, '_nav_face_combo', None),
                      getattr(self, '_adv_face_combo', None),
                      getattr(self, '_adv_target_combo', None),
                      getattr(self, '_adv_merge_combo', None)):
            if combo is None:
                continue
            for i in range(combo.count()):
                txt = combo.itemText(i)
                if txt.startswith('+'):
                    continue
                fid = combo.itemData(i)
                if fid is not None:
                    combo.setItemText(i, get(fid))

        # Update label picker buttons
        # Button text is constant "Assign Label", no need to update here.
        pass

        if hasattr(self, '_section_model') and self._section_model:
            self._build_section_table()

    def _show_face_labels_dialog(self):
        """Open cross-stimulus face labels editor dialog."""
        processed_dir = os.path.join(self.current_project_path, "processed_stimuli_data")
        all_stimuli_face_ids = {}

        if os.path.isdir(processed_dir):
            for fname in os.listdir(processed_dir):
                if fname.endswith('_landmarks.parquet') or fname.endswith('_landmarks.csv'):
                    base = fname.replace('_landmarks.parquet', '').replace('_landmarks.csv', '')
                    # Find matching stimulus file
                    stimuli_dir = os.path.join(self.current_project_path, "stimuli")
                    stim_basename = None
                    if os.path.isdir(stimuli_dir):
                        for sf in os.listdir(stimuli_dir):
                            if os.path.splitext(sf)[0] == base:
                                stim_basename = sf
                                break
                    if stim_basename is None:
                        stim_basename = base

                    try:
                        if fname.endswith('.parquet'):
                            df = pd.read_parquet(os.path.join(processed_dir, fname))
                        else:
                            df = pd.read_csv(os.path.join(processed_dir, fname))
                        if 'Face_ID' in df.columns:
                            all_stimuli_face_ids[stim_basename] = sorted(
                                df['Face_ID'].unique().tolist())
                    except Exception:
                        pass

        all_labels = self._project_settings.get("video_preprocessor", "face_labels") or {} if self._project_settings else {}

        dialog = FaceLabelsDialog(self, all_labels, all_stimuli_face_ids)
        dialog.exec()

        # Refresh current display after dialog closes
        self._refresh_face_display_names()
        self._update_face_selection_ui()

    # ------------------------------------------------------------------

    def _on_timeline_clicked(self, frame_idx):
        """Jump to the frame at the clicked position on the timeline."""
        csv_idx = max(1, min(frame_idx, self._total_frames))
        slider_idx = csv_idx - 1
        self.slider.setValue(slider_idx)

    def _update_timeline_position(self, slider_value):
        """Update all face timelines playheads and section label when slider moves."""
        if hasattr(self, '_face_timelines'):
            for ft in self._face_timelines:
                ft.set_current_frame(slider_value + 1)
        if hasattr(self, '_nav_face_combo') and self._nav_face_combo.isVisible():
            fid = self._nav_face_combo.currentData()
            if fid is not None:
                self._update_section_label(fid)

    def _goto_prev_detection(self):
        """Jump to the previous frame that has landmark data for any face."""
        if not self._sorted_frame_indices:
            return
        current_csv = self.slider.value() + 1
        for fi in reversed(self._sorted_frame_indices):
            if fi < current_csv:
                self.slider.setValue(fi - 1)
                return

    def _goto_next_detection(self):
        """Jump to the next frame that has landmark data for any face."""
        if not self._sorted_frame_indices:
            return
        current_csv = self.slider.value() + 1
        for fi in self._sorted_frame_indices:
            if fi > current_csv:
                self.slider.setValue(fi - 1)
                return
        self._refresh_frame()

    def _seek_and_show(self, frame_idx):
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._current_frame = frame
        self._display_frame(frame, frame_idx)

    def _display_frame(self, frame, frame_idx):
        """Render a BGR frame (with optional overlays) into the QLabel."""
        h, w = frame.shape[:2]
        self._displayed_frame_index = frame_idx
        self._update_template_button()

        # Issue 4: Only copy frame when overlays are actually needed
        csv_frame_idx = frame_idx + 1
        faces_data = self._landmarks_by_frame.get(csv_frame_idx)

        show_landmarks = self.chk_landmarks.isChecked()
        show_delaunay = self.chk_delaunay.isChecked()
        show_margin = self.chk_contour_margin.isChecked()
        has_landmarks = faces_data is not None

        # Zero-copy path: no overlay needed and landmarks exist
        if not show_landmarks and not show_delaunay and not show_margin and has_landmarks:
            display = frame
        else:
            display = frame.copy()

        if has_landmarks and (show_landmarks or show_delaunay or show_margin):
            for face_id, pts_norm in faces_data.items():
                color_bgr = self._face_colors.get(face_id, (200, 200, 200))

                # Convert normalised [0,1] coords to pixel coords
                pts_px = pts_norm.copy()
                pts_px[:, 0] *= (w - 1)
                pts_px[:, 1] *= (h - 1)
                pts_int = pts_px.astype(np.int32)

                if show_delaunay:
                    try:
                        # Use precomputed Delaunay connectivity from the average face template.
                        # Ensures consistent triangle topology across all faces and videos.
                        if _delaunay_tri_cfg is not None:
                            simplices = _delaunay_tri_cfg
                            if isinstance(simplices, list):
                                simplices = np.array(simplices)
                            # Only use triangles whose vertex indices fit the per-frame landmark count.
                            n_pts = len(pts_px)
                            mask = (simplices < n_pts).all(axis=1)
                            simplices = simplices[mask]
                        else:
                            # Fallback: compute from current frame if avg face unavailable.
                            tri = Delaunay(pts_px)
                            simplices = tri.simplices

                        for simplex in simplices:
                            p0 = tuple(pts_int[simplex[0]])
                            p1 = tuple(pts_int[simplex[1]])
                            p2 = tuple(pts_int[simplex[2]])
                            cv2.line(display, p0, p1, color_bgr, 1, cv2.LINE_AA)
                            cv2.line(display, p1, p2, color_bgr, 1, cv2.LINE_AA)
                            cv2.line(display, p2, p0, color_bgr, 1, cv2.LINE_AA)
                    except Exception:
                        pass

                if show_margin:
                    self._draw_contour_margin(display, pts_norm, w, h, color_bgr)

                if show_landmarks:
                    for pt in pts_int:
                        cv2.circle(display, tuple(pt), 2, color_bgr, -1, cv2.LINE_AA)

                    if len(pts_int) > 0:
                        min_x, min_y = np.min(pts_int, axis=0)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = max(0.4, min(w, h) / 1000.0)
                        thickness = max(1, int(scale * 2))
                        cv2.putText(display, self._face_label(face_id), (min_x, max(0, min_y - 10)), font, scale, color_bgr, thickness, cv2.LINE_AA)

        elif not has_landmarks and self._landmarks_df is not None:
            # No landmarks for this frame — draw notice (display was already copied)
            text = "No landmarks for this frame"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = min(w, h) / 800.0
            thickness = max(1, int(scale * 2))
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            overlay = display.copy()
            pad = 10
            cv2.rectangle(overlay, (0, 0), (w, th + baseline + pad * 2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
            tx = (w - tw) // 2
            ty = th + pad
            cv2.putText(display, text, (tx, ty), font, scale,
                        (0, 160, 255), thickness, cv2.LINE_AA)

        # Convert BGR → RGB → QImage → QPixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Store original pixmap and immediately scale to current stage size
        self._original_frame_pixmap = pixmap
        self._scale_and_set_pixmap()

        # Update time label
        secs = frame_idx / self._fps if self._fps > 0 else 0
        mins = int(secs // 60)
        secs_rem = secs - mins * 60
        self.lbl_time.setText(
            f"Frame {frame_idx + 1} / {self._total_frames}   "
            f"({mins:02d}:{secs_rem:05.2f})"
        )

    def _draw_contour_margin(self, display, pts_norm, w, h, color_bgr):
        """Draw the detected face outline (thin) and the margin outline used in transformation."""
        if _delaunay_tri_cfg is None or landmark_extraction is None or len(pts_norm) < 478:
            return
        if getattr(self, '_outline_indices', None) is None:
            self._outline_indices = landmark_extraction.facial_outline_indices(np.asarray(_delaunay_tri_cfg))
        outline = self._outline_indices
        expanded = landmark_extraction.expand_face_contour(
            pts_norm[:478], outline, self.spin_contour_margin.value() / 100.0)
        # Outline indices are unordered; the outline is star-shaped, so sort by angle.
        ring = pts_norm[outline, :2]
        order = np.argsort(np.arctan2(*(ring - ring.mean(axis=0)).T[::-1]))
        scale = np.array([w - 1, h - 1])
        for poly, thickness in ((ring, 1), (expanded[outline, :2], 2)):
            px = (poly[order] * scale).astype(np.int32)
            cv2.polylines(display, [px], True, color_bgr, thickness, cv2.LINE_AA)

    def _scale_and_set_pixmap(self):
        """Scale the original frame pixmap to current label size and set on label."""
        if self._original_frame_pixmap is None:
            return
        target_size = self.lbl_frame.size()
        if target_size.isEmpty():
            return
        scaled = self._original_frame_pixmap.scaled(
            target_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_frame.setPixmap(scaled)

    def _refresh_frame(self):
        """Re-render the current frame (e.g. after toggling overlay)."""
        idx = self.slider.value()
        if self._current_frame is not None:
            self._display_frame(self._current_frame, idx)
        elif self._original_frame_pixmap is not None:
            # Only resize needed (no overlay toggle)
            self._scale_and_set_pixmap()

    # --- slider ---
    def _on_slider_changed(self, value):
        # Keep frame nav in sync with slider
        self.spin_frame_nav.blockSignals(True)
        self.spin_frame_nav.setValue(value + 1)
        self.spin_frame_nav.blockSignals(False)
        if self._playing:
            return  # don't fight the timer
        self._seek_and_show(value)

    # --- playback ---
    def _toggle_playback(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self, end_frame=None):
        if self._cap is None:
            return
        self._play_end_frame = end_frame
        self._playing = True
        self.btn_play.setText("⏹ Stop")
        interval = max(1, int(1000 / self._fps))
        self._play_timer.start(interval)

    def _stop_playback(self):
        """Standard method to stop playback and reset UI state."""
        self._playing = False
        self._play_end_frame = None
        self._is_playing_section = False
        
        if hasattr(self, '_playing_row') and self._playing_row != -1:
            old_row = self._playing_row
            self._playing_row = -1
            if hasattr(self, '_section_model'):
                self._section_model.dataChanged.emit(
                    self._section_model.index(old_row, 5),
                    self._section_model.index(old_row, 5)
                )

        self._play_timer.stop()
        self.btn_play.setText("▶ Play")

    def _play_tick(self):
        next_idx = self.slider.value() + 1
        if self._play_end_frame is not None and next_idx > self._play_end_frame:
            if self._is_playing_section and hasattr(self, '_play_section_start'):
                next_idx = self._play_section_start
            else:
                self._stop_playback()
                return
        elif next_idx >= self._total_frames:
            self._stop_playback()
            return
            
        self.slider.setValue(next_idx)
        self._seek_and_show(next_idx)

    def _play_current_section(self):
        if self._playing and self._is_playing_section:
            self._stop_playback()
            return
            
        fid = self._nav_face_combo.currentData()
        if fid is None: return
        
        sections = self._compute_face_sections(fid)
        if not sections: return
        
        current_csv = self.slider.value() + 1
        target_sec = None
        for s, e in sections:
            if s <= current_csv <= e:
                target_sec = (s, e)
                break
        
        if not target_sec:
            # Not in a section, try to find the next upcoming one
            for s, e in sections:
                if s > current_csv:
                    target_sec = (s, e)
                    break
            # Or fallback to the first section
            if not target_sec:
                target_sec = sections[0]
                
        start_frame, end_frame = target_sec
        
        self.slider.setValue(start_frame - 1)
        self._seek_and_show(start_frame - 1)
        
        # Start playback with an end frame
        self._is_playing_section = True
        self._play_section_start = start_frame - 1
        self._btn_play_sec.setText("⏸ Pause Section")
        self._start_playback(end_frame=end_frame - 1)

    # --- step buttons ---
    def _step_forward(self):
        self._stop_playback()
        nxt = min(self.slider.value() + 1, self._total_frames - 1)
        self.slider.setValue(nxt)

    def _step_backward(self):
        self._stop_playback()
        prev = max(self.slider.value() - 1, 0)
        self.slider.setValue(prev)

    def _on_frame_nav_editing_finished(self):
        """Jump to the frame number typed in the navigation spin box."""
        frame = self.spin_frame_nav.value()  # 1-based
        if frame < 1 or frame > self._total_frames:
            return
        self._stop_playback()
        target = frame - 1  # convert to 0-based slider index
        if self._is_image:
            return
        self.slider.setValue(target)
        self._seek_and_show(target)

    # --- advanced operations ---

    def _adv_on_start_changed(self):
        """When Start changes, clamp End if it falls below Start."""
        if self._adv_end_spin.value() < self._adv_start_spin.value():
            self._adv_end_spin.setValue(self._adv_start_spin.value())
        self._adv_update_info()

    def _adv_set_current_frame(self):
        """Set the active frame range spin to the current slider position."""
        current = self.slider.value() + 1  # 1-based
        # Update the focused spin box, default to Start
        if self._adv_active_range_spin == "end":
            self._adv_end_spin.setValue(current)
        else:
            self._adv_start_spin.setValue(current)
        # Enforce End >= Start after setting
        if self._adv_end_spin.value() < self._adv_start_spin.value():
            self._adv_end_spin.setValue(self._adv_start_spin.value())
        self._adv_update_info()

    def _adv_get_range(self):
        """Validate and return (face_id, start, end) or None on error."""
        fid = self._adv_face_combo.currentData()
        start = self._adv_start_spin.value()
        end = self._adv_end_spin.value()
        if start > end:
            show_warning(self, "Invalid Frame Range",
                                "Start frame must be ≤ End frame.")
            return None
        if fid is None:
            show_warning(self, "No Face Selected",
                                "Select a face from the Filter dropdown.")
            return None
        return fid, start, end

    def _adv_update_info(self):
        """Update the info label with data summary for the selected range."""
        result = self._adv_get_range()
        if result is None:
            self._adv_info_lbl.setText("Fix the range above to see info.")
            return
        fid, start, end = result
        if self._landmarks_df is None:
            self._adv_info_lbl.setText("No landmark data loaded.")
            return
        mask = (
            (self._landmarks_df['Face_ID'] == fid) &
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        rows = int(mask.sum())
        present = sorted(
            self._landmarks_df.loc[mask, 'Frame_Index'].unique().tolist()
        ) if rows else []
        all_frames = set(range(start, end + 1))
        missing = sorted(all_frames - set(present)) if present else list(range(start, end + 1))

        # Check for other faces in range
        other_mask = (
            (self._landmarks_df['Face_ID'] != fid) &
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        other_fids = sorted(self._landmarks_df.loc[other_mask, 'Face_ID'].unique().tolist())

        info_parts = [f"Face {fid}: {rows} rows, frames {start}–{end}"]
        if len(present) > 0 and len(present) < (end - start + 1):
            info_parts.append(f"present on {len(present)} frames")
            if len(missing) <= 10:
                info_parts.append(f"missing {missing}")
            else:
                info_parts.append(f"missing {len(missing)} frames")
        elif len(present) == (end - start + 1):
            info_parts.append("full coverage")
        if other_fids:
            info_parts.append(f"other faces: {', '.join('Face '+str(f) for f in other_fids)}")
        self._adv_info_lbl.setText(" | ".join(info_parts))

    def _adv_update_action_labels(self):
        """Update action button text and description labels based on selected face."""
        fid = self._adv_face_combo.currentData()
        if fid is None:
            return
        face_text = self._face_label(fid)

        # Reassign description
        self._adv_swap_desc.setText(
            f"Renames {face_text} in the selected frames to the target ID. Other faces unchanged."
        )

        # Merge description
        self._adv_merge_desc.setText(
            f"Replaces ALL face data in range with the target ID. "
            f"If {face_text} shares a frame with another face, the other face's data is deleted. NOT reversible."
        )

        # Remove button text
        self._adv_declare_gap_btn.setText(f"Remove {face_text} from the selected frame range")

        # Remove description
        self._adv_remove_desc.setText(
            f"Removes {face_text} from the selected frame range. "
            f"Creates a gap to be interpolated in the transformation step. Data is deleted permanently."
        )

        # Re-analyze description
        self._adv_reanalyze_desc.setText(
            f"Re-runs face detection for {face_text} in the selected frames. "
            f"New face ID(s) will be assigned. Use Merge or Reassign to consolidate."
        )

    def _adv_clear_notice(self):
        """Clear the persistent re-analysis notice label."""
        if hasattr(self, '_adv_notice_lbl'):
            self._adv_notice_lbl.setText("")

    def _adv_swap_face_id(self):
        """Swap Face ID for all rows of the filtered face in the frame range."""
        result = self._adv_get_range()
        if result is None:
            return
        fid, start, end = result
        if self._landmarks_df is None or not self._landmarks_data_path:
            show_warning(self, "No Landmark Data", "No landmark data is loaded. Run Face Extraction first, or load an existing landmark file.")
            return
        new_fid = self._adv_target_combo.currentData()
        if fid == new_fid:
            show_warning(self, "No Change",
                                    "Source and target Face ID are the same. Choose a different target.")
            return
        mask = (
            (self._landmarks_df['Face_ID'] == fid) &
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        rows_affected = int(mask.sum())
        if rows_affected == 0:
            show_warning(self, "No Data in Range",
                                    f"Face {fid} has no data in frames {start}–{end}.")
            return
        collision_mask = (
            (self._landmarks_df['Face_ID'] == new_fid) &
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        collision_count = int(collision_mask.sum())
        if collision_count > 0:
            show_warning(
                self, "Collision Detected",
                f"Face {new_fid} already has {collision_count} rows in frames {start}–{end}.\n\n"
                f"Swapping would create duplicates. Declare the conflicting face as a gap first, "
                f"or choose a different target ID."
            )
            return
        if not ask_confirm(self, "Reassign Face ID?",
            f"Reassign Face {fid} → Face {new_fid} for {rows_affected} rows "
            f"in frames {start}–{end}?\n\n"
            f"Only the filtered face is affected. Other faces in this range are unchanged.\n\n"
            f"This modifies the data file."):
            return
        self._landmarks_df.loc[mask, 'Face_ID'] = new_fid
        for f_idx in range(start, end + 1):
            if f_idx in self._landmarks_by_frame:
                pts = self._landmarks_by_frame[f_idx].pop(fid, None)
                if pts is not None:
                    self._landmarks_by_frame[f_idx][new_fid] = pts
        self._landmarks_df = self._landmarks_df.sort_values(
            ['Face_ID', 'Frame_Index']
        ).reset_index(drop=True)
        self._section_cache.clear()
        self._save_landmarks_to_file()
        QTimer.singleShot(0, lambda: self._reload_after_glue())

    def _adv_merge_to_id(self):
        """Merge ALL faces in the frame range into a single target Face ID.

        When multiple faces coexist on the same frame, only the last one
        (by sort order) is kept — earlier landmark rows are dropped.
        This prevents silent duplicate (Frame_Index, Face_ID) pairs from
        corrupting the downstream transformation step.
        """
        result = self._adv_get_range()
        if result is None:
            return
        _, start, end = result
        if self._landmarks_df is None or not self._landmarks_data_path:
            show_warning(self, "No Landmark Data", "No landmark data is loaded. Run Face Extraction first, or load an existing landmark file.")
            return
        new_fid = self._adv_merge_combo.currentData()
        range_mask = (
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        faces_in_range = sorted(
            self._landmarks_df.loc[range_mask, 'Face_ID'].unique().tolist()
        )
        if not faces_in_range:
            show_warning(self, "No Data in Range",
                                    f"No face data exists in frames {start}–{end}.")
            return
        if new_fid in faces_in_range:
            show_warning(
                self, "Target Already Present",
                f"Face {new_fid} already has data in this range. "
                "Choose a different target ID or declare the conflicting face "
                "as a gap first.\n\n"
                f"Faces in range: {', '.join('Face '+str(f) for f in faces_in_range)}"
            )
            return

        # Detect frames where multiple faces coexist — merging will drop data.
        range_df = self._landmarks_df.loc[range_mask]
        frames_with_multiple = (
            range_df.groupby('Frame_Index')['Face_ID']
            .nunique()
            .loc[lambda s: s > 1]
            .index.tolist()
        )
        frames_to_drop = len(frames_with_multiple)
        rows_affected = int(range_mask.sum())

        # Build explicit warning
        merge_msg = (
            f"Merge ALL face data in frames {start}–{end} into Face {new_fid}?\n\n"
            f"Faces in range: {', '.join('Face '+str(f) for f in faces_in_range)}\n"
            f"Total rows affected: {rows_affected}"
        )
        if frames_to_drop > 0:
            drop_list = frames_with_multiple[:20]
            merge_msg += (
                f"\n\n{status_icon_warning} {frames_to_drop} frame(s) have multiple faces. "
                f"Only the last face per frame will be kept; "
                f"earlier landmark data will be permanently DELETED.\n"
                f"Affected frames: {', '.join(str(f) for f in drop_list)}"
                + (f" ... and {len(frames_with_multiple)-20} more" if len(frames_with_multiple) > 20 else "")
            )
        else:
            merge_msg += "\n\nNo frames have overlapping faces — safe to merge."

        if not ask_confirm(self, "Merge All Faces in Range?", merge_msg):
            return

        # Reassign all faces in range to the target ID
        self._landmarks_df.loc[range_mask, 'Face_ID'] = new_fid

        # Drop duplicate (Frame_Index, Face_ID) rows — keep last
        self._landmarks_df = self._landmarks_df.drop_duplicates(
            subset=['Frame_Index', 'Face_ID'], keep='last'
        )

        # Update spatial cache for affected frames
        for f_idx in range(start, end + 1):
            if f_idx in self._landmarks_by_frame:
                for old_fid in list(self._landmarks_by_frame[f_idx].keys()):
                    if old_fid != new_fid:
                        pts = self._landmarks_by_frame[f_idx].pop(old_fid)
                        self._landmarks_by_frame[f_idx][new_fid] = pts
        self._landmarks_df = self._landmarks_df.sort_values(
            ['Face_ID', 'Frame_Index']
        ).reset_index(drop=True)
        self._section_cache.clear()
        self._save_landmarks_to_file()
        QTimer.singleShot(0, lambda: self._reload_after_glue())

    def _adv_declare_gap(self):
        """Declare a frame range as a gap by removing data for the selected face."""
        result = self._adv_get_range()
        if result is None:
            return
        fid, start, end = result
        if self._landmarks_df is None or not self._landmarks_data_path:
            show_warning(self, "No Landmark Data", "No landmark data is loaded. Run Face Extraction first, or load an existing landmark file.")
            return
        mask = (
            (self._landmarks_df['Face_ID'] == fid) &
            (self._landmarks_df['Frame_Index'] >= start) &
            (self._landmarks_df['Frame_Index'] <= end)
        )
        rows_to_remove = int(mask.sum())
        if rows_to_remove == 0:
            show_warning(self, "No Data in Range",
                                    f"Face {fid} has no data in frames {start}–{end} "
                                    f"to remove.")
            return
        if not ask_confirm(self, "Mark as Gap?",
            f"Remove {rows_to_remove} row(s) of Face {fid} in frames {start}–{end}?\n\n"
            f"This modifies the data file."):
            return
        # Use the section delete worker for robust multi-file deletion
        tasks = [(fid, start, end)]
        self._overlay.set_status("Declaring gap...")
        self._overlay.show()
        self._delete_worker = _SectionDeleteWorker(
            self._landmarks_data_path, tasks
        )
        self._delete_worker.finished.connect(self._on_adv_gap_declared)
        self._delete_worker.start()

    def _on_adv_gap_declared(self, success, msg):
        """Reload after gap declaration completes."""
        self._overlay.hide()
        if success:
            self._adv_notice_lbl.setText("")  # Clear stale notice after follow-up
            show_success(self, "Gap Declared",
                                    f"{status_icon_success} {msg}")
            video_path = self.combo_video.itemData(
                self.combo_video.currentIndex()
            )
            if video_path and os.path.isfile(video_path):
                self._load_landmarks(video_path)
        else:
            show_error(self, "Operation Failed",
                                 f"Failed to declare gap:\n{msg}")

    # --- re-analyze face detection in range ---
    def _adv_reanalyze_face_range(self):
        """Re-run MediaPipe face detection on the selected frame range.

        The newly detected face(s) receive new Face_ID(s), and the
        existing data for the filtered face in this range is replaced.
        """
        result = self._adv_get_range()
        if result is None:
            return
        fid, start, end = result
        if self._landmarks_df is None or not self._landmarks_data_path:
            show_warning(self, "No Landmark Data", "No landmark data is loaded. Run Face Extraction first, or load an existing landmark file.")
            return

        video_path = self.combo_video.itemData(self.combo_video.currentIndex())
        if not video_path or not os.path.isfile(video_path):
            show_warning(self, "No Video Loaded", "No valid video is loaded. Select a video from the dropdown before re-analyzing.")
            return

        # Determine the next available Face ID for the new detection
        new_fid = (max(self._all_face_ids) if self._all_face_ids else 1) + 1

        frame_count = end - start + 1

        if not ask_confirm(self, "Re-analyze Face Detection?",
            f"Re-run face detection on frames {start}–{end} ({frame_count} frames)?\n\n"
            f"Existing data for {self._face_label(fid)} in this range will be "
            f"replaced with newly detected face(s) starting at Face {new_fid}.\n\n"
            f"This is irreversible."):
            return

        # Clear any previous notice and store expected new FIDs for the callback
        self._adv_notice_lbl.setText("")
        num_faces = self.spin_faces.value()
        self._reanalysis_new_fids = list(range(new_fid, new_fid + num_faces))

        self._overlay.set_status("Re-analyzing face detection...")
        self._overlay.show()
        self._reanalysis_worker = _ReAnalysisWorker(
            video_path, start, end, num_faces,
            fid, new_fid, self._landmarks_data_path
        )
        self._reanalysis_worker.finished.connect(self._on_reanalysis_finished)
        self._reanalysis_worker.start()

    def _on_reanalysis_finished(self, success, msg):
        """Reload landmarks after re-analysis completes."""
        self._overlay.hide()
        if success:
            show_success(self, "Re-analysis Complete", msg)
            # Show persistent notice with follow-up actions
            new_fids = getattr(self, '_reanalysis_new_fids', [])
            if new_fids:
                fid_list = ', '.join('Face ' + str(f) for f in new_fids)
                self._adv_notice_lbl.setText(
                    "ATTENTION: New face(s) detected: %s. "
                    "Use Reassign or Merge above to consolidate into an existing Face ID, "
                    "or leave as-is to keep as separate faces." % fid_list
                )
            video_path = self.combo_video.itemData(
                self.combo_video.currentIndex()
            )
            if video_path and os.path.isfile(video_path):
                self._load_landmarks(video_path)
        else:
            show_error(self, "Re-analysis Failed", msg)

    # --- delete selected sections ---
    def _delete_selected_sections(self):
        """Delete the selected section(s) from all three data files."""
        selected = self._section_table.selectionModel().selectedRows()
        if not selected:
            show_warning(self, "Nothing Selected",
                                    "Select one or more rows in the table to delete.")
            return

        # Collect delete tasks from the model
        tasks = []
        for idx in selected:
            row = idx.row()
            if row >= len(self._section_model._sections):
                continue
            sec = self._section_model._sections[row]
            tasks.append((sec['face_id'], sec['start'], sec['end']))

        # Confirm with user
        detail = "\n".join(f"  Face {fid}, frames {s}–{e}" for fid, s, e in tasks)
        if not ask_confirm(self, "Delete Landmark Data?",
            f"Delete data for the following section(s)?\n\n{detail}\n\n"
            "This removes rows from landmarks, blendshapes, and\n"
            "transformation files. The preview will reload after."):
            return

        if not hasattr(self, '_landmarks_data_path') or not self._landmarks_data_path:
            show_warning(self, "No Landmark Data", "No landmark file is loaded. Run Face Extraction first, or load an existing landmark file.")
            return

        self._overlay.set_status("Deleting section data...")
        self._overlay.show()
        self._delete_worker = _SectionDeleteWorker(self._landmarks_data_path, tasks)
        self._delete_worker.finished.connect(self._on_section_delete_finished)
        self._delete_worker.start()

    def _on_section_delete_finished(self, success, msg):
        """Reload landmark data after section deletion completes."""
        self._overlay.hide()
        if success:
            show_success(self, "Deletion Complete", msg)
            # Reload to refresh in-memory data and UI
            video_path = self.combo_video.itemData(self.combo_video.currentIndex())
            if video_path and os.path.isfile(video_path):
                self._load_landmarks(video_path)
        else:
            show_error(self, "Deletion Failed", f"Delete failed:\n{msg}")

    def closeEvent(self, event: QCloseEvent):
        for attr in ('worker', '_merge_worker', '_delete_worker', '_load_worker',
                      '_data_save_worker'):
            w = getattr(self, attr, None)
            if w and isinstance(w, QThread) and w.isRunning():
                w.wait()


# ---------------------------------------------------------------------------
# Background video loader — opens video and reads first frame to avoid UI freeze
# ---------------------------------------------------------------------------
class _VideoLoaderThread(QThread):
    finished = pyqtSignal(bool, object, int, float, object)  # success, cap, total_frames, fps, first_frame

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit(False, None, 0, 0.0, None)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Read the first frame so we can display it immediately on return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            self.finished.emit(False, None, 0, 0.0, None)
            return

        self.finished.emit(True, cap, total_frames, fps, first_frame)


# Timeline section colors — distinct from Face ID label colors.
# Based on project_manager.py highlight palette, boosted for dark bg visibility.
_TIMELINE_COLORS = [
    QColor(130, 100, 190),   # Purple
    QColor(100, 140, 210),   # Blue
    QColor(100, 190, 140),   # Green
    QColor(200, 160, 90),    # Amber
    QColor(100, 200, 200),   # Teal
]

class _FaceTimelineBar(QWidget):
    clicked = pyqtSignal(int)  # frame_idx (1-based)

    def __init__(self, face_ids, face_sections, total_frames, face_colors,
                 current_frame=1):
        super().__init__()
        self._face_ids = face_ids
        self._face_sections = face_sections  # {face_id: [(s,e), (s,e), ...]}
        self._total_frames = max(total_frames, 1)
        self._face_colors = face_colors
        self._current_frame = current_frame
        self.setMinimumHeight(20)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_face_sections(self, sections_dict):
        """Update face sections data and repaint."""
        self._face_sections = sections_dict
        self.update()

    def set_current_frame(self, frame_idx):
        """Update the playhead position and repaint."""
        if self._current_frame != frame_idx:
            self._current_frame = frame_idx
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background with light outline for visibility on dark theme
        painter.fillRect(0, 0, w, h, QColor(40, 42, 48))
        painter.setPen(QColor(80, 84, 90))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._face_ids:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(painter.boundingRect(0, 0, w, h,
                            Qt.AlignmentFlag.AlignCenter, 0,
                            "No faces detected"), "No faces detected")
            return

        # Draw each face's sections
        pad = 2
        row_h = max(4, (h - pad * 2) // max(len(self._face_ids), 1))
        for row, fid in enumerate(self._face_ids):
            y = pad + row * row_h
            # Use timeline-specific colors (distinct from Face ID label colors)
            color = _TIMELINE_COLORS[row % len(_TIMELINE_COLORS)]
            sections = self._face_sections.get(fid, [])

            # Draw detection sections
            for s, e in sections:
                x1 = int((s - 1) / self._total_frames * (w - 1))
                x2 = int((e - 1) / self._total_frames * (w - 1))
                rect = QRect(x1, y, max(1, x2 - x1), row_h - 1)
                painter.fillRect(rect, color)

            # Draw gap markers between consecutive sections
            for i in range(len(sections) - 1):
                gap_end = int(sections[i][1])
                gap_start = int(sections[i + 1][0])
                gx1 = int((gap_end) / self._total_frames * (w - 1))
                gx2 = int((gap_start - 1) / self._total_frames * (w - 1))
                gw = max(1, gx2 - gx1 + 1)
                painter.fillRect(QRect(gx1, y, gw, row_h - 1), QColor(80, 160, 80))

        # Reset pen after section drawing
        painter.setPen(Qt.PenStyle.NoPen)

        # Current position indicator (white playhead line)
        pos_x = int((self._current_frame - 1) / self._total_frames * (w - 1))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawLine(pos_x, 0, pos_x, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            frac = event.position().x() / max(self.width(), 1)
            frame = int(frac * self._total_frames) + 1
            frame = max(1, min(frame, self._total_frames))
            self.clicked.emit(frame)
