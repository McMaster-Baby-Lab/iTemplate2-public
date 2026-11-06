"""
core.transformation — MediaPipe inference, Numba Delaunay, and coordinate
transformation logic.

Everything in this module is pure Python — no PyQt imports.
"""

import cv2
import concurrent.futures
import mediapipe as mp
import numpy as np
import os
import logging
import pandas as pd

from mediapipe.tasks.python import vision
from numba import njit, prange

from core.config import (
    detector, options, avg_landmarks, delaunay_tri, create_detector,
)
from core.io_utils import resolve_file_in_folder, atomic_save_parquet

logger = logging.getLogger('iTemplate2')


def _read_dataframe_auto(file_path):
    """Read CSV or Parquet based on file extension."""
    if str(file_path).lower().endswith('.parquet'):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _write_dataframe_auto(df, file_path):
    """Write CSV or Parquet based on file extension."""
    if str(file_path).lower().endswith('.parquet'):
        atomic_save_parquet(df, file_path)
    else:
        df.to_csv(file_path, index=False)


# ---------------------------------------------------------------------------
# Frame index helpers
# ---------------------------------------------------------------------------

def read_frame_index_data(file_path):
    """Read a CSV or Parquet file and return a DataFrame requiring 'Frame_Index'.

    Attempts to parse the file as Parquet if the extension is ``.parquet``,
    otherwise falls back to CSV. Returns ``None`` if the file is missing,
    empty, or lacks the required column.

    Parameters
    ----------
    file_path : str or PathLike
        Path to the data file (CSV or Parquet).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with at least a ``Frame_Index`` column, or ``None`` on error.

    Raises
    ------
    ValueError
        If the file does not contain a ``Frame_Index`` column.
    """
    try:
        if str(file_path).lower().endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
            
        if 'Frame_Index' not in df.columns:
            raise ValueError("The data file does not contain a 'Frame_Index' column.")
        return df
    except FileNotFoundError:
        logger.error("File not found at %s", file_path)
        return None
    except pd.errors.EmptyDataError:
        logger.error("The file at %s is empty", file_path)
        return None
    except ValueError as e:
        logger.error("Error: %s", e)
        return None

def find_missing_and_replace(df, method='nearest', max_gap=None):
    """Detect missing frame indices and compute replacement values.

    Finds gaps in the ``Frame_Index`` column and returns a DataFrame mapping
    each missing index to a replacement value. When *max_gap* is set, gaps
    larger than the threshold are excluded from the fill.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing a ``Frame_Index`` column.
    method : str, optional
        Fill strategy: ``'nearest'`` (default) replaces the first half of a gap
        with the preceding index and the second half with the following index.
        ``'preceding'`` fills the entire gap with the index immediately before it.
    max_gap : int or None, optional
        Maximum gap size (in frames) to fill. Gaps larger than this are skipped.
        ``None`` (default) fills all gaps regardless of size.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``Missing_Index`` and ``Replacement_Index``,
        or ``None`` if no gaps are found (or all gaps exceed *max_gap*).
    """
    if df is None:
        return [], None

    # Ensure 'Frame_Index' is sorted
    df = df.sort_values(by='Frame_Index').reset_index(drop=True)

    expected_indices = pd.Series(range(1, df['Frame_Index'].max() + 1))
    missing_indices = sorted(list(set(expected_indices) - set(df['Frame_Index'])))

    if not missing_indices:
        logger.info("No missing indices found.")
        return None

    # Group missing_indices based on intervals
    grouped_missing_indices = []
    current_group = []

    if missing_indices:  # Check if missing_indices is not empty
        current_group.append(missing_indices[0])
        for i in range(1, len(missing_indices)):
            if missing_indices[i] - missing_indices[i-1] == 1:
                current_group.append(missing_indices[i])
            else:
                grouped_missing_indices.append(current_group)
                current_group = [missing_indices[i]]
        grouped_missing_indices.append(current_group)

    # Filter out groups exceeding max_gap
    if max_gap is not None and max_gap > 0:
        kept_groups = []
        skipped_count = 0
        skipped_total_frames = 0
        for group in grouped_missing_indices:
            if len(group) > max_gap:
                skipped_count += 1
                skipped_total_frames += len(group)
                logger.info(
                    "Skipping gap of %d frames (frames %d-%d), exceeds max_gap=%d",
                    len(group), group[0], group[-1], max_gap
                )
            else:
                kept_groups.append(group)
        grouped_missing_indices = kept_groups
        if skipped_count > 0:
            logger.info(
                "Total: %d gap(s) skipped, %d frame(s) left as NaN",
                skipped_count, skipped_total_frames
            )

        # Narrow missing_indices to only those from kept groups
        if grouped_missing_indices:
            missing_indices = sorted(set(item for g in grouped_missing_indices for item in g))
        else:
            logger.info("All gaps exceed max_gap=%d, no replacement performed.", max_gap)
            return None

    # Replace indices based on method
    replaced_indices = []
    if method == 'nearest':
        total_groups = len(grouped_missing_indices)
        for i, group in enumerate(grouped_missing_indices):
            # Progress bar
            bar_length = 50
            filled_length = int(bar_length * (i + 1) / total_groups)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            progress = ((i + 1) / total_groups) * 100
            print(f'\rProcessing missing video frame {i+1}/{total_groups} [{bar}] {progress:.2f}%', end='')
            # Calculate the median of the 'Frame_Index' column within the group
            middle_index = np.array(group).mean()
            # Find the index of the middle value in the group
            group_array = np.array(group)
            min_index = group_array.min()
            max_index = group_array.max()

            # Vectorized conditions
            condition_le = group_array <= middle_index
            condition_gt = group_array > middle_index

            # Initialize replacement values with NaN
            replacement_value = np.full(len(group), np.nan)

            # Apply replacement logic based on conditions
            replacement_value[condition_le] = np.where((min_index - 1) >= df['Frame_Index'].min(), np.full(condition_le.sum(), min_index - 1), np.full(condition_le.sum(), max_index + 1))
            replacement_value[condition_gt] = np.where((max_index + 1) <= df['Frame_Index'].max(), np.full(condition_gt.sum(), max_index + 1), np.full(condition_gt.sum(), min_index - 1))

            replaced_indices.extend(replacement_value.astype(int).tolist())

    else:  # default to preceding method
        for group in grouped_missing_indices:
            replacement_value = group[0] - 1
            replaced_indices.extend([replacement_value] * len(group))

    missing_df = pd.DataFrame({'Missing_Index': missing_indices, 'Replacement_Index': replaced_indices})

    return missing_df


def glue_landmark_gap(file_path, face_id, gap_end_frame, gap_start_frame, fps=30.0):
    """Linearly interpolate landmark rows across a single gap.

    Fills frames ``gap_end_frame+1`` through ``gap_start_frame-1`` with
    linearly interpolated landmark coordinates for the given face,
    permanently writing the merged DataFrame back to *file_path*.

    Parameters
    ----------
    file_path : str or PathLike
        Path to the landmark file (CSV or Parquet).
    face_id : int or str
        The ``Face_ID`` value to interpolate.
    gap_end_frame : int
        Frame index of the last frame before the gap.
    gap_start_frame : int
        Frame index of the first frame after the gap.
    fps : float, optional
        Video frame rate (kept for API compatibility; not used in interpolation).

    Returns
    -------
    dict
        Dictionary with keys ``success`` (bool), ``frames_added`` (int),
        and ``message`` (str) describing the outcome.
    """
    if gap_start_frame <= gap_end_frame + 1:
        return {'success': False, 'frames_added': 0,
                'message': 'Gap is empty or adjacent frames, nothing to do.'}

    df = _read_dataframe_auto(file_path)
    face_df = df.loc[df['Face_ID'] == face_id]

    end_row = face_df.loc[face_df['Frame_Index'] == gap_end_frame]
    start_row = face_df.loc[face_df['Frame_Index'] == gap_start_frame]

    if end_row.empty:
        return {'success': False, 'frames_added': 0,
                'message': f'Boundary frame {gap_end_frame} not found '
                f'for Face_ID {face_id}.'}
    if start_row.empty:
        return {'success': False, 'frames_added': 0,
                'message': f'Boundary frame {gap_start_frame} not found '
                f'for Face_ID {face_id}.'}

    # Identify interpolable columns
    interp_cols = [c for c in df.columns if c.startswith('x_') or c.startswith('y_')
                   or c.startswith('z_') or c == 'Timestamp']

    n_frames = gap_start_frame - gap_end_frame
    num_new = n_frames - 1
    frame_indices = np.arange(gap_end_frame + 1, gap_start_frame)
    t = np.arange(1, num_new + 1) / n_frames  # interpolation weights

    end_vals = end_row.iloc[0][interp_cols].values.astype(np.float64)
    start_vals = start_row.iloc[0][interp_cols].values.astype(np.float64)

    # Vectorized interpolation: shape (num_new, len(interp_cols))
    deltas = start_vals - end_vals
    interp_values = end_vals[np.newaxis, :] + t[:, np.newaxis] * deltas[np.newaxis, :]

    # NaN mask: if either boundary is NaN, all interpolated values are NaN
    nan_mask = np.isnan(end_vals) | np.isnan(start_vals)
    if nan_mask.any():
        interp_values[:, nan_mask] = np.nan

    # Build DataFrame
    new_df = pd.DataFrame(interp_values, columns=interp_cols)
    new_df = pd.concat([
        pd.Series(frame_indices, name='Frame_Index'),
        pd.Series([face_id] * num_new, name='Face_ID'),
        new_df
    ], axis=1)
    new_df = new_df[df.columns]

    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)
    _write_dataframe_auto(df, file_path)

    return {'success': True, 'frames_added': num_new,
            'message': f'Added {num_new} interpolated frames '
            f'(frames {gap_end_frame + 1}-{gap_start_frame - 1}) '
            f'for Face_ID {face_id}.'}


def glue_landmark_gaps_batch(file_path, face_id, gaps, fps=30.0):
    """Interpolate all gaps for a single face in a single read/write pass.

    Reads the landmark file once, fills every gap with linearly interpolated
    coordinates, and writes back once — more efficient than calling
    :func:`glue_landmark_gap` repeatedly.

    Parameters
    ----------
    file_path : str or PathLike
        Path to the landmark file (CSV or Parquet).
    face_id : int or str
        The ``Face_ID`` value to interpolate.
    gaps : list of tuple
        List of ``(gap_end_frame, gap_start_frame)`` tuples defining the
        boundaries of each gap.
    fps : float, optional
        Video frame rate (kept for API compatibility; not used in interpolation).

    Returns
    -------
    dict
        Dictionary with keys ``success`` (bool), ``frames_added`` (int),
        and ``message`` (str) describing the outcome.
    """
    if not gaps:
        return {'success': False, 'frames_added': 0,
                'message': 'No gaps to fill.'}

    df = _read_dataframe_auto(file_path)
    face_df = df.loc[df['Face_ID'] == face_id]

    interp_cols = [c for c in df.columns if c.startswith('x_') or c.startswith('y_')
                   or c.startswith('z_') or c == 'Timestamp']

    all_new_frames = []
    total_added = 0

    for gap_end_frame, gap_start_frame in gaps:
        if gap_start_frame <= gap_end_frame + 1:
            continue

        end_row = face_df.loc[face_df['Frame_Index'] == gap_end_frame]
        start_row = face_df.loc[face_df['Frame_Index'] == gap_start_frame]

        if end_row.empty or start_row.empty:
            continue

        n_frames = gap_start_frame - gap_end_frame
        num_new = n_frames - 1
        frame_indices = np.arange(gap_end_frame + 1, gap_start_frame)
        t = np.arange(1, num_new + 1) / n_frames

        end_vals = end_row.iloc[0][interp_cols].values.astype(np.float64)
        start_vals = start_row.iloc[0][interp_cols].values.astype(np.float64)

        deltas = start_vals - end_vals
        interp_values = end_vals[np.newaxis, :] + t[:, np.newaxis] * deltas[np.newaxis, :]

        nan_mask = np.isnan(end_vals) | np.isnan(start_vals)
        if nan_mask.any():
            interp_values[:, nan_mask] = np.nan

        row_dict = {
            'Frame_Index': frame_indices,
            'Face_ID': np.full(num_new, face_id, dtype=np.int64),
        }
        for col_idx, col in enumerate(interp_cols):
            row_dict[col] = interp_values[:, col_idx]
        all_new_frames.append(pd.DataFrame(row_dict))
        total_added += num_new

    if not all_new_frames:
        return {'success': False, 'frames_added': 0,
                'message': 'No valid gaps to fill.'}

    new_rows = pd.concat(all_new_frames, ignore_index=True)
    new_rows = new_rows[df.columns]
    df = pd.concat([df, new_rows], ignore_index=True)
    df = df.sort_values(['Face_ID', 'Frame_Index']).reset_index(drop=True)
    _write_dataframe_auto(df, file_path)

    return {'success': True, 'frames_added': total_added,
            'message': f'Added {total_added} interpolated frames for Face_ID {face_id}.'}


def _glue_face_gaps_worker(args):
    """Picklable worker for parallel gap interpolation.

    Unpacks arguments and delegates to :func:`glue_landmark_gaps_batch`.
    Exists as a top-level function so ``multiprocessing`` can pickle it.

    Parameters
    ----------
    args : tuple
        ``(file_path, face_id, gaps, fps)`` passed to
        :func:`glue_landmark_gaps_batch`.

    Returns
    -------
    dict
        Result dictionary from :func:`glue_landmark_gaps_batch`.
    """
    file_path, face_id, gaps, fps = args
    return glue_landmark_gaps_batch(file_path, face_id, gaps, fps)


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def add_boundary_points(landmarks):
    """Add boundary points around facial landmarks to span the full frame.

    Appends eight points along the image border (corners and edge midpoints)
    so the Delaunay triangulation covers the entire frame, not just the face.
    For 3-D landmarks the boundary points receive a large Z value (2.0) to
    place them behind the face.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape ``(N, 2)`` or ``(N, 3)`` with the detected face
        landmark coordinates.

    Returns
    -------
    np.ndarray
        Array of shape ``(N+8, 2)`` or ``(N+8, 3)`` with boundary points
        appended. Returns *landmarks* unchanged if the dimensionality is
        neither 2 nor 3.
    """

    # Determine dimensionality by looking at the first landmark
    if len(landmarks[0]) == 2:
        boundary_pts = [
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5),
            (1.0, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.5)
        ]
    elif len(landmarks[0]) == 3:
        # Use large Z value (2.0) to place boundary behind face
        boundary_pts = [
            (0.0, 0.0, 2.0), (0.5, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 0.5, 2.0),
            (1.0, 1.0, 2.0), (0.5, 1.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.5, 2.0)
        ]
    else:
        # Fallback or error
        return landmarks

    landmarks_extended = np.concatenate((landmarks, boundary_pts), axis=0)
    return landmarks_extended


def facial_outline_indices(tri_indices):
    """Return landmark indices on the outer edge of the facial triangles.

    An outline edge belongs to exactly one facial triangle (all vertices
    below 478); its other side is a peripheral triangle.
    """
    facial = np.asarray(tri_indices)[np.all(np.asarray(tri_indices) < 478, axis=1)]
    edges = np.sort(np.concatenate([facial[:, [0, 1]], facial[:, [1, 2]], facial[:, [0, 2]]]), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique_edges[counts == 1])


def expand_face_contour(landmarks, outline_indices, margin):
    """Scale the stimulus face outline away from its centre by *margin* (0.05 = 5%).

    Only the stimulus-side mesh is expanded; the template keeps its landmarks.
    Gaze within the margin therefore falls into facial triangles and maps
    inside the template face contour instead of through peripheral triangles.
    """
    if not margin:
        return landmarks
    out = np.array(landmarks, dtype=np.float32, copy=True)
    ring = out[outline_indices, :2]
    centre = ring.mean(axis=0)
    out[outline_indices, :2] = np.clip(centre + (ring - centre) * (1.0 + margin), 0.0, 1.0)
    return out


def detect_face_landmarks(frame):
    """Detect facial landmarks in a single image frame.

    Converts the frame from BGR to RGB, runs the global MediaPipe
    FaceLandmarker, and extracts both 2-D and 3-D landmark coordinates
    for every detected face. Lazily initialises the global detector if
    it has not yet been created.

    Parameters
    ----------
    frame : np.ndarray
        BGR image of shape ``(H, W, 3)`` as returned by OpenCV.

    Returns
    -------
    list of tuple
        One ``(landmarks_2d, landmarks_3d)`` tuple per detected face.
        Each element is a ``list[tuple]`` of length 478.

    Raises
    ------
    RuntimeError
        If the global MediaPipe FaceLandmarker fails to initialize.
    """
    import core.config as _cfg
    if _cfg.detector is None:
        try:
             _cfg.detector = vision.FaceLandmarker.create_from_options(_cfg.options)
        except Exception as e:
             raise RuntimeError(
                 f"MediaPipe FaceLandmarker failed to initialize: {e}\n\n"
                 "Ensure 'face_landmarker.task' is present in the iTemplate2 root directory.\n"
                 "Download it from: https://storage.googleapis.com/mediapipe-models/"
                 "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
             ) from e

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = _cfg.detector.detect(mp_image)

    all_faces = []
    if hasattr(results, 'face_landmarks') and results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            lms_2d = []
            lms_3d = []
            for landmark in face_landmarks:
                lms_2d.append((landmark.x, landmark.y))
                lms_3d.append((landmark.x, landmark.y, landmark.z))
            all_faces.append((lms_2d, lms_3d))
    return all_faces


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def get_detector(num_faces=None):
    """Obtain a MediaPipe FaceLandmarker detector instance.

    Returns the shared global detector when *num_faces* is ``None``.
    Otherwise creates a fresh detector tuned for the requested number of
    simultaneous faces, which is required when spawning parallel workers.

    Parameters
    ----------
    num_faces : int or None
        Maximum number of faces to detect per frame. ``None`` (default)
        returns the shared global detector.

    Returns
    -------
    vision.FaceLandmarker
        A MediaPipe FaceLandmarker instance.
    """
    import core.config as _cfg
    if num_faces is None:
        return _cfg.get_face_detector()
    return _cfg.create_detector(num_faces=num_faces)

def _process_video_chunk(args):
    """Process a contiguous range of video frames for face landmarks.

    Opens the video, seeks to *start_frame*, and runs the MediaPipe
    FaceLandmarker on every frame through *end_frame*. Disables internal
    multi-threading (OpenCV, OMP, MKL) to avoid core contention when
    multiple workers run in parallel.

    Parameters
    ----------
    args : tuple
        ``(input_path, start_frame, end_frame, num_faces)`` where
        *input_path* is the video file path, *start_frame* and
        *end_frame* are 0-based inclusive frame indices, and *num_faces*
        is passed to :func:`get_detector`.

    Returns
    -------
    tuple of pd.DataFrame
        ``(df_landmarks, df_blendshapes, df_transformation)`` — three
        DataFrames (possibly empty) with one row per detected face per
        frame.
    """
    input_path, start_frame, end_frame, num_faces = args
    
    # Disable internal multi-threading to avoid core contention between workers
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Cache detector per (pid, num_faces) to avoid re-creating per call
    global _chunk_detector_cache
    if '_chunk_detector_cache' not in globals():
        _chunk_detector_cache = {}
    cache_key = (os.getpid(), num_faces)
    if cache_key not in _chunk_detector_cache:
        _chunk_detector_cache[cache_key] = get_detector(num_faces)
    local_detector = _chunk_detector_cache[cache_key]
    
    landmarks_data = []
    blendshapes_data = []
    trans_data = []
    
    for frame_idx in range(start_frame + 1, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_sec = frame_idx / max(fps, 1.0)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = local_detector.detect(mp_image)
        
        if results.face_landmarks:
            for idx, ls in enumerate(results.face_landmarks):
                face_id = idx + 1
                
                # Optimized Landmarks collection: return a dict but minimize key creation
                # Actually, building a large dict is still slow. 
                # Let's use a flat list for landmarks and keys only for metadata.
                lm_dict = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                for i, lm in enumerate(ls):
                    lm_dict[f'x_{i}'] = lm.x
                    lm_dict[f'y_{i}'] = lm.y
                    lm_dict[f'z_{i}'] = lm.z
                landmarks_data.append(lm_dict)
                
                # Blendshapes
                if results.face_blendshapes and idx < len(results.face_blendshapes):
                    bs = results.face_blendshapes[idx]
                    bs_dict = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                    for item in bs:
                        bs_dict[item.category_name] = item.score
                    blendshapes_data.append(bs_dict)
                
                # Matrices
                if results.facial_transformation_matrixes and idx < len(results.facial_transformation_matrixes):
                    mat = results.facial_transformation_matrixes[idx]
                    flat = mat.flatten() if isinstance(mat, np.ndarray) else [x for r in mat for x in r]
                    t_dict = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                    for i, val in enumerate(flat):
                        t_dict[f'Matrix_{i}'] = val
                    
                    rm = np.array([
                        [flat[0], flat[1], flat[2]],
                        [flat[4], flat[5], flat[6]],
                        [flat[8], flat[9], flat[10]]
                    ])
                    yaw = np.arcsin(-rm[2, 0])
                    if np.cos(yaw) > 1e-6:
                        pitch = np.arctan2(rm[2, 1], rm[2, 2])
                        roll = np.arctan2(rm[1, 0], rm[0, 0])
                    else:
                        roll = np.arctan2(-rm[0, 1], rm[1, 1])
                        pitch = 0
                    t_dict['roll_deg'] = np.degrees(roll)
                    t_dict['pitch_deg'] = np.degrees(pitch)
                    t_dict['yaw_deg'] = np.degrees(yaw)
                    trans_data.append(t_dict)
                    
    cap.release()
    
    # Build DataFrames inside the worker to offload processing
    df_l = pd.DataFrame(landmarks_data) if landmarks_data else pd.DataFrame()
    df_b = pd.DataFrame(blendshapes_data) if blendshapes_data else pd.DataFrame()
    df_t = pd.DataFrame(trans_data) if trans_data else pd.DataFrame()
    
    return df_l, df_b, df_t


def process_video(input_path, output_dir=None, progress_callback=None, export_csv=True, num_faces=None):
    """Extract face landmarks, blendshapes, and transformation matrices from a video.

    Runs MediaPipe face detection on every frame, applies a rolling
    5-frame smoother to 2-D landmark coordinates per face, and writes
    the results to Parquet (and optionally CSV) in *output_dir*.

    Parameters
    ----------
    input_path : str or PathLike
        Path to the input video file.
    output_dir : str or PathLike, optional
        Directory for output files. Defaults to a ``processed_stimuli_data``
        sub-folder next to the input video.
    progress_callback : callable, optional
        Called as ``callback(current, total)`` after processing starts
        and completes.
    export_csv : bool, optional
        If ``True`` (default), also write CSV copies alongside Parquet.
    num_faces : int or None, optional
        Maximum number of faces to detect per frame. ``None`` uses the
        global default detector.

    Returns
    -------
    list of str
        Paths to the three output files: landmarks, blendshapes,
        and transformation matrices.

    Raises
    ------
    ValueError
        If the video cannot be opened.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "processed_stimuli_data")

    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    # Canonical outputs are parquet for speed; CSV is optional compatibility export.
    out_landmarks = os.path.join(output_dir, f"{base_filename}_landmarks.parquet")
    out_blendshapes = os.path.join(output_dir, f"{base_filename}_blendshapes.parquet")
    out_trans = os.path.join(output_dir, f"{base_filename}_transformation.parquet")
    out_landmarks_csv = os.path.join(output_dir, f"{base_filename}_landmarks.csv")
    out_blendshapes_csv = os.path.join(output_dir, f"{base_filename}_blendshapes.csv")
    out_trans_csv = os.path.join(output_dir, f"{base_filename}_transformation.csv")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if progress_callback:
        progress_callback(0, frame_count)

    # Reuse the chunk logic for consistency (single chunk)
    l_data, b_data, t_data = _process_video_chunk((input_path, 0, frame_count, num_faces))

    if progress_callback:
        progress_callback(frame_count, frame_count)

    # Save
    if not l_data.empty:
        df_lm = l_data
        xy_cols = [c for c in df_lm.columns if c.startswith('x_') or c.startswith('y_')]
        # Smooth per face so rolling mean doesn't cross face boundaries
        for _, group_df in df_lm.groupby('Face_ID'):
            smoothed = group_df[xy_cols].rolling(
                window=5, center=True, min_periods=1
            ).mean()
            df_lm.loc[group_df.index, xy_cols] = smoothed
        atomic_save_parquet(df_lm, out_landmarks)
        if export_csv:
            df_lm.to_csv(out_landmarks_csv, index=False)
    if not b_data.empty:
        atomic_save_parquet(b_data, out_blendshapes)
        if export_csv:
            b_data.to_csv(out_blendshapes_csv, index=False)
    if not t_data.empty:
        atomic_save_parquet(t_data, out_trans)
        if export_csv:
            t_data.to_csv(out_trans_csv, index=False)
    
    return [out_landmarks, out_blendshapes, out_trans]

def process_image(input_path, output_dir=None, export_csv=True, num_faces=None):
    """Extract face landmarks, blendshapes, and transformation matrices from an image.

    Runs MediaPipe face detection on a single image and writes the
    results to Parquet (and optionally CSV) in *output_dir*.

    Parameters
    ----------
    input_path : str or PathLike
        Path to the input image file.
    output_dir : str or PathLike, optional
        Directory for output files. Defaults to a ``processed_stimuli_data``
        sub-folder next to the input image.
    export_csv : bool, optional
        If ``True`` (default), also write CSV copies alongside Parquet.
    num_faces : int or None, optional
        Maximum number of faces to detect. ``None`` uses the global
        default detector.

    Returns
    -------
    list of str
        Paths to the three output files: landmarks, blendshapes,
        and transformation matrices.

    Raises
    ------
    ValueError
        If the image cannot be read.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "processed_stimuli_data")

    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    out_landmarks = os.path.join(output_dir, f"{base_filename}_landmarks.parquet")
    out_blendshapes = os.path.join(output_dir, f"{base_filename}_blendshapes.parquet")
    out_trans = os.path.join(output_dir, f"{base_filename}_transformation.parquet")
    out_landmarks_csv = os.path.join(output_dir, f"{base_filename}_landmarks.csv")
    out_blendshapes_csv = os.path.join(output_dir, f"{base_filename}_blendshapes.csv")
    out_trans_csv = os.path.join(output_dir, f"{base_filename}_transformation.csv")

    local_detector = get_detector(num_faces)

    image = cv2.imread(input_path)
    if image is None:
         raise ValueError(f"Could not read image: {input_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    results = local_detector.detect(mp_image)

    landmarks_data = []
    blendshapes_data = []
    trans_data = []

    if results.face_landmarks:
        for idx, ls in enumerate(results.face_landmarks):
            face_id = idx + 1
            frame_idx = 1
            timestamp_sec = 0.0

            row_l = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
            for i, lm in enumerate(ls):
                row_l[f'x_{i}'] = lm.x
                row_l[f'y_{i}'] = lm.y
                row_l[f'z_{i}'] = lm.z
            landmarks_data.append(row_l)

            if results.face_blendshapes and idx < len(results.face_blendshapes):
                bs = results.face_blendshapes[idx]
                row_b = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                for item in bs:
                    row_b[item.category_name] = item.score
                blendshapes_data.append(row_b)

            if results.facial_transformation_matrixes and idx < len(results.facial_transformation_matrixes):
                mat = results.facial_transformation_matrixes[idx]
                flat = mat.flatten() if isinstance(mat, np.ndarray) else [x for r in mat for x in r]
                row_t = {'Frame_Index': frame_idx, 'Timestamp': timestamp_sec, 'Face_ID': face_id}
                for i, val in enumerate(flat):
                    row_t[f'Matrix_{i}'] = val

                rm = np.array([
                    [flat[0], flat[1], flat[2]],
                    [flat[4], flat[5], flat[6]],
                    [flat[8], flat[9], flat[10]]
                ])
                yaw = np.arcsin(-rm[2, 0])
                if np.cos(yaw) > 1e-6:
                    pitch = np.arctan2(rm[2, 1], rm[2, 2])
                    roll = np.arctan2(rm[1, 0], rm[0, 0])
                else:
                    roll = np.arctan2(-rm[0, 1], rm[1, 1])
                    pitch = 0
                row_t['roll_deg'] = np.degrees(roll)
                row_t['pitch_deg'] = np.degrees(pitch)
                row_t['yaw_deg'] = np.degrees(yaw)

                trans_data.append(row_t)

    if landmarks_data:
        df_lm = pd.DataFrame(landmarks_data)
        atomic_save_parquet(df_lm, out_landmarks)
        if export_csv:
            df_lm.to_csv(out_landmarks_csv, index=False)
    if blendshapes_data:
        df_b = pd.DataFrame(blendshapes_data)
        atomic_save_parquet(df_b, out_blendshapes)
        if export_csv:
            df_b.to_csv(out_blendshapes_csv, index=False)
    if trans_data:
        df_t = pd.DataFrame(trans_data)
        atomic_save_parquet(df_t, out_trans)
        if export_csv:
            df_t.to_csv(out_trans_csv, index=False)

    return [out_landmarks, out_blendshapes, out_trans]

def batch_process(files, output_dir=None, progress_main_cb=None, progress_file_cb=None):
    """Process a list of video and/or image files sequentially.

    Inspects each file extension to decide whether to call
    :func:`process_video` or :func:`process_image`. Unsupported file
    types are skipped with a warning.

    Parameters
    ----------
    files : list of str
        Paths to the video or image files to process.
    output_dir : str or PathLike, optional
        Shared output directory for all files. Passed through to the
        individual processing functions.
    progress_main_cb : callable, optional
        Called as ``cb(current_file_index, total_files)`` after each
        file completes.
    progress_file_cb : callable, optional
        Passed to :func:`process_video` as its ``progress_callback``
        for per-frame progress within each video.

    Returns
    -------
    list
        Empty list (placeholder; results are written to disk).
    """
    results = []  # Unused return in original
    for i, f in enumerate(files):
        if progress_main_cb: progress_main_cb(i+1, len(files))

        ext = os.path.splitext(f)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            process_image(f, output_dir)
            if progress_file_cb: progress_file_cb(1, 1)
        elif ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
            process_video(f, output_dir, progress_file_cb)
        else:
             logger.warning("Skipping unsupported file: %s", f)

    return []  # Placeholder


# ---------------------------------------------------------------------------
# Numba-accelerated simplex search
# ---------------------------------------------------------------------------

@njit(parallel=True)
def find_simplex_numba(points, vertices_3d, tri_indices):
    """
    Find which triangle each point belongs to using parallel processing.
    Handles occlusion by selecting the triangle with the minimum average Z-value (closest to camera).
    """
    N = points.shape[0]
    M = tri_indices.shape[0]
    simplex_ids = np.full(N, -1, dtype=np.int32)

    for i in prange(N):
        px = points[i, 0]
        py = points[i, 1]

        best_tri = -1
        min_avg_z = 10000.0

        for j in range(M):
            t1 = tri_indices[j, 0]
            t2 = tri_indices[j, 1]
            t3 = tri_indices[j, 2]

            x1 = vertices_3d[t1, 0]
            y1 = vertices_3d[t1, 1]
            x2 = vertices_3d[t2, 0]
            y2 = vertices_3d[t2, 1]
            x3 = vertices_3d[t3, 0]
            y3 = vertices_3d[t3, 1]

            # AABB early-out: skip triangle if point is outside bounding box
            min_x = min(x1, x2, x3)
            max_x = max(x1, x2, x3)
            min_y = min(y1, y2, y3)
            max_y = max(y1, y2, y3)
            if px < min_x or px > max_x or py < min_y or py > max_y:
                continue

            d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
            d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
            d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            if not (has_neg and has_pos):
                z1 = vertices_3d[t1, 2]
                z2 = vertices_3d[t2, 2]
                z3 = vertices_3d[t3, 2]
                avg_z = (z1 + z2 + z3) / 3.0

                if avg_z < min_avg_z:
                    min_avg_z = avg_z
                    best_tri = j

        if best_tri != -1:
            simplex_ids[i] = best_tri

    return simplex_ids


@njit
def find_simplex_numba_serial(points, vertices_3d, tri_indices):
    """
    Serial fallback for environments where Numba's workqueue threading layer
    cannot be called concurrently from multiple Python threads.
    """
    N = points.shape[0]
    M = tri_indices.shape[0]
    simplex_ids = np.full(N, -1, dtype=np.int32)

    for i in range(N):
        px = points[i, 0]
        py = points[i, 1]

        best_tri = -1
        min_avg_z = 10000.0

        for j in range(M):
            t1 = tri_indices[j, 0]
            t2 = tri_indices[j, 1]
            t3 = tri_indices[j, 2]

            x1 = vertices_3d[t1, 0]
            y1 = vertices_3d[t1, 1]
            x2 = vertices_3d[t2, 0]
            y2 = vertices_3d[t2, 1]
            x3 = vertices_3d[t3, 0]
            y3 = vertices_3d[t3, 1]

            # AABB early-out: skip triangle if point is outside bounding box
            min_x = min(x1, x2, x3)
            max_x = max(x1, x2, x3)
            min_y = min(y1, y2, y3)
            max_y = max(y1, y2, y3)
            if px < min_x or px > max_x or py < min_y or py > max_y:
                continue

            d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
            d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
            d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            if not (has_neg and has_pos):
                z1 = vertices_3d[t1, 2]
                z2 = vertices_3d[t2, 2]
                z3 = vertices_3d[t3, 2]
                avg_z = (z1 + z2 + z3) / 3.0

                if avg_z < min_avg_z:
                    min_avg_z = avg_z
                    best_tri = j

        if best_tri != -1:
            simplex_ids[i] = best_tri

    return simplex_ids


# ---------------------------------------------------------------------------
# Vectorized / parallel coordinate transformation
# ---------------------------------------------------------------------------

def _find_mapping_triangles(points, landmarks, tri_indices,
                            use_parallel_simplex=True, mapping_mode='all'):
    """Find facial matches before peripheral matches, preserving triangle IDs.

    Indices 0--477 are facial landmarks; 478--485 are frame boundaries.
    A peripheral triangle contains at least one boundary vertex.
    """
    if mapping_mode not in ('all', 'facial', 'peripheral'):
        raise ValueError(f"Unknown mapping mode: {mapping_mode}")
    search = find_simplex_numba if use_parallel_simplex else find_simplex_numba_serial
    result = np.full(len(points), -1, dtype=np.int32)
    is_facial = np.all(tri_indices < 478, axis=1)
    modes = ('facial', 'peripheral') if mapping_mode == 'all' else (mapping_mode,)
    for mode in modes:
        triangle_ids = np.flatnonzero(is_facial if mode == 'facial' else ~is_facial)
        rows = np.flatnonzero(result == -1)
        if not len(triangle_ids) or not len(rows):
            continue
        matches = search(points[rows], landmarks, tri_indices[triangle_ids])
        valid = matches >= 0
        result[rows[valid]] = triangle_ids[matches[valid]]
    return result


def _write_face_mapping(data, indices, points, simplex_ids, tri_indices,
                        face_id, x_col, y_col, face_id_col, face_label_col):
    """Write unclaimed samples and distinguish face identity from off-face gaze."""
    unfilled = data.loc[indices, x_col].isna().to_numpy()
    rows = indices[unfilled]
    if not len(rows):
        return
    on_face = np.all(tri_indices[simplex_ids[unfilled]] < 478, axis=1)
    if 'gaze_region' not in data.columns:
        data['gaze_region'] = pd.Series('unmapped', index=data.index, dtype='string')
    data.loc[rows, [x_col, y_col]] = points[unfilled]
    data.loc[rows[on_face], face_id_col] = face_id
    data.loc[rows[on_face], face_label_col] = f"Face {face_id}"
    data.loc[rows[~on_face], face_id_col] = pd.NA
    data.loc[rows[~on_face], face_label_col] = 'Off-face'
    data.loc[rows[on_face], 'gaze_region'] = 'on_face'
    data.loc[rows[~on_face], 'gaze_region'] = 'off_face'


def transform_per_group_vectorized(data, stimuliFolder, landmarkFolder, videoName,
                                   face_id=1, x_col='transformed_x', y_col='transformed_y',
                                   face_id_col='face_id', face_label_col='face_label',
                                   use_parallel_simplex=True, mapping_mode='all', template=None,
                                   contour_margin=0.0):
    """Transform gaze coordinates for a single video stimulus.

    For each frame, maps raw gaze positions (``stimuli_x``, ``stimuli_y``)
    onto the normalized face template using affine transforms derived from
    per-frame MediaPipe landmarks and Delaunay triangulation. Prefers
    pre-computed landmark files; falls back to real-time detection.

    Writes results only to rows not yet claimed by a prior face
    (first-match-wins), so that multiple sequential face passes do not
    overwrite each other.

    Parameters
    ----------
    data : pd.DataFrame
        Gaze data containing ``stimuli_x``, ``stimuli_y``, and
        ``_video_frame_index`` columns.
    stimuliFolder : str or PathLike
        Directory containing the original video stimuli.
    landmarkFolder : str or PathLike or None
        Directory containing pre-computed landmark files (Parquet/CSV).
    videoName : str
        Stimulus filename used to locate the landmark or video file.
    face_id : int, optional
        Which ``Face_ID`` to extract from the landmark file (default ``1``).
    x_col : str, optional
        Output column name for transformed X (default ``'transformed_x'``).
    y_col : str, optional
        Output column name for transformed Y (default ``'transformed_y'``).
    face_id_col : str, optional
        Output column for the integer face identifier.
    face_label_col : str, optional
        Output column for the human-readable face label.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with transformed coordinates written in place.

    Raises
    ------
    RuntimeError
        If the average face template has not been loaded.
    """
    import core.config as _cfg

    if x_col not in data.columns:
        data[x_col] = np.nan
    if y_col not in data.columns:
        data[y_col] = np.nan

    _delaunay_tri = template.triangles if template is not None else _cfg.delaunay_tri
    if _delaunay_tri is None:
        raise RuntimeError(
            "Average face template (average_face_data.h5) is not loaded. "
            "Ensure it is present in the iTemplate2 root directory."
        )
    if isinstance(_delaunay_tri, list):
        _delaunay_tri = np.array(_delaunay_tri)
        _cfg.delaunay_tri = _delaunay_tri
    tri_indices = _delaunay_tri.astype(np.int32)
    outline_indices = facial_outline_indices(tri_indices) if contour_margin else None

    _avg_landmarks = template.landmarks if template is not None else _cfg.avg_landmarks
    avg_dim = _avg_landmarks.shape[1]

    if avg_dim == 2:
        boundary_pts_tpl = [(0,0), (0.5,0), (1,0), (1,0.5), (1,1), (0.5,1), (0,1), (0,0.5)]
    else:
        boundary_pts_tpl = [(0,0,0), (0.5,0,0), (1,0,0), (1,0.5,0), (1,1,0), (0.5,1,0), (0,1,0), (0,0.5,0)]

    boundary_arr_tpl = np.array(boundary_pts_tpl, dtype=np.float32)
    avg_landmarks_extended = np.vstack([_avg_landmarks, boundary_arr_tpl])
    avg_landmarks_2d = avg_landmarks_extended[:, :2]

    landmark_map = {}
    use_precomputed_landmarks = False

    if landmarkFolder is not None:
        landmark_file_path = resolve_file_in_folder(
            videoName, landmarkFolder, suffix='_landmarks', extensions=['.parquet', '.csv']
        )
        if landmark_file_path and os.path.exists(landmark_file_path):
            try:
                landmarks_all = _read_dataframe_auto(landmark_file_path)
                use_precomputed_landmarks = True

                cols_x = [f'x_{i}' for i in range(478)]
                cols_y = [f'y_{i}' for i in range(478)]
                cols_z = [f'z_{i}' for i in range(478)]

                if cols_x[0] in landmarks_all.columns:
                    if 'Face_ID' in landmarks_all.columns:
                        face_data = landmarks_all[landmarks_all['Face_ID'] == face_id]
                    else:
                        face_data = landmarks_all

                    lx = face_data[cols_x].values
                    ly = face_data[cols_y].values
                    lz = face_data[cols_z].values
                    l_stack = np.dstack((lx, ly, lz)).astype(np.float32)
                    frame_indices = face_data['Frame_Index'].values
                    landmark_map = {idx: l_stack[i] for i, idx in enumerate(frame_indices)}
                else:
                     use_precomputed_landmarks = False
            except Exception as e:
                logger.error("Error loading landmarks: %s", e)
                use_precomputed_landmarks = False

    cap = None
    if not use_precomputed_landmarks:
        video_path = resolve_file_in_folder(videoName, stimuliFolder)
        if video_path is None:
            logger.error("Could not find video file '%s' in %s", videoName, stimuliFolder)
            return data
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file %s", video_path)
            return data

    frame_groups = data.groupby('_video_frame_index')

    current_frame_index = -1

    for i, (idx, frame_data) in enumerate(frame_groups):
        if pd.isna(idx):
            continue

        frame_idx = int(idx) - 1
        landmarks_frame_3d = None

        if use_precomputed_landmarks:
            landmarks_frame_3d = landmark_map.get(frame_idx + 1)
            if landmarks_frame_3d is not None:
                landmarks_frame_3d = add_boundary_points(landmarks_frame_3d).astype(np.float32)
            else:
                continue
        else:
            if frame_idx != current_frame_index:
                if frame_idx == current_frame_index + 1:
                    ret, frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                if not ret: continue
                current_frame_index = frame_idx

                all_faces = detect_face_landmarks(frame)
                if not all_faces: continue
                # Select the face for the requested face_id (default first face)
                face_idx = face_id - 1
                if face_idx < 0 or face_idx >= len(all_faces):
                    continue
                lms, lms_3d = all_faces[face_idx]

                landmarks_frame_3d = add_boundary_points(np.array(lms_3d))
                landmarks_frame_3d = np.array(landmarks_frame_3d, dtype=np.float32)
            else:
                 pass

        if landmarks_frame_3d is None:
            continue
        landmarks_frame_3d = expand_face_contour(landmarks_frame_3d, outline_indices, contour_margin)

        points_2d = landmarks_frame_3d[:, :2].copy()

        gaze_points = frame_data[['stimuli_x', 'stimuli_y']].values.astype(np.float32)
        valid_gaze_mask = np.isfinite(gaze_points).all(axis=1)
        if not np.any(valid_gaze_mask):
            continue

        gaze_points_valid = gaze_points[valid_gaze_mask]
        original_indices = frame_data.index[valid_gaze_mask]

        simplex_indices = _find_mapping_triangles(
            gaze_points_valid, landmarks_frame_3d, tri_indices,
            use_parallel_simplex, mapping_mode)

        inside_mask = simplex_indices != -1
        if not np.any(inside_mask):
            continue

        simplex_indices = simplex_indices[inside_mask]
        points_inside = gaze_points_valid[inside_mask]
        indices_inside = original_indices[inside_mask]

        transformed_points = np.zeros_like(points_inside)

        unique_simplices = np.unique(simplex_indices)

        for sim_idx in unique_simplices:
            mask_tri = simplex_indices == sim_idx
            tri_vertex_indices = tri_indices[sim_idx]

            src_tri = points_2d[tri_vertex_indices].astype(np.float32)
            dst_tri = avg_landmarks_2d[tri_vertex_indices].astype(np.float32)

            affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)

            pts = points_inside[mask_tri]
            m00, m01, m02 = affine_matrix[0]
            m10, m11, m12 = affine_matrix[1]

            tp_x = m00 * pts[:, 0] + m01 * pts[:, 1] + m02
            tp_y = m10 * pts[:, 0] + m11 * pts[:, 1] + m12

            transformed_points[mask_tri, 0] = tp_x
            transformed_points[mask_tri, 1] = tp_y

        _write_face_mapping(data, indices_inside, transformed_points,
                            simplex_indices, tri_indices, face_id,
                            x_col, y_col, face_id_col, face_label_col)

    if cap is not None:
        cap.release()
    return data


def transform_per_group_img(data, stimuliFolder, landmarkFolder, imgName,
                            face_id=1, x_col='transformed_x', y_col='transformed_y',
                            face_id_col='face_id', face_label_col='face_label',
                            use_parallel_simplex=True, mapping_mode='all', template=None,
                            contour_margin=0.0):
    """Transform gaze coordinates for a single static image stimulus.

    Loads pre-computed landmarks (or detects them in real time) and maps
    raw gaze positions onto the normalized face template via affine
    triangle transforms. Like the video variant, only fills rows not
    already claimed by a prior face.

    Parameters
    ----------
    data : pd.DataFrame
        Gaze data containing ``stimuli_x`` and ``stimuli_y`` columns.
    stimuliFolder : str or PathLike
        Directory containing the original image stimuli.
    landmarkFolder : str or PathLike or None
        Directory containing pre-computed landmark files (Parquet/CSV).
    imgName : str
        Stimulus filename used to locate the landmark or image file.
    face_id : int, optional
        Which ``Face_ID`` to extract from the landmark file (default ``1``).
    x_col : str, optional
        Output column name for transformed X (default ``'transformed_x'``).
    y_col : str, optional
        Output column name for transformed Y (default ``'transformed_y'``).
    face_id_col : str, optional
        Output column for the integer face identifier.
    face_label_col : str, optional
        Output column for the human-readable face label.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with transformed coordinates written in place.

    Raises
    ------
    RuntimeError
        If the average face template has not been loaded.
    """
    import core.config as _cfg

    landmarks_frame_3d = None
    if landmarkFolder is not None:
        try:
            landmark_file_path = resolve_file_in_folder(
                imgName, landmarkFolder, suffix='_landmarks', extensions=['.parquet', '.csv']
            )
            if landmark_file_path and os.path.exists(landmark_file_path):
                 landmarks_all = _read_dataframe_auto(landmark_file_path)
                 cols_x = [f'x_{i}' for i in range(478)]
                 if cols_x[0] in landmarks_all.columns:
                     if 'Face_ID' in landmarks_all.columns:
                         face_data = landmarks_all[landmarks_all['Face_ID'] == face_id]
                     else:
                         face_data = landmarks_all
                     if len(face_data) == 0:
                         return data
                     lx = face_data[cols_x].values[0]
                     ly = face_data[[f'y_{i}' for i in range(478)]].values[0]
                     lz = face_data[[f'z_{i}' for i in range(478)]].values[0]
                     landmarks_frame_3d = np.dstack((lx, ly, lz))[0].astype(np.float32)
        except:
             pass

    if landmarks_frame_3d is None:
        img_path = resolve_file_in_folder(imgName, stimuliFolder)
        if not img_path:
            logger.warning("Image not found: %s", imgName)
            return data

        image = cv2.imread(img_path)
        if image is None: return data

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_faces = detect_face_landmarks(image_rgb)

        if not all_faces: return data
        face_idx = face_id - 1
        if face_idx < 0 or face_idx >= len(all_faces):
            return data
        lms, lms_3d = all_faces[face_idx]
        landmarks_frame_3d = np.array(lms_3d, dtype=np.float32)

    landmarks_frame_3d = add_boundary_points(landmarks_frame_3d).astype(np.float32)

    _delaunay_tri = template.triangles if template is not None else _cfg.delaunay_tri
    if _delaunay_tri is None:
        raise RuntimeError(
            "Average face template (average_face_data.h5) is not loaded. "
            "Ensure it is present in the iTemplate2 root directory."
        )
    if isinstance(_delaunay_tri, list):
        _delaunay_tri = np.array(_delaunay_tri)
        _cfg.delaunay_tri = _delaunay_tri
    tri_indices = _delaunay_tri.astype(np.int32)

    _avg_landmarks = template.landmarks if template is not None else _cfg.avg_landmarks

    boundary_pts_tpl = [(0,0), (0.5,0), (1,0), (1,0.5), (1,1), (0.5,1), (0,1), (0,0.5)]
    if _avg_landmarks.shape[1] == 3:
         boundary_pts_tpl = [(0,0,0), (0.5,0,0), (1,0,0), (1,0.5,0), (1,1,0), (0.5,1,0), (0,1,0), (0,0.5,0)]

    avg_landmarks_extended = np.vstack([_avg_landmarks, np.array(boundary_pts_tpl)])
    avg_landmarks_2d = avg_landmarks_extended[:, :2]

    if contour_margin:
        landmarks_frame_3d = expand_face_contour(
            landmarks_frame_3d, facial_outline_indices(tri_indices), contour_margin)
    points_2d = landmarks_frame_3d[:, :2].copy()

    gaze_points = data[['stimuli_x', 'stimuli_y']].values.astype(np.float32)
    valid_gaze_mask = np.isfinite(gaze_points).all(axis=1)
    if not np.any(valid_gaze_mask): return data

    gaze_points_valid = gaze_points[valid_gaze_mask]
    original_indices = data.index[valid_gaze_mask]

    simplex_indices = _find_mapping_triangles(
        gaze_points_valid, landmarks_frame_3d, tri_indices,
        use_parallel_simplex, mapping_mode)

    inside_mask = simplex_indices != -1
    if not np.any(inside_mask): return data

    simplex_indices = simplex_indices[inside_mask]
    points_inside = gaze_points_valid[inside_mask]
    indices_inside = original_indices[inside_mask]

    transformed_points = np.zeros_like(points_inside)
    unique_simplices = np.unique(simplex_indices)

    for sim_idx in unique_simplices:
        mask_tri = simplex_indices == sim_idx
        tri_vertex_indices = tri_indices[sim_idx]

        src_tri = points_2d[tri_vertex_indices].astype(np.float32)
        dst_tri = avg_landmarks_2d[tri_vertex_indices].astype(np.float32)

        affine_matrix = cv2.getAffineTransform(src_tri, dst_tri)

        pts = points_inside[mask_tri]
        m00, m01, m02 = affine_matrix[0]
        m10, m11, m12 = affine_matrix[1]

        tp_x = m00 * pts[:, 0] + m01 * pts[:, 1] + m02
        tp_y = m10 * pts[:, 0] + m11 * pts[:, 1] + m12

        transformed_points[mask_tri, 0] = tp_x
        transformed_points[mask_tri, 1] = tp_y

    _write_face_mapping(data, indices_inside, transformed_points,
                        simplex_indices, tri_indices, face_id,
                        x_col, y_col, face_id_col, face_label_col)
    return data


def transform_coordinates_parallel(data, type='video', stimuliFolder='data', landmarkFolder=None,
                                   num_workers=None, face_ids=None, template=None,
                                   contour_margin=0.0):
    """Dispatch parallel coordinate transformation across stimuli and faces.

    Groups gaze data by stimulus name, then processes each group (in parallel
    via a thread pool) using either :func:`transform_per_group_vectorized`
    (video) or :func:`transform_per_group_img` (image). Iterates over *face_ids*
    sequentially, first checking facial triangles across all selected faces,
    then mapping remaining samples through peripheral triangles. A valid face
    assignment is never overwritten by another face's peripheral mapping.

    Parameters
    ----------
    data : pd.DataFrame
        Gaze data with a stimulus name column (``videoName``, ``stimulusName``,
        or ``STIMULI_NAME``) and ``stimuli_x`` / ``stimuli_y`` coordinates.
    type : str, optional
        ``'video'`` (default) or ``'image'``. Selects the transformation
        function and the group column.
    stimuliFolder : str or PathLike, optional
        Directory containing original stimuli files.
    landmarkFolder : str or PathLike or None, optional
        Directory containing pre-computed landmark files.
    num_workers : int or None, optional
        Number of threads in the pool. ``None`` uses ``os.cpu_count()``.
    face_ids : list of int or None, optional
        Face IDs to process (default ``[1]``). Processed sequentially;
        first facial match claims a point when facial regions overlap.
        Peripheral mapping is deferred until all faces have been checked.
    contour_margin : float, optional
        Fraction by which each stimulus face outline is enlarged before
        facial triangles are matched (default ``0``). Absorbs gaze error
        near the face edge; template landmarks are unchanged.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with columns ``transformed_x``, ``transformed_y``,
        ``face_id``, ``face_label``, and ``gaze_region`` added/filled.
        ``gaze_region`` is ``on_face``, ``off_face``, or ``unmapped``.
        Off-face samples retain peripheral coordinates but no attended face ID.

    Raises
    ------
    ValueError
        If the required stimulus name column is missing.
    """
    if face_ids is None:
        face_ids = [1]

    renamed_col = False

    if type == 'video':
        if 'videoName' not in data.columns and 'stimulusName' in data.columns:
            data = data.rename(columns={'stimulusName': 'videoName'})
            renamed_col = True
        if 'videoName' not in data.columns:
            raise ValueError("videoName/stimulusName column not found.")
        group_col = 'videoName'
        func = transform_per_group_vectorized

    elif type == 'image':
        if 'STIMULI_NAME' not in data.columns and 'stimulusName' in data.columns:
            data = data.rename(columns={'stimulusName': 'STIMULI_NAME'})
            renamed_col = True
        if 'STIMULI_NAME' not in data.columns:
            raise ValueError("STIMULI_NAME/stimulusName column not found.")
        group_col = 'STIMULI_NAME'
        func = transform_per_group_img

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    # Avoid concurrent calls into Numba's parallel workqueue from multiple
    # Python threads. When the outer dispatcher uses >1 worker thread, use the
    # serial Numba simplex kernel inside each task.
    use_parallel_simplex = num_workers == 1

    # Long-format output: fixed columns regardless of how many faces are processed.
    # face_id uses nullable Int64 to avoid NaN coercing integers to floats.
    x_col         = 'transformed_x'
    y_col         = 'transformed_y'
    face_id_col   = 'face_id'
    face_label_col = 'face_label'

    data[x_col]          = np.nan
    data[y_col]          = np.nan
    data[face_id_col]    = pd.array([pd.NA] * len(data), dtype='Int64')
    data[face_label_col] = pd.NA
    data['gaze_region'] = pd.Series('unmapped', index=data.index, dtype='string')
    data['selected_face_ids'] = ','.join(str(int(fid)) for fid in face_ids)
    # A fresh transformation invalidates prior manual decisions.
    data.drop(columns=['automatic_face_id', 'review_face_id', 'assignment_review_status',
                       'overlap_face_ids', 'overlap_section', 'automatic_transformed_x',
                       'automatic_transformed_y', 'automatic_face_label', 'automatic_gaze_region'], errors='ignore', inplace=True)

    # Face ownership must be settled before the full-frame peripheral meshes
    # are considered. The same ordering resolves facial overlaps and chooses
    # a display reference for off-face samples without assigning a face ID.
    for mapping_mode in ('facial', 'peripheral'):
        for fid in face_ids:
            remaining = data.loc[data[x_col].isna()]
            grouped = remaining.groupby(group_col, group_keys=False)
            tasks = [(group.copy(), stimuliFolder, landmarkFolder, name,
                      fid, x_col, y_col, face_id_col, face_label_col,
                      use_parallel_simplex, mapping_mode, template, contour_margin)
                     for name, group in grouped]
            if not tasks:
                continue
            if num_workers == 1:
                face_results = [func(*t) for t in tasks]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    face_results = list(executor.map(
                        func, *[col for col in zip(*tasks)]))
            face_combined = pd.concat(face_results)
            mapped = face_combined.index[face_combined[x_col].notna()]
            for col in (x_col, y_col, face_id_col, face_label_col, 'gaze_region'):
                data.loc[mapped, col] = face_combined.loc[mapped, col]

    if renamed_col:
        if type == 'video' and 'videoName' in data.columns:
            data = data.rename(columns={'videoName': 'stimulusName'})
        elif type == 'image' and 'STIMULI_NAME' in data.columns:
            data = data.rename(columns={'STIMULI_NAME': 'stimulusName'})

    return data
