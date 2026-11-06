"""
core.spatial_prep — Coordinate normalization (screen-based and wearable)
and frame-index assignment.

Extracted from CoordinateSetup widget @staticmethods.
Everything in this module is pure Python — no PyQt imports.
"""

import os
import logging

import numpy as np
import pandas as pd

from core.io_utils import atomic_save_parquet

logger = logging.getLogger('iTemplate2')


# ---------------------------------------------------------------------------
# Screen-based normalization
# ---------------------------------------------------------------------------

def normalize_screen(gaze_path, stim_dimensions, center_x, center_y,
                     progress_callback=None):
    """Normalize gaze coordinates to a 0-1 range relative to stimulus dimensions.

    For each unique stimulus in the data, locates its width and height in
    ``stim_dimensions`` (trying the full name, the name without extension,
    and the basename without extension), then centers the stimulus on
    ``(center_x, center_y)`` and computes normalized coordinates:

    ``stimuli_x = (x - left) / width``
    ``stimuli_y = (y - top) / height``

    Rows whose stimulus is not found in ``stim_dimensions`` are left as NaN.

    Parameters
    ----------
    gaze_path : str
        Path to the Parquet file containing gaze data. Must have columns
        ``x``, ``y``, and ``stimulusName``.
    stim_dimensions : dict
        Mapping from stimulus name to a dict with keys ``width`` and
        ``height`` (and optionally ``fps``, ``frames``).
    center_x : float
        Screen center X coordinate; stimuli are assumed centered here.
    center_y : float
        Screen center Y coordinate; stimuli are assumed centered here.
    progress_callback : callable, optional
        Optional callable for GUI progress updates. Receives an int 0-100.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``matched`` (int): number of unique stimuli with valid dimensions.
        - ``total`` (int): total number of unique stimuli.
        - ``unmatched`` (list of str): stimulus names that had no matching
          entry in ``stim_dimensions``.

    Raises
    ------
    ValueError
        If the data file is missing required columns (``x``, ``y``, or
        ``stimulusName``).
    """
    df = pd.read_parquet(gaze_path)

    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("Data file must have 'x' and 'y' columns.")
    if 'stimulusName' not in df.columns:
        raise ValueError("Data file must have a 'stimulusName' column.")

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['stimuli_x'] = np.nan
    df['stimuli_y'] = np.nan

    unique_stims = df['stimulusName'].dropna().unique()
    n_stims = len(unique_stims)
    matched = 0
    unmatched = []

    for i, stim_name in enumerate(unique_stims):
        if progress_callback:
            progress_callback(int((i / max(n_stims, 1)) * 90))
        stim_name_str = str(stim_name)
        dims = None
        if stim_name_str in stim_dimensions:
            dims = stim_dimensions[stim_name_str]
        else:
            base = os.path.splitext(stim_name_str)[0]
            if base in stim_dimensions:
                dims = stim_dimensions[base]
            else:
                stripped = stim_name_str.replace('\\', '/').rsplit('/', 1)[-1]
                stripped_base = os.path.splitext(stripped)[0]
                if stripped_base in stim_dimensions:
                    dims = stim_dimensions[stripped_base]

        if dims is None:
            unmatched.append(stim_name_str)
            continue

        stim_w = dims['width']
        stim_h = dims['height']
        if stim_w == 0 or stim_h == 0:
            unmatched.append(stim_name_str)
            continue

        left = center_x - (stim_w / 2)
        top = center_y - (stim_h / 2)

        mask = df['stimulusName'] == stim_name
        df.loc[mask, 'stimuli_x'] = np.round(
            ((df.loc[mask, 'x'] - left) / stim_w).astype(np.float32), 4
        )
        df.loc[mask, 'stimuli_y'] = np.round(
            ((df.loc[mask, 'y'] - top) / stim_h).astype(np.float32), 4
        )
        matched += 1

    if progress_callback:
        progress_callback(95)
    atomic_save_parquet(df, gaze_path)
    if progress_callback:
        progress_callback(100)
    return {'matched': matched, 'total': len(unique_stims), 'unmatched': unmatched}


# ---------------------------------------------------------------------------
# Wearable normalization
# ---------------------------------------------------------------------------

def normalize_wearable(gaze_path, stim_dimensions, scene_col, scene_keyword,
                       progress_callback=None):
    """Normalize wearable eye-tracking gaze coordinates by video dimensions.

    Divides raw gaze coordinates by the video width and height to produce
    a 0-1 normalized space. Matches each stimulusName to an entry in
    ``stim_dimensions`` by full name or name without extension. Optionally
    filters to rows where a scene column matches a scene keyword before
    processing.

    Normalized coordinates:

    ``stimuli_x = x / video_width``
    ``stimuli_y = y / video_height``

    Parameters
    ----------
    gaze_path : str
        Path to the Parquet file containing gaze data. Must have columns
        ``x``, ``y``, and ``stimulusName``.
    stim_dimensions : dict
        Mapping from stimulus/video filename to a dict with keys ``width``
        and ``height`` (and optionally ``fps``, ``frames``).
    scene_col : str or None
        Column name to filter on. Only rows where this column equals
        ``scene_keyword`` are processed. Skipped if empty or not in data.
    scene_keyword : str or None
        Value to match against ``scene_col``. Skipped if empty.
    progress_callback : callable, optional
        Optional callable for GUI progress updates. Receives an int 0-100.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``matched`` (int): number of unique stimuli with valid dimensions.
        - ``total`` (int): total number of unique stimuli (after scene filter).
        - ``unmatched`` (list of str): stimulus names that had no matching
          entry in ``stim_dimensions``.

    Raises
    ------
    ValueError
        If the data file is missing required columns (``x``, ``y``, or
        ``stimulusName``).
    """
    df = pd.read_parquet(gaze_path)

    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("Data file must have 'x' and 'y' columns.")
    if 'stimulusName' not in df.columns:
        raise ValueError("Wearable mode requires a 'stimulusName' column.")

    # Optional: filter to rows matching the scene keyword
    if scene_col and scene_keyword and scene_col in df.columns:
        scene_mask = df[scene_col].astype(str) == scene_keyword
    else:
        scene_mask = pd.Series(True, index=df.index)

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['stimuli_x'] = np.nan
    df['stimuli_y'] = np.nan

    unique_stims = df.loc[scene_mask, 'stimulusName'].dropna().unique()
    n_stims = len(unique_stims)
    matched = 0
    unmatched = []

    for i, stim_name in enumerate(unique_stims):
        if progress_callback:
            progress_callback(int((i / max(n_stims, 1)) * 90))
        stim_str = str(stim_name)
        # Match stimulusName to video filename
        dims = None
        if stim_str in stim_dimensions:
            dims = stim_dimensions[stim_str]
        else:
            base = os.path.splitext(stim_str)[0]
            if base in stim_dimensions:
                dims = stim_dimensions[base]

        if dims is None:
            unmatched.append(stim_str)
            continue

        vid_w = dims['width']
        vid_h = dims['height']
        if vid_w == 0 or vid_h == 0:
            unmatched.append(stim_str)
            continue

        mask = (df['stimulusName'] == stim_name) & scene_mask
        df.loc[mask, 'stimuli_x'] = np.round(
            (df.loc[mask, 'x'] / vid_w).astype(np.float32), 4
        )
        df.loc[mask, 'stimuli_y'] = np.round(
            (df.loc[mask, 'y'] / vid_h).astype(np.float32), 4
        )
        matched += 1

    if progress_callback:
        progress_callback(95)
    atomic_save_parquet(df, gaze_path)
    if progress_callback:
        progress_callback(100)
    return {'matched': matched, 'total': len(unique_stims), 'unmatched': unmatched}


# ---------------------------------------------------------------------------
# Frame index assignment
# ---------------------------------------------------------------------------

def assign_frames(gaze_path, frame_rate, delay, stim_dimensions,
                  wearable=False, scene_col=None, scene_keywords=None,
                  progress_callback=None):
    """
    Assign _video_frame_index so that frame 1 starts at each STIM_ONSET row.

    Screen-based mode:
      Groups by Participant_ID × stimulusName.  Within each group, every
      STIM_ONSET row resets the counter so that the onset row = frame 1.
      If a stimulus appears multiple times for a participant (multiple
      trials), each trial gets its own frame sequence.

    Wearable mode:
      Groups by Participant_ID only.  Frame 1 starts at the first
      TIMESTAMP of each participant.
    """
    df = pd.read_parquet(gaze_path)
    logger.debug("assign_frames File: %s", gaze_path)
    logger.debug("assign_frames Columns: %s", list(df.columns))
    logger.debug("assign_frames wearable=%s, frame_rate=%s, delay=%s", wearable, frame_rate, delay)
    logger.debug("assign_frames 'trial_event' in columns: %s", 'trial_event' in df.columns)
    if 'trial_event' in df.columns:
        vc = df['trial_event'].value_counts(dropna=False)
        logger.debug("assign_frames trial_event value_counts:\n%s", vc.head(10))

    if 'TIMESTAMP' not in df.columns:
        raise ValueError("Frame index assignment requires a TIMESTAMP column.")

    df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='coerce')

    delay_ms = delay
    df['_video_frame_index'] = -1

    # ── Wearable mode ──
    if wearable:
        if 'Participant_ID' not in df.columns:
            raise ValueError("Wearable mode requires a 'Participant_ID' column.")

        # Optionally filter to scene-video rows
        if scene_col and scene_keywords and scene_col in df.columns:
            scene_mask = df[scene_col].astype(str).isin(scene_keywords)
        else:
            scene_mask = pd.Series(True, index=df.index)

        work_df = df.loc[scene_mask]
        if work_df.empty:
            raise ValueError("No rows match the scene filter — check column and keywords.")

        # Build group columns: Participant_ID, and stimulusName if present
        group_cols = ['Participant_ID']
        if 'stimulusName' in df.columns:
            group_cols.append('stimulusName')

        has_trial_event = 'trial_event' in df.columns

        groups = work_df.groupby(group_cols, group_keys=False).groups
        n_groups = len(groups)

        for gi, (name, grp_idx) in enumerate(groups.items()):
            if progress_callback:
                progress_callback(int((gi / max(n_groups, 1)) * 90))
            group = df.loc[grp_idx]

            # Resolve lookup key for stim_dimensions
            lookup_key = str(name) if not isinstance(name, tuple) else str(name[0])
            dims = stim_dimensions.get(lookup_key) or stim_dimensions.get(
                os.path.splitext(lookup_key)[0]
            )
            current_fps = frame_rate
            max_f = float('inf')
            if dims:
                if dims.get('fps', 0) > 0:
                    current_fps = dims['fps']
                if dims.get('frames', 0) > 0:
                    max_f = dims['frames']

            ms_per_frame = 1000.0 / current_fps

            # Check for STIM_ONSET
            if has_trial_event:
                onset_mask = group['trial_event'] == 'STIM_ONSET'
                onset_positions = group.index[onset_mask]
            else:
                onset_positions = pd.Index([])

            if len(onset_positions) == 0:
                logger.debug("Wearable group '%s': NO STIM_ONSET, using TIMESTAMP.min()", name)
                start_ts = group['TIMESTAMP'].min()
                raw = ((group['TIMESTAMP'] - start_ts - delay_ms) / ms_per_frame).astype(int) + 1
                valid = (raw >= 1) & (
                    raw <= max_f if max_f != float('inf') else True
                )
                raw = raw.where(valid, -1)
                df.loc[grp_idx, '_video_frame_index'] = raw
            else:
                logger.debug("Wearable group '%s': %d STIM_ONSET(s) at %s", name, len(onset_positions), onset_positions.tolist())
                group_positions = group.index.tolist()
                for t, onset_idx in enumerate(onset_positions):
                    onset_pos = group_positions.index(onset_idx)
                    if t + 1 < len(onset_positions):
                        next_onset_idx = onset_positions[t + 1]
                        end_pos = group_positions.index(next_onset_idx)
                    else:
                        end_pos = len(group_positions)
                    trial_indices = group_positions[onset_pos:end_pos]
                    trial_rows = df.loc[trial_indices]
                    start_ts = df.loc[onset_idx, 'TIMESTAMP']
                    raw = ((trial_rows['TIMESTAMP'] - start_ts - delay_ms) / ms_per_frame).astype(int) + 1
                    valid = (raw >= 1) & (
                        raw <= max_f if max_f != float('inf') else True
                    )
                    raw = raw.where(valid, -1)
                    df.loc[trial_indices, '_video_frame_index'] = raw
    else:
        if 'stimulusName' not in df.columns:
            raise ValueError("Screen-based mode requires a 'stimulusName' column.")

        group_cols = []
        if 'Participant_ID' in df.columns:
            group_cols.append('Participant_ID')
        group_cols.append('stimulusName')

        has_trial_event = 'trial_event' in df.columns

        groups = df.groupby(group_cols, group_keys=False).groups
        n_groups = len(groups)

        for gi, (name, grp_idx) in enumerate(groups.items()):
            if progress_callback:
                progress_callback(int((gi / max(n_groups, 1)) * 90))
            group = df.loc[grp_idx]

            # Resolve stimulus key for dimensions / fps lookup
            if isinstance(name, tuple):
                stim_name = str(name[group_cols.index('stimulusName')])
            else:
                stim_name = str(name)

            dims = stim_dimensions.get(stim_name) or stim_dimensions.get(
                os.path.splitext(stim_name)[0]
            )
            current_fps = frame_rate
            max_f = float('inf')
            if dims:
                if dims.get('fps', 0) > 0:
                    current_fps = dims['fps']
                if dims.get('frames', 0) > 0:
                    max_f = dims['frames']

            ms_per_frame = 1000.0 / current_fps

            # Find STIM_ONSET rows within this group
            if has_trial_event:
                onset_mask = group['trial_event'] == 'STIM_ONSET'
                onset_positions = group.index[onset_mask]
            else:
                onset_positions = pd.Index([])

            if len(onset_positions) == 0:
                # No onset markers — use first timestamp of the group
                logger.debug("Group '%s': NO STIM_ONSET found, using TIMESTAMP.min()", name)
                start_ts = group['TIMESTAMP'].min()
                raw = ((group['TIMESTAMP'] - start_ts - delay_ms) / ms_per_frame).astype(int) + 1
                valid = (raw >= 1) & (
                    raw <= max_f if max_f != float('inf') else True
                )
                raw = raw.where(valid, -1)
                df.loc[grp_idx, '_video_frame_index'] = raw
            else:
                # Process each trial (STIM_ONSET → next STIM_ONSET or end)
                logger.debug("Group '%s': %d STIM_ONSET(s) found at indices %s", name, len(onset_positions), onset_positions.tolist())
                group_positions = group.index.tolist()
                for t, onset_idx in enumerate(onset_positions):
                    onset_pos = group_positions.index(onset_idx)
                    if t + 1 < len(onset_positions):
                        next_onset_idx = onset_positions[t + 1]
                        end_pos = group_positions.index(next_onset_idx)
                    else:
                        end_pos = len(group_positions)
                    trial_indices = group_positions[onset_pos:end_pos]
                    trial_rows = df.loc[trial_indices]
                    start_ts = df.loc[onset_idx, 'TIMESTAMP']
                    raw = ((trial_rows['TIMESTAMP'] - start_ts - delay_ms) / ms_per_frame).astype(int) + 1
                    valid = (raw >= 1) & (
                        raw <= max_f if max_f != float('inf') else True
                    )
                    raw = raw.where(valid, -1)
                    df.loc[trial_indices, '_video_frame_index'] = raw

    atomic_save_parquet(df, gaze_path)

    n_valid = int((df['_video_frame_index'] >= 0).sum())
    max_frame = int(df['_video_frame_index'].max())
    return {'n_valid': n_valid, 'max_frame': max_frame, 'fps': frame_rate}
