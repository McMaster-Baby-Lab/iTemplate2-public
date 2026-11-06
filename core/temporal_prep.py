"""
core.temporal_prep — Trial-level data manipulation (onset/offset marking,
sorting, keyword filtering).

Extracted from TemporalPreprocessing widget @staticmethods.
Everything in this module is pure Python — no PyQt imports.
"""

import logging

import numpy as np
import pandas as pd

from core.io_utils import atomic_save_parquet

logger = logging.getLogger('iTemplate2')


# ---------------------------------------------------------------------------
# Keyword-based STIM_ONSET tagging
# ---------------------------------------------------------------------------

def apply_keyword_filter(gaze_path,
                         inc_col, include_kws,
                         exc_col, exclude_kws,
                         progress_callback=None):
    """Tag the first row of contiguous keyword-matching blocks as STIM_ONSET.

    Places a STIM_ONSET tag on the first row of each contiguous block where
    the include and exclude criteria are satisfied. A new STIM_ONSET is also
    placed whenever the stimulus identifier (stimulusName or videoName) changes
    within a matched region.

    Filtering logic (include and exclude may reference different columns):
    - Include: row is kept if the specified column value IS IN include_kws.
    - Exclude: row is kept if the specified column value IS NOT IN exclude_kws.
    - Both criteria are AND-combined when both are active.
    - The special keyword '<empty>' matches NaN or blank cells.

    Parameters
    ----------
    gaze_path : str
        Path to the Parquet file containing gaze data. Read in place and
        saved back after modification.
    inc_col : str or None
        Column name to check against ``include_kws``. If empty or None,
        the include criterion is skipped.
    include_kws : list of str or None
        Keywords that satisfy the include criterion. The special value
        ``'<empty>'`` matches NaN / blank cells.
    exc_col : str or None
        Column name to check against ``exclude_kws``. If empty or None,
        the exclude criterion is skipped.
    exclude_kws : list of str or None
        Keywords that cause a row to be excluded. Skipped if empty or None.
    progress_callback : callable, optional
        Optional callable for GUI progress updates. Receives an int 0-100.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``n_blocks`` (int): number of STIM_ONSET tags placed.
        - ``n_matching_rows`` (int): total rows that matched the filter.
        - ``inc_col``, ``include_kws``, ``exc_col``, ``exclude_kws``:
          echoing the filter parameters back to the caller.

    Raises
    ------
    ValueError
        If ``inc_col`` or ``exc_col`` is specified but not found in the data.
    """
    df = pd.read_parquet(gaze_path)
    if progress_callback:
        progress_callback(20)

    if 'trial_event' not in df.columns:
        df['trial_event'] = ""

    # --- helper to build a boolean mask, handling <empty> ---
    def _build_mask(dataframe, col, keywords, negate=False):
        if not col or not keywords:
            return None
        if col not in dataframe.columns:
            raise ValueError(f"Column '{col}' not found in data.")
        col_str = dataframe[col].astype(str)
        is_empty = dataframe[col].isna() | (col_str.str.strip() == '')

        normal_kws = [k for k in keywords if k != '<empty>']
        want_empty = '<empty>' in keywords

        if normal_kws and want_empty:
            mask = col_str.isin(normal_kws) | is_empty
        elif normal_kws:
            mask = col_str.isin(normal_kws)
        else:  # only <empty>
            mask = is_empty

        return ~mask if negate else mask

    # Build combined mask
    inc_mask = _build_mask(df, inc_col, include_kws, negate=False)
    exc_mask = _build_mask(df, exc_col, exclude_kws, negate=True)  # True = NOT in

    if inc_mask is not None and exc_mask is not None:
        match_mask = inc_mask & exc_mask
    elif inc_mask is not None:
        match_mask = inc_mask
    else:
        match_mask = exc_mask

    # Detect block starts.  A new STIM_ONSET is placed when:
    #   1. current row matches but previous row didn't, OR
    #   2. current row matches and the stimulusName/videoName changed.
    shifted = match_mask.shift(1, fill_value=False)
    block_starts = match_mask & ~shifted

    # Also break on stimulusName / videoName changes within matched rows
    stim_col = None
    for candidate in ['stimulusName', 'videoName']:
        if candidate in df.columns:
            stim_col = candidate
            break
    if stim_col is not None:
        stim_changed = df[stim_col] != df[stim_col].shift(1)
        block_starts = block_starts | (match_mask & stim_changed)

    n_blocks = int(block_starts.sum())
    df.loc[block_starts, 'trial_event'] = 'STIM_ONSET'

    if progress_callback:
        progress_callback(80)
    atomic_save_parquet(df, gaze_path)
    if progress_callback:
        progress_callback(100)
    return {
        'n_blocks': n_blocks,
        'n_matching_rows': int(match_mask.sum()),
        'inc_col': inc_col,
        'include_kws': include_kws,
        'exc_col': exc_col,
        'exclude_kws': exclude_kws,
    }


# ---------------------------------------------------------------------------
# Temporal sorting
# ---------------------------------------------------------------------------

def sort_data(gaze_path, primary, secondary, progress_callback=None):
    """Sort gaze data by Participant_ID and temporal columns, then save.

    Reads the Parquet file, coerces the primary and secondary temporal
    columns to numeric, sorts by Participant_ID (if present) followed by
    the temporal columns, and writes back to the same path.

    Parameters
    ----------
    gaze_path : str
        Path to the Parquet file containing gaze data. Read in place and
        saved back after sorting.
    primary : str
        Primary temporal column name (e.g. ``'TIMESTAMP'``). Must exist in
        the data to be used as a sort key.
    secondary : str
        Secondary temporal column name (e.g. ``'sample_number'``). Prefix
        with ``"--"`` to indicate no secondary column. Must exist in the
        data to be used as a sort key.
    progress_callback : callable, optional
        Optional callable for GUI progress updates. Receives an int 0-100.

    Returns
    -------
    dict
        Dictionary with key:
        - ``sort_cols`` (list of str): column names used for sorting.

    Raises
    ------
    ValueError
        If neither the primary nor the secondary column (nor Participant_ID)
        is found in the data, leaving no valid sort columns.
    """
    df = pd.read_parquet(gaze_path)
    if progress_callback:
        progress_callback(20)

    sort_cols = []
    if 'Participant_ID' in df.columns:
        sort_cols.append('Participant_ID')

    if primary in df.columns:
        df[primary] = pd.to_numeric(df[primary], errors='coerce')
        sort_cols.append(primary)

    if not secondary.startswith("--") and secondary in df.columns:
        df[secondary] = pd.to_numeric(df[secondary], errors='coerce')
        sort_cols.append(secondary)

    if not sort_cols:
        raise ValueError("No valid sort columns found.")

    if not secondary.startswith("--") and secondary in df.columns:
        max_secondary = df[secondary].max() if df[secondary].max() > 0 else 1
        multiplier = 10 ** (len(str(int(max_secondary))) + 1)
        df['_temporal_order'] = df[primary] * multiplier + df[secondary]
    else:
        df['_temporal_order'] = df[primary]

    df = df.sort_values(by=sort_cols)
    if progress_callback:
        progress_callback(80)
    atomic_save_parquet(df, gaze_path)
    if progress_callback:
        progress_callback(100)
    return {'sort_cols': sort_cols}


# ---------------------------------------------------------------------------
# Trial onset / offset marker application
# ---------------------------------------------------------------------------

def apply_markers(gaze_path, onset_col, onset_markers, offset_config,
                  onset_offset_ms=0.0, progress_callback=None):
    """Mark STIM_ONSET and STIM_OFFSET rows based on column values or durations.

    Scans the specified onset column for marker values and tags matching
    rows as STIM_ONSET. Optionally shifts the onset forward by
    ``onset_offset_ms`` milliseconds using binary search on the TIMESTAMP
    column. For offsets, supports two modes:

    - **Column mode**: tag rows where an offset column matches given markers.
    - **Duration mode**: place STIM_OFFSET a fixed number of milliseconds
      after each STIM_ONSET using a temporal column.

    When Participant_ID is present, searches are scoped per participant
    to avoid cross-participant bleed.

    Parameters
    ----------
    gaze_path : str
        Path to the Parquet file containing gaze data. Read in place and
        saved back after modification.
    onset_col : str
        Column name to scan for onset marker values.
    onset_markers : list of str
        String values in ``onset_col`` that indicate a trial onset.
    offset_config : dict
        Configuration for offset marking. Two modes:
        - ``{'mode': 'column', 'col': str, 'markers': list of str}``
        - ``{'mode': 'duration', 'temporal_col': str, 'duration_ms': float}``
    onset_offset_ms : float, default 0.0
        Additional milliseconds to shift the STIM_ONSET forward from the
        raw marker row. Requires a TIMESTAMP column in the data.
    progress_callback : callable, optional
        Optional callable for GUI progress updates. Receives an int 0-100.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``n_onset`` (int): number of STIM_ONSET tags in the saved file.
        - ``n_offset`` (int): number of STIM_OFFSET tags in the saved file.

    Raises
    ------
    ValueError
        If ``onset_offset_ms > 0`` but no TIMESTAMP column exists, or if
        the duration mode temporal column is missing.
    """
    df = pd.read_parquet(gaze_path)
    if progress_callback:
        progress_callback(10)

    if 'trial_event' not in df.columns:
        df['trial_event'] = ""

    # Mark raw onset rows first
    raw_onset_mask = df[onset_col].astype(str).isin(onset_markers)

    if onset_offset_ms > 0:
        # Shift onset forward by onset_offset_ms using TIMESTAMP
        ts_col = None
        for c in df.columns:
            if c.lower() == 'timestamp':
                ts_col = c
                break
        if ts_col is None:
            raise ValueError(
                "Onset offset requires a TIMESTAMP column in your data."
            )
        df[ts_col] = pd.to_numeric(df[ts_col], errors='coerce')

        has_pid = 'Participant_ID' in df.columns
        onset_indices = df.index[raw_onset_mask].tolist()

        if has_pid:
            # Group by participant once, then use searchsorted per onset
            for pid, pid_group in df.groupby('Participant_ID'):
                ts_values = pid_group[ts_col].values
                pid_indices = pid_group.index.values
                # Get onsets belonging to this participant
                pid_onsets = [oi for oi in onset_indices
                             if df.loc[oi, 'Participant_ID'] == pid]
                for oi in pid_onsets:
                    onset_ts = df.loc[oi, ts_col]
                    if pd.isna(onset_ts):
                        continue
                    target_ts = onset_ts + onset_offset_ms
                    # Find first row in this participant where ts >= target and index >= oi
                    pos = np.searchsorted(ts_values, target_ts, side='left')
                    while pos < len(pid_indices) and pid_indices[pos] < oi:
                        pos += 1
                    if pos < len(pid_indices):
                        df.loc[pid_indices[pos], 'trial_event'] = 'STIM_ONSET'
        else:
            ts_values = df[ts_col].values
            df_indices = df.index.values
            for oi in onset_indices:
                onset_ts = df.loc[oi, ts_col]
                if pd.isna(onset_ts):
                    continue
                target_ts = onset_ts + onset_offset_ms
                pos = np.searchsorted(ts_values, target_ts, side='left')
                while pos < len(df_indices) and df_indices[pos] < oi:
                    pos += 1
                if pos < len(df_indices):
                    df.loc[df_indices[pos], 'trial_event'] = 'STIM_ONSET'
    else:
        df.loc[raw_onset_mask, 'trial_event'] = 'STIM_ONSET'

    if progress_callback:
        progress_callback(50)

    if offset_config['mode'] == 'duration':
        temporal_col = offset_config['temporal_col']
        duration_ms = offset_config['duration_ms']

        if temporal_col not in df.columns:
            raise ValueError(f"Temporal column '{temporal_col}' not found in data.")

        df[temporal_col] = pd.to_numeric(df[temporal_col], errors='coerce')
        onset_indices = df.index[df['trial_event'] == 'STIM_ONSET'].tolist()

        has_pid = 'Participant_ID' in df.columns

        if has_pid:
            # Group by participant once, use searchsorted
            for pid, pid_group in df.groupby('Participant_ID'):
                ts_values = pid_group[temporal_col].values
                pid_indices = pid_group.index.values
                pid_onsets = [oi for oi in onset_indices
                             if df.loc[oi, 'Participant_ID'] == pid]
                for onset_idx in pid_onsets:
                    onset_ts = df.loc[onset_idx, temporal_col]
                    if pd.isna(onset_ts):
                        continue
                    target_ts = onset_ts + duration_ms
                    pos = np.searchsorted(ts_values, target_ts, side='left')
                    while pos < len(pid_indices) and pid_indices[pos] <= onset_idx:
                        pos += 1
                    if pos < len(pid_indices):
                        df.loc[pid_indices[pos], 'trial_event'] = 'STIM_OFFSET'
        else:
            ts_values = df[temporal_col].values
            df_indices = df.index.values
            for onset_idx in onset_indices:
                onset_ts = df.loc[onset_idx, temporal_col]
                if pd.isna(onset_ts):
                    continue
                target_ts = onset_ts + duration_ms
                pos = np.searchsorted(ts_values, target_ts, side='left')
                while pos < len(df_indices) and df_indices[pos] <= onset_idx:
                    pos += 1
                if pos < len(df_indices):
                    df.loc[df_indices[pos], 'trial_event'] = 'STIM_OFFSET'
    else:
        offset_col = offset_config['col']
        offset_markers = offset_config['markers']
        df.loc[df[offset_col].astype(str).isin(offset_markers), 'trial_event'] = 'STIM_OFFSET'

    if progress_callback:
        progress_callback(90)
    atomic_save_parquet(df, gaze_path)
    if progress_callback:
        progress_callback(100)

    n_onset = int((df['trial_event'] == 'STIM_ONSET').sum())
    n_offset = int((df['trial_event'] == 'STIM_OFFSET').sum())
    return {'n_onset': n_onset, 'n_offset': n_offset}
