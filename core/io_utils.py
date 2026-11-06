"""
core.io_utils — File I/O, Parquet conversion, and data cleaning.

Everything in this module is pure Python — no PyQt imports.
"""

import os
import time
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('iTemplate2')


# ---------------------------------------------------------------------------
# Resolve stimulus file paths
# ---------------------------------------------------------------------------

def resolve_file_in_folder(name, folder, suffix='', extensions=None):
    """
    Resolve a stimulus name (with or without extension) to a real file path.

    Parameters
    ----------
    name : str
        Stimulus name, e.g. 'T_song_teapot_bg' or 'T_song_teapot_bg.webm'
    folder : str
        Directory to search in
    suffix : str
        Suffix to append before extension, e.g. '_landmarks' or '_transformation'
    extensions : list or None
        Allowed extensions to try, e.g. ['.csv']. If None, tries common
        video/image extensions.

    Returns
    -------
    str or None
        Full path to the resolved file, or None if not found.
    """
    if extensions is None:
        extensions = ['.parquet', '.npy', '.webm', '.mp4', '.avi', '.mov', '.mkv',
                      '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.csv']

    base = os.path.splitext(name)[0]  # strip extension if present

    # 1. Try the name as-is (already has extension)
    if suffix:
        candidate = os.path.join(folder, base + suffix)
    else:
        candidate = os.path.join(folder, name)
    if os.path.isfile(candidate):
        return candidate

    # 2. Try each extension
    for ext in extensions:
        if suffix:
            candidate = os.path.join(folder, base + suffix + ext)
        else:
            candidate = os.path.join(folder, base + ext)
        if os.path.isfile(candidate):
            return candidate

    # 3. Glob fallback: match by prefix in the folder
    if os.path.isdir(folder):
        target_prefix = (base + suffix).lower()
        for f in os.listdir(folder):
            if f.lower().startswith(target_prefix):
                return os.path.join(folder, f)

    return None


# ---------------------------------------------------------------------------
# Atomic Parquet I/O
# ---------------------------------------------------------------------------

def atomic_save_parquet(df, filepath, compression='snappy', **kwargs):
    """
    Write a DataFrame to Parquet atomically.

    Writes to a temporary file first, then swaps it into place with
    ``os.replace``.  This prevents data corruption if the process
    crashes or disk space runs out mid-write.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    filepath : str
        Output path.
    compression : str
        Compression type (default ``'snappy'``).
    **kwargs
        Additional arguments passed to ``df.to_parquet``.
    """
    temp_path = filepath + '.tmp'
    try:
        df.to_parquet(temp_path, engine='pyarrow', compression=compression, **kwargs)
        os.replace(temp_path, filepath)
    except Exception:
        # Clean up the partial temp file on any failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# ---------------------------------------------------------------------------
# Stimulus column cleaning
# ---------------------------------------------------------------------------

def clean_stimulus_column(df, strip_path=False, strip_ext=False, col='stimulusName',
                          remove_prefix='', remove_suffix='', add_prefix='', add_suffix=''):
    """Clean stimulus name column by stripping paths, extensions, and adjusting prefixes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the stimulus name column. Modified in-place.
    strip_path : bool
        If True, strip directory path keeping only the filename.
    strip_ext : bool
        If True, remove the file extension.
    col : str
        Column name to clean (default ``'stimulusName'``).
    remove_prefix, remove_suffix : str
        Literal strings to remove from the start/end of each name.
    add_prefix, add_suffix : str
        Literal strings to prepend/append to each name.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, modified in-place.
    """
    if col not in df.columns:
        return df
    if strip_path:
        df[col] = df[col].astype(str).apply(lambda x: x.replace('\\', '/').rsplit('/', 1)[-1] if '/' in x or '\\' in x else x)
    if strip_ext:
        df[col] = df[col].astype(str).apply(lambda x: os.path.splitext(x)[0])
    if remove_prefix:
        df[col] = df[col].astype(str).apply(lambda x: x[len(remove_prefix):] if x.startswith(remove_prefix) else x)
    if remove_suffix:
        df[col] = df[col].astype(str).apply(lambda x: x[:-len(remove_suffix)] if x.endswith(remove_suffix) else x)
    if add_prefix:
        df[col] = add_prefix + df[col].astype(str)
    if add_suffix:
        df[col] = df[col].astype(str) + add_suffix
    return df


# ---------------------------------------------------------------------------
# File reading and cleaning
# ---------------------------------------------------------------------------

def read_and_clean_file(csv_path, rename_map=None, strip_stim_path=False, strip_stim_ext=False,
                        copy_map=None):
    """Read a CSV, TXT, or Parquet file and apply cleaning and column renaming.

    Parameters
    ----------
    csv_path : str
        Path to the input file (.csv, .txt, or .parquet).
    rename_map : dict or None
        ``{old_name: new_name}`` — renames columns, removing the original.
    strip_stim_path : bool
        If True, strip directory path from the ``stimulusName`` column.
    strip_stim_ext : bool
        If True, remove file extension from the ``stimulusName`` column.
    copy_map : dict or None
        ``{source_col: new_col}`` — copies column data to a new column, keeping the original.

    Returns
    -------
    pd.DataFrame or None
        Cleaned DataFrame, or None if the file could not be read.
    """
    if not os.path.exists(csv_path):
        logger.error("File not found: %s", csv_path)
        return None

    logger.info("Reading file: %s...", csv_path)
    start_time = time.time()
    try:
        if csv_path.lower().endswith('.txt'):
            sep = '\t'
        else:
            sep = ','

        if csv_path.lower().endswith('.parquet'):
             df = pd.read_parquet(csv_path)
        else:
             df = pd.read_csv(csv_path, sep=sep, low_memory=False)

    except Exception as e:
        logger.error("Error reading file: %s", e)
        return None

    read_time = time.time() - start_time
    logger.info("Read complete in %.2f seconds. Rows: %d", read_time, len(df))

    # ---- CLEANUP LOGIC ----
    # 0. Rename Columns if map provided
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 0b. Copy columns if copy_map provided (create new column, keep original)
    if copy_map:
        for src, dst in copy_map.items():
            if src in df.columns:
                df[dst] = df[src]
                logger.info("  Copied column '%s' → '%s' (original preserved)", src, dst)

    # 1. Replace common missing value indicators
    df.replace(['.', 'nan', 'NaN'], float('nan'), inplace=True)

    # 2. Convert specific columns to numeric
    numeric_cols = ['TIMESTAMP', 'x', 'y']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Generic numeric conversion
    for col in df.columns:
        if col not in numeric_cols and df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                if not converted.isna().all():
                     df[col] = converted
            except Exception:
                pass

    # 4. Convert TRIAL_INDEX to nullable integer (supports NaN)
    if 'TRIAL_INDEX' in df.columns:
        df['TRIAL_INDEX'] = pd.to_numeric(df['TRIAL_INDEX'], errors='coerce').astype('Int64')

    # 5. Clean stimulusName
    if strip_stim_path or strip_stim_ext:
        clean_stimulus_column(df, strip_path=strip_stim_path, strip_ext=strip_stim_ext)

    return df


# ---------------------------------------------------------------------------
# Parquet conversion utilities
# ---------------------------------------------------------------------------

def convert_csv_to_parquet(csv_path, parquet_path=None, compression='snappy', rename_map=None,
                           strip_stim_path=False, strip_stim_ext=False, copy_map=None):
    """Convert a single CSV, TXT, or Parquet file to Parquet format.

    Reads the file, applies column renaming, numeric conversion, and stimulus
    name cleaning, sorts by session and timestamp, and writes the result to a
    Parquet file via atomic save.

    Parameters
    ----------
    csv_path : str
        Path to the input file (.csv, .txt, or .parquet).
    parquet_path : str or None
        Output Parquet path. If None, defaults to replacing the input file's
        extension with '.parquet'.
    compression : str
        Parquet compression type (default ``'snappy'``).
    rename_map : dict or None
        ``{old_name: new_name}`` — renames columns before saving.
    strip_stim_path : bool
        If True, strip directory paths from the ``stimulusName`` column.
    strip_stim_ext : bool
        If True, remove the file extension from the ``stimulusName`` column.
    copy_map : dict or None
        ``{source_col: new_col}`` — copies column data to a new column while
        keeping the original.

    Returns
    -------
    None
        Saves to disk; returns early (no output) if the file cannot be read.
    """
    df = read_and_clean_file(csv_path, rename_map, strip_stim_path, strip_stim_ext,
                             copy_map=copy_map)
    if df is None:
        return

    # Sorting
    sort_cols = []
    if 'RECORDING_SESSION_LABEL' in df.columns:
        sort_cols.append('RECORDING_SESSION_LABEL')
    if 'TIMESTAMP' in df.columns:
        sort_cols.append('TIMESTAMP')
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)

    if parquet_path is None:
        parquet_path = os.path.splitext(csv_path)[0] + '.parquet'

    logger.info("Writing Parquet: %s...", parquet_path)
    try:
        atomic_save_parquet(df, parquet_path, compression=compression)
        logger.info("Saved to: %s", parquet_path)

        # Stats
        file_size = os.path.getsize(csv_path) / (1024 * 1024)
        pq_size = os.path.getsize(parquet_path) / (1024 * 1024)
        logger.info("Original: %.2f MB, Parquet: %.2f MB", file_size, pq_size)
    except Exception as e:
        logger.error("Error writing Parquet: %s", e)

def convert_multiple_to_parquet(file_list, output_path, compression='snappy', rename_map=None,
                                strip_stim_path=False, strip_stim_ext=False, copy_map=None):
    """Read multiple files, concatenate, and save as a single Parquet file.

    Each file is read and cleaned independently (column renaming, numeric
    conversion, stimulus name cleaning), then all DataFrames are concatenated
    into one, sorted by session and timestamp, and written via atomic save.

    Parameters
    ----------
    file_list : list of str
        Paths to the input files (.csv, .txt, or .parquet).
    output_path : str
        Output Parquet file path.
    compression : str
        Parquet compression type (default ``'snappy'``).
    rename_map : dict or None
        ``{old_name: new_name}`` — renames columns in each file before merging.
    strip_stim_path : bool
        If True, strip directory paths from the ``stimulusName`` column.
    strip_stim_ext : bool
        If True, remove the file extension from the ``stimulusName`` column.
    copy_map : dict or None
        ``{source_col: new_col}`` — copies column data to a new column while
        keeping the original.

    Returns
    -------
    None
        Saves to disk; returns early (no output) if no valid data is found.
    """
    all_dfs = []
    logger.info("Processing %d files for concatenation...", len(file_list))

    for f in file_list:
        df = read_and_clean_file(f, rename_map, strip_stim_path, strip_stim_ext,
                                copy_map=copy_map)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No valid data found to merge.")
        return

    logger.info("Concatenating data...")
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Sorting
    sort_cols = []
    if 'RECORDING_SESSION_LABEL' in full_df.columns:
        sort_cols.append('RECORDING_SESSION_LABEL')
    if 'TIMESTAMP' in full_df.columns:
        sort_cols.append('TIMESTAMP')
    if sort_cols:
        logger.info("Sorting by: %s", sort_cols)
        full_df.sort_values(by=sort_cols, inplace=True)

    logger.info("Writing concatenated Parquet: %s...", output_path)
    try:
        atomic_save_parquet(full_df, output_path, compression=compression)
        logger.info("Saved to: %s", output_path)
    except Exception as e:
        logger.error("Error writing Parquet: %s", e)

def clean_stimulus_in_parquet(parquet_path, strip_path=False, strip_ext=False,
                              remove_prefix='', remove_suffix='', add_prefix='', add_suffix=''):
    """Post-import tool: clean the ``stimulusName`` column in an existing Parquet file.

    Loads a Parquet file, applies path, extension, prefix, and suffix
    cleaning to the ``stimulusName`` column, and saves the result back
    in place using an atomic write.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file to modify in place.
    strip_path : bool
        If True, strip directory paths, keeping only the filename.
    strip_ext : bool
        If True, remove the file extension.
    remove_prefix, remove_suffix : str
        Literal strings to remove from the start/end of each name.
    add_prefix, add_suffix : str
        Literal strings to prepend/append to each name.

    Returns
    -------
    bool
        ``True`` if the file was cleaned and saved, ``False`` if the
        ``stimulusName`` column is absent.
    """
    df = pd.read_parquet(parquet_path)
    if 'stimulusName' not in df.columns:
        logger.warning("No stimulusName column found.")
        return False
    clean_stimulus_column(df, strip_path=strip_path, strip_ext=strip_ext,
                          remove_prefix=remove_prefix, remove_suffix=remove_suffix,
                          add_prefix=add_prefix, add_suffix=add_suffix)
    atomic_save_parquet(df, parquet_path)
    logger.info("Cleaned and saved: %s", parquet_path)
    return True

def remap_columns_in_parquet(parquet_path, rename_map, copy_map=None):
    """Post-import tool: rename and copy columns in an existing Parquet file.

    Loads a Parquet file, deduplicates any duplicate column names (keeping
    the first occurrence), applies column renaming and copying, and saves
    the result back in place using an atomic write.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file to modify in place.
    rename_map : dict
        ``{old_name: new_name}`` — renames columns, removing the original.
    copy_map : dict or None
        ``{source_col: new_col}`` — copies column data to a new column,
        keeping the original.

    Returns
    -------
    bool
        ``True`` on success.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)

    # Check for duplicate column names and deduplicate
    col_names = table.column_names
    seen = {}
    drop_indices = []
    for i, name in enumerate(col_names):
        if name in seen:
            drop_indices.append(i)
            logger.info("  Dropping duplicate column '%s' at index %d", name, i)
        else:
            seen[name] = i

    if drop_indices:
        # Remove duplicate columns (keep first occurrence)
        keep_indices = [i for i in range(len(col_names)) if i not in drop_indices]
        table = table.select(keep_indices)
        logger.info("  Removed %d duplicate column(s)", len(drop_indices))

    df = table.to_pandas()
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if copy_map:
        for src, dst in copy_map.items():
            if src in df.columns:
                df[dst] = df[src]
                logger.info("  Copied column '%s' → '%s' (original preserved)", src, dst)
    atomic_save_parquet(df, parquet_path)
    logger.info("Remapped columns and saved: %s", parquet_path)
    return True


def convert_timestamp_in_parquet(parquet_path, source_unit):
    """
    Post-import tool: converts the TIMESTAMP column from *source_unit* to
    milliseconds (ms) and saves the result back to the same parquet file.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file to modify in-place.
    source_unit : str
        One of:
            'ns'  – nanoseconds  (divide by 1e6)
            'us'  – microseconds (divide by 1e3)
            'ms'  – milliseconds (divide by 1)
            's'   – seconds      (multiply by 1000)
        Any other string is interpreted as a numeric divisor
        (e.g. '1000' means divide by 1000).

    Returns
    -------
    bool
        True on success, False if TIMESTAMP column is missing.

    Raises
    ------
    ValueError
        If *source_unit* is not recognised and cannot be parsed as a number.
    """
    # Map unit names to a multiplier that converts the raw value to cx
    _unit_factors = {
        'ns': 1e-6,   # 1 ns  * 1e-6  = 1e-6 cx  → divide raw by 1e6
        'us': 1e-3,   # 1 µs  * 1e-3  = 1e-3 cx
        'ms': 1.0,    # 1 ms  * 1   = 1  cx
        's':  1000.0,  # 1 s   * 1000   = 1000  cx
    }

    unit_key = source_unit.strip().lower()
    if unit_key in _unit_factors:
        factor = _unit_factors[unit_key]
    else:
        try:
            divisor = float(unit_key)
            if divisor == 0:
                raise ValueError("Divisor cannot be zero.")
            factor = 1.0 / divisor
        except ValueError:
            raise ValueError(
                f"Unknown source unit '{source_unit}'. "
                "Use 'ns', 'us', 'ms', 's', or a numeric divisor."
            )

    df = pd.read_parquet(parquet_path)
    if 'TIMESTAMP' not in df.columns:
        logger.warning("No TIMESTAMP column found in %s", parquet_path)
        return False

    df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='coerce') * factor
    inf_count = int(np.isinf(df['TIMESTAMP']).sum())
    if inf_count > 0:
        logger.warning(
            "Clamped %d infinite TIMESTAMP values to NaN in %s (source unit=%s, factor=%.2e)",
            inf_count, parquet_path, source_unit, factor
        )
        df.loc[np.isinf(df['TIMESTAMP']), 'TIMESTAMP'] = np.nan
    atomic_save_parquet(df, parquet_path)
    logger.info(
        "Converted TIMESTAMP from '%s' to cx (factor=%.2e) and saved: %s",
        source_unit, factor, parquet_path
    )
    return True
