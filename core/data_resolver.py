"""Helpers for resolving step-default dataset paths within a project."""

from __future__ import annotations

import os
from typing import Optional


def _is_within_dir(path: str, directory: str) -> bool:
    if not path or not directory:
        return False
    try:
        path_abs = os.path.abspath(path)
        dir_abs = os.path.abspath(directory)
        return os.path.commonpath([path_abs, dir_abs]) == dir_abs
    except ValueError:
        return False


def is_in_project_subdir(project_path: str, path: str, subdir: str) -> bool:
    """Return True when *path* lives inside *project_path/subdir*."""
    if not project_path or not path:
        return False
    return _is_within_dir(path, os.path.join(project_path, subdir))


def is_imported_data_file(project_path: str, path: str) -> bool:
    return is_in_project_subdir(project_path, path, "imported_parquet")


def is_transformed_data_file(project_path: str, path: str) -> bool:
    return is_in_project_subdir(project_path, path, "transformed_data")


def resolve_latest_imported_parquet(project_path: str) -> Optional[str]:
    """Return the newest parquet file in imported_parquet/, if any."""
    if not project_path:
        return None
    imported_dir = os.path.join(project_path, "imported_parquet")
    if not os.path.isdir(imported_dir):
        return None
    parquet_files = [
        os.path.join(imported_dir, name)
        for name in os.listdir(imported_dir)
        if name.lower().endswith(".parquet")
    ]
    if not parquet_files:
        return None
    return max(parquet_files, key=os.path.getmtime)


def dataset_prefix_from_source(source_path: str) -> str:
    """Return the canonical dataset prefix used to derive step outputs."""
    base = os.path.splitext(os.path.basename(source_path or ""))[0]
    if base.endswith("_transformed"):
        base = base[:-len("_transformed")]
    return base


def build_expected_transformed_path(project_path: str, source_path: str) -> str:
    """Build the exact transformed-data path expected for a source dataset."""
    prefix = dataset_prefix_from_source(source_path)
    if not project_path or not prefix:
        return ""
    return os.path.join(project_path, "transformed_data", f"{prefix}_transformed.parquet")


def resolve_existing_transformed_file(project_path: str, source_path: str) -> Optional[str]:
    """Resolve the transformed file for a source dataset.

    Prefers the exact expected path. If that file is missing, falls back to a
    single unambiguous transformed candidate with the same prefix.
    """
    expected = build_expected_transformed_path(project_path, source_path)
    if expected and os.path.isfile(expected):
        return expected

    transformed_dir = os.path.join(project_path, "transformed_data")
    prefix = dataset_prefix_from_source(source_path)
    if not os.path.isdir(transformed_dir) or not prefix:
        return None

    candidates = []
    prefix_lower = prefix.lower()
    for name in os.listdir(transformed_dir):
        if not name.lower().endswith((".parquet", ".csv")):
            continue
        stem = os.path.splitext(name)[0].lower()
        if stem == f"{prefix_lower}_transformed":
            candidates.append(os.path.join(transformed_dir, name))
            continue
        if (stem.startswith(prefix_lower + "__") or stem.startswith(prefix_lower + "_transformed_")) and "_transformed" in stem:
            candidates.append(os.path.join(transformed_dir, name))

    if len(candidates) == 1:
        return candidates[0]
    return None
