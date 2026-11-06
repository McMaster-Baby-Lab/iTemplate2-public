# Architecture

iTemplate2 follows a **Model–View separation**: all pure-Python data logic lives in the `core/` package (no PyQt imports), while the GUI code in `gui/` only handles user interaction. This makes the core logic independently testable and scriptable.

```
iTemplate2-public/
├── core/                        # Pure-Python data logic (no PyQt)
│   ├── __init__.py              # Re-exports all public APIs
│   ├── config.py                # MediaPipe init, average face data, constants
│   ├── io_utils.py              # File I/O, Parquet conversion, column cleaning
│   ├── transformation.py        # Numba simplex, affine transforms, batch processing
│   ├── temporal_prep.py         # Keyword filter, sort, onset/offset markers
│   ├── spatial_prep.py          # Screen/wearable normalization, frame assignment
│   ├── data_resolver.py         # Step-default dataset path resolution within a project
│   ├── template_manager.py      # Project-scoped templates and dataset provenance
│   ├── template_detection.py    # Isolated MediaPipe detection for custom template faces
│   └── overlap_review.py        # Overlapping-face detection and auditable review decisions
├── gui/                         # PyQt6 GUI layer
│   ├── main.py                  # Main window, sidebar, page routing
│   └── widgets/                 # One widget per pipeline step
├── logging_config.py            # Centralized logging setup
└── start_gui.py                 # Entry point
```

### Key Modules

| Module | Purpose |
|---|---|
| `core.config` | Initializes the MediaPipe FaceLandmarker, loads the average face template and Delaunay triangulation, and exposes constants (`width`, `height`). |
| `core.io_utils` | Parquet/CSV file I/O (`atomic_save_parquet`, `read_and_clean_file`), data conversion (`convert_csv_to_parquet`, `convert_multiple_to_parquet`), and column cleaning utilities. |
| `core.transformation` | Face landmark extraction (`process_video`, `process_image`), gap-aware frame index remapping (`find_missing_and_replace` with `max_gap`), manual gap interpolation (`glue_landmark_gap`), Numba-accelerated Delaunay point-in-triangle search, and parallel coordinate transformation (`transform_coordinates_parallel`). |
| `core.temporal_prep` | Trial segmentation functions: `apply_keyword_filter`, `sort_data`, and `apply_markers` for STIM_ONSET/OFFSET tagging. |
| `core.spatial_prep` | Coordinate normalization (`normalize_screen`, `normalize_wearable`) and video frame index assignment (`assign_frames`). |
| `core.data_resolver` | Resolves default input/output dataset paths inside a project (imported, transformed, and AOI files). |
| `core.template_manager` | Stores project-scoped, immutable face templates and records dataset provenance and review decisions. |
| `core.template_detection` | Runs MediaPipe image detection in an isolated process so native detector failures cannot close the GUI. |
| `core.overlap_review` | Detects ambiguous facial matches (`scan_overlaps`) and applies sample-level decisions (`apply_decision`). |
