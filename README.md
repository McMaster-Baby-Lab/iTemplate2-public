# iTemplate2

**An open-source Python toolbox for automatic eye-movement analysis of dynamic facial stimuli.**

[Project page](https://mcmaster-baby-lab.github.io/iTemplate2-public/) · [User guide](docs/user-guide.md) · [Report an issue](https://github.com/McMaster-Baby-Lab/iTemplate2-public/issues)

![iTemplate2 face extraction screen](docs/assets/face-extraction.webp)

iTemplate2 co-registers eye-tracking gaze data onto a normalised face template. It lets researchers analyse where participants looked on **dynamic face stimuli (videos)** or **static face images**, even when faces move, turn, or share the scene with other people. A full desktop GUI walks you through the whole pipeline, from importing raw gaze data to exporting analysis-ready tables.

## What it does

1. **Extracts facial landmarks** from your stimulus videos or images with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
2. **Maps raw gaze coordinates** into a common, template-normalised face space, so gaze on different faces, frames and stimuli is directly comparable.
3. **Visualises** gaze over the stimulus and as per-participant-normalised heatmaps.
4. **Analyses areas of interest (AOIs)** drawn on the template face or on the original stimulus images.
5. **Exports** the results to CSV.

![Pipeline overview](docs/assets/flowchart.webp)

## Key features

- Face landmark extraction for video and image stimuli, with gap detection and repair
- Automatic gaze-to-template coordinate transformation, including multi-face scenes
- Review tool for frames where facial regions overlap
- Custom template faces, in addition to the built-in average face
- Works with screen-based and wearable eye trackers, independent of the tracker's make or model (for example Tobii, SR Research, Pupil Labs)
- Flexible data import with column mapping and stimulus-name cleaning
- Temporal preprocessing with onset/offset trial segmentation
- Interactive heatmaps and gaze playback, with video and image export
- Dual-context AOI analysis (template face and stimulus images)
- Project-based workflow with saved settings and a timestamped processing log

## Installation

iTemplate2 runs on macOS, Windows and Linux and requires **Python 3.10, 3.11, 3.12 or 3.13**. Python 3.14 is not recommended because some dependencies, especially MediaPipe, may not yet provide compatible wheels on every platform.

### 1. Get the code

```bash
git clone https://github.com/McMaster-Baby-Lab/iTemplate2-public.git
cd iTemplate2-public
```

### 2. Create an environment

**Option A (recommended): venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Option B: conda**

```bash
conda env create -f environment.yml
conda activate iTemplate2
```

> **Linux:** PyQt6 needs the `xcb-util-cursor` package. Install it first, for example `sudo apt install xcb-util-cursor` (Debian/Ubuntu) or `sudo dnf install xcb-util-cursor` (Fedora/RHEL).

The MediaPipe face model (`face_landmarker.task`) and the average face template (`average_face_data.h5`) are included in the repository root.

### 3. Launch the app

```bash
python start_gui.py
```

A sidebar on the left lists every processing step, with **Settings** at the bottom.

## Getting started

1. Read [Before You Begin: Know Your Data](docs/user-guide.md#before-you-begin-know-your-data) to check that you know your dataset's columns, timing and stimulus names.
2. In **Step 0**, create a project and import your gaze data.
3. Follow Steps 1 to 6 in the sidebar: face extraction, data review, temporal preprocessing and coordinate setup, transformation, visualisation, AOI analysis and export.

The [user guide](docs/user-guide.md) covers every step, the parameter reference, the required data columns and troubleshooting.

## Documentation

| Document | Contents |
|---|---|
| [User guide](docs/user-guide.md) | Step-by-step GUI workflow, parameters, data requirements, logging, troubleshooting |
| [Gap management](docs/gap_management.md) | Understanding landmark gaps, choosing a threshold, using the glue tool |
| [Architecture](docs/architecture.md) | Code layout and the `core/` modules for developers |

## Citation

If you use iTemplate2 in your research, please cite:

> Xiao, N. G., Román-Caballero, R., Fang, W., Ripley, S., Han, D., & Trainor, L. J. (in press). iTemplate2: An open-source Python toolbox for automatic eye-movement analysis of dynamic facial stimuli.

<!-- TODO: add journal, volume, year and DOI once available -->

## Authors

Developed by Dr. Gabriel (Naiqi) Xiao with contributions from Dr. Wei Fang, Dr. Rafael Román-Caballero, Sara Ripley and Dr. Laurel J. Trainor, at the McMaster Baby Lab, McMaster University.

## Support

Questions, bug reports and feature requests are welcome on the [issue tracker](https://github.com/McMaster-Baby-Lab/iTemplate2-public/issues). If something goes wrong in the app, `processing_log.txt` in your project folder records every step, including errors.

## License

<!-- TODO: choose a license and add a LICENSE file -->
A license has not been chosen yet. Until a `LICENSE` file is added to this repository, all rights are reserved by the authors.
