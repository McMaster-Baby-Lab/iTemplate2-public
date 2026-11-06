---
title: "Gap Management"
format:
  html:
    theme: lumen
    page-layout: full
    fontsize: 1.1em
    linestretch: 1.7
    toc: true
    toc-location: left
---

**Gap Management — controlling how the pipeline handles missing face detections in video stimuli.**

During face landmark extraction, there are frames where the face is not detected (e.g., the participant turned away, occlusion, or poor lighting). These create *gaps* — sequences of missing frame indices between contiguous detection sections. Gap management provides two complementary features to handle these cases.

---

## Understanding Gaps

When landmarks are extracted from a video, each detected frame produces a row in the landmark CSV (`_landmarks.csv`). A "section" is a contiguous block of frames where the face was detected. The space between two sections is a "gap."

**Example:** If Face 1 was detected on frames `1-45` and `78-200`, there are two sections with a gap of 32 frames between them (frames 46-77).

| Scenario | Gap Size | Recommended Approach |
|----------|----------|---------------------|
| Brief blink, slight head turn | 1-10 frames (~0.3s) | Auto-fill with nearest-neighbor remapping |
| Extended occlusion, face turned away | Several seconds | Skip auto-fill; use "Glue Gap" if needed |
| Face left the scene entirely | 10+ seconds | Leave as NaN; do not fill |

---

## Feature 1: Max Gap Threshold (Automatic)

**Location:** Step 3 — Transformation page

The **Max Gap Fill (seconds)** control sets an upper limit on which gaps are automatically filled during the transformation pipeline's Stage 1 (Frame Index Remapping).

### How It Works

- Gaps **smaller than** the threshold are filled by nearest-neighbor remapping (existing behavior). The gaze rows in those frames are mapped to the closest valid frame's coordinates.
- Gaps **larger than** the threshold are **skipped entirely**. Gaze rows in those frames remain as NaN, meaning they will not be transformed and will not contribute to visualization or AOI analysis.
- Setting the value to **60.0 seconds** (maximum) disables the threshold — all gaps are filled as before.

### Example

With a 30 FPS video and a threshold of **2.0 seconds** (default):

| Gap | Frames | Seconds | Auto-filled? |
|-----|--------|---------|-------------|
| Frames 100-105 | 6 | ~0.2s | Yes |
| Frames 200-250 | 51 | ~1.7s | Yes |
| Frames 300-400 | 101 | ~3.4s | No |
| Frames 500-900 | 401 | ~13.4s | No |

### Where to Find It

1. Open the **Transformation** page (Step 3 in the sidebar)
2. Look for the **Max Gap Fill (seconds)** field below the Parallel Workers control
3. Enter a value between 0.1 and 60.0 seconds

The value is automatically saved to `project_settings.json` and persists across sessions.

### Parameters

| Parameter | Key | Default | Description |
|---|---|---|---|
| Max Gap Fill (seconds) | `processing.max_gap_seconds` | `2.0` | Gaps larger than this (in seconds) are not auto-filled during Stage 1. Set to 60.0 to disable. |
| Default FPS | `processing.default_fps` | `30.0` | Frames per second used to convert seconds to frame count. |

---

## Feature 2: Manual Gap Glue (Interactive)

**Location:** Step 1 — Face Extraction page, under the Face Selection / Timeline area

The **Gap Glue** feature allows you to manually bridge a specific gap by linearly interpolating all 478 MediaPipe landmark coordinates between the two boundary frames. Unlike auto-fill (which remaps gaze rows to an existing frame), glue creates entirely new landmark rows in the CSV file.

### When to Use Glue

- A gap occurred due to a brief tracking loss but the face was present the whole time
- You want to ensure landmark data exists for visualization in the preview (over the gap area)
- You've already identified a gap through the timeline bar or section navigation

### How to Glue a Gap

1. Navigate to **Step 1: Face Extraction**
2. In the **Face Selection** area, locate the **Gap** dropdown below the section navigation controls
3. Select the face from the "Face" dropdown to see its gaps
4. The Gap dropdown will list all gaps for that face with frame ranges, frame count, and duration:
   - `Frames 101→105 (5 frames, 0.17s)`
   - `Frames 200→350 (149 frames, 4.97s)`
5. Select the gap you want to fill
6. Click **Glue Gap**
7. Confirm the dialog — it shows the frames to be interpolated, the face ID, and the CSV file that will be modified

### What Happens

- The pipeline reads the landmark CSV and identifies the boundary rows at each end of the gap
- For each missing frame between the boundaries, it computes a new row by linearly interpolating:
  - All 478 landmark x, y, z coordinates (`x_0` through `x_477`, etc.)
  - The `Timestamp` column
- The new rows are written **permanently** to the landmark CSV file
- After a successful glue, the landmark data is reloaded, and the timeline, sections, and gap list all update to reflect the merged sections

### Visual Example

Before glue (gap of 3 frames at 30 FPS):

```
Frame 100  |  Face 1  |  x_0=0.51  y_0=0.52  (last frame of section A)
--- GAP: frames 101, 102, 103 missing ---
Frame 104  |  Face 1  |  x_0=0.54  y_0=0.53  (first frame of section B)
```

After glue (3 interpolated rows added):

```
Frame 100  |  Face 1  |  x_0=0.510  y_0=0.520  (original)
Frame 101  |  Face 1  |  x_0=0.520  y_0=0.523  (interpolated, t=1/4)
Frame 102  |  Face 1  |  x_0=0.530  y_0=0.527  (interpolated, t=2/4)
Frame 103  |  Face 1  |  x_0=0.540  y_0=0.530  (interpolated, t=3/4)
Frame 104  |  Face 1  |  x_0=0.540  y_0=0.530  (original)
```

### Important Notes

- **Permanent change:** The glue operation modifies the landmark CSV file on disk. There is no undo — consider committing to version control before gluing, or keep a backup of the original landmarks.
- **NaN propagation:** If either boundary frame has NaN values for a landmark coordinate, the interpolated rows will also have NaN for that coordinate.
- **Gap at start/end of video:** Gaps at the very beginning or end of a face's data (with no adjacent section) cannot be glued because there is no boundary data to interpolate between. These will not appear in the gap dropdown.
- **After gluing:** If you've already run the Transformation step (Step 3), you should re-run it so the newly glued landmark frames are available for gaze mapping.

---

## How the Two Features Work Together

The recommended workflow when processing video stimuli:

1. **Set a conservative Max Gap Fill threshold** (e.g., 2.0 seconds) in the Transformation page
2. **Run the Transformation pipeline** — small gaps are auto-filled; large gaps remain as NaN
3. **Review the output** — check the log console for "Skipping gap of N frames" messages
4. **For large gaps that should be filled**, use the Gap Glue tool in Step 1 to manually bridge them
5. **Re-run Transformation** if you glued gaps after the initial pipeline run

This two-tier approach ensures that small, expected gaps (blinks, micro-movements) are handled automatically, while larger gaps get individual review rather than being silently filled with potentially incorrect data.

---

## Technical Details

### Max Gap Fill — Implementation

The gap threshold is enforced in `core.transformation.find_missing_and_replace()`. During Stage 1, gaps are grouped as contiguous sequences of missing frame indices. Each group is measured in frames and compared against `max_gap_frames` (computed as `max_gap_seconds * default_fps`). Groups exceeding the threshold are excluded from the remapping step, leaving those gaze rows unchanged (NaN).

### Gap Glue — Implementation

The glue operation is implemented as `core.transformation.glue_landmark_gap(data_path, face_id, gap_end_frame, gap_start_frame, fps)`. It reads the data file, filters to the specified face, identifies the boundary rows, and linearly interpolates all landmark coordinate columns (`x_*`, `y_*`, `z_*`) and the `Timestamp` column. The interpolation parameter for frame `f` is `t = (f - gap_end_frame) / (gap_start_frame - gap_end_frame)`, applied as `new_value = end_value + t * (start_value - end_value)`.
