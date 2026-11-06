"""Detect ambiguous facial matches and apply auditable sample-level decisions."""
import cv2
import numpy as np
import pandas as pd
from core import config
from core.transformation import _find_mapping_triangles, expand_face_contour, facial_outline_indices


def scan_overlaps(data, landmarks, face_ids, is_video=True, cancelled=lambda: False, template=None,
                  contour_margin=0.0):
    """Return candidate mappings keyed by source row index, using facial triangles only.

    Requires extracted landmarks (not fresh detections) so reviewed IDs refer to
    the same faces as transformation. Frame indices are one-based throughout.
    """
    if not data.index.is_unique:
        raise ValueError('Gaze rows must have unique indices.')
    required = ['Face_ID'] + [f'{axis}_{i}' for axis in 'xyz' for i in range(478)]
    if is_video:
        required += ['Frame_Index']
    missing = [col for col in required if col not in landmarks]
    if missing:
        raise ValueError('Landmark file lacks required face/frame or coordinate columns: ' + ', '.join(missing[:5]))
    if is_video and '_video_frame_index' not in data:
        raise ValueError('Video gaze data lacks _video_frame_index. Run transformation first.')
    if template is None and config.delaunay_tri is None:
        raise ValueError('The face template is unavailable.')
    triangles = np.asarray(template.triangles if template is not None else config.delaunay_tri, dtype=np.int32)
    from core.template_manager import BOUNDARY
    target = template.vertices if template is not None else np.vstack([np.asarray(config.avg_landmarks)[:478, :2], BOUNDARY])
    selected = landmarks[landmarks.Face_ID.isin(face_ids)].copy()
    if is_video:
        selected['Frame_Index'] = pd.to_numeric(selected.Frame_Index, errors='coerce')
        if selected.duplicated(['Frame_Index', 'Face_ID']).any():
            raise ValueError('Duplicate face IDs within a frame. Correct face IDs before review.')
        lm_groups = {frame: group for frame, group in selected.groupby('Frame_Index')}
        frames = pd.to_numeric(data['_video_frame_index'], errors='coerce')
    else:
        if selected.duplicated('Face_ID').any():
            raise ValueError('Image landmark file has duplicate face IDs.')
        lm_groups = {1: selected}
        frames = pd.Series(1, index=data.index)
    # Detect stimulus overlap independently of gaze. A frame can overlap even
    # when all its samples look at the background or no samples exist at all.
    facial = triangles[np.all(triangles < 478, axis=1)]
    outline = facial_outline_indices(triangles) if contour_margin else None
    overlap_meshes, outlines = {}, {}
    # Frames lacking a finite mesh for any selected face cannot rule out
    # overlap with that face. Their samples are counted as unchecked, but
    # overlaps among the remaining faces are still reviewed.
    incomplete_frames = set()
    for frame, frame_lm in lm_groups.items():
        if cancelled():
            raise InterruptedError('Overlap detection cancelled.')
        if not np.isfinite(frame) or frame < 1 or frame != int(frame):
            continue
        meshes = {}
        for fid in face_ids:
            rows = frame_lm[frame_lm.Face_ID == fid]
            if rows.empty:
                incomplete_frames.add(frame)
                continue
            row = rows.iloc[0]
            xyz = np.column_stack([row[[f'{axis}_{i}' for i in range(478)]].to_numpy(dtype=np.float32)
                                   for axis in 'xyz'])
            if np.isfinite(xyz).all():
                # Same expanded outline as transformation, so candidates match.
                meshes[fid] = expand_face_contour(xyz, outline, contour_margin)
            else:
                incomplete_frames.add(frame)
        ids = list(meshes)
        overlapping = any(_meshes_overlap(meshes[left][:, :2], meshes[right][:, :2], facial)
                          for i, left in enumerate(ids) for right in ids[i + 1:])
        if overlapping:
            overlap_meshes[int(frame)] = meshes
            outlines[int(frame)] = {fid: cv2.convexHull(xyz[:, :2].copy()).reshape(-1, 2)
                                    for fid, xyz in meshes.items()}
    overlap_frames = sorted(overlap_meshes)
    intervals = []
    for frame in overlap_frames:
        if intervals and frame == intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], frame)
        else:
            intervals.append((frame, frame))
    overlaps, section_samples = {}, {}
    unchecked_rows = set(data.index[~np.isfinite(frames) | frames.isna()])
    for frame, group in data.groupby(frames):
        if cancelled():
            raise InterruptedError('Overlap detection cancelled.')
        if (not np.isfinite(frame) or frame < 1 or frame != int(frame)
                or frame not in lm_groups):
            unchecked_rows.update(group.index)
            continue
        if frame in incomplete_frames:
            unchecked_rows.update(group.index)
        if frame not in overlap_meshes:
            continue
        # Include every sample from overlapping sections, including off-face
        # and invalid gaze; only finite facial matches offer face assignments.
        candidates = {key: {} for key in group.index}
        gaze = group[['stimuli_x', 'stimuli_y']].to_numpy(dtype=np.float32)
        finite = np.isfinite(gaze).all(axis=1)
        row_ids = group.index[finite]
        gaze = gaze[finite]
        for fid, xyz in overlap_meshes[frame].items():
            matches = _find_mapping_triangles(gaze, xyz, triangles, False, 'facial')
            for triangle_id in np.unique(matches[matches >= 0]):
                vertices = triangles[triangle_id]
                matrix = cv2.getAffineTransform(xyz[vertices, :2].copy(), target[vertices].astype(np.float32))
                hit = np.flatnonzero(matches == triangle_id)
                mapped = gaze[hit] @ matrix[:, :2].T + matrix[:, 2]
                for pos, coords in zip(hit, mapped):
                    candidates[row_ids[pos]][fid] = coords
        section_samples.update(candidates)
        overlaps.update({key: value for key, value in candidates.items() if len(value) > 1})
    return {'overlaps': overlaps, 'section_samples': section_samples,
            'overlap_frames': overlap_frames, 'intervals': intervals,
            'outlines': outlines, 'unchecked': len(unchecked_rows)}


def _meshes_overlap(left, right, facial_triangles):
    """Positive-area intersection of facial triangles; touching is not overlap.

    Hull and bounding-box checks are only acceleration filters. The final test
    uses actual triangles, so peripheral meshes and hull gaps cannot trigger it.
    """
    tolerance = 1e-10  # normalized stimulus area
    left_hull = cv2.convexHull(left.copy())
    right_hull = cv2.convexHull(right.copy())
    if cv2.intersectConvexConvex(left_hull, right_hull)[0] <= tolerance:
        return False
    a, b = left[facial_triangles], right[facial_triangles]
    amin, amax = a.min(axis=1), a.max(axis=1)
    bmin, bmax = b.min(axis=1), b.max(axis=1)
    for triangle, lower, upper in zip(a, amin, amax):
        possible = np.flatnonzero(np.all(bmax > lower, axis=1) & np.all(bmin < upper, axis=1))
        for other in possible:
            if cv2.intersectConvexConvex(triangle.copy(), b[other].copy())[0] > tolerance:
                return True
    return False


def apply_decision(data, row_id, candidates, decision):
    """decision is a candidate face ID, 'exclude', or 'automatic'."""
    if isinstance(decision, int) and decision not in candidates:
        raise ValueError('The chosen face is not a facial match for this gaze.')
    if decision not in ('automatic', 'exclude') and not isinstance(decision, int):
        raise ValueError('Unknown assignment decision.')
    source_columns = ['face_id', 'transformed_x', 'transformed_y', 'face_label', 'gaze_region']
    captured = 'automatic_gaze_region' in data and pd.notna(data.at[row_id, 'automatic_gaze_region'])
    if not captured:
        original_fid = data.at[row_id, 'automatic_face_id'] if 'automatic_face_id' in data else pd.NA
        if pd.notna(original_fid) and int(original_fid) in candidates:
            original_fid = int(original_fid)
            original = {'face_id': original_fid, 'transformed_x': candidates[original_fid][0],
                        'transformed_y': candidates[original_fid][1],
                        'face_label': f'Face {original_fid}', 'gaze_region': 'on_face'}
        else:
            original = {col: data.at[row_id, col] for col in source_columns}
        for col in source_columns:
            audit_col = 'automatic_' + col
            if audit_col not in data:
                if col == 'face_id':
                    data[audit_col] = pd.array([pd.NA] * len(data), dtype='Int64')
                elif col in ('transformed_x', 'transformed_y'):
                    data[audit_col] = np.nan
                else:
                    data[audit_col] = pd.Series(pd.NA, index=data.index, dtype='string')
            data.at[row_id, audit_col] = original[col]
    if 'assignment_review_status' not in data:
        data['assignment_review_status'] = 'not_required'
    if 'review_face_id' not in data:
        data['review_face_id'] = pd.array([pd.NA] * len(data), dtype='Int64')
    if 'overlap_face_ids' not in data:
        data['overlap_face_ids'] = ''
    data.at[row_id, 'overlap_face_ids'] = ','.join(map(str, candidates))
    data.at[row_id, 'review_face_id'] = pd.NA
    if decision == 'exclude':
        data.loc[row_id, ['transformed_x', 'transformed_y']] = np.nan
        data.at[row_id, 'face_id'] = pd.NA
        data.at[row_id, 'face_label'] = 'Excluded overlap'
        data.at[row_id, 'gaze_region'] = 'unmapped'
        data.at[row_id, 'assignment_review_status'] = 'excluded'
        return
    if decision == 'automatic':
        for col in source_columns:
            data.at[row_id, col] = data.at[row_id, 'automatic_' + col]
        data.at[row_id, 'assignment_review_status'] = 'unreviewed' if len(candidates) > 1 else 'context'
        return
    fid = decision
    data.loc[row_id, ['transformed_x', 'transformed_y']] = candidates[fid]
    data.at[row_id, 'face_id'] = fid
    data.at[row_id, 'face_label'] = f'Face {fid}'
    data.at[row_id, 'gaze_region'] = 'on_face'
    data.at[row_id, 'assignment_review_status'] = 'assigned'
    data.at[row_id, 'review_face_id'] = fid
