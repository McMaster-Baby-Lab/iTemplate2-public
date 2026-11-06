"""Project-scoped, immutable template assets and dataset provenance."""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

BOUNDARY = np.array([(0,0),(.5,0),(1,0),(1,.5),(1,1),(.5,1),(0,1),(0,.5)], dtype=np.float32)

# Largest share of reference-mesh area allowed to be orientation-flipped in a custom target.
MAX_FOLDED_AREA_SHARE = .03

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()

@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    image: np.ndarray
    landmarks: np.ndarray
    triangles: np.ndarray
    boundary: np.ndarray
    content_hash: str
    provenance: dict

    def __post_init__(self):
        for field in ('image', 'landmarks', 'triangles', 'boundary'):
            value = np.array(getattr(self, field), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    @property
    def width(self): return self.image.shape[1]
    @property
    def height(self): return self.image.shape[0]
    @property
    def vertices(self): return np.vstack([self.landmarks[:, :2], self.boundary])

class TemplateManager:
    def __init__(self, project_path):
        self.project_path = Path(project_path)

    def load(self, template_id='default'):
        from core import config
        if template_id == 'default':
            if config.average_face is None:
                raise ValueError('The default template asset is missing.')
            return TemplateSpec('default', config.average_face, config.avg_landmarks[:478, :2],
                                np.asarray(config.delaunay_tri, dtype=np.int32), BOUNDARY,
                                array_hash(config.average_face, config.avg_landmarks[:478, :2], np.asarray(config.delaunay_tri)),
                                {'name': 'Default Chicago Face Database average face', 'source_kind': 'default'})
        # IDs are opaque directory names, never arbitrary paths.
        if not template_id or any(c not in '0123456789abcdef' for c in template_id):
            raise ValueError('Invalid template ID.')
        directory = self.project_path / 'templates' / template_id
        manifest = json.loads((directory / 'manifest.json').read_text())
        with h5py.File(directory / 'template.h5', 'r') as hf:
            image = hf['average_face'][:]; lm = hf['avg_landmarks'][:]
            tri = hf['delaunay_tri'][:]; boundary = hf['boundary_points'][:]
        content = array_hash(image, lm, tri, boundary)
        if content != manifest['template_hash'] or template_id != manifest['template_id']:
            raise ValueError('Template asset failed its integrity check.')
        if image.ndim != 3 or image.shape[2] != 4 or min(image.shape[:2]) < 2:
            raise ValueError('Template image is invalid.')
        self.validate(lm, tri, boundary)
        return TemplateSpec(template_id, image, lm, tri, boundary, content, manifest)

    def validate(self, landmarks, triangles, boundary=BOUNDARY):
        ref = self.load('default')
        lm = np.asarray(landmarks)
        if lm.shape != (478, 2) or not np.isfinite(lm).all():
            raise ValueError('Template must contain 478 finite facial landmarks.')
        if (lm < 0).any() or (lm > 1).any():
            raise ValueError('Template landmarks must lie within the image.')
        if not np.array_equal(triangles, ref.triangles) or not np.array_equal(boundary, BOUNDARY):
            raise ValueError('Template topology is incompatible with the default mesh.')
        if len(np.unique(np.round(lm.astype(np.float64), 9), axis=0)) != len(lm):
            raise ValueError('The selected face has duplicate landmark positions. Choose a clearer, frontal face.')
        vertices = np.vstack([lm, boundary])
        def areas(v):
            t = v[ref.triangles]
            u, w = t[:,1]-t[:,0], t[:,2]-t[:,0]
            return u[:,0]*w[:,1]-u[:,1]*w[:,0]
        a, original = areas(vertices), areas(ref.vertices)
        if np.any(np.abs(a) < 1e-10):
            raise ValueError('The selected face has collapsed mesh triangles. Choose a clearer, frontal face.')
        # The reference mesh has hundreds of near-degenerate eyelid/lip triangles that flip
        # with any ordinary expression or proportion difference. Those flips do not affect
        # the mapping (each triangle maps affinely and source membership is independent of
        # the target), so only reject folding that covers a meaningful share of the mesh.
        folded = a * original <= 0
        folded_share = np.abs(original[folded]).sum() / np.abs(original).sum()
        if folded_share > MAX_FOLDED_AREA_SHARE:
            raise ValueError(f'The selected face does not match the reference mesh ({folded_share:.0%} of the mesh '
                             'is folded). Choose a clearer, frontal face.')

    def create(self, image_bgr, landmarks_xyz, provenance):
        image = np.asarray(image_bgr)
        xyz = np.asarray(landmarks_xyz, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or xyz.shape != (478,3) or not np.isfinite(xyz).all():
            raise ValueError('A valid image and 478 extracted XYZ landmarks are required.')
        h, w = image.shape[:2]
        xy = xyz[:,:2] * [w,h]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        margin = (hi-lo) * .15
        x0,y0 = np.floor(lo-margin).astype(int)
        x1,y1 = np.ceil(hi+margin).astype(int)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Face crop is empty.')
        lm = ((xy-[x0,y0])/[x1-x0,y1-y0]).astype(np.float32)
        default = self.load()
        self.validate(lm, default.triangles)
        # A face near the image border keeps its margin: pad by edge replication
        # rather than clipping, so landmarks never touch the boundary points.
        pad = int(max(0, -x0, -y0, x1-w, y1-h))
        padded = cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE) if pad else image
        crop = cv2.cvtColor(padded[y0+pad:y1+pad,x0+pad:x1+pad], cv2.COLOR_BGR2BGRA)
        content = array_hash(crop,lm,default.triangles,BOUNDARY)
        tid = content[:32]
        root = self.project_path / 'templates'; root.mkdir(parents=True,exist_ok=True)
        dest = root / tid
        if dest.exists(): return self.load(tid)
        manifest = dict(provenance, template_id=tid, template_hash=content, schema_version=1,
                        created_at=datetime.now(timezone.utc).isoformat(), original_dimensions=[w,h],
                        crop_bounds=[int(x0),int(y0),int(x1),int(y1)],
                        topology_hash=array_hash(default.triangles), coordinate_convention='normalized_xy',
                        application_version='iTemplate2', detector_provenance=provenance.get('detector_provenance','unavailable'))
        tmp = Path(tempfile.mkdtemp(prefix='.template-',dir=root))
        try:
            with h5py.File(tmp/'template.h5','w') as hf:
                for name,val in [('average_face',crop),('avg_landmarks',lm),('delaunay_tri',default.triangles),
                                 ('boundary_points',BOUNDARY),('source_landmarks_xyz',xyz)]:
                    hf.create_dataset(name,data=val)
                hf.attrs.update(schema_version=1,template_id=tid,width=crop.shape[1],height=crop.shape[0],
                                coordinate_convention='normalized_xy',topology_hash=manifest['topology_hash'])
            (tmp/'manifest.json').write_text(json.dumps(manifest,indent=2))
            if not cv2.imwrite(str(tmp/'source.png'),image) or not cv2.imwrite(str(tmp/'preview.png'),crop):
                raise ValueError('Could not save template images.')
            os.replace(tmp,dest)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
        return self.load(tid)

    def for_dataset(self,path):
        metadata = dataset_manifest(path)
        spec = self.load(metadata.get('template_id','default'))
        if metadata.get('template_hash') and metadata['template_hash'] != spec.content_hash:
            raise ValueError('Dataset template hash does not match its asset.')
        return spec

def dataset_manifest(path):
    if str(path).lower().endswith('.parquet'):
        meta = pq.read_schema(path).metadata or {}
        if b'itemplate' in meta:
            return json.loads(meta[b'itemplate'])
        if 'template_id' in pq.read_schema(path).names:
            ids = set(pq.read_table(path,columns=['template_id']).column('template_id').to_pylist()) - {None}
            if len(ids) != 1: raise ValueError('Dataset contains inconsistent template IDs.')
            return {'template_id':ids.pop()}
    sidecar = Path(path).with_suffix('.manifest.json')
    return json.loads(sidecar.read_text()) if sidecar.exists() else {}

def save_dataset(data,path,manifest):
    """Atomic parquet publishing with embedded, portable provenance."""
    import pyarrow as pa
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    table = pa.Table.from_pandas(data,preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b'itemplate'] = json.dumps(manifest).encode()
    table = table.replace_schema_metadata(metadata)
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    sidecar = path.with_suffix('.manifest.json')
    tmp_json = tmp.with_suffix('.json')
    try:
        pq.write_table(table,tmp,compression='snappy')
        tmp_json.write_text(json.dumps(manifest,indent=2))
        os.replace(tmp_json,sidecar)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True); tmp_json.unlink(missing_ok=True)

def context_fingerprint(manifest):
    keys = ('source_dataset_hash','landmark_hashes','face_ids','stimuli_type','max_gap_frames')
    payload = {k:manifest.get(k) for k in keys}
    # Added only when set, so decisions saved before this option keep their fingerprint.
    if manifest.get('contour_margin'): payload['contour_margin'] = manifest['contour_margin']
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def save_review_decisions(project_path, data, manifest):
    if 'gaze_sample_id' not in data or 'assignment_review_status' not in data:
        return
    items = []
    for _, row in data[data.assignment_review_status.isin(['assigned','excluded'])].iterrows():
        items.append({'sample_id':str(row.gaze_sample_id), 'action':str(row.assignment_review_status),
                      'face_id':int(row.review_face_id) if row.assignment_review_status == 'assigned' else None})
    directory = Path(project_path)/'review_decisions'; directory.mkdir(parents=True,exist_ok=True)
    path = directory/(context_fingerprint(manifest)+'.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'context':context_fingerprint(manifest),
                               'source_dataset_hash':manifest.get('source_dataset_hash'),
                               'decisions':items},indent=2))
    os.replace(tmp,path)

def replay_review_decisions(project_path,data,manifest,landmark_folder,template):
    """Recompute reviewed coordinates, never reuse coordinates from an old target."""
    from core.overlap_review import scan_overlaps, apply_decision
    from core.io_utils import resolve_file_in_folder
    path = Path(project_path)/'review_decisions'/(context_fingerprint(manifest)+'.json')
    if not path.exists():
        # A changed source context must not silently reuse earlier decisions.
        stale_ids = set()
        for previous in (Path(project_path)/'review_decisions').glob('*.json'):
            saved = json.loads(previous.read_text())
            if saved.get('source_dataset_hash') == manifest.get('source_dataset_hash'):
                stale_ids.update(d['sample_id'] for d in saved.get('decisions',[]))
        affected = data.gaze_sample_id.astype(str).isin(stale_ids)
        if affected.any(): data.loc[affected,'assignment_review_status'] = 'requires_review'
        return data
    decisions = {d['sample_id']:d for d in json.loads(path.read_text())['decisions']}
    stim_col = next(c for c in ('stimulusName','videoName','STIMULI_NAME') if c in data)
    for stim, group in data.groupby(stim_col):
        selected = group[group.gaze_sample_id.astype(str).isin(decisions)]
        if selected.empty: continue
        lm_path = resolve_file_in_folder(stim,landmark_folder,suffix='_landmarks',extensions=['.parquet','.csv'])
        if not lm_path: continue
        import pandas as pd
        lm = pd.read_parquet(lm_path) if lm_path.endswith('.parquet') else pd.read_csv(lm_path)
        scan = scan_overlaps(selected,lm,manifest['face_ids'],manifest['stimuli_type']=='video',template=template,
                             contour_margin=manifest.get('contour_margin') or 0.0)
        for idx,row in selected.iterrows():
            d=decisions[str(row.gaze_sample_id)]
            candidates=scan['section_samples'].get(idx,{})
            if d['action']=='excluded':
                apply_decision(data,idx,candidates,'exclude')
            elif d['face_id'] in candidates:
                apply_decision(data,idx,candidates,d['face_id'])
            else:
                data.loc[idx,'assignment_review_status']='requires_review'
    return data
