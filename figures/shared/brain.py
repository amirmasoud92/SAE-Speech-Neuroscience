"""Cortical anatomy for the map panels: the fsaverage pial surface in lateral view.

- **Surface:** fsaverage pial, shaded by sulcal depth and a single light source.
- **Regions:** the analysis's region definitions (``ROI_MAP``, as in ``config/config.yaml``),
  painted from the Destrieux annotation on the same surface.

Triangles are projected orthographically onto the sagittal plane and painted back to front, so
the surface occludes itself correctly. A region's rim is drawn as a band of its own boundary
triangles. Electrodes are drawn at the nearest pial vertex to their MNI coordinate (display
only). nilearn downloads the fsaverage surfaces and the Destrieux atlas on first use.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

import style as S

LIGHT = np.array([-1.0, 0.35, 0.55])                 # left view; x is mirrored for right
ROI_MAP = {
    "EAC": ["G_temp_sup-G_T_transv", "G_temp_sup-Plan_polar", "G_temp_sup-Plan_tempo"],
    "STG": ["G_temp_sup-Lateral", "S_temporal_sup"],
    "MTG": ["G_temporal_middle", "S_temporal_inf"],
    "IFG": ["G_front_inf-Opercular", "G_front_inf-Triangul", "G_front_inf-Orbital", "S_front_inf"],
    "PRC": ["G_precentral", "S_precentral-inf-part", "S_precentral-sup-part", "S_central"],
    "MFG": ["G_front_middle", "S_front_middle"],
    "TMP": ["Pole_temporal"],
}


def _destrieux_dir() -> Path:
    from nilearn import datasets
    datasets.fetch_atlas_surf_destrieux()
    return Path(datasets.get_data_dirs()[0]) / "destrieux_surface"


@lru_cache(maxsize=2)
def _mesh(hemi: str):
    import nibabel as nib
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    m = surface.load_surf_mesh(fs[f"pial_{hemi}"])
    V = np.asarray(getattr(m, "coordinates", None) if hasattr(m, "coordinates") else m[0], float)
    F = np.asarray(getattr(m, "faces", None) if hasattr(m, "faces") else m[1], int)
    sulc = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), float)
    lab, _, names = nib.freesurfer.read_annot(str(_destrieux_dir() / f"{hemi}.aparc.a2009s.annot"))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    if len(lab) != len(V):
        # the annotation is on fsaverage5; carry it to the full mesh by nearest neighbour on
        # the registration sphere (the meshes share that sphere)
        from scipy.spatial import cKDTree
        fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
        s5 = surface.load_surf_mesh(fs5[f"sphere_{hemi}"]).coordinates
        sf = surface.load_surf_mesh(fs[f"sphere_{hemi}"]).coordinates
        assert len(s5) == len(lab), "Destrieux annotation matches neither mesh"
        lab = lab[cKDTree(s5).query(sf)[1]]
    return V, F, sulc, lab, names


@lru_cache(maxsize=4)
def _vertex_roi_cached(hemi: str, key: tuple) -> np.ndarray:
    """Region per vertex, with the upsampled boundaries smoothed by a mesh mode filter."""
    from scipy import sparse
    V, F, sulc, lab, names = _mesh(hemi)
    rois = [r for r, _ in key]
    code = np.zeros(len(V), int)                       # 0 = no region
    for k, (roi, parcels) in enumerate(key, 1):
        ids = [i for i, n in enumerate(names) if n in set(parcels)]
        code[np.isin(lab, ids)] = k
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V),) * 2).tocsr()
    A = ((A + A.T) > 0).astype(float) + sparse.eye(len(V))
    for _ in range(6):
        onehot = sparse.csr_matrix((np.ones(len(V)), (np.arange(len(V)), code)),
                                   shape=(len(V), len(rois) + 1))
        code = np.asarray((A @ onehot).argmax(1)).ravel()
    return np.array([""] + rois, dtype=object)[code]


def _vertex_roi(hemi: str, roi_map: dict) -> np.ndarray:
    return _vertex_roi_cached(hemi, tuple((r, tuple(p)) for r, p in roi_map.items()))


def project(xyz: np.ndarray, hemi: str) -> np.ndarray:
    """MNI (x, y, z) to lateral-view page coordinates (anterior left for the left hemisphere)."""
    xyz = np.asarray(xyz, float)
    s = -1.0 if hemi == "left" else 1.0
    return np.c_[s * xyz[:, 1], xyz[:, 2]]


def lateral(ax, hemi: str, roi_map: dict = ROI_MAP, tint: tuple = (), rim: tuple = (),
            tint_alpha: float = 0.38, rim_alpha: float = 0.95, lightness: float = 1.0,
            warm: tuple = (1.0, 0.988, 0.962), shade_floor: float = 0.58,
            tint_colors: dict | None = None):
    """Paint one hemisphere in lateral view into ``ax``. Returns the face centroids (2-D) and
    the face region labels of the visible surface, for placing labels."""
    V, F, sulc, lab, names = _mesh(hemi)
    vroi = _vertex_roi(hemi, roi_map)
    s = -1.0 if hemi == "left" else 1.0
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    if np.mean(n[T[:, :, 0].mean(1) * s > 50][:, 0] * s) < 0:   # normals point outwards
        n = -n
    front = n[:, 0] * s > 0
    T, n, Fv = T[front], n[front], F[front]
    depth = -T[:, :, 0].mean(1) * s
    order = np.argsort(-depth)                                 # far first
    T, n, Fv = T[order], n[order], Fv[order]

    L = np.array([s * abs(LIGHT[0]), LIGHT[1], LIGHT[2]])
    L = L / np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1)
    sf = sulc[Fv].mean(1)
    g = (0.86 - 0.10 * np.tanh(sf * 1.2)) * lightness
    shade = (shade_floor + (1 - shade_floor) * lam)[:, None]
    rgb = np.repeat(g[:, None], 3, 1) * shade * np.asarray(warm)[None, :]

    froi = vroi[Fv]
    face_roi = np.where((froi[:, 0] == froi[:, 1]) & (froi[:, 1] == froi[:, 2]), froi[:, 0], "")
    boundary = ~((froi[:, 0] == froi[:, 1]) & (froi[:, 1] == froi[:, 2]))
    for r in tint:
        m = face_roi == r
        c = np.array(to_rgb((tint_colors or {}).get(r, S.REGION[r])))
        rgb[m] = rgb[m] * (1 - tint_alpha) + tint_alpha * c * shade[m]
    for r in rim:
        c = np.array(to_rgb(S.REGION[r]))
        m = boundary & (froi == r).any(1)
        rgb[m] = rgb[m] * (1 - rim_alpha) + rim_alpha * c
    P = np.stack([s * T[:, :, 1], T[:, :, 2]], -1)
    pc = PolyCollection(P, facecolors=np.clip(rgb, 0, 1), edgecolors=np.clip(rgb, 0, 1),
                        linewidths=0.12, antialiaseds=False, rasterized=True, zorder=1)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_xlim(P[..., 0].min() - 2, P[..., 0].max() + 2)
    ax.set_ylim(P[..., 1].min() - 2, P[..., 1].max() + 2)
    ax.axis("off")
    return P.mean(1), face_roi


def snap(xyz: np.ndarray, hemi: str) -> np.ndarray:
    """Page coordinates of each electrode's nearest pial vertex (display only)."""
    from scipy.spatial import cKDTree
    V = _mesh(hemi)[0]
    if len(xyz) == 0:
        return np.zeros((0, 2))
    return project(V[cKDTree(V).query(np.asarray(xyz, float))[1]], hemi)


def roi_anchor(cent: np.ndarray, face_roi: np.ndarray, roi: str) -> np.ndarray:
    """A visible point inside a region (the median of its visible face centroids)."""
    m = face_roi == roi
    return np.median(cent[m], axis=0) if m.any() else np.array([np.nan, np.nan])
