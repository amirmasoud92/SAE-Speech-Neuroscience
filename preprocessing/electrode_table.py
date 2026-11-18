"""Electrode table: coordinates, Destrieux labels and regions of interest.

Joins each participant's ``channels.tsv``, the MNI and native ``electrodes.tsv`` files and
the channel list of the high-gamma derivative. Every localised electrode is labelled with
the Destrieux parcel of the nearest labelled atlas voxel (with the distance to it), and the
label is mapped to a region of interest:

* language network: IFG, STG, middle temporal gyrus (MTG) and temporal pole (TMP);
* control set: early auditory cortex (EAC), precentral cortex (PRC) and middle frontal
  gyrus (MFG);
* ``other`` for every remaining parcel.

An electrode is analysis-eligible when it is a good ECoG channel, present in the derivative
and localised. Writes ``electrodes_master.csv``, which ``preprocessing/neural_targets.py``
reads.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy.spatial import KDTree

mne.set_log_level("ERROR")
log = logging.getLogger("electrode_table")

SUBJECTS: list[str] = [f"sub-{i:02d}" for i in range(1, 10)]


# Destrieux (2009) label stems (hemisphere prefix stripped) that map into each region.
ROI_MAP: dict[str, tuple[str, ...]] = {
    "EAC": ("G_temp_sup-G_T_transv", "G_temp_sup-Plan_polar", "G_temp_sup-Plan_tempo"),
    "STG": ("G_temp_sup-Lateral", "S_temporal_sup"),
    "MTG": ("G_temporal_middle", "S_temporal_inf"),
    "IFG": ("G_front_inf-Opercular", "G_front_inf-Triangul", "G_front_inf-Orbital",
            "S_front_inf"),
    "PRC": ("G_precentral", "S_precentral-inf-part", "S_precentral-sup-part", "S_central"),
    "MFG": ("G_front_middle", "S_front_middle"),
    "TMP": ("Pole_temporal",),
}


def coarse_roi(destrieux_label: str) -> str:
    """Map a lateralised Destrieux label (e.g. 'L G_temp_sup-Lateral') to a coarse ROI.

    Returns 'other' when the label falls outside the seven ROIs in ROI_MAP.
    """
    stem = destrieux_label.split(maxsplit=1)[-1].strip()
    for roi, members in ROI_MAP.items():
        if stem in members:
            return roi
    return "other"


def destrieux_labels(coords_mm: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Nearest-labelled-voxel Destrieux lookup for electrode coordinates.

    Parameters
    ----------
    coords_mm : (n_electrodes, 3) float array
        Electrode positions in MNI152NLin2009aSym space, millimetres.

    Returns
    -------
    labels : list of str, length n_electrodes
        Lateralised Destrieux label of the nearest labelled voxel.
    dist_mm : (n_electrodes,) float array
        Euclidean distance to that voxel in millimetres. Large values flag electrodes
        far from any parcel (white matter, off-brain).
    """
    atlas = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
    amap = image.load_img(atlas["maps"])
    adata = amap.get_fdata().astype(int)
    labels_df = atlas["labels"]
    lut = dict(zip(labels_df.index, labels_df["name"]))

    known = {n.split(maxsplit=1)[-1].strip() for n in labels_df["name"]}
    missing = {m for members in ROI_MAP.values() for m in members} - known
    if missing:
        raise ValueError(f"ROI_MAP references labels absent from the atlas: {sorted(missing)}")

    ijk = np.nonzero(adata)
    xyz = np.vstack(image.coord_transform(*ijk, amap.affine)).T
    dist, nn = KDTree(xyz).query(coords_mm, k=1)
    ids = adata[ijk[0][nn], ijk[1][nn], ijk[2][nn]]
    return [lut[i] for i in ids], dist


def build_electrode_table(bids_root: Path) -> pd.DataFrame:
    """Join channels.tsv, both electrodes.tsv coordinate spaces, and the ecogprep derivative."""
    frames = []
    for sub in SUBJECTS:
        ieeg = bids_root / sub / "ieeg"
        ch = pd.read_csv(ieeg / f"{sub}_task-podcast_channels.tsv", sep="\t")
        ch["name"] = ch["name"].astype(str).str.strip()

        mni = pd.read_csv(ieeg / f"{sub}_space-MNI152NLin2009aSym_electrodes.tsv", sep="\t")
        mni["name"] = mni["name"].astype(str).str.strip()
        mni = mni.rename(columns={"x": "mni_x", "y": "mni_y", "z": "mni_z",
                                  "size": "contact_area_mm2", "group": "array_group"})

        nat = pd.read_csv(ieeg / f"{sub}_space-Other_electrodes.tsv", sep="\t")
        nat["name"] = nat["name"].astype(str).str.strip()
        nat = nat.rename(columns={"x": "nat_x", "y": "nat_y", "z": "nat_z"})
        nat = nat[["name", "nat_x", "nat_y", "nat_z"]]

        fif = (bids_root / "derivatives" / "ecogprep" / sub / "ieeg"
               / f"{sub}_task-podcast_desc-highgamma_ieeg.fif")
        in_deriv = set(mne.io.read_raw_fif(fif, preload=False).ch_names)

        df = ch.merge(mni, on="name", how="outer").merge(nat, on="name", how="outer")
        df.insert(0, "subject", sub)
        df["in_channels_tsv"] = df["type"].notna()
        df["in_electrodes_tsv"] = df["mni_x"].notna()
        df["in_native_tsv"] = df["nat_x"].notna()
        df["in_derivative"] = df["name"].isin(in_deriv)
        frames.append(df)

    tab = pd.concat(frames, ignore_index=True)
    tab["array_type"] = tab["array_group"].map(
        {"G": "grid", "EG": "grid_hybrid_1mm", "S": "strip", "D": "depth"})

    loc = tab["in_electrodes_tsv"].to_numpy()
    labels = np.array(["n/a"] * len(tab), dtype=object)
    dists = np.full(len(tab), np.nan)
    lab, dist = destrieux_labels(tab.loc[loc, ["mni_x", "mni_y", "mni_z"]].to_numpy(float))
    labels[loc] = lab
    dists[loc] = dist
    tab["destrieux"] = labels
    tab["destrieux_dist_mm"] = dists
    tab["roi"] = [coarse_roi(x) if x != "n/a" else "n/a" for x in labels]
    tab["hemisphere"] = [x.split()[0] if x != "n/a" else "n/a" for x in labels]

    # Analysis-eligible = good ECoG channel, present in the derivative, and localised.
    tab["analysis_eligible"] = (
        (tab["status"] == "good") & (tab["type"] == "ECOG")
        & tab["in_derivative"] & tab["in_electrodes_tsv"])
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="for example results/electrodes")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tab = build_electrode_table(args.bids_root)
    tab.to_csv(args.out / "electrodes_master.csv", index=False)
    log.info("electrodes_master.csv: %d rows", len(tab))


if __name__ == "__main__":
    main()
