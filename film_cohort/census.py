"""Brain Treebank census from metadata only (no neural samples are read).

Counts, per participant and region, the contacts an analysis can use: Destrieux label mapped
through the primary cohort's region map, not listed as corrupted, and with at least one
same-shaft neighbour for re-referencing (Laplacian if both neighbours exist, bipolar if one).
Also counts words with a neural sample per participant and the films each participant
watched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from utils.config import load_config
from utils.paths import BTB_ROOT, BTB_ALIGNMENT

BTB = BTB_ROOT
ALIGN = BTB_ALIGNMENT
NO_ATLAS = {"sub_5"}


def shaft(nm: str):
    m = re.match(r"^(.*?)(\d+)$", nm)
    return (m.group(1), int(m.group(2))) if m else (nm, -1)


def census(cfg) -> pd.DataFrame:
    roi_of = {lab: roi for roi, labs in cfg.electrodes["roi_map"].items() for lab in labs}
    loc = pd.read_csv(BTB / "localization" / "localization" / "elec_coords_full.csv")
    bad = json.load(open(BTB / "corrupted_elec.json"))
    rows = []
    for sub in sorted(loc.Subject.unique(), key=lambda s: int(s.split("_")[1])):
        lab = json.load(open(BTB / "electrode_labels" / "electrode_labels" / sub /
                             "electrode_labels.json"))
        names = [x.rstrip("*") for x in lab if not x.upper().startswith("TRIG")]
        good = set(names) - {b.strip("*#_") for b in bad.get(sub, [])} - set(bad.get(sub, []))
        reg = loc[loc.Subject == sub].set_index("Electrode").Region.to_dict()
        for nm in sorted(good):
            s, c = shaft(nm)
            nb = sum(f"{s}{c + d}" in good for d in (-1, 1))
            rows.append({"subject": sub, "electrode": nm, "region": reg.get(nm, ""),
                         "roi": roi_of.get(reg.get(nm, ""), "other"),
                         "ref": "lap" if nb == 2 else ("bip" if nb == 1 else "mono"),
                         "atlas_valid": sub not in NO_ATLAS})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    C = census(cfg)
    use = C[(C.ref != "mono") & C.atlas_valid]
    tab = use.pivot_table(index="subject", columns="roi", values="electrode",
                          aggfunc="count", fill_value=0)
    A = pd.read_csv(ALIGN)
    words = A.dropna(subset=["neural_sample"]).groupby("sub").agg(
        words=("start", "size"), films=("film", lambda f: ", ".join(sorted(f.unique()))))
    tab = tab.join(words, how="left")
    out = cfg.results("film_replication")
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "btb_contact_census.csv", index=False)
    tab.to_csv(out / "btb_census_by_subject.csv")
    pd.set_option("display.width", 250)
    print(tab.to_string())
    print("\ntotals:", use.roi.value_counts().to_dict())


if __name__ == "__main__":
    main()
