"""Contrasts between window arms: Matryoshka, static embeddings and pooling.

Reads ``controls/window_arms/<convention>_<arm>.csv``. Every contrast is the median
per-electrode difference with a participant-clustered bootstrap (10,000 resamples; 95% interval,
two-sided P); equivalence bounds use the 90% interval against the equivalence margin.

* Matryoshka dictionary against the dense stream and against GemmaScope;
* Gemma-2-2B static input embeddings against the classical band and the contextual arms;
* pooling by mean, maximum and last token, for the sparse and dense arms.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent
MARGIN = 0.0155                       # the equivalence margin, from the reference arms
CONVS = ("LEG", "TO")


def load(cfg, conv, arm):
    f = cfg.results("controls", "window_arms",
                    f"{conv}_{arm.replace(':', '_').replace(' ', '_')}.csv")
    return pd.read_csv(f) if f.exists() else None


def contrast(cfg, conv, a, b, seed, label=None):
    A, B = load(cfg, conv, a), load(cfg, conv, b)
    if A is None or B is None:
        return None
    assert (A.ch.to_numpy() == B.ch.to_numpy()).all()
    st, boot = PS.electrode_stat(A.r.to_numpy() - B.r.to_numpy(), A.subject.to_numpy(), seed)
    return {"convention": conv, "contrast": label or f"{a} - {b}", "mean_r_a": A.r.mean(),
            "mean_r_b": B.r.mean(), "ratio_of_means": A.r.mean() / B.r.mean(),
            "n_a_higher": int((A.r > B.r).sum()), **st}, boot


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    R = cfg.results("controls")
    s = cfg.seed + 2000

    # ------------------------------------------------------------------- static arm
    rows = []
    for conv in CONVS:
        for i, (a, b) in enumerate([("SAE-85", "classical-85"),
                                    ("Static embedding-85", "classical-85"),
                                    ("SAE-85", "Static embedding-85"),
                                    ("Gemma dense-300", "Static embedding-300"),
                                    ("SAE-300", "Static embedding-300"),
                                    ("GPT-2 XL-300", "Static embedding-300"),
                                    ("SAE-300", "Gemma dense-300"),
                                    ("SAE-300", "GPT-2 XL-300")]):
            c = contrast(cfg, conv, a, b, s + i)
            if c:
                rows.append({"table": "LLM vs classical (85)" if "85" in a else
                             "attribution (300)", **c[0]})
    A4 = pd.DataFrame(rows)
    (R / "static_embeddings").mkdir(exist_ok=True)
    A4.to_csv(R / "static_embeddings" / "a4_static_rows.csv", index=False)

    # --------------------------------------------------------------- Matryoshka arm
    rows, eq = [], []
    for conv in CONVS:
        for i, (a, b) in enumerate([("Matryoshka-85", "classical-85"),
                                    ("Matryoshka-300", "Gemma dense-300"),
                                    ("SAE-300", "Gemma dense-300")]):
            c = contrast(cfg, conv, a, b, s + 100 + i)
            if not c:
                continue
            rows.append(c[0])
            if "Gemma dense" in b:
                st, boot = c
                eq.append({"convention": conv, "contrast": st["contrast"], "margin": MARGIN,
                           "estimate": st["estimate"], "ci90_lo": st["ci90_lo"],
                           "ci90_hi": st["ci90_hi"],
                           "p_nonsuperiority": float((boot >= MARGIN).mean()),
                           "p_lower": float((boot <= -MARGIN).mean()),
                           "tost_p": float(max((boot >= MARGIN).mean(), (boot <= -MARGIN).mean())),
                           "nonsuperiority": bool(st["ci90_hi"] < MARGIN),
                           "equivalence": bool(st["ci90_lo"] > -MARGIN and st["ci90_hi"] < MARGIN),
                           "test": "participant-clustered bootstrap, 90% CI against the "
                                   "equivalence margin; one-sided tests (TOST)",
                           "n_electrodes": st["n_electrodes"],
                           "n_participants": st["n_participants"]})
        G, D = load(cfg, conv, "GPT-2 XL-300"), load(cfg, conv, "Gemma dense-300")
        if G is not None and D is not None:
            eq.append({"convention": conv, "contrast": "margin re-derived under this "
                       "convention (dense - GPT-2 XL, mean r)",
                       "estimate": float(D.r.mean() - G.r.mean())})
    (R / "matryoshka").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(R / "matryoshka" / "a2_matched.csv", index=False)
    pd.DataFrame(eq).to_csv(R / "matryoshka" / "a2_equivalence.csv", index=False)
    bat = pd.read_csv(cfg.results("layer_and_model_battery", "a35_a36_battery.csv"))
    mrow = bat[bat.family.str.contains("Matryoshka")].iloc[0].to_dict()
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_matryoshka32k.hdf5", "r") as h:
        attrs = {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in h.attrs.items()}
        Fw = h["sae_word_max"][...]
        shells = h["shell_bounds"][...].tolist()
    info = {"release": "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40",
            "layer": 12, "width": int(Fw.shape[1]), "nominal_l0": 40,
            "word_l0_mean": float((Fw > 0).sum(1).mean()), "shell_bounds": shells,
            "file_attrs": attrs, "a36_battery_row": mrow}
    json.dump(info, open(R / "matryoshka" / "a2_matryoshka_info.json", "w"), indent=1, default=str)

    # ---------------------------------------------------------------------- pooling
    arm = {("SAE", "max"): "SAE-300", ("SAE", "mean"): "pool:sae_mean-300",
           ("SAE", "last"): "pool:sae_last-300", ("dense", "mean"): "Gemma dense-300",
           ("dense", "max"): "pool:dense_max-300", ("dense", "last"): "pool:dense_last-300"}
    rows, inter = [], []
    for conv in CONVS:
        r = {k: load(cfg, conv, v) for k, v in arm.items()}
        if any(v is None for v in r.values()):
            continue
        subj = r[("SAE", "max")].subject.to_numpy()
        gap = {p: r[("SAE", p)].r.to_numpy() - r[("dense", p)].r.to_numpy()
               for p in ("mean", "max", "last")}
        for i, p in enumerate(("mean", "max", "last")):
            st, _ = PS.electrode_stat(gap[p], subj, s + 200 + i)
            rows.append({"convention": conv, "pooling": p, "contrast": f"SAE({p}) - dense({p})",
                         "mean_r_sae": r[("SAE", p)].r.mean(),
                         "mean_r_dense": r[("dense", p)].r.mean(), **st})
        for i, (p, q) in enumerate((("max", "mean"), ("last", "mean"), ("last", "max"))):
            st, _ = PS.electrode_stat(gap[p] - gap[q], subj, s + 210 + i)
            inter.append({"convention": conv, "contrast": f"gap({p}) - gap({q})", **st})
    T5, I5 = pd.DataFrame(rows), pd.DataFrame(inter)
    if len(I5):
        I5["p_holm"] = np.concatenate([holm(g.p_two_sided.to_numpy())
                                       for _, g in I5.groupby("convention", sort=False)])
    (R / "pooling").mkdir(exist_ok=True)
    T5.to_csv(R / "pooling" / "a5_pooling_gaps.csv", index=False)
    I5.to_csv(R / "pooling" / "a5_pooling_interactions.csv", index=False)

    cols = ["convention", "contrast", "mean_r_a", "mean_r_b", "estimate", "ci95_lo", "ci95_hi",
            "p_two_sided", "n_electrodes", "n_participants"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("A4\n", A4[["table"] + cols].to_string(index=False))
        print("A2\n", pd.DataFrame(rows and []).to_string() if False else "")
        print(pd.read_csv(R / "matryoshka" / "a2_matched.csv")[cols].to_string(index=False))
        print(pd.read_csv(R / "matryoshka" / "a2_equivalence.csv").to_string(index=False))
        print("A5\n", T5.to_string(index=False) if len(T5) else "(pending)")
        print(I5.to_string(index=False) if len(I5) else "")


if __name__ == "__main__":
    main()
