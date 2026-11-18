# Figure source data and panel scripts

Each data-driven panel of the paper's figures has its own folder:

```
figures/figure_<N>/panel_<x>/
    data.csv     the values plotted in the panel
    *.csv        companion tables for statistics printed on the panel
    plot.py      a standalone script that draws the panel from the files in its folder
```

Run any script from anywhere, for example `python figures/figure_4/panel_f/plot.py`. It writes
`panel.pdf` next to itself; set `PANEL_OUT=<folder>` to write elsewhere and `PANEL_FORMAT=png` to
change the format. Shared colours and helpers are in `shared/style.py`; cortical-surface panels also
use `shared/brain.py`, which needs `nilearn` and downloads the fsaverage surface and Destrieux atlas
on first use. Where one script serves several panels, the same script is copied into each folder and
reads its panel from the folder name.

| Figure | Panel folders |
|---|---|
| 1 | a (spectrogram, dense stream, sparse code, unique variance), b, c, d, e, f |
| 2 | a, b, c, d (lag profiles, change in median), e |
| 3 | a, b, c, d, e |
| 4 | a, b, c, d, e, f, g (IFG, STG) |
| 5 | a, b |
| 6 | a, b, c |
| 7 | a (raw, partialled), b, c |
| 8 | a, b, c, d |
| 9 | a, b, c, d (within, between) |
| 10 | a, b, c, d (raw, corrected), e, f |
| S1 | a, b, c |
| S2 | a–f |
| S3 | a, b |
| S4 | a, b |
| S5 | single panel (`panel_a`) |
| S6 | between types, within types (`panel_between`, `panel_within`) |
| S7 | a, b |

Schematic elements (pipeline glyphs in Fig. 1a, the schematic in Fig. 7b, the decomposition
header of Fig. 9), panel letters, titles and anatomical icons have no script; the scripts use plain
text labels in place of icons. Figures are assembled from the individual panels.
