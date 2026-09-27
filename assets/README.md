# Assets

Figures and media used by [the README](../README.md) and by the project page
([`index.html`](../index.html)). The root `.gitignore` drops `*.png`, `*.jpg`,
`*.pdf` and `*.mp4`; `assets/.gitignore` and `assets/site/.gitignore` re-allow
the extensions this directory needs, so add new media types in those files
rather than at the repository root.

## Figures

| File | Content |
|---|---|
| `teaser.png` | Teaser. The two DreamGen tasks with per-rollout Model Laziness annotations, Standard SFT against EVEWorld. |
| `method_overview.png` | Overall architecture: IGR builds count-edited supervision pairs from clean clips, TIA aligns the target instance across frames at the probed layer $\ell^\star$. |
| `teaser.svg` | Vector redraw of the teaser, used where a resolution-independent figure is preferable. |

Both PNGs are rendered from the camera-ready vector figures at 200 dpi and
capped at 2400 px on the long edge.

## `qualitative/`

Panels behind the quantitative claims. Each file is rendered at 150 dpi and
capped at 2000 px wide.

| File | Content |
|---|---|
| `fig_igr_weightmap24.png` | Regional weighting in IGR: the restoration region and the pasted-instance region carry three times the reconstruction weight before unit-mean normalization. |
| `fig_mechanism.png` | Controlled restoration analysis. Retention of an injected duplicate drops from 92--99% (Standard SFT) to 23--42% after 50 IGR steps across corruption strengths $\alpha$ and $\sigma$; the directional cosine follows. |
| `fig_persistence.png` | Persistence length and cumulative onset of occlusion-aware count violations across models. |
| `fig_mlr_occlusion_montage.png` | One rollout with target detections and robot occlusion across the sampled timestamps. |
| `fig_mlr_count_example.png` | MLR counting examples: red marks a retained persistent violation, green an under-count excluded by the occlusion check. |
| `fig_qual_dup1.png`, `fig_qual_dup2.png` | Cross-frame target consistency. Standard SFT and IGR-only duplicate, drift or deform the target; EVEWorld preserves its identity. |
| `570_{sft,igr,eve}_mlr.png` | Per-timestamp audit of task 570 over the 24 fixed MLR timestamps: instance inconsistency, cross-frame inconsistency, physical consistency. |
| `637_{sft,igr,eve}_mlr.png` | Same audit for task 637. |

## `site/`

Media for the project page: `figures/` holds the downscaled web variants of the
paper figures, `videos/hero` the looped banner, `videos/gr1` and `videos/wa` the
qualitative comparisons (AgiBot-GR1 and WorldArena rollouts), and
`videos/posters/` the first-frame stills that keep the grid layout from jumping
while clips load.
