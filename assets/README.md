# Assets

Figures used by [the README](../README.md). The project-page media lives in the
`gh-pages` branch, so nothing here is served as part of the website. The root
`.gitignore` drops `*.png`, `*.jpg`, `*.pdf` and `*.mp4`; `assets/.gitignore`
re-allows the extensions this directory needs, so add new media types in that
file rather than at the repository root.

## Figures

| File | Content |
|---|---|
| `teaser.png` | Teaser. The two DreamGen tasks with per-rollout Model Laziness annotations, Standard SFT against EVEWorld. |
| `comparison.png` | Component contrast on one clip: Standard SFT, IGR only and EVEWorld side by side. It is the figure the README opens with. |
| `method_overview.png` | Overall architecture: IGR builds count-edited supervision pairs from clean clips, TIA aligns the target instance across frames at the probed layer $\ell^\star$. |
| `teaser.svg` | Vector redraw of the teaser, used where a resolution-independent figure is preferable. |

The PNGs are rendered from the camera-ready vector figures at 200 dpi and
capped at 2400 px on the long edge.
