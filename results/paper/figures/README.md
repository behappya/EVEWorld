# Figures

The released images are rendered from the camera-ready vector figures, so a figure can be
regenerated at another resolution without keeping any intermediate PDF in the repository.
The sources live in the paper source tree (`paper/figures/`, kept local while the paper is
under review); the releases live under `assets/`, where the root `.gitignore` re-allows
`*.png`, and the project page reads the vector copies from `assets/site/figures/`.

## Releases

Each release is one camera-ready figure, kept at the stem it was exported from:

- `figures/fig1.pdf` → `assets/comparison.png` — the component contrast: Standard SFT, IGR
  only and EVEWorld on one clip.
- `figures/fig2.pdf` → `assets/teaser.png` — teaser: the two DreamGen tasks with per-rollout
  Model Laziness annotations, Standard SFT against EVEWorld.
- `figures/fig3.pdf` → `assets/method_overview.png` — overall architecture: IGR builds
  count-perturbed supervision pairs, TIA aligns the target at the probed layer.
- `figures/fig_igr_weightmap24.pdf` → `assets/qualitative/fig_igr_weightmap24.png` —
  regional weighting in IGR and the restoration region it produces.
- `figures/fig_mechanism.pdf` → `assets/qualitative/fig_mechanism.png` — the controlled
  restoration analysis behind IGR.
- `figures/fig_persistence.pdf` → `assets/qualitative/fig_persistence.png` — persistence
  length and cumulative onset of count violations across models.
- `figures/fig_mlr_occlusion_montage.pdf` → `assets/qualitative/fig_mlr_occlusion_montage.png`
  — one rollout with target detections and robot occlusion over the sampled timestamps.
- `figures/fig_mlr_count_example.pdf` → `assets/qualitative/fig_mlr_count_example.png` — a
  worked count violation of the MLR rule.
- `figures/fig_qual_dup{1,2}.pdf` → `assets/qualitative/fig_qual_dup{1,2}.png` — the
  duplication failure cases of the appendix.
- `figures/{570,637}_{sft,igr,eve}_mlr.pdf` → `assets/qualitative/{570,637}_*_mlr.png` — the
  two qualitative clips, Standard SFT against IGR against EVEWorld.

The two top-level images are rendered at 200 dpi and capped at 2400 px on the long edge, the
qualitative panels at 150 dpi and capped at 2000 px, so a release is readable at full width
on the project page without shipping a raster larger than the page needs.

## Re-rendering

The renderer is PyMuPDF for the page raster and Pillow for the cap, both already installed
in the evaluation environment:

```python
import fitz
from PIL import Image

RENDERS = [
    ("paper/figures/fig1.pdf", "assets/comparison.png", 200, 2400),
    ("paper/figures/fig2.pdf", "assets/teaser.png", 200, 2400),
    ("paper/figures/fig3.pdf", "assets/method_overview.png", 200, 2400),
    ("paper/figures/fig_igr_weightmap24.pdf",
     "assets/qualitative/fig_igr_weightmap24.png", 150, 2000),
]

for src, dst, dpi, max_edge in RENDERS:
    page = fitz.open(src)[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    img.save(dst)
```

Save the snippet next to the paper sources and run it with the evaluation interpreter; each
entry rewrites one release in place. The site copies under `assets/site/figures/` are
exported as vector SVG from the same sources (`page.get_svg_image(text_as_path=True)`) and
follow the paper figures whenever they change, so a figure only ever has one source of
truth.
