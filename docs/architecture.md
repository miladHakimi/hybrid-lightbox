# Minimum viable architecture

The dependency direction is deliberately one-way:

```text
Qt UI / CLI -> project model -> optics -> geometry/STL
```

- `model.py` owns RGB pixels, boolean image-space masks, layer settings, and project persistence.
- `optics.py` maps normalized luminance to physical millimetres and evaluates combined transmission.
- `geometry.py` converts occupied pixels into an exposed voxel-union surface in millimetres. Watertightness checks remain available for diagnostics, but a failed check does not block export.
- `cli.py` is the headless reproducibility path; `app.py` is only interaction and display.
- `ai.py` is an optional adapter around local Depth Anything V2 and SAM 2 inference. Its pure depth/segment assignment functions remain testable without PyTorch.

Coordinates are explicit: masks use `(row, column)` pixels, imported and AI imagery defaults to a 768 px cap, and export resamples masks/thickness to an independent physical mesh spacing (0.12 mm by default). Sampling below nozzle width preserves smooth curves and narrow gaps for the slicer. Maximum-detail defaults disable print smoothing, sparse-feature removal, and automatic hole filling. Exported X/Y and per-layer Z are millimetres. Separate STL files share the same X/Y origin. Assembly spacing is project metadata and does not offset printable geometry.

## Important assumptions

The display preview is a Beer–Lambert approximation, not a promise of printed brightness. Optimized rear light paths assume a fixed 0.4 mm nozzle and expand cutouts to at least that width. Luminance-to-depth detail remains nozzle-resolved, while the STL XY sampling grid may be finer to retain boundary shape. The default structural mode gives every floating island one 3.2 mm-wide, 1.6 mm-deep connection along a weighted route: existing material is cheapest, regions occluded by foreground layers are strongly preferred, and visible empty regions are expensive. A solid 0.6 mm backing membrane and legacy horizontal bridges remain optional. Rear openings stay optically open in the default hidden-bridge mode.
