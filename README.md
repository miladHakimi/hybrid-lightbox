# Hybrid Lightbox Designer

An offline macOS-first Phase 1 prototype for turning an image into three aligned hybrid lithophane/shadow-box STL layers. The image pipeline, optics, geometry, persistence, CLI, and Qt UI are separate modules.

## What works

- JPEG, PNG, and WebP import at up to 768 px with two or more deterministic starter layers.
- Multi-image projects with per-image cached segmentation lists. Every physical layer records its source image, source-space mask, and independent output-space X/Y, width, and height transforms.
- Adjacent logical layers can be linked into a single physical output layer. Grey link controls keep layers separate; green links fuse the transformed masks into one STL while preserving independent editing and transforms.
- AI object-segmentation review before layer assignment. Refine any proposed segment or physical layer with a brush, or use the magic wand to select a connected color area with adjustable sensitivity and inverse deselection.
- Thirty-step undo/redo history for brush strokes, wand selections, crops, segment application, automatic assignment, and other destructive mask edits.
- Custom keybindings for undo/redo, selecting tools, changing canvas views, and adding or erasing selected segments. Open **Edit → Keybindings…** to assign them; undo/redo start with their standard shortcuts and changes persist across launches.
- Replace, add, or erase an edited segment on the selected physical layer; layer masks may overlap.
- Silhouette, lithophane, and hybrid layer behavior with thickness, gamma, and attenuation controls.
- Occlusion-aware rear opening: material behind a selected foreground lithophane region is removed.
- Transmission preview using `T = exp(-sum(mu_i * t_i))`.
- Editable `.hbox` projects (a documented ZIP container with JSON, PNG, and NumPy masks).
- Automatic workspace recovery after restarting the app, plus explicit Save, Load, New, and Reset actions.
- One aligned, millimetre-scaled binary STL per layer, a combined multi-object 3MF model, and a PNG backlight preview. Watertightness is reported as a diagnostic but does not block export.
- Occlusion-aware bridges connect islands using short printable paths that preferentially sit behind foreground silhouettes. Optional solid-backing and legacy bridge modes remain available.
- Every layer has a reinforced 1 mm outer border that is 3 mm thick by default.
- Print smoothing rounds noisy mask/segment boundaries and softens abrupt thickness changes before STL generation (0.3 mm by default, with 0.1 mm thickness quantization).

This alpha does not claim calibrated physical color or a photorealistic 3D renderer.

## Setup and launch (macOS)

Python 3.11–3.13 is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
hybrid-lightbox
```

For automatic Phase 2 depth and object segmentation, install the optional local-AI stack:

```bash
python -m pip install -e '.[ai]'
```

The first AI run downloads the Apache-2.0 Depth Anything V2 Small and the selected SAM 2.1 checkpoint. **Accurate (Large)** is the default; **Base (Base+)** and Fast reduce memory and runtime. Models run locally after download; no paid API is used. In the app, click **Segment image** to see SAM's proposed objects over the source image. Select a segment to inspect it, left-drag to add pixels, and right-drag to erase. Choose **Auto-assign to layers** for depth-based layering, or select a physical layer and use **Replace**, **Add**, or **Erase** for manual assignment. Selecting a layer displays its constituent segments in color. Segment edits are included in saved `.hbox` projects.

On macOS, local AI uses CPU inference by default to avoid native Metal command-buffer crashes that can abort the entire application before Python can recover. Advanced users can explicitly opt back into Apple GPU inference with `LIGHTBOX_AI_DEVICE=mps hybrid-lightbox`; `LIGHTBOX_AI_DEVICE=cpu` restores the stable default.

For the optional HQ-SAM 2 Large backend, which favors finer object boundaries, install its native package and select **HQ (HQ-SAM 2 Large)** in the app:

```bash
SAM2_BUILD_CUDA=0 python -m pip install -e '.[hq-ai]'
```

`SAM2_BUILD_CUDA=0` avoids attempting to compile the optional CUDA extension on macOS. The first HQ run downloads its large checkpoint and can use substantially more memory than the regular modes.

Open an image and use **Add image** to bring more source images into the project. Selecting a source shows only that image and its cached segments in the left panel; masks belonging to other images are never overlaid in Source Image view. **Remove** deletes the selected source after confirmation. Layers tied to a removed source have their masks cleared so unrelated image coordinates are never reused accidentally. Applying a segment or painting a layer binds that layer to the selected source. Use **Output X**, **Output Y**, **Output width**, and **Output height** in layer settings to move or resize the masked content on the final canvas without changing the mask on its source image. Use **Subtract** beside the brush to choose another layer and remove its mask from the current layer. Use **Add layer…** to choose any other layer and add its mask to the current layer. Use **Fill gaps** to replace the selected layer's output with the full final canvas minus everything covered by the transformed outputs of the layers before it, then enter an extra pixel margin to overlap the earlier outputs and prevent seams. Select its material behavior, optimize rear openings, inspect the backlit preview, validate, then export. Import the STL files together in Bambu Studio without changing their X/Y position.

Cropping is source-local: it crops only the selected image, that image's cached segments and depth map, and logical-layer masks tied to that image. The cropped content is fitted back onto the unchanged output canvas, so other source images and their layer masks are untouched.

Choose **Transform** from the canvas tool menu to edit the selected logical layer with the mouse. Drag inside its green bounding box to move it, or drag the lower-right handle to resize its width and height. The numeric output transform controls remain synchronized for precise adjustments.

Choose **Pen** to draw independent material directly on the final output canvas. Pen strokes are stored separately from the source-image mask, so source cropping and piece transforms do not move or resize them. Right-drag erases only pen strokes. Pen output participates in layer linking, structural previews, and STL export.

The current workspace is saved automatically after edits and when the app closes, then restored on the next launch. Use **Save** to keep a portable `.hbox` copy, **Load** to reopen one, **New** to start from another image, or **File → Reset workspace** to clear the current and automatically restored state.

## Reproducible CLI demonstration

```bash
python scripts/make_sample.py
lightbox-cli generate samples/sample.png --optimize -o demo_output
```

The output contains `layer_1.stl` through `layer_3.stl`, a combined `model.3mf` with each layer as a separate aligned object, `backlit_preview.png`, and `project.hbox`.

Inspect an image without writing geometry, or recursively test a customer folder:

```bash
lightbox-cli inspect image.png --optimize
lightbox-cli batch /path/to/images --optimize -o batch_output
lightbox-cli generate image.png --ai --optimize -o ai_output
lightbox-cli generate image.png --ai --sam-points '0.25,0.60;0.50,0.35;0.75,0.60' -o character_layers
lightbox-cli generate image.png --ai --segmentation-quality accurate -o accurate_layers
lightbox-cli generate image.png --ai --segmentation-quality base -o base_layers
lightbox-cli generate image.png --ai --segmentation-quality hq -o hq_layers
lightbox-cli generate image.png --layers 5 --ai -o five_layers
lightbox-cli generate image.png --image-resolution 1024 --mesh-resolution 0.30 -o high_detail
lightbox-cli generate image.png --structure hidden_bridges -o hidden_bridges
lightbox-cli generate image.png --structure backing --backing-thickness 0.6 -o solid_layers
lightbox-cli generate image.png --frame-width 1 --frame-thickness 3 -o reinforced
lightbox-cli generate image.png --print-smoothing 0.3 -o smoothed
```

Small automatically generated masks are filtered by area. The desktop app exposes a **Min segment** threshold (100 px by default), and the CLI equivalent is `--min-segment-area PIXELS`. Set it to `0` to disable filtering.

## Tests

```bash
pytest
```

The suite covers luminance-to-thickness mapping, minimum thickness, physical scaling/alignment, Beer–Lambert transmission, cutout effects, component detection, mesh watertightness, STL serialization, and project round-tripping.

## Contributing and security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Please report security issues privately as described in [SECURITY.md](SECURITY.md), rather than in a public issue.

## License

Hybrid Lightbox Designer is available under the [MIT License](LICENSE).

## Design and research notes

See [docs/architecture.md](docs/architecture.md) and [docs/research.md](docs/research.md). No code from the reference projects is copied into this implementation.

For customer-image production, use the independent, maximum-five-round review
loop in [docs/iterative-review.md](docs/iterative-review.md). It requires a fresh
strict review of print integrity, semantic object grouping, layer registration,
and optical quality after every complete regeneration.
