# Reference implementation review

Reviewed on 2026-10-21 from the projects' public repositories. These are design references only; Phase 1 has no source-level dependency on them.

| Project | Observed implementation | License / reuse decision |
|---|---|---|
| My3DShadowBox | Python application centered on image processing and 3D shadow-box generation. Useful workflow reference, but its repository does not present a root license file. | Treat as all-rights-reserved; do not copy code. |
| Shadowbox Studio | Browser/PWA implementation with a build script, aimed at tonal-band layered acrylic output. | MIT; reusable in principle, but no code is incorporated into this independently implemented prototype. |
| Lithophane Studio | Browser and Python-launchable lithophane generator with Three.js preview, client-side image processing, watertight STL/OBJ export, and geometry tests. | MIT; suitable as a future algorithm reference, but no code incorporated. |
| Depth Anything V2 | PyTorch monocular relative/metric depth inference. The Small model is the practical optional candidate for a distributable Phase 2. | Code and Small weights are Apache-2.0; Base/Large/Giant weights are CC-BY-NC-4.0 and should not be a commercial default. |
| SAM 2 | PyTorch image/video segmentation with automatic mask generation and promptable predictors. Strong optional Phase 2 component, but substantially increases packaging size. | Apache-2.0; suitable as an optional local backend with notices. |

Maintenance snapshots are not product guarantees. Before bundling a future model, pin a revision and record the exact code and checkpoint licenses in third-party notices.

## Resulting decisions

Phase 1 uses NumPy/Pillow and an original exposed-voxel mesher. This keeps manual editing fully offline, avoids license ambiguity, and gives deterministic geometry. PySide6 provides the desktop shell. AI backends remain optional adapters for Phase 2 rather than assumptions embedded in the project format.
