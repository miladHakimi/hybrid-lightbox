"""Render reproducible README screenshots and an animated workflow preview."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from lightbox.app import MainWindow
from lightbox.model import Project


PHASES = (
    ("Image", "01-import.png", "Phase 1/8 — Source image imported"),
    ("Segmentations", "02-segmentation-review.png", "Phase 2/8 — Object segments ready for review"),
    ("Depth map", "03-depth-map.png", "Phase 3/8 — Relative depth estimated locally"),
    ("Layer segments", "04-layer-assignment.png", "Phase 4/8 — Segments assigned to printable layers"),
    ("Selected mask", "05-mask-editing.png", "Phase 5/8 — Layer mask ready for brush refinement"),
    ("Connector view", "06-structure-review.png", "Phase 6/8 — Structure and connector paths reviewed"),
    ("Backlit preview", "07-backlit-preview.png", "Phase 7/8 — Backlit transmission preview"),
    ("Backlit preview", "08-export-ready.png", "Phase 8/8 — Validation complete; design ready to export"),
)


def demonstration_project(sample_path: Path) -> Project:
    """Build deterministic stand-ins for each local-AI workflow phase."""
    project = Project.from_image(sample_path, size=None, layer_count=3)
    image = project.image
    circle = np.all(image == (55, 65, 80), axis=2)
    eye = np.all(image == (225, 220, 200), axis=2)
    beam = np.all(image == (125, 140, 150), axis=2)
    base = np.all(image == (25, 30, 40), axis=2)
    project.segment_masks = [circle | eye, beam, base]
    yy, xx = np.indices(image.shape[:2])
    project.depth_map = ((xx / max(1, image.shape[1] - 1)) * .65 +
                         (yy / max(1, image.shape[0] - 1)) * .35).astype(np.float32)
    project.layers[0].mask = circle | eye
    project.layers[1].mask = beam
    project.layers[2].mask = base | circle | beam
    project._store_active_source()
    return project


def generate_media(sample_path: Path, output_dir: Path) -> list[Path]:
    """Render every app workflow phase and an animated overview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Hybrid Lightbox Designer")
    window = MainWindow()
    window.resize(1440, 900)
    window.set_project(demonstration_project(sample_path))
    window.show()
    app.processEvents()

    screenshots: list[Path] = []
    for view, filename, status in PHASES:
        window.view_box.setCurrentText(view)
        window.refresh()
        window.statusBar().showMessage(status)
        app.processEvents()
        destination = output_dir / filename
        if not window.grab().save(str(destination), "PNG"):
            raise RuntimeError(f"Could not save {destination}")
        screenshots.append(destination)

    frames = [Image.open(path).convert("RGB") for path in screenshots]
    gif_path = output_dir / "workflow.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:] + [frames[0]],
        duration=[1300, 1500, 1300, 1500, 1500, 1700, 2200, 1800, 700],
        loop=0,
        optimize=True,
    )
    for frame in frames:
        frame.close()

    window.close()
    QCoreApplication.processEvents()
    return screenshots + [gif_path]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for path in generate_media(root / "samples" / "sample.png", root / "docs" / "media"):
        print(path.relative_to(root))
