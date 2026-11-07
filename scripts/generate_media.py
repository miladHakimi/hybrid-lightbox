"""Render reproducible README screenshots and an animated workflow preview."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from lightbox.app import MainWindow
from lightbox.model import Project


VIEWS = (
    ("Image", "editor-source.png"),
    ("Selected mask", "editor-layers.png"),
    ("Backlit preview", "backlit-preview.png"),
)


def generate_media(sample_path: Path, output_dir: Path) -> list[Path]:
    """Render three app screenshots and a GIF using the original demo image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Hybrid Lightbox Designer")
    window = MainWindow()
    window.resize(1440, 900)
    window.set_project(Project.from_image(sample_path, size=None, layer_count=3))
    window.show()
    app.processEvents()

    screenshots: list[Path] = []
    for view, filename in VIEWS:
        window.view_box.setCurrentText(view)
        window.refresh()
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
        append_images=frames[1:] + [frames[1]],
        duration=[1400, 1400, 2200, 800],
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
