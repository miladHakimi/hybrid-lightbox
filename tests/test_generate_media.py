import os
import subprocess
import sys
import textwrap

from PIL import Image

from scripts.make_sample import make_sample


def test_generate_media_creates_screenshots_and_animated_gif(tmp_path):
    sample = make_sample(tmp_path / "sample.png")
    output = tmp_path / "media"
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from scripts.generate_media import generate_media

        generate_media(Path({str(sample)!r}), Path({str(output)!r}))
        os._exit(0)
        """
    )
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-c", script], env=environment)
    assert result.returncode == 0
    paths = sorted(output.iterdir())

    assert [path.name for path in paths] == [
        "01-import.png",
        "02-segmentation-review.png",
        "03-depth-map.png",
        "04-layer-assignment.png",
        "05-mask-editing.png",
        "06-structure-review.png",
        "07-backlit-preview.png",
        "08-export-ready.png",
        "workflow.gif",
    ]
    for path in paths:
        assert path.stat().st_size > 0
    with Image.open(paths[-1]) as animation:
        assert animation.is_animated
        assert animation.n_frames == 9
        assert animation.size == (1440, 900)
