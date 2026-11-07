from PIL import Image

from scripts.make_sample import make_sample


def test_make_sample_creates_deterministic_demo_image(tmp_path):
    first = make_sample(tmp_path / "first.png")
    second = make_sample(tmp_path / "second.png")

    with Image.open(first) as image:
        assert image.size == (96, 72)
        assert image.mode == "RGB"
    assert first.read_bytes() == second.read_bytes()
