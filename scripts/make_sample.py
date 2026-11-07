from pathlib import Path
from PIL import Image, ImageDraw


def make_sample(output: Path) -> Path:
    """Create a small, original geometric image for demos and tests."""
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 72), (235, 220, 185))
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 12, 62, 58), fill=(55, 65, 80))
    draw.ellipse((28, 22, 38, 32), fill=(225, 220, 200))
    draw.polygon([(52, 22), (86, 36), (52, 50)], fill=(125, 140, 150))
    draw.rectangle((0, 58, 96, 72), fill=(25, 30, 40))
    image.save(output)
    return output


if __name__ == "__main__":
    sample = Path(__file__).resolve().parents[1] / "samples" / "sample.png"
    print(make_sample(sample))
