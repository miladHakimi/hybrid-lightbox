from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

out=Path(__file__).resolve().parents[1]/"samples"; out.mkdir(exist_ok=True)
im=Image.new("RGB",(96,72),(235,220,185)); d=ImageDraw.Draw(im)
d.ellipse((18,12,62,58),fill=(55,65,80)); d.ellipse((28,22,38,32),fill=(225,220,200))
d.polygon([(52,22),(86,36),(52,50)],fill=(125,140,150)); d.rectangle((0,58,96,72),fill=(25,30,40))
im.save(out/"sample.png")
print(out/"sample.png")
