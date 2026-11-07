from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import copy
import io
import json
import zipfile
from collections import deque

import numpy as np
from PIL import Image, ImageEnhance


class LayerMode(str, Enum):
    SILHOUETTE = "silhouette"
    LITHOPHANE = "lithophane"
    HYBRID = "hybrid"


def magic_wand_region(image: np.ndarray, x: int, y: int, sensitivity: int = 20) -> np.ndarray:
    """Return the connected color region around a seed pixel.

    Sensitivity is deliberately expressed as a friendly 0–100 UI value. The
    comparison happens in RGB space, while connectivity prevents similarly
    colored but separate objects elsewhere in the image from being selected.
    """
    pixels = np.asarray(image, dtype=np.int32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Magic wand requires an RGB image")
    height, width = pixels.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Magic wand seed is outside the image")
    sensitivity = min(100, max(0, int(sensitivity)))
    tolerance = 2.0 + sensitivity * 2.0
    delta = pixels[..., :3] - pixels[y, x, :3]
    eligible = np.sum(delta * delta, axis=2, dtype=np.int32) <= tolerance * tolerance
    selected = np.zeros((height, width), dtype=bool)
    selected[y, x] = True
    queue = deque([(y, x)])
    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < height and 0 <= nx < width and eligible[ny, nx] and not selected[ny, nx]:
                selected[ny, nx] = True
                queue.append((ny, nx))
    return selected


@dataclass
class Material:
    minimum_mm: float = 0.4
    maximum_mm: float = 1.6
    attenuation_per_mm: float = 0.75
    gamma: float = 1.0
    invert: bool = False

    def validate(self) -> None:
        if self.minimum_mm <= 0 or self.maximum_mm < self.minimum_mm:
            raise ValueError("Thickness range must be positive and ordered")
        if self.attenuation_per_mm <= 0 or self.gamma <= 0:
            raise ValueError("Attenuation and gamma must be positive")


@dataclass
class Layer:
    name: str
    mask: np.ndarray
    mode: LayerMode = LayerMode.HYBRID
    material: Material = field(default_factory=Material)
    enabled: bool = True
    source_index: int = 0
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0
    scale_x: float | None = None
    scale_y: float | None = None
    fuse_with_next: bool = False
    output_additions: np.ndarray | None = None
    rotation_degrees: float = 0.0
    output_subtractions: np.ndarray | None = None

    @property
    def output_scale_x(self) -> float:
        return self.scale if self.scale_x is None else self.scale_x

    @property
    def output_scale_y(self) -> float:
        return self.scale if self.scale_y is None else self.scale_y


@dataclass
class SourceImage:
    """An imported image and its cached, image-local AI results."""
    name: str
    image: np.ndarray
    segment_masks: list[np.ndarray] = field(default_factory=list)
    depth_map: np.ndarray | None = None


@dataclass
class Project:
    image: np.ndarray
    layers: list[Layer]
    width_mm: float = 120.0
    spacing_mm: float = 8.0
    brightness: float = 1.0
    contrast: float = 1.0
    depth_map: np.ndarray | None = None
    mesh_resolution_mm: float = 0.12
    structure_mode: str = "hidden_bridges"
    backing_thickness_mm: float = 0.6
    frame_width_mm: float = 2.0
    frame_thickness_mm: float = 3.0
    segment_masks: list[np.ndarray] = field(default_factory=list)
    print_smoothing_mm: float = 0.0
    sources: list[SourceImage] = field(default_factory=list)
    active_source_index: int = 0

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [SourceImage("Image 1", self.image, self.segment_masks, self.depth_map)]
        self.active_source_index = min(max(0, self.active_source_index), len(self.sources) - 1)
        canvas_shape = self.sources[0].image.shape[:2]
        for layer in self.layers:
            if layer.output_additions is None:
                layer.output_additions = np.zeros(canvas_shape, dtype=bool)
            if layer.output_subtractions is None:
                layer.output_subtractions = np.zeros(canvas_shape, dtype=bool)
        self._load_active_source()

    @property
    def active_source(self) -> SourceImage:
        return self.sources[self.active_source_index]

    def _store_active_source(self) -> None:
        if not self.sources:
            return
        source = self.sources[self.active_source_index]
        source.image = self.image
        source.segment_masks = self.segment_masks
        source.depth_map = self.depth_map

    def _load_active_source(self) -> None:
        source = self.sources[self.active_source_index]
        self.image = source.image
        self.segment_masks = source.segment_masks
        self.depth_map = source.depth_map

    def select_source(self, index: int) -> None:
        if not 0 <= index < len(self.sources):
            raise IndexError("Source image index is out of range")
        self._store_active_source()
        self.active_source_index = index
        self._load_active_source()

    def add_source(self, path: str | Path, size: int | None = None) -> int:
        """Import an image into the canvas size; its segmentation cache starts empty."""
        image = Image.open(path).convert("RGB")
        if size is not None:
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas_h, canvas_w = self.sources[0].image.shape[:2]
        image.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        fitted = Image.new("RGB", (canvas_w, canvas_h))
        fitted.paste(image, ((canvas_w - image.width) // 2, (canvas_h - image.height) // 2))
        self._store_active_source()
        self.sources.append(SourceImage(Path(path).name, np.asarray(fitted, dtype=np.uint8)))
        return len(self.sources) - 1

    def remove_source(self, index: int) -> list[int]:
        """Remove a source and clear logical layers whose masks belonged to it."""
        if len(self.sources) <= 1:
            raise ValueError("A project must contain at least one source image")
        if not 0 <= index < len(self.sources):
            raise IndexError("Source image index is out of range")
        self._store_active_source()
        del self.sources[index]
        fallback = min(index, len(self.sources) - 1)
        cleared = []
        for layer_index, layer in enumerate(self.layers):
            if layer.source_index == index:
                layer.mask = np.zeros(self.sources[fallback].image.shape[:2], dtype=bool)
                layer.source_index = fallback
                cleared.append(layer_index)
            elif layer.source_index > index:
                layer.source_index -= 1
        if self.active_source_index == index:
            self.active_source_index = fallback
        elif self.active_source_index > index:
            self.active_source_index -= 1
        self._load_active_source()
        return cleared

    def assign_layer_source(self, layer_index: int, source_index: int) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("Layer index is out of range")
        if not 0 <= source_index < len(self.sources):
            raise IndexError("Source image index is out of range")
        self.layers[layer_index].source_index = source_index

    def prepare_layer_for_source(self, layer_index: int, source_index: int) -> bool:
        """Bind a layer to one image, clearing masks from its previous image."""
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("Layer index is out of range")
        if not 0 <= source_index < len(self.sources):
            raise IndexError("Source image index is out of range")
        layer = self.layers[layer_index]
        if layer.source_index == source_index:
            return False
        layer.source_index = source_index
        layer.mask = np.zeros(self.sources[source_index].image.shape[:2], dtype=bool)
        return True

    def transform_layer(self, index: int, mode: str, delta_x: float, delta_y: float) -> None:
        """Apply a mouse-sized output-space movement, resize, or rotation delta."""
        if not 0 <= index < len(self.layers):
            raise IndexError("Layer index is out of range")
        layer = self.layers[index]
        if mode == "move":
            layer.offset_x += delta_x
            layer.offset_y += delta_y
            return
        if mode == "rotate":
            layer.rotation_degrees = (layer.rotation_degrees + delta_x + 180) % 360 - 180
            return
        if mode != "resize":
            raise ValueError("Transform mode must be move, resize, or rotate")
        ys, xs = np.where(layer.mask)
        if not len(xs):
            return
        layer.scale_x = max(.05, min(10, layer.output_scale_x + delta_x / max(1, xs.max() - xs.min() + 1)))
        layer.scale_y = max(.05, min(10, layer.output_scale_y + delta_y / max(1, ys.max() - ys.min() + 1)))

    def output_mask(self, index: int) -> np.ndarray:
        layer = self.layers[index]
        transformed = transform_array(layer.mask.astype(np.uint8) * 255, layer.offset_x,
                                      layer.offset_y, layer.output_scale_x,
                                      layer.output_scale_y, layer.rotation_degrees,
                                      nearest=True) > 127
        additions = (np.zeros_like(transformed) if layer.output_additions is None
                     else np.asarray(layer.output_additions, dtype=bool))
        subtractions = (np.zeros_like(transformed) if layer.output_subtractions is None
                        else np.asarray(layer.output_subtractions, dtype=bool))
        return (transformed | additions) & ~subtractions

    def output_image(self, index: int) -> np.ndarray:
        layer = self.layers[index]
        source = self.sources[min(layer.source_index, len(self.sources) - 1)].image
        return transform_array(source, layer.offset_x, layer.offset_y,
                               layer.output_scale_x, layer.output_scale_y,
                               layer.rotation_degrees)

    @classmethod
    def from_image(cls, path: str | Path, size: int | None = 768, layer_count: int = 3) -> "Project":
        if layer_count < 2: raise ValueError("Layer count must be at least 2")
        image = Image.open(path).convert("RGB")
        if size is not None:
            if size < 1: raise ValueError("Image resolution must be positive")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.uint8)
        lum = luminance(arr)
        # Deterministic overlapping starter masks. Overlap is intentional: a
        # foreground region may require an opening in material behind it.
        masks = [lum < (i + 1) / layer_count for i in range(layer_count - 1)]
        masks.append(np.ones_like(lum, dtype=bool))
        layers = [Layer(f"Layer {i + 1}", m.copy()) for i, m in enumerate(masks)]
        return cls(arr, layers, sources=[SourceImage(Path(path).name, arr)])

    def set_layer_count(self, count: int) -> None:
        if count < 2: raise ValueError("Layer count must be at least 2")
        if count == len(self.layers): return
        lum = luminance(self.adjusted_image())
        old = self.layers
        layers = []
        for i in range(count):
            mask = np.ones_like(lum, bool) if i == count - 1 else lum < (i + 1) / count
            if i < len(old):
                old[i].mask = mask; layers.append(old[i])
            else:
                template = old[0]
                material = Material(**vars(template.material))
                layers.append(Layer(f"Layer {i + 1}", mask, template.mode, material, template.enabled))
        self.layers = layers
        for i, layer in enumerate(self.layers): layer.name = f"Layer {i + 1}"

    def add_blank_layer(self) -> None:
        template = self.layers[0]
        material = Material(**vars(template.material))
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        self.layers.append(Layer(f"Layer {len(self.layers) + 1}", mask, template.mode, material,
                                 template.enabled, self.active_source_index,
                                 output_additions=np.zeros_like(mask)))

    def fill_layer_mask(self, index: int, source_index: int | None = None) -> int:
        """Select every pixel in a layer mask for the requested source image."""
        if not 0 <= index < len(self.layers):
            raise IndexError("Layer index is out of range")
        source_index = self.active_source_index if source_index is None else source_index
        if not 0 <= source_index < len(self.sources):
            raise IndexError("Source image index is out of range")
        self.prepare_layer_for_source(index, source_index)
        mask = self.layers[index].mask
        added = int(mask.size - np.count_nonzero(mask))
        mask[:] = True
        return added

    def remove_layer(self, index: int) -> None:
        if len(self.layers) <= 2: raise ValueError("A project must contain at least 2 layers")
        if not 0 <= index < len(self.layers): raise IndexError("Layer index is out of range")
        del self.layers[index]
        for i, layer in enumerate(self.layers): layer.name = f"Layer {i + 1}"

    def move_layer(self, source: int, destination: int) -> None:
        """Move a layer to a new front-to-back position."""
        if not 0 <= source < len(self.layers): raise IndexError("Source layer index is out of range")
        if not 0 <= destination < len(self.layers): raise IndexError("Destination layer index is out of range")
        if source == destination: return
        self.layers.insert(destination, self.layers.pop(source))
        for i, layer in enumerate(self.layers): layer.name = f"Layer {i + 1}"

    def toggle_layer_link(self, index: int) -> bool:
        """Toggle the physical link between a layer and the layer behind it."""
        if not 0 <= index < len(self.layers) - 1:
            raise IndexError("Only adjacent layers can be linked")
        self.layers[index].fuse_with_next = not self.layers[index].fuse_with_next
        return self.layers[index].fuse_with_next

    def update_layer_settings(self, index: int, mode: LayerMode, material: Material,
                              apply_to_all: bool = False) -> list[int]:
        """Update one layer, or copy the same independent settings to every layer."""
        if not 0 <= index < len(self.layers): raise IndexError("Layer index is out of range")
        material.validate()
        indices = list(range(len(self.layers))) if apply_to_all else [index]
        for target in indices:
            self.layers[target].mode = mode
            self.layers[target].material = copy.deepcopy(material)
        return indices

    def physical_layer_groups(self) -> list[list[int]]:
        """Return adjacent logical layers grouped into physical output layers."""
        groups: list[list[int]] = []
        for index in range(len(self.layers)):
            if not groups or not self.layers[index - 1].fuse_with_next:
                groups.append([])
            groups[-1].append(index)
        return groups

    def physical_mask(self, group: list[int]) -> np.ndarray:
        if not group:
            raise ValueError("A physical layer group cannot be empty")
        return np.logical_or.reduce([self.output_mask(index) for index in group])

    def subtract_layer(self, target: int, source: int) -> int:
        """Subtract the source's transformed output from the target's output."""
        if not 0 <= target < len(self.layers): raise IndexError("Target layer index is out of range")
        if not 0 <= source < len(self.layers): raise IndexError("Source layer index is out of range")
        if target == source: raise ValueError("A layer cannot be subtracted from itself")
        target_output = self.output_mask(target)
        overlap = target_output & self.output_mask(source)
        layer = self.layers[target]
        if layer.output_subtractions is None:
            layer.output_subtractions = np.zeros_like(target_output)
        layer.output_subtractions |= overlap
        return int(overlap.sum())

    def add_layer(self, target: int, source: int) -> int:
        """Add every pixel in the source layer to the target layer."""
        if not 0 <= target < len(self.layers): raise IndexError("Target layer index is out of range")
        if not 0 <= source < len(self.layers): raise IndexError("Source layer index is out of range")
        if target == source: raise ValueError("A layer cannot be added to itself")
        target_mask = self.layers[target].mask
        before = int(target_mask.sum())
        target_mask |= self.layers[source].mask
        return int(target_mask.sum()) - before

    def fill_uncovered(self, target: int, overlap_pixels: int = 0) -> int:
        """Replace a layer's output with final-canvas gaps plus an overlap margin."""
        if not 0 <= target < len(self.layers): raise IndexError("Target layer index is out of range")
        if overlap_pixels < 0: raise ValueError("Overlap pixels cannot be negative")
        mask = np.ones(self.image.shape[:2], dtype=bool)
        for index in range(target):
            mask &= ~self.output_mask(index)
        if overlap_pixels:
            from .geometry import dilate_mask
            mask = dilate_mask(mask, int(overlap_pixels))
        # Gaps are output-space geometry. Storing them as output additions keeps
        # the result exact even when the target itself is translated, scaled, or
        # clipped by the output canvas.
        self.layers[target].mask = np.zeros_like(self.layers[target].mask, dtype=bool)
        self.layers[target].output_additions = mask
        self.layers[target].output_subtractions = np.zeros_like(mask)
        return int(mask.sum())

    def crop(self, left: int, top: int, right: int, bottom: int) -> None:
        """Crop only the active source while preserving the shared output canvas."""
        height, width = self.image.shape[:2]
        left, right = sorted((max(0, int(left)), min(width, int(right))))
        top, bottom = sorted((max(0, int(top)), min(height, int(bottom))))
        if right - left < 2 or bottom - top < 2:
            raise ValueError("Crop must be at least 2 × 2 pixels")
        self._store_active_source()
        source = self.active_source
        source.image = crop_resize(source.image, left, top, right, bottom, (height, width))
        source.segment_masks = [crop_resize(mask, left, top, right, bottom, (height, width), nearest=True)
                                for mask in source.segment_masks]
        if source.depth_map is not None:
            source.depth_map = crop_resize(source.depth_map, left, top, right, bottom, (height, width))
        for layer in self.layers:
            if layer.source_index == self.active_source_index:
                # Cropping then fitting the selected region back to the source
                # canvas is itself a source-space transform. Compose it with
                # the layer's output transform so already-positioned content
                # does not jump or expand on the final canvas.
                old_scale_x, old_scale_y = layer.output_scale_x, layer.output_scale_y
                crop_center_x = (left + right) / 2.0
                crop_center_y = (top + bottom) / 2.0
                layer.offset_x += old_scale_x * (crop_center_x - width / 2.0)
                layer.offset_y += old_scale_y * (crop_center_y - height / 2.0)
                layer.scale_x = old_scale_x * (right - left) / width
                layer.scale_y = old_scale_y * (bottom - top) / height
                layer.mask = crop_resize(layer.mask, left, top, right, bottom,
                                         (height, width), nearest=True)
        self._load_active_source()

    @property
    def height_mm(self) -> float:
        h, w = self.image.shape[:2]
        return self.width_mm * h / w

    @property
    def pixel_mm(self) -> float:
        return self.width_mm / self.image.shape[1]

    def remove_thin_features_from_all_masks(self, minimum_width_mm: float = 0.4) -> int:
        """Clean every physical layer and generated segment mask in place."""
        from .geometry import remove_thin_features_from_masks
        masks = ([layer.mask for layer in self.layers] +
                 [layer.output_additions for layer in self.layers] + list(self.segment_masks))
        before = sum(int(mask.sum()) for mask in masks)
        cleaned = remove_thin_features_from_masks(masks, self.pixel_mm, minimum_width_mm)
        layer_count = len(self.layers)
        for layer, mask in zip(self.layers, cleaned[:layer_count]):
            layer.mask = mask
        for layer, mask in zip(self.layers, cleaned[layer_count:layer_count * 2]):
            layer.output_additions = mask
        self.segment_masks = cleaned[layer_count * 2:]
        return before - sum(int(mask.sum()) for mask in cleaned)

    def adjusted_image(self) -> np.ndarray:
        im = Image.fromarray(self.image)
        im = ImageEnhance.Brightness(im).enhance(self.brightness)
        im = ImageEnhance.Contrast(im).enhance(self.contrast)
        return np.asarray(im)

    def save(self, path: str | Path) -> None:
        self._store_active_source()
        manifest = {
            "version": 2, "width_mm": self.width_mm, "spacing_mm": self.spacing_mm,
            "brightness": self.brightness, "contrast": self.contrast,
            "mesh_resolution_mm": self.mesh_resolution_mm,
            "structure_mode": self.structure_mode,
            "backing_thickness_mm": self.backing_thickness_mm,
            "frame_width_mm": self.frame_width_mm,
            "frame_thickness_mm": self.frame_thickness_mm,
            "print_smoothing_mm": self.print_smoothing_mm,
            "active_source_index": self.active_source_index,
            "sources": [{"name": source.name} for source in self.sources],
            "layers": [{"name": x.name, "mode": x.mode.value, "enabled": x.enabled,
                        "source_index": x.source_index, "offset_x": x.offset_x,
                        "offset_y": x.offset_y, "scale": x.scale,
                        "scale_x": x.output_scale_x, "scale_y": x.output_scale_y,
                        "rotation_degrees": x.rotation_degrees,
                        "fuse_with_next": x.fuse_with_next,
                        "material": vars(x.material)} for x in self.layers],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.json", json.dumps(manifest, indent=2))
            b = io.BytesIO(); Image.fromarray(self.image).save(b, "PNG")
            zf.writestr("image.png", b.getvalue())
            layer_arrays = {f"mask{i}": x.mask for i, x in enumerate(self.layers)}
            layer_arrays.update({f"addition{i}": (x.output_additions if x.output_additions is not None else np.zeros(self.image.shape[:2], bool)) for i, x in enumerate(self.layers)})
            layer_arrays.update({f"subtraction{i}": (x.output_subtractions if x.output_subtractions is not None else np.zeros(self.image.shape[:2], bool)) for i, x in enumerate(self.layers)})
            b = io.BytesIO(); np.savez_compressed(b, **layer_arrays)
            zf.writestr("masks.npz", b.getvalue())
            if self.depth_map is not None:
                b = io.BytesIO(); np.save(b, self.depth_map.astype(np.float32)); zf.writestr("depth.npy", b.getvalue())
            if self.segment_masks:
                b = io.BytesIO(); np.savez_compressed(b, **{f"segment{i}": x for i, x in enumerate(self.segment_masks)})
                zf.writestr("segments.npz", b.getvalue())
            for i, source in enumerate(self.sources):
                b = io.BytesIO(); Image.fromarray(source.image).save(b, "PNG")
                zf.writestr(f"sources/{i}/image.png", b.getvalue())
                if source.segment_masks:
                    b = io.BytesIO(); np.savez_compressed(b, **{f"segment{j}": mask for j, mask in enumerate(source.segment_masks)})
                    zf.writestr(f"sources/{i}/segments.npz", b.getvalue())
                if source.depth_map is not None:
                    b = io.BytesIO(); np.save(b, source.depth_map.astype(np.float32))
                    zf.writestr(f"sources/{i}/depth.npy", b.getvalue())

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("project.json"))
            image = np.asarray(Image.open(io.BytesIO(zf.read("image.png"))).convert("RGB"))
            masks = np.load(io.BytesIO(zf.read("masks.npz")))
            layers = []
            for i, item in enumerate(meta["layers"]):
                addition = masks[f"addition{i}"].astype(bool) if f"addition{i}" in masks.files else None
                subtraction = masks[f"subtraction{i}"].astype(bool) if f"subtraction{i}" in masks.files else None
                layers.append(Layer(item["name"], masks[f"mask{i}"].astype(bool),
                                    LayerMode(item["mode"]), Material(**item["material"]), item["enabled"],
                                    item.get("source_index", 0), item.get("offset_x", 0.0),
                                    item.get("offset_y", 0.0), item.get("scale", 1.0),
                                    item.get("scale_x"), item.get("scale_y"),
                                    item.get("fuse_with_next", False), addition,
                                    item.get("rotation_degrees", 0.0), subtraction))
            depth = np.load(io.BytesIO(zf.read("depth.npy"))) if "depth.npy" in zf.namelist() else None
            segments = []
            if "segments.npz" in zf.namelist():
                saved = np.load(io.BytesIO(zf.read("segments.npz")))
                segments = [saved[key].astype(bool) for key in sorted(saved.files, key=lambda key: int(key[7:]))]
            sources = []
            for i, source_meta in enumerate(meta.get("sources", [])):
                source_image = np.asarray(Image.open(io.BytesIO(zf.read(f"sources/{i}/image.png"))).convert("RGB"))
                source_segments = []
                segment_name = f"sources/{i}/segments.npz"
                if segment_name in zf.namelist():
                    saved = np.load(io.BytesIO(zf.read(segment_name)))
                    source_segments = [saved[key].astype(bool) for key in sorted(saved.files, key=lambda key: int(key[7:]))]
                depth_name = f"sources/{i}/depth.npy"
                source_depth = np.load(io.BytesIO(zf.read(depth_name))) if depth_name in zf.namelist() else None
                sources.append(SourceImage(source_meta.get("name", f"Image {i + 1}"), source_image, source_segments, source_depth))
        return cls(image, layers, meta["width_mm"], meta["spacing_mm"],
                   meta.get("brightness", 1.0), meta.get("contrast", 1.0), depth,
                   meta.get("mesh_resolution_mm", 0.12), meta.get("structure_mode", "hidden_bridges"),
                   meta.get("backing_thickness_mm", 0.6), meta.get("frame_width_mm", 1.0),
                   meta.get("frame_thickness_mm", 3.0), segments, meta.get("print_smoothing_mm", 0.0),
                   sources, meta.get("active_source_index", 0))


def transform_array(array: np.ndarray, offset_x: float, offset_y: float,
                    scale_x: float, scale_y: float | None = None,
                    rotation_degrees: float = 0.0,
                    nearest: bool = False) -> np.ndarray:
    """Render an image-local array into the fixed output canvas."""
    scale_y = scale_x if scale_y is None else scale_y
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Layer scales must be positive")
    source = np.asarray(array)
    height, width = source.shape[:2]
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(source)
    scaled = image.resize((max(1, round(width * scale_x)), max(1, round(height * scale_y))), mode)
    transformed = scaled.rotate(rotation_degrees, resample=mode, expand=True)
    if source.ndim == 2:
        canvas = Image.new(image.mode, (width, height), 0)
    else:
        canvas = Image.new(image.mode, (width, height), (0,) * source.shape[2])
    left = round((width - transformed.width) / 2 + offset_x)
    top = round((height - transformed.height) / 2 + offset_y)
    canvas.paste(transformed, (left, top))
    return np.asarray(canvas)


def crop_resize(array: np.ndarray, left: int, top: int, right: int, bottom: int,
                shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Crop an image-local asset and fit it back onto the unchanged canvas."""
    source = np.asarray(array)
    cropped = source[top:bottom, left:right]
    boolean = source.dtype == bool
    image = Image.fromarray(cropped.astype(np.uint8) * 255) if boolean else Image.fromarray(cropped)
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = np.asarray(image.resize((shape[1], shape[0]), resampling))
    return resized > 127 if boolean else resized.astype(source.dtype, copy=False)


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32) / 255.0
    return rgbf[..., 0] * 0.2126 + rgbf[..., 1] * 0.7152 + rgbf[..., 2] * 0.0722
