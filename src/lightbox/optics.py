from __future__ import annotations
import numpy as np
from PIL import Image
from .model import Project, LayerMode, luminance


# The light-path optimizer targets the project's standard FDM setup.  Keeping
# this explicit prevents image-resolution-sized cutouts that a slicer would
# discard when using the assumed nozzle.
OPTIMIZER_NOZZLE_DIAMETER_MM = 0.4


def _minimum_width_mask(mask: np.ndarray, pixel_mm: float, width_mm: float) -> np.ndarray:
    """Expand selected cells to at least ``width_mm`` on the sampled grid."""
    if pixel_mm <= 0:
        raise ValueError("Pixel size must be positive")
    radius = max(0, int(np.ceil((width_mm / pixel_mm - 1.0) / 2.0)))
    source = np.asarray(mask, dtype=bool)
    if radius == 0 or not source.any():
        return source.copy()
    padded = np.pad(source, radius, constant_values=False)
    offsets = ((dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if dy * dy + dx * dx <= radius * radius)
    return np.logical_or.reduce([
        padded[radius + dy:radius + dy + source.shape[0],
               radius + dx:radius + dx + source.shape[1]]
        for dy, dx in offsets
    ])


def nozzle_resolved_values(values: np.ndarray, pixel_mm: float) -> np.ndarray:
    """Limit an image-space field to the XY detail reproducible by the nozzle."""
    source = np.asarray(values, dtype=np.float32)
    if pixel_mm <= 0:
        raise ValueError("Pixel size must be positive")
    if pixel_mm >= OPTIMIZER_NOZZLE_DIAMETER_MM:
        return source.copy()
    height, width = source.shape
    physical_width = width * pixel_mm
    physical_height = height * pixel_mm
    target_width = max(1, int(np.floor(physical_width / OPTIMIZER_NOZZLE_DIAMETER_MM)))
    target_height = max(1, int(np.floor(physical_height / OPTIMIZER_NOZZLE_DIAMETER_MM)))
    if (target_height, target_width) == source.shape:
        return source.copy()
    image = Image.fromarray(source, mode="F")
    sampled = image.resize((target_width, target_height), Image.Resampling.BOX)
    return np.asarray(sampled.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def thickness_map(luminance_map: np.ndarray, minimum: float, maximum: float,
                  gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    """Map [0,1] luminance to thickness; dark pixels are thicker by default."""
    y = np.clip(luminance_map, 0.0, 1.0)
    if invert:
        y = 1.0 - y
    darkness = np.power(1.0 - y, gamma)
    return minimum + darkness * (maximum - minimum)


def layer_thickness(project: Project, index: int) -> np.ndarray:
    layer = project.layers[index]
    mat = layer.material; mat.validate()
    lum = nozzle_resolved_values(luminance(project.output_image(index)), project.pixel_mm)
    if layer.mode == LayerMode.SILHOUETTE:
        values = np.full(lum.shape, mat.maximum_mm, dtype=np.float32)
    else:
        values = thickness_map(lum, mat.minimum_mm, mat.maximum_mm, mat.gamma, mat.invert)
    return np.where(project.output_mask(index) & layer.enabled, values, 0.0).astype(np.float32)


def combined_transmission(project: Project) -> np.ndarray:
    optical_depth = np.zeros(project.image.shape[:2], dtype=np.float32)
    for group in project.physical_layer_groups():
        group_depth = np.zeros(project.image.shape[:2], dtype=np.float32)
        for i in group:
            layer = project.layers[i]
            if layer.enabled:
                thickness=layer_thickness(project,i)
                if project.structure_mode=="backing": thickness=np.where(project.output_mask(i),thickness,project.backing_thickness_mm)
                group_depth = np.maximum(group_depth, layer.material.attenuation_per_mm * thickness)
        optical_depth += group_depth
    return np.exp(-optical_depth)


def optimize_rear_openings(project: Project, foreground_index: int = 0) -> int:
    """Open rear layers with light paths printable by a fixed 0.4 mm nozzle."""
    foreground = project.layers[foreground_index]
    if foreground.mode == LayerMode.SILHOUETTE:
        return 0
    region = _minimum_width_mask(
        project.output_mask(foreground_index), project.pixel_mm, OPTIMIZER_NOZZLE_DIAMETER_MM)
    changed = 0
    for rear_index, rear in enumerate(project.layers[foreground_index + 1:], foreground_index + 1):
        # Optimization is destructive and therefore only safe in source space
        # when the layer has no output transform.
        if rear.offset_x or rear.offset_y or rear.output_scale_x != 1.0 or rear.output_scale_y != 1.0 or rear.rotation_degrees:
            continue
        overlap = project.output_mask(rear_index) & region
        changed += int(overlap.sum())
        rear.mask[overlap] = False
    return changed


def preview_u8(project: Project) -> np.ndarray:
    # White diffuser luminance modulated by predicted transmission.
    return np.clip(combined_transmission(project) * 255.0, 0, 255).astype(np.uint8)
