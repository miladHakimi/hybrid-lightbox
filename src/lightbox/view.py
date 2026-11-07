from __future__ import annotations

import numpy as np

from .geometry import add_frame_and_bridges, add_hidden_bridges


def automatic_view(current, proposed):
    """Keep an explicitly selected connector view during automatic UI changes."""
    return current if current == "Connector view" else proposed


def view_for_tool(current, previous, tool):
    """Keep transform/output coordinates separate from source editing views."""
    if tool in ("Transform", "Pen"):
        return "Selected mask"
    if current == "Selected mask":
        return "Selected segment" if previous == "Selected segment" else "Image"
    return current


def supports_drag_paint(tool_mode):
    """Tools that should emit paint continuously while a mouse button is held."""
    return tool_mode in ("brush", "pen")


def brush_cursor_diameter(radius, displayed_width, source_width):
    """Return the screen-space diameter of an image-space brush footprint."""
    if source_width <= 0 or displayed_width <= 0:
        return max(2, round(radius * 2))
    return max(2, round(radius * 2 * displayed_width / source_width))


def valid_segment_indices(indices, count):
    """Discard stale UI selections after switching image-local segment caches."""
    return sorted({index for index in indices if 0 <= index < count})


class ConnectorCache:
    """Cache connector masks until an editor mutation invalidates them."""
    def __init__(self, generator=None):
        self.generator = generator or connector_masks
        self.value = None

    def invalidate(self):
        self.value = None

    def get(self, project):
        if self.value is None:
            self.value = self.generator(project)
        return self.value


def mask_outline(mask):
    """Return the one-pixel inner edge of a boolean mask."""
    inside = np.asarray(mask, bool)
    interior = inside.copy()
    interior[1:, :] &= inside[:-1, :]
    interior[:-1, :] &= inside[1:, :]
    interior[:, 1:] &= inside[:, :-1]
    interior[:, :-1] &= inside[:, 1:]
    return inside & ~interior


def selection_overlay(image, mask):
    """Show a mask over its source image without hiding image detail."""
    array = np.asarray(image).copy()
    selected = np.asarray(mask, bool)
    if not selected.any():
        return array
    tint = np.array([139, 108, 255], np.float32)
    array[selected] = np.clip(
        array[selected].astype(np.float32) * .58 + tint * .42, 0, 255
    ).astype(np.uint8)
    array[mask_outline(selected)] = [220, 210, 255]
    return array


def image_view_preview(image, mask):
    """Render the source image with the editable layer mask always visible."""
    return selection_overlay(image, mask)


def source_image_preview(image, mask, mask_source_index, active_source_index):
    """Overlay a mask only when it belongs to the displayed source image."""
    if mask_source_index != active_source_index:
        return np.asarray(image).copy()
    return image_view_preview(image, mask)


def connector_masks(project):
    """Return image-space structural additions without changing layer masks."""
    additions = []
    source_masks = [project.output_mask(i) for i in range(len(project.layers))]
    for index, layer in enumerate(project.layers):
        source = source_masks[index] if layer.enabled else np.zeros_like(source_masks[index])
        if project.structure_mode == "backing":
            structural = np.ones_like(source)
        elif project.structure_mode == "hidden_bridges":
            foreground = (np.logical_or.reduce(source_masks[:index])
                          if index else np.zeros_like(source))
            structural = add_hidden_bridges(
                source, foreground, project.pixel_mm,
                frame_width_mm=project.frame_width_mm,
            )
        else:
            structural = add_frame_and_bridges(source)
        additions.append(structural & ~source)
    return additions


def connector_view_preview(image, layer_mask, connectors):
    """Render layer material and its generated connectors in distinct colors."""
    array = np.asarray(image).astype(np.float32) * .2
    mask = np.asarray(layer_mask, bool)
    added = np.asarray(connectors, bool)
    array[mask] = array[mask] * .25 + np.array([139, 108, 255]) * .75
    array[added] = array[added] * .15 + np.array([255, 166, 74]) * .85
    return np.clip(array, 0, 255).astype(np.uint8)
