from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Callable, Sequence
import numpy as np
from PIL import Image


DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SAM_MODELS = {
    "fast": "facebook/sam2.1-hiera-tiny",
    "base": "facebook/sam2.1-hiera-base-plus",
    "balanced": "facebook/sam2.1-hiera-base-plus",
    "accurate": "facebook/sam2.1-hiera-large",
}


def select_ai_device(torch, requested: str | None = None,
                     platform_name: str | None = None):
    """Choose a stable inference device; Metal is explicit opt-in on macOS."""
    requested = (requested or os.environ.get("LIGHTBOX_AI_DEVICE", "cpu")).lower()
    platform_name = platform_name or sys.platform
    mps_available = (hasattr(torch.backends, "mps") and
                     torch.backends.mps.is_available())
    if requested == "mps":
        return "mps" if mps_available else -1
    if requested == "auto" and platform_name != "darwin" and mps_available:
        return "mps"
    return -1
SAM_MODEL = SAM_MODELS["accurate"]
HQ_SAM_REPOSITORY = "lkeab/hq-sam"
HQ_SAM_CHECKPOINT = "sam2.1_hq_hiera_large.pt"
SEGMENTATION_QUALITIES = (*SAM_MODELS, "hq")


class AIUnavailable(RuntimeError):
    pass


def normalize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    result = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(result[np.isfinite(result)], [1, 99])
    return np.clip((result - lo) / max(float(hi - lo), 1e-6), 0, 1)


def _erode(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Small dependency-free erosion used to avoid noisy depth at mask edges."""
    result = np.asarray(mask, bool).copy()
    for _ in range(iterations):
        if not result.any(): break
        padded = np.pad(result, 1, constant_values=False)
        result = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] &
                  padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return result


def _segment_depth(depth: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return a robust interior depth and its 10–90% spread."""
    interior = _erode(mask)
    # Thin objects can disappear under erosion; retain enough samples to make
    # their original mask a more reliable estimate than a handful of pixels.
    sample = interior if np.count_nonzero(interior) >= min(16, np.count_nonzero(mask)) else mask
    values = np.asarray(depth[sample], np.float32)
    values = values[np.isfinite(values)]
    if not len(values): return .5, 1.
    lo, median, hi = np.percentile(values, [10, 50, 90])
    return float(median), float(hi - lo)


def _depth_centers(depth: np.ndarray, layer_count: int) -> np.ndarray:
    """Find deterministic, populated depth planes with one-dimensional k-means."""
    values = np.asarray(depth, np.float32).ravel()
    values = values[np.isfinite(values)]
    if not len(values): return np.linspace(1, 0, layer_count, dtype=np.float32)
    # Cap work for large source images while sampling across the sorted
    # distribution (rather than according to scan order).
    values = np.sort(values)
    if len(values) > 8192:
        values = values[np.linspace(0, len(values) - 1, 8192).astype(int)]
    centers = np.quantile(values, (np.arange(layer_count) + .5) / layer_count)
    for _ in range(24):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.array([values[labels == i].mean() if np.any(labels == i) else centers[i]
                            for i in range(layer_count)])
        if np.allclose(updated, centers, atol=1e-5): break
        centers = updated
    return np.sort(centers)[::-1].astype(np.float32)  # front/near first


def _majority_smooth(labels: np.ndarray, fixed: np.ndarray, layer_count: int) -> np.ndarray:
    """Remove isolated depth-bin pixels without moving pixels owned by segments."""
    result = labels.copy()
    for _ in range(2):
        padded = np.pad(result, 1, mode="edge")
        neighbours = (padded[:-2, 1:-1], padded[2:, 1:-1],
                      padded[1:-1, :-2], padded[1:-1, 2:])
        counts = np.stack([sum(n == i for n in neighbours) for i in range(layer_count)])
        winner = np.argmax(counts, axis=0)
        strongest = np.max(counts, axis=0) >= 3
        change = ~fixed & strongest
        result[change] = winner[change]
    return result


def assign_depth_layers(depth: np.ndarray, segments: Sequence[np.ndarray] | None,
                        layer_count: int = 3) -> list[np.ndarray]:
    """Assign objects to adaptive depth planes and coherently fill uncovered areas."""
    if layer_count < 2: raise ValueError("At least two layers are required")
    depth = np.asarray(depth, np.float32)
    if depth.ndim != 2: raise ValueError("Depth map must be two-dimensional")
    centers = _depth_centers(depth, layer_count)
    # Per-pixel adaptive fallback. It also gives every pixel a deterministic
    # answer if SAM finds nothing useful.
    safe_depth = np.nan_to_num(depth, nan=float(np.median(centers)))
    labels = np.argmin(np.abs(safe_depth[..., None] - centers), axis=2).astype(np.int16)
    valid = []
    for mask in segments or ():
        mask = np.asarray(mask, bool)
        if mask.shape != depth.shape or not mask.any(): continue
        estimate, spread = _segment_depth(depth, mask)
        layer = int(np.argmin(np.abs(centers - estimate)))
        # The signature makes all tie-breaking independent of SAM's list order.
        ys, xs = np.nonzero(mask)
        signature = (round(estimate, 7), round(spread, 7), int(mask.sum()),
                     int(ys.min()), int(xs.min()), mask.tobytes())
        valid.append((mask, layer, signature))
    fixed = np.zeros(depth.shape, bool)
    if valid:
        # Resolve overlaps by local depth fit. A stable signature handles exact
        # ties, so permuting otherwise identical SAM output cannot change masks.
        order = sorted(range(len(valid)), key=lambda i: valid[i][2])
        best_cost = np.full(depth.shape, np.inf, np.float32)
        for original in order:
            mask, layer, _signature = valid[original]
            local_cost = np.abs(safe_depth[mask] - centers[layer])
            better = local_cost < best_cost[mask]
            if not np.any(better): continue
            locations = np.flatnonzero(mask)
            chosen = locations[better]
            labels.flat[chosen] = layer
            best_cost.flat[chosen] = local_cost[better]
        fixed = np.isfinite(best_cost)
    labels = _majority_smooth(labels, fixed, layer_count)
    return [labels == i for i in range(layer_count)]


def assign_prompted_layers(depth: np.ndarray, segments: Sequence[np.ndarray],
                           layer_count: int = 3) -> list[np.ndarray]:
    """Map user-prompted objects to layers by point order, then fill the scene by depth."""
    labels = np.minimum(((1.0 - depth) * layer_count).astype(np.int16), layer_count - 1)
    claimed = np.zeros(depth.shape, bool)
    for index, mask in enumerate(segments[:layer_count]):
        mask = np.asarray(mask, bool)
        region = mask & ~claimed
        labels[region] = index; claimed |= region
    return [labels == i for i in range(layer_count)]


def filter_segments(segments: Sequence[np.ndarray], shape: tuple[int, int],
                    minimum_area: int = 0) -> list[np.ndarray]:
    """Discard undersized/invalid masks and collapse near-duplicates."""
    if minimum_area < 0: raise ValueError("Minimum segment area cannot be negative")
    unique = []
    for mask in sorted(segments, key=lambda item: int(np.asarray(item, bool).sum()), reverse=True):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape or np.count_nonzero(mask) < max(1, minimum_area): continue
        duplicate = any(np.count_nonzero(mask & other) / max(np.count_nonzero(mask | other), 1) > .92
                        for other in unique)
        if not duplicate: unique.append(mask)
    return unique


@dataclass
class LocalAI:
    depth_model: str = DEPTH_MODEL
    segmentation_model: str = SAM_MODEL
    use_segmentation: bool = True
    segmentation_quality: str = "accurate"
    minimum_segment_area: int = 0

    def __post_init__(self):
        if self.segmentation_quality not in SEGMENTATION_QUALITIES:
            raise ValueError(f"Unknown segmentation quality: {self.segmentation_quality}")
        if self.minimum_segment_area < 0:
            raise ValueError("Minimum segment area cannot be negative")
        # An explicitly supplied model still wins; the default tracks quality.
        if self.segmentation_model == SAM_MODEL:
            self.segmentation_model = SAM_MODELS.get(self.segmentation_quality, SAM_MODEL)

    def _imports(self):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise AIUnavailable("Install the local AI extras with: pip install -e '.[ai]'") from exc
        return torch, pipeline

    def _segment_hq(self, image: Image.Image, torch, device,
                    points: Sequence[tuple[float, float]] | None,
                    notify: Callable[[str], None]) -> list[np.ndarray]:
        try:
            from huggingface_hub import hf_hub_download
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise AIUnavailable(
                "Install HQ-SAM 2 with: SAM2_BUILD_CUDA=0 python -m pip install -e '.[hq-ai]'"
            ) from exc
        model_device = device if device != -1 else "cpu"
        notify("Downloading HQ-SAM 2 Large…")
        checkpoint = hf_hub_download(repo_id=HQ_SAM_REPOSITORY, filename=HQ_SAM_CHECKPOINT)
        notify("Loading HQ-SAM 2 Large…")
        model = build_sam2("configs/sam2.1/sam2.1_hq_hiera_l.yaml", checkpoint,
                           device=model_device, apply_postprocessing=True)
        if points:
            predictor = SAM2ImagePredictor(model)
            predictor.set_image(image)
            segments = []
            for x, y in points:
                masks, scores, _logits = predictor.predict(
                    point_coords=np.asarray([[x * image.width, y * image.height]], np.float32),
                    point_labels=np.asarray([1], np.int32), multimask_output=True)
                segments.append(np.asarray(masks[int(np.argmax(scores))], dtype=bool))
            return segments
        notify("Segmenting objects with HQ-SAM 2…")
        generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=48, points_per_batch=16,
            pred_iou_thresh=.82, stability_score_thresh=.88, crop_n_layers=1,
            min_mask_region_area=self.minimum_segment_area)
        return [np.asarray(item["segmentation"], dtype=bool) for item in generator.generate(np.asarray(image))]

    def decompose(self, rgb: np.ndarray, layer_count: int = 3,
                  progress: Callable[[str], None] | None = None,
                  points: Sequence[tuple[float, float]] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        depth, segments = self.segment_image(rgb, progress, points)
        notify = progress or (lambda _message: None)
        notify("Assigning objects to depth layers…")
        masks = assign_prompted_layers(depth, segments, layer_count) if points and segments else assign_depth_layers(depth, segments, layer_count)
        return depth, masks

    def segment_image(self, rgb: np.ndarray,
                      progress: Callable[[str], None] | None = None,
                      points: Sequence[tuple[float, float]] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        """Estimate depth and return the editable object masks before layer assignment."""
        torch, pipeline = self._imports(); notify = progress or (lambda _message: None)
        device = select_ai_device(torch)
        image = Image.fromarray(rgb)
        if sys.platform == "darwin" and device == -1:
            notify("Using CPU inference for stable macOS segmentation…")
        notify("Loading Depth Anything V2 Small…")
        depth_pipe = pipeline("depth-estimation", model=self.depth_model, device=device)
        notify("Estimating relative depth…")
        raw = depth_pipe(image)["predicted_depth"]
        if hasattr(raw, "detach"): raw = raw.detach().cpu().numpy()
        depth = normalize_depth(np.squeeze(raw), rgb.shape[:2])
        segments = []
        if self.use_segmentation:
            if self.segmentation_quality == "hq":
                segments = self._segment_hq(image, torch, device, points, notify)
            elif points:
                notify(f"Segmenting prompted objects with SAM 2.1 {self.segmentation_quality.title()}…")
                from transformers import Sam2Model, Sam2Processor
                model_device = device if device != -1 else "cpu"
                model = Sam2Model.from_pretrained(self.segmentation_model).to(model_device)
                processor = Sam2Processor.from_pretrained(self.segmentation_model)
                pixel_points = [[[[x * image.width, y * image.height]] for x, y in points]]
                labels = [[[1] for _point in points]]
                inputs = processor(images=image, input_points=pixel_points, input_labels=labels, return_tensors="pt").to(model_device)
                # SAM proposes three masks per point. Keeping only its first
                # proposal can select a part instead of the complete character.
                with torch.inference_mode(): outputs = model(**inputs, multimask_output=True)
                raw_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
                scores = outputs.iou_scores.detach().cpu().numpy()[0]
                segments = []
                for candidates, candidate_scores in zip(raw_masks, scores):
                    best = int(np.argmax(candidate_scores))
                    segments.append(np.asarray(candidates[best], dtype=bool))
            else:
                notify(f"Loading SAM 2.1 {self.segmentation_quality.title()}…")
                mask_pipe = pipeline("mask-generation", model=self.segmentation_model, device=device)
                notify("Segmenting objects…")
                points_per_crop = 48 if self.segmentation_quality == "accurate" else 32
                output = mask_pipe(image, points_per_batch=16, points_per_crop=points_per_crop,
                                   pred_iou_thresh=.82, stability_score_thresh=.88,
                                   crop_n_layers=1)
                segments = []
                for mask in output.get("masks", []):
                    if hasattr(mask, "detach"): mask = mask.detach().cpu().numpy()
                    segments.append(np.asarray(mask, dtype=bool).squeeze())
        # Present useful masks first and remove near-duplicates produced by SAM.
        unique = filter_segments(segments, rgb.shape[:2], self.minimum_segment_area)
        notify(f"Found {len(unique)} editable segment{'s' if len(unique) != 1 else ''}…")
        return depth, unique


def apply_ai_decomposition(project, depth: np.ndarray, masks: Sequence[np.ndarray]) -> None:
    if len(masks) != len(project.layers): raise ValueError("AI mask count must match project layers")
    project.depth_map = np.asarray(depth, dtype=np.float32)
    for layer, mask in zip(project.layers, masks): layer.mask = np.asarray(mask, dtype=bool)
