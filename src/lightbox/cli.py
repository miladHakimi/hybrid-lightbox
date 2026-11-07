from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from .model import Project
from .optics import (OPTIMIZER_NOZZLE_DIAMETER_MM, combined_transmission,
                     layer_thickness, optimize_rear_openings, preview_u8)
from .geometry import (voxel_surface, write_binary_stl, write_3mf, is_watertight,
                       add_frame_and_bridges, add_hidden_bridges, connected_components,
                       remove_small_features, smooth_print_data)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_STL_DIMENSIONS_MM = (109.0, 144.0, 2.65)
DEFAULT_CONNECTOR_DEPTH_MM = 1.6


def scale_mesh_to_dimensions(triangles: np.ndarray, width_mm: float,
                             height_mm: float, depth_mm: float) -> np.ndarray:
    """Scale a mesh to an exact width, height, and depth in millimetres."""
    dimensions=np.asarray((width_mm,height_mm,depth_mm),dtype=np.float32)
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions<=0):
        raise ValueError("STL width, height, and depth must be positive")
    triangles=np.asarray(triangles,dtype=np.float32)
    minimum=triangles.min(axis=(0,1));maximum=triangles.max(axis=(0,1));extent=maximum-minimum
    if np.any(extent<=0):raise ValueError("Cannot resize an STL with an empty dimension")
    return (triangles-minimum)*(dimensions/extent)


def mesh_inputs(project: Project, mask: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
    """Resample image-space data to a physical mesh grid independent of AI resolution."""
    source_h,source_w=mask.shape
    effective_resolution=project.mesh_resolution_mm
    if effective_resolution <= 0:
        raise ValueError("Mesh resolution must be positive")
    target_w=min(source_w,max(2,round(project.width_mm/effective_resolution)))
    # A small nearest-neighbour reduction can erase single-pixel cutouts such
    # as lettering and line art while barely reducing the mesh size. Keep the
    # source grid in that case so fine positive and negative details survive.
    if target_w < source_w and target_w / source_w >= .75:
        target_w=source_w
    target_h=max(2,round(source_h*target_w/source_w))
    if (target_h,target_w)==mask.shape:return mask,heights,project.width_mm/target_w
    resized_mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((target_w,target_h),Image.Resampling.NEAREST))>0
    resized_heights=np.asarray(Image.fromarray(heights.astype(np.float32),mode="F").resize((target_w,target_h),Image.Resampling.BILINEAR),dtype=np.float32)
    # Bilinear sampling blends zero-height background into the edge of a
    # printable mask. Without restoring the material floor, boundary cells can
    # become far thinner than the configured minimum and look ragged or even
    # disappear in a slicer.
    printable_heights=heights[mask]
    minimum=float(printable_heights.min()) if printable_heights.size else 0.0
    resized_heights=np.where(resized_mask,np.maximum(resized_heights,minimum),0.0).astype(np.float32)
    return resized_mask,resized_heights,project.width_mm/target_w


def resize_mask(mask: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    if mask.shape==shape:return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((shape[1],shape[0]),Image.Resampling.NEAREST))>0


def reinforce_frame(mask: np.ndarray, heights: np.ndarray, pixel_mm: float,
                    width_mm: float, thickness_mm: float) -> tuple[np.ndarray,np.ndarray]:
    width=max(1,int(np.ceil(width_mm/pixel_mm)));frame=np.zeros_like(mask,bool)
    frame[:width,:]=True;frame[-width:,:]=True;frame[:,:width]=True;frame[:,-width:]=True
    mask=mask|frame;heights=np.where(frame,np.maximum(heights,thickness_mm),heights)
    return mask,heights


def reinforce_connectors(base_mask: np.ndarray, structural_mask: np.ndarray,
                         heights: np.ndarray,
                         depth_mm: float = DEFAULT_CONNECTOR_DEPTH_MM) -> np.ndarray:
    """Give generated connectors a sturdy depth without thinning existing material."""
    connectors=structural_mask & ~base_mask
    return np.where(connectors,np.maximum(heights,depth_mm),heights)


def export(project: Project, output: Path, minimum_feature_area_mm2: float = 0.0,
           maximum_hole_area_mm2: float = 0.0,
           dimensions_mm: tuple[float,float,float] | None = None) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True); paths=[];structural_masks=[];meshes=[]
    project.save(output / "project.hbox")
    cleaned_masks=[remove_small_features(project.output_mask(i),project.pixel_mm,minimum_feature_area_mm2,maximum_hole_area_mm2)
                   for i, layer in enumerate(project.layers)]
    groups=project.physical_layer_groups();physical_masks=[]
    for i, group in enumerate(groups):
        layer=project.layers[group[0]]
        source_mask=np.logical_or.reduce([cleaned_masks[index] for index in group]);physical_masks.append(source_mask)
        raw_heights=np.maximum.reduce([layer_thickness(project,index) for index in group])
        minimum=min(project.layers[index].material.minimum_mm for index in group)
        heights=np.where(source_mask,np.maximum(raw_heights,minimum),0.0)
        if project.structure_mode=="backing":
            mask=np.ones_like(source_mask,bool);heights=np.where(source_mask,heights,project.backing_thickness_mm)
            mask,heights,pixel_mm=mesh_inputs(project,mask,heights)
        elif project.structure_mode=="hidden_bridges":
            mask=source_mask if layer.enabled else np.zeros_like(source_mask,bool)
            mask,heights,pixel_mm=mesh_inputs(project,mask,heights)
            mask,heights=smooth_print_data(mask,heights,pixel_mm,project.print_smoothing_mm)
            foreground=np.logical_or.reduce(physical_masks[:i]) if i else np.zeros_like(source_mask,bool)
            hidden=resize_mask(foreground,mask.shape);final_mask=add_hidden_bridges(mask,hidden,pixel_mm,frame_width_mm=project.frame_width_mm)
            heights=reinforce_connectors(mask,final_mask,heights);mask=final_mask
        else:
            mask=add_frame_and_bridges(source_mask) if layer.enabled else source_mask
            heights=np.where(mask & ~source_mask,layer.material.minimum_mm,heights)
            mask,heights,pixel_mm=mesh_inputs(project,mask,heights)
            mask,heights=smooth_print_data(mask,heights,pixel_mm,project.print_smoothing_mm)
            final_mask=add_frame_and_bridges(mask)
            heights=reinforce_connectors(mask,final_mask,heights);mask=final_mask
        if project.structure_mode=="backing":
            mask,heights=smooth_print_data(mask,heights,pixel_mm,project.print_smoothing_mm)
        mask,heights=reinforce_frame(mask,heights,pixel_mm,project.frame_width_mm,project.frame_thickness_mm)
        tris=voxel_surface(mask, heights, pixel_mm)
        if dimensions_mm is not None:
            tris=scale_mesh_to_dimensions(tris,*dimensions_mm)
        path=output/f"layer_{i+1}.stl"; write_binary_stl(path,tris,f"layer_{i+1}"); paths.append(path)
        meshes.append(tris)
        structural_masks.append(mask)
    write_3mf(output/"model.3mf",meshes,[f"layer_{i+1}" for i in range(len(meshes))])
    Image.fromarray(preview_u8(project)).save(output/"backlit_preview.png")
    for i, source_mask in enumerate(physical_masks):
        Image.fromarray((source_mask.astype(np.uint8) * 255)).save(output/f"layer_{i+1}_mask.png")
        Image.fromarray((structural_masks[i].astype(np.uint8)*255)).save(output/f"layer_{i+1}_structure.png")
    if project.depth_map is not None:
        Image.fromarray(np.clip(project.depth_map * 255, 0, 255).astype(np.uint8)).save(output/"depth_map.png")
    return paths


def inspect(project: Project, minimum_feature_area_mm2: float = 0.0,
            maximum_hole_area_mm2: float = 0.0) -> dict:
    """Return machine-readable optical, structural, and mesh diagnostics."""
    transmission = combined_transmission(project)
    layers = []
    cleaned_masks=[remove_small_features(project.output_mask(i),project.pixel_mm,minimum_feature_area_mm2,maximum_hole_area_mm2)
                   for i, layer in enumerate(project.layers)]
    for i, layer in enumerate(project.layers):
        source_mask=cleaned_masks[i]
        raw_heights=layer_thickness(project,i)
        heights = np.where(source_mask,np.maximum(raw_heights,layer.material.minimum_mm),0.0)
        hidden=None
        if project.structure_mode=="backing":
            printable=np.ones_like(source_mask,bool);heights=np.where(source_mask,heights,project.backing_thickness_mm)
            printable,heights,mesh_pixel_mm=mesh_inputs(project,printable,heights)
            printable,heights=smooth_print_data(printable,heights,mesh_pixel_mm,project.print_smoothing_mm)
            base_printable=printable.copy()
        elif project.structure_mode=="hidden_bridges":
            printable=source_mask if layer.enabled else np.zeros_like(source_mask,bool)
            printable,heights,mesh_pixel_mm=mesh_inputs(project,printable,heights)
            printable,heights=smooth_print_data(printable,heights,mesh_pixel_mm,project.print_smoothing_mm)
            base_printable=printable.copy()
            foreground=np.logical_or.reduce(cleaned_masks[:i]) if i else np.zeros_like(source_mask,bool)
            hidden=resize_mask(foreground,printable.shape);final_mask=add_hidden_bridges(printable,hidden,mesh_pixel_mm,frame_width_mm=project.frame_width_mm)
            heights=reinforce_connectors(printable,final_mask,heights);printable=final_mask
        else:
            printable=add_frame_and_bridges(source_mask) if layer.enabled else source_mask
            heights=np.where(printable & ~source_mask,layer.material.minimum_mm,heights)
            printable,heights,mesh_pixel_mm=mesh_inputs(project,printable,heights)
            printable,heights=smooth_print_data(printable,heights,mesh_pixel_mm,project.print_smoothing_mm)
            base_printable=printable.copy()
            final_mask=add_frame_and_bridges(printable)
            heights=reinforce_connectors(printable,final_mask,heights);printable=final_mask
        printable,heights=reinforce_frame(printable,heights,mesh_pixel_mm,project.frame_width_mm,project.frame_thickness_mm)
        triangles = voxel_surface(printable, heights, mesh_pixel_mm)
        added=printable & ~base_printable
        frame_px=max(1,int(np.ceil(project.frame_width_mm/mesh_pixel_mm)));interior=np.ones_like(added,bool);interior[:frame_px,:]=False;interior[-frame_px:,:]=False;interior[:,:frame_px]=False;interior[:,-frame_px:]=False
        bridge_added=added & interior
        layers.append({
            "index": i + 1,
            "name": layer.name,
            "mode": layer.mode.value,
            "mask_pixels": int(source_mask.sum()),
            "removed_sparse_pixels": int(np.count_nonzero(project.output_mask(i) & ~source_mask)),
            "components_before_bridging": len(connected_components(source_mask)),
            "components_after_bridging": len(connected_components(printable)),
            "structural_added_pixels": int(added.sum()),
            "bridge_pixels_excluding_frame": int(bridge_added.sum()),
            "hidden_bridge_fraction": float((bridge_added & hidden).sum()/bridge_added.sum()) if hidden is not None and bridge_added.any() else None,
            "triangles": int(len(triangles)),
            "watertight": is_watertight(triangles),
            "minimum_thickness_mm": float(heights[printable].min()) if printable.any() else 0.0,
            "maximum_thickness_mm": float(heights[printable].max()) if printable.any() else 0.0,
        })
    return {
        "image_pixels": [int(project.image.shape[1]), int(project.image.shape[0])],
        "physical_mm": [project.width_mm, project.height_mm],
        "pixel_mm": project.pixel_mm,
        "mesh_resolution_mm": project.mesh_resolution_mm,
        "structure_mode": project.structure_mode,
        "backing_thickness_mm": project.backing_thickness_mm,
        "frame_width_mm": project.frame_width_mm,
        "frame_thickness_mm": project.frame_thickness_mm,
        "print_smoothing_mm": project.print_smoothing_mm,
        "optimizer_nozzle_diameter_mm": OPTIMIZER_NOZZLE_DIAMETER_MM,
        "transmission": {"minimum": float(transmission.min()), "maximum": float(transmission.max()), "mean": float(transmission.mean())},
        "valid": all(layer["watertight"] and layer["components_after_bridging"] == 1 for layer in layers),
        "layers": layers,
    }


def generate_one(image: Path, output: Path, width: float, optimize: bool,
                 ai: bool = False, segment: bool = True,
                 points: list[tuple[float,float]] | None = None, layers: int = 3,
                 image_resolution: int = 768, mesh_resolution: float = .12,
                 structure: str = "hidden_bridges", backing_thickness: float = .6,
                 frame_width: float = 1.0, frame_thickness: float = 3.0,
                 segmentation_quality: str = "accurate",
                 minimum_feature_area: float = 0.0,
                 maximum_hole_area: float = 0.0,
                 minimum_segment_area: int = 100,
                 print_smoothing: float = 0.0) -> dict:
    project = Project.from_image(image, size=image_resolution, layer_count=layers); project.width_mm = width; project.mesh_resolution_mm=mesh_resolution; project.structure_mode=structure; project.backing_thickness_mm=backing_thickness; project.frame_width_mm=frame_width; project.frame_thickness_mm=frame_thickness; project.print_smoothing_mm=print_smoothing
    backend = "luminance"
    if ai:
        from .ai import LocalAI, apply_ai_decomposition
        depth, masks = LocalAI(use_segmentation=segment, segmentation_quality=segmentation_quality,
                               minimum_segment_area=minimum_segment_area).decompose(
            project.image, len(project.layers), lambda message: print(message, file=sys.stderr, flush=True), points)
        apply_ai_decomposition(project, depth, masks); backend = "depth-anything-v2+sam2" if segment else "depth-anything-v2"
    opened = optimize_rear_openings(project) if optimize else 0
    output.mkdir(parents=True, exist_ok=True)
    paths = export(project, output, minimum_feature_area, maximum_hole_area); report = inspect(project, minimum_feature_area, maximum_hole_area)
    report.update({"source": str(image), "output": str(output), "decomposition": backend, "rear_pixels_opened": opened,
                   "exports": [str(path) for path in paths]+[str(output/"model.3mf")]})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def discover_images(path: Path) -> list[Path]:
    if path.is_file(): return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_points(value: str | None) -> list[tuple[float,float]] | None:
    if not value: return None
    result=[]
    for pair in value.split(";"):
        x,y=(float(part.strip()) for part in pair.split(","))
        if not 0<=x<=1 or not 0<=y<=1: raise argparse.ArgumentTypeError("SAM points must be normalized from 0 to 1")
        result.append((x,y))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(prog="lightbox-cli", description="Generate and validate hybrid lightbox projects")
    sub=p.add_subparsers(dest="command",required=True)
    one=sub.add_parser("generate",help="generate one image"); one.add_argument("image",type=Path); one.add_argument("-o","--output",type=Path,default=Path("demo_output"))
    batch=sub.add_parser("batch",help="recursively process all supported images in a folder"); batch.add_argument("input",type=Path); batch.add_argument("-o","--output",type=Path,default=Path("batch_output"))
    check=sub.add_parser("inspect",help="analyze one image without writing geometry"); check.add_argument("image",type=Path)
    for parser in (one,batch,check):
        parser.add_argument("--width",type=float,default=120.0); parser.add_argument("--optimize",action="store_true")
        parser.add_argument("--ai",action="store_true",help="use local Depth Anything V2 and SAM 2")
        parser.add_argument("--depth-only",action="store_true",help="use depth AI without SAM 2 segmentation")
        parser.add_argument("--sam-points",type=parse_points,help="object centers x,y;x,y… normalized 0–1; order maps to layers")
        parser.add_argument("--segmentation-quality",choices=("fast","base","balanced","accurate","hq"),default="accurate",help="SAM 2.1 checkpoint, or optional HQ-SAM 2 Large for finer boundaries")
        parser.add_argument("--min-segment-area",type=int,default=100,metavar="PIXELS",help="discard SAM segments smaller than this pixel area; use 0 to disable")
        parser.add_argument("--layers",type=int,default=3,metavar="COUNT",help="physical layer count (minimum 2)")
        parser.add_argument("--image-resolution",type=int,default=768,help="maximum source/AI image dimension")
        parser.add_argument("--mesh-resolution",type=float,default=.12,help="target STL sample spacing in millimetres; values below nozzle width preserve smoother curves")
        parser.add_argument("--structure",choices=("hidden_bridges","backing","bridges"),default="hidden_bridges",help="occlusion-aware bridges, solid backing, or legacy bridges")
        parser.add_argument("--backing-thickness",type=float,default=.6,help="solid backing thickness in millimetres")
        parser.add_argument("--frame-width",type=float,default=1.0,help="reinforced outer border width in millimetres")
        parser.add_argument("--frame-thickness",type=float,default=3.0,help="reinforced outer border thickness in millimetres")
        parser.add_argument("--print-smoothing",type=float,default=0.0,help="mask and thickness smoothing radius in millimetres; use 0 to preserve maximum detail")
        parser.add_argument("--min-feature-area",type=float,default=0.0,help="remove islands and pinholes smaller than this area in mm²; default preserves all features")
        parser.add_argument("--max-hole-area",type=float,default=0.0,help="fill enclosed holes smaller than this area in mm²; default preserves all holes")
    a=p.parse_args(argv)
    if a.command=="inspect":
        project=Project.from_image(a.image,size=a.image_resolution,layer_count=a.layers); project.width_mm=a.width; project.mesh_resolution_mm=a.mesh_resolution; project.structure_mode=a.structure; project.backing_thickness_mm=a.backing_thickness; project.frame_width_mm=a.frame_width; project.frame_thickness_mm=a.frame_thickness; project.print_smoothing_mm=a.print_smoothing
        if a.ai:
            from .ai import LocalAI, apply_ai_decomposition
            depth,masks=LocalAI(use_segmentation=not a.depth_only,segmentation_quality=a.segmentation_quality,minimum_segment_area=a.min_segment_area).decompose(project.image,len(project.layers),lambda m: print(m,file=sys.stderr,flush=True),a.sam_points)
            apply_ai_decomposition(project,depth,masks)
        if a.optimize: optimize_rear_openings(project)
        print(json.dumps(inspect(project,a.min_feature_area,a.max_hole_area),indent=2)); return 0
    if a.command=="generate":
        report=generate_one(a.image,a.output,a.width,a.optimize,a.ai,not a.depth_only,a.sam_points,a.layers,a.image_resolution,a.mesh_resolution,a.structure,a.backing_thickness,a.frame_width,a.frame_thickness,a.segmentation_quality,a.min_feature_area,a.max_hole_area,a.min_segment_area,a.print_smoothing); print(json.dumps(report,indent=2)); return 0 if report["valid"] else 2
    images=discover_images(a.input)
    if not images: p.error(f"no JPEG, PNG, or WebP images found in {a.input}")
    reports=[]
    for image in images:
        relative=image.relative_to(a.input) if a.input.is_dir() else Path(image.name)
        target=a.output/relative.with_suffix("")
        try: reports.append(generate_one(image,target,a.width,a.optimize,a.ai,not a.depth_only,a.sam_points,a.layers,a.image_resolution,a.mesh_resolution,a.structure,a.backing_thickness,a.frame_width,a.frame_thickness,a.segmentation_quality,a.min_feature_area,a.max_hole_area,a.min_segment_area,a.print_smoothing))
        except Exception as exc: reports.append({"source":str(image),"output":str(target),"valid":False,"error":str(exc)})
    summary={"input":str(a.input),"output":str(a.output),"images":len(images),"passed":sum(bool(x.get("valid")) for x in reports),"failed":sum(not bool(x.get("valid")) for x in reports),"results":reports}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/"batch_report.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2)); return 0 if summary["failed"]==0 else 2


if __name__ == "__main__":
    main()
