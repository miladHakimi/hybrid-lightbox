from __future__ import annotations
from pathlib import Path
import math
import struct
import heapq
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree.ElementTree import Element, SubElement, tostring
import numpy as np


def voxel_surface(mask: np.ndarray, heights: np.ndarray, pixel_mm: float) -> np.ndarray:
    """Triangulated surface of a mask extrusion with a continuous height field."""
    h, w = mask.shape
    tris: list[list[list[float]]] = []
    def quad(a, b, c, d):
        tris.extend([[a, b, c], [a, c, d]])
    # Cell-centred image values are averaged onto grid vertices. This makes
    # neighboring top facets share exact edges (no STL T-junctions).
    vertex_z = np.zeros((h + 1, w + 1), dtype=np.float32)
    vertex_n = np.zeros((h + 1, w + 1), dtype=np.int16)
    for y, x in np.argwhere(mask):
        for vy, vx in ((y,x),(y,x+1),(y+1,x),(y+1,x+1)):
            vertex_z[vy,vx] += heights[y,x]; vertex_n[vy,vx] += 1
    np.divide(vertex_z, vertex_n, out=vertex_z, where=vertex_n > 0)
    for y, x in np.argwhere(mask):
        x0, x1 = x * pixel_mm, (x + 1) * pixel_mm
        # Flip image Y so exported layers keep visual orientation.
        y0, y1 = (h - y - 1) * pixel_mm, (h - y) * pixel_mm
        z00=float(vertex_z[y+1,x]); z10=float(vertex_z[y+1,x+1])
        z11=float(vertex_z[y,x+1]); z01=float(vertex_z[y,x])
        quad([x0,y0,z00],[x1,y0,z10],[x1,y1,z11],[x0,y1,z01])
        quad([x0,y1,0],[x1,y1,0],[x1,y0,0],[x0,y0,0])
        neighbors = [(-1,0,"top"),(1,0,"bottom"),(0,-1,"left"),(0,1,"right")]
        for dy, dx, side in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny,nx]: continue
            if side == "left": quad([x0,y0,0],[x0,y0,z00],[x0,y1,z01],[x0,y1,0])
            elif side == "right": quad([x1,y1,0],[x1,y1,z11],[x1,y0,z10],[x1,y0,0])
            elif side == "top": quad([x0,y1,0],[x0,y1,z01],[x1,y1,z11],[x1,y1,0])
            else: quad([x1,y0,0],[x1,y0,z10],[x0,y0,z00],[x0,y0,0])
    return np.asarray(tris, dtype=np.float32).reshape((-1, 3, 3))


def _normal(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); length = np.linalg.norm(n)
    return n / length if length else n


def write_ascii_stl(path: str | Path, triangles: np.ndarray, name: str = "layer") -> None:
    if len(triangles) == 0: raise ValueError("Cannot export an empty layer")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            n = _normal(tri); f.write(f"  facet normal {n[0]:.8g} {n[1]:.8g} {n[2]:.8g}\n    outer loop\n")
            for v in tri: f.write(f"      vertex {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


def write_binary_stl(path: str | Path, triangles: np.ndarray, name: str = "layer") -> None:
    """Write compact binary STL; Bambu Studio interprets coordinates as millimetres."""
    if len(triangles) == 0: raise ValueError("Cannot export an empty layer")
    vectors=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(vectors,axis=1);normals=np.divide(vectors,lengths[:,None],out=np.zeros_like(vectors),where=lengths[:,None]>0)
    records=np.empty(len(triangles),dtype=[("normal","<f4",(3,)),("vertices","<f4",(9,)),("attribute","<u2")])
    records["normal"]=normals;records["vertices"]=triangles.reshape(-1,9);records["attribute"]=0
    header=(f"Hybrid Lightbox {name}".encode("ascii")[:80]).ljust(80,b"\0")
    with open(path,"wb") as f:
        f.write(header);f.write(struct.pack("<I",len(triangles)));records.tofile(f)


def write_3mf(path: str | Path, meshes, names=None) -> None:
    """Write aligned triangle meshes as separate objects in a 3MF package."""
    meshes=[np.asarray(mesh,dtype=np.float32) for mesh in meshes]
    names=list(names) if names is not None else [f"layer_{i+1}" for i in range(len(meshes))]
    if len(names)!=len(meshes):raise ValueError("Every 3MF mesh must have a name")
    namespace="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    model=Element("model",{"xmlns":namespace,"unit":"millimeter","xml:lang":"en-US"})
    resources=SubElement(model,"resources");build=SubElement(model,"build")
    for object_id,(triangles,name) in enumerate(zip(meshes,names),1):
        if triangles.ndim!=3 or triangles.shape[1:]!=(3,3):raise ValueError("3MF meshes must have shape (n, 3, 3)")
        obj=SubElement(resources,"object",{"id":str(object_id),"type":"model","name":str(name)})
        mesh=SubElement(obj,"mesh");vertices=SubElement(mesh,"vertices");faces=SubElement(mesh,"triangles")
        unique,index=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
        for x,y,z in unique:SubElement(vertices,"vertex",{"x":f"{x:.9g}","y":f"{y:.9g}","z":f"{z:.9g}"})
        for v1,v2,v3 in index.reshape(-1,3):SubElement(faces,"triangle",{"v1":str(v1),"v2":str(v2),"v3":str(v3)})
    # Aligned material layers are parts of one printable object, not separate
    # build items occupying the same coordinates.  Slicers such as Bambu
    # Studio otherwise treat the layers as overlapping models and may hide or
    # repair away fine geometry in an earlier layer.
    assembly_id=len(meshes)+1
    assembly=SubElement(resources,"object",{"id":str(assembly_id),"type":"model","name":"model"})
    components=SubElement(assembly,"components")
    for object_id in range(1,assembly_id):
        SubElement(components,"component",{"objectid":str(object_id)})
    SubElement(build,"item",{"objectid":str(assembly_id)})
    content_types=b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
    relationships=b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    with ZipFile(path,"w",ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml",content_types)
        package.writestr("_rels/.rels",relationships)
        package.writestr("3D/3dmodel.model",b'<?xml version="1.0" encoding="UTF-8"?>'+tostring(model,encoding="utf-8"))


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, bool); components = []
    for sy, sx in np.argwhere(mask):
        if seen[sy,sx]: continue
        stack=[(int(sy),int(sx))]; seen[sy,sx]=True; pts=[]
        while stack:
            y,x=stack.pop(); pts.append((y,x))
            for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True; stack.append((ny,nx))
        components.append(np.asarray(pts))
    return components


def remove_small_features(mask: np.ndarray, pixel_mm: float,
                          minimum_area_mm2: float = 0.8,
                          maximum_hole_area_mm2: float | None = None) -> np.ndarray:
    """Remove tiny islands and fill tiny enclosed holes by physical area."""
    out = np.asarray(mask, bool).copy()
    if minimum_area_mm2 <= 0:
        return out
    minimum_pixels = max(1, math.ceil(minimum_area_mm2 / (pixel_mm * pixel_mm)))
    hole_area = minimum_area_mm2 if maximum_hole_area_mm2 is None else maximum_hole_area_mm2
    maximum_hole_pixels = max(0, math.ceil(hole_area / (pixel_mm * pixel_mm)))
    for component in connected_components(out):
        if len(component) < minimum_pixels:
            out[component[:, 0], component[:, 1]] = False
    # A background component touching the image edge is open space, not a hole.
    h, w = out.shape
    for component in connected_components(~out):
        touches_edge = np.any((component[:, 0] == 0) | (component[:, 0] == h - 1) |
                              (component[:, 1] == 0) | (component[:, 1] == w - 1))
        if not touches_edge and len(component) < maximum_hole_pixels:
            out[component[:, 0], component[:, 1]] = True
    return out


def remove_thin_features(mask: np.ndarray, pixel_mm: float,
                         minimum_width_mm: float = 0.4) -> np.ndarray:
    """Remove mask regions too narrow to contain the requested print width.

    This is a binary opening with a circular physical-width kernel.  Unlike the
    area filter above, it removes long whiskers and narrow islands even when
    they contain many pixels.
    """
    source = np.asarray(mask, bool)
    if pixel_mm <= 0:
        raise ValueError("Pixel size must be positive")
    if minimum_width_mm < 0:
        raise ValueError("Minimum width cannot be negative")
    if minimum_width_mm == 0 or not source.any():
        return source.copy()
    radius = max(1, math.ceil((minimum_width_mm / pixel_mm - 1.0) / 2.0))
    return _binary_morphology(
        _binary_morphology(source, radius, False), radius, True)


def remove_thin_features_from_masks(masks, pixel_mm: float,
                                    minimum_width_mm: float = 0.4) -> list[np.ndarray]:
    """Apply the physical-width filter independently to every supplied mask."""
    return [remove_thin_features(mask, pixel_mm, minimum_width_mm) for mask in masks]


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    return [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1) if dy * dy + dx * dx <= radius * radius]


def _binary_morphology(mask: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    """Small dependency-free disk dilation/erosion."""
    source = np.asarray(mask, bool)
    if radius <= 0:
        return source.copy()
    padded = np.pad(source, radius, constant_values=False)
    views = [padded[radius + dy:radius + dy + source.shape[0],
                    radius + dx:radius + dx + source.shape[1]]
             for dy, dx in _disk_offsets(radius)]
    return np.logical_or.reduce(views) if dilate else np.logical_and.reduce(views)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a boolean mask by a circular pixel radius."""
    if radius < 0:
        raise ValueError("Dilation radius cannot be negative")
    return _binary_morphology(mask, int(radius), True)


def _binary_majority(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, bool)
    padded = np.pad(source, radius, constant_values=False)
    offsets = _disk_offsets(radius)
    votes = sum(padded[radius + dy:radius + dy + source.shape[0],
                       radius + dx:radius + dx + source.shape[1]].astype(np.uint16)
                for dy, dx in offsets)
    return votes >= (len(offsets) // 2 + 1)


def _gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma)); kernel /= kernel.sum()
    result = np.asarray(values, np.float32)
    for axis in (0, 1):
        pads = [(0, 0), (0, 0)]; pads[axis] = (radius, radius)
        padded = np.pad(result, pads, mode="constant")
        result = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)
    return result.astype(np.float32)


def smooth_print_data(mask: np.ndarray, heights: np.ndarray, pixel_mm: float,
                      smoothing_mm: float = 0.3,
                      thickness_step_mm: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Round mask noise and low-pass/quantize heights on the physical mesh grid."""
    out_mask = np.asarray(mask, bool).copy()
    out_heights = np.asarray(heights, np.float32).copy()
    if smoothing_mm < 0 or thickness_step_mm < 0:
        raise ValueError("Print smoothing and thickness step cannot be negative")
    if smoothing_mm > 0 and out_mask.any():
        radius = max(1, int(round(smoothing_mm / pixel_mm)))
        # Opening removes narrow whiskers; closing then heals equally narrow notches.
        out_mask = _binary_morphology(_binary_morphology(out_mask, radius, False), radius, True)
        out_mask = _binary_morphology(_binary_morphology(out_mask, radius, True), radius, False)
        out_mask = _binary_majority(out_mask, radius)
        blur_radius = max(0.5, smoothing_mm / pixel_mm)
        weights = _gaussian_blur(mask.astype(np.float32), blur_radius)
        values = _gaussian_blur((out_heights * mask).astype(np.float32), blur_radius)
        out_heights = np.divide(values, weights, out=np.zeros_like(values), where=weights > 1e-6)
    positive = heights[mask]
    if positive.size:
        minimum = float(positive.min())
        if thickness_step_mm > 0:
            out_heights = minimum + np.round((out_heights - minimum) / thickness_step_mm) * thickness_step_mm
        out_heights = np.where(out_mask, np.maximum(out_heights, minimum), 0.0)
    else:
        out_heights.fill(0)
    return out_mask, out_heights.astype(np.float32)


def add_frame_and_bridges(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Return one connected printable mask by adding a rim and short left bridges."""
    out = mask.copy(); out[:width,:]=True; out[-width:,:]=True; out[:,:width]=True; out[:,-width:]=True
    for component in connected_components(out):
        if np.any((component[:,0] < width) | (component[:,0] >= out.shape[0]-width) |
                  (component[:,1] < width) | (component[:,1] >= out.shape[1]-width)):
            continue
        y,x = component[np.argmin(component[:,1])]
        out[int(y), :int(x)+1] = True
    # Resolve corner-only contacts, which are non-manifold in a triangle mesh.
    # Filling the two alternate cells is conservative and deterministic.
    changed = True
    while changed:
        changed = False
        for y in range(out.shape[0]-1):
            for x in range(out.shape[1]-1):
                a,b,c,d = out[y,x],out[y,x+1],out[y+1,x],out[y+1,x+1]
                if a and d and not b and not c:
                    out[y,x+1]=out[y+1,x]=True; changed=True
                elif b and c and not a and not d:
                    out[y,x]=out[y+1,x+1]=True; changed=True
    return out


def add_hidden_bridges(mask: np.ndarray, hidden: np.ndarray, pixel_mm: float,
                       bridge_width_mm: float = 3.2, frame_width_mm: float = 1.2) -> np.ndarray:
    """Connect islands to the frame with one sturdy, preferably hidden route."""
    h,w=mask.shape
    frame=max(1,math.ceil(frame_width_mm/pixel_mm))
    # Paths begin one pixel wide, so dilate by the radius needed to meet or
    # exceed the requested physical connector width on the sampled grid.
    radius=max(0,math.ceil((bridge_width_mm/pixel_mm-1)/2))
    out=mask.copy();out[:frame,:]=True;out[-frame:,:]=True;out[:,:frame]=True;out[:,-frame:]=True
    components=connected_components(out)
    if len(components)<=1:return out
    border=np.zeros((h,w),bool);border[:frame,:]=True;border[-frame:,:]=True;border[:,:frame]=True;border[:,-frame:]=True
    open_background=np.zeros_like(out)
    for background_component in connected_components(~mask):
        by,bx=background_component[:,0],background_component[:,1]
        if np.any((by==0)|(by==h-1)|(bx==0)|(bx==w-1)):
            open_background[by,bx]=True
    dist=np.full((h,w),np.inf,np.float64);previous=np.full((h,w,2),-1,np.int32);queue=[]
    for y,x in np.argwhere(border):dist[y,x]=0;heapq.heappush(queue,(0.0,int(y),int(x)))
    while queue:
        score,y,x=heapq.heappop(queue)
        if score!=float(dist[y,x]):continue
        for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if not (0<=ny<h and 0<=nx<w):continue
            step=.05 if out[ny,nx] else (1.0 if hidden[ny,nx] else 25.0)
            candidate=score+step
            if candidate<float(dist[ny,nx]):
                dist[ny,nx]=candidate;previous[ny,nx]=[y,x];heapq.heappush(queue,(candidate,ny,nx))
    paths=np.zeros_like(out)
    for component in components:
        if np.any(border[component[:,0],component[:,1]]):continue
        # White counters enclosed inside negative lettering are intentional
        # cutout detail, not loose exterior pieces requiring a connector.
        adjacent_to_open_background=False
        for cy,cx in component:
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<h and 0<=nx<w and open_background[ny,nx]:
                    adjacent_to_open_background=True;break
            if adjacent_to_open_background:break
        if not adjacent_to_open_background:continue
        values=dist[component[:,0],component[:,1]];y,x=map(int,component[int(np.argmin(values))])
        guard=0
        while not border[y,x] and guard<h*w:
            paths[y,x]=True;py,px=previous[y,x]
            if py<0:break
            y,x=int(py),int(px);guard+=1
    if radius:
        expanded=paths.copy()
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                if dx*dx+dy*dy>radius*radius:continue
                ys=slice(max(0,dy),min(h,h+dy));xs=slice(max(0,dx),min(w,w+dx))
                src_y=slice(max(0,-dy),min(h,h-dy));src_x=slice(max(0,-dx),min(w,w-dx))
                expanded[ys,xs]|=paths[src_y,src_x]
        paths=expanded
    out|=paths
    # Eliminate corner-only contacts introduced by dilation/path intersections.
    # Limit repairs to generated paths; applying this rule across the complete
    # source mask fills intentional one-pixel diagonal cutouts and lettering.
    repair_region=paths.copy()
    changed=True
    while changed:
        changed=False
        for y in range(h-1):
            for x in range(w-1):
                if not repair_region[y:y+2,x:x+2].any():continue
                a,b,c,d=out[y,x],out[y,x+1],out[y+1,x],out[y+1,x+1]
                if a and d and not b and not c:out[y,x+1]=out[y+1,x]=True;changed=True
                elif b and c and not a and not d:out[y,x]=out[y+1,x+1]=True;changed=True
    return out


def edge_counts(triangles: np.ndarray, precision: int = 6) -> dict[tuple, int]:
    counts = {}
    for tri in np.round(triangles, precision):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edge = tuple(sorted((tuple(a),tuple(b))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def is_watertight(triangles: np.ndarray) -> bool:
    return bool(len(triangles)) and all(v == 2 for v in edge_counts(triangles).values())
