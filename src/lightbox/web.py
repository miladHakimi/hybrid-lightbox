from __future__ import annotations

import argparse
import json
import mimetypes
from io import BytesIO
import tempfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

from .cli import export
from .model import Layer, LayerMode, Material, Project, SourceImage
from .optics import preview_u8


WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


class ProjectValidationError(ValueError):
    pass


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectValidationError(f"{name} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ProjectValidationError(f"{name} must be between {minimum} and {maximum}")
    return result


def validate_web_project(payload: Any) -> dict[str, Any]:
    """Validate and normalize the browser's portable composition document."""
    if not isinstance(payload, dict):
        raise ProjectValidationError("project must be an object")
    canvas = payload.get("canvas")
    sources = payload.get("sources")
    layers = payload.get("layers")
    if not isinstance(canvas, dict):
        raise ProjectValidationError("canvas must be an object")
    if not isinstance(sources, list) or not isinstance(layers, list):
        raise ProjectValidationError("sources and layers must be arrays")
    if not 1 <= len(layers) <= 12:
        raise ProjectValidationError("a project must have between 1 and 12 layers")

    normalized_sources = []
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ProjectValidationError("each source must be an object")
        source_id = source.get("id")
        name = source.get("name")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ProjectValidationError("source IDs must be unique non-empty strings")
        if not isinstance(name, str) or not name.strip():
            raise ProjectValidationError("source names must be non-empty strings")
        source_ids.add(source_id)
        normalized_sources.append({"id": source_id, "name": name.strip()[:160]})

    layer_ids: set[str] = set()
    piece_ids: set[str] = set()
    normalized_layers = []
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ProjectValidationError("each layer must be an object")
        layer_id = layer.get("id")
        if not isinstance(layer_id, str) or not layer_id or layer_id in layer_ids:
            raise ProjectValidationError("layer IDs must be unique non-empty strings")
        layer_ids.add(layer_id)
        pieces = layer.get("pieces", [])
        if not isinstance(pieces, list) or len(pieces) > 100:
            raise ProjectValidationError("each layer may contain at most 100 pieces")
        normalized_pieces = []
        for piece in pieces:
            if not isinstance(piece, dict):
                raise ProjectValidationError("each piece must be an object")
            piece_id = piece.get("id")
            source_id = piece.get("sourceId")
            if not isinstance(piece_id, str) or not piece_id or piece_id in piece_ids:
                raise ProjectValidationError("piece IDs must be unique non-empty strings")
            if source_id not in source_ids:
                raise ProjectValidationError("every piece must reference a known source")
            piece_ids.add(piece_id)
            normalized_pieces.append({
                "id": piece_id,
                "sourceId": source_id,
                "x": _number(piece.get("x", 0.5), "piece.x", -2, 3),
                "y": _number(piece.get("y", 0.5), "piece.y", -2, 3),
                "width": _number(piece.get("width", 0.5), "piece.width", 0.01, 5),
                "height": _number(piece.get("height", 0.5), "piece.height", 0.01, 5),
                "rotation": _number(piece.get("rotation", 0), "piece.rotation", -3600, 3600),
                "opacity": _number(piece.get("opacity", 1), "piece.opacity", 0.05, 1),
            })
        normalized_layers.append({
            "id": layer_id,
            "name": str(layer.get("name") or f"Layer {layer_index + 1}")[:80],
            "linkedToPrevious": bool(layer.get("linkedToPrevious", False)) if layer_index else False,
            "visible": bool(layer.get("visible", True)),
            "mode": str(layer.get("mode", "hybrid")),
            "minimumMm": _number(layer.get("minimumMm", .4), "layer.minimumMm", .1, 10),
            "maximumMm": _number(layer.get("maximumMm", 1.6), "layer.maximumMm", .1, 20),
            "attenuation": _number(layer.get("attenuation", .75), "layer.attenuation", .01, 20),
            "gamma": _number(layer.get("gamma", 1), "layer.gamma", .05, 10),
            "invert": bool(layer.get("invert", False)),
            "penStrokes": _validate_strokes(layer.get("penStrokes", [])),
            "pieces": normalized_pieces,
        })

    return {
        "version": 1,
        "canvas": {
            "widthMm": _number(canvas.get("widthMm", 180), "canvas.widthMm", 20, 1000),
            "heightMm": _number(canvas.get("heightMm", 120), "canvas.heightMm", 20, 1000),
            "meshResolutionMm": _number(canvas.get("meshResolutionMm", .6), "canvas.meshResolutionMm", .1, 2),
        },
        "sources": normalized_sources,
        "layers": normalized_layers,
    }


def _validate_strokes(strokes: Any) -> list[dict[str, Any]]:
    if not isinstance(strokes, list) or len(strokes) > 1000:
        raise ProjectValidationError("pen strokes must be an array of at most 1000 items")
    normalized = []
    for stroke in strokes:
        if not isinstance(stroke, dict) or not isinstance(stroke.get("points"), list):
            raise ProjectValidationError("each pen stroke must contain points")
        points = stroke["points"]
        if not 1 <= len(points) <= 5000:
            raise ProjectValidationError("each pen stroke must contain 1 to 5000 points")
        normalized.append({
            "mode": "erase" if stroke.get("mode") == "erase" else "add",
            "radius": _number(stroke.get("radius", .01), "stroke.radius", .001, .25),
            "points": [[_number(point[0], "stroke.x", -1, 2),
                        _number(point[1], "stroke.y", -1, 2)]
                       for point in points if isinstance(point, list) and len(point) == 2],
        })
    return normalized


def web_project_to_project(document: dict[str, Any], source_images: dict[str, bytes],
                           piece_masks: dict[str, bytes], max_pixels: int = 320) -> Project:
    """Flatten the browser's nondestructive pieces into authoritative Python layers."""
    document = validate_web_project(document)
    aspect = document["canvas"]["widthMm"] / document["canvas"]["heightMm"]
    width = max_pixels if aspect >= 1 else max(32, round(max_pixels * aspect))
    height = max_pixels if aspect < 1 else max(32, round(max_pixels / aspect))
    sources: list[SourceImage] = []
    layers: list[Layer] = []
    for layer_index, item in enumerate(document["layers"]):
        composite = Image.new("RGB", (width, height), "white")
        occupied = Image.new("L", (width, height), 0)
        for piece in item["pieces"]:
            raw = source_images.get(piece["sourceId"])
            if raw is None:
                raise ProjectValidationError(f"source image {piece['sourceId']} has not been uploaded")
            source = Image.open(BytesIO(raw)).convert("RGBA")
            if piece["id"] in piece_masks:
                edited = Image.open(BytesIO(piece_masks[piece["id"]])).convert("L")
                edited = edited.resize(source.size, Image.Resampling.NEAREST)
                alpha = Image.fromarray(np.minimum(np.asarray(source.getchannel("A")), np.asarray(edited)).astype(np.uint8))
            else:
                alpha = source.getchannel("A")
            alpha = alpha.point(lambda value: round(value * piece["opacity"]))
            target = (max(1, round(piece["width"] * width)), max(1, round(piece["height"] * height)))
            source = source.resize(target, Image.Resampling.LANCZOS)
            alpha = alpha.resize(target, Image.Resampling.LANCZOS)
            if piece["rotation"]:
                source = source.rotate(-piece["rotation"], Image.Resampling.BICUBIC, expand=True)
                alpha = alpha.rotate(-piece["rotation"], Image.Resampling.BICUBIC, expand=True)
            left = round(piece["x"] * width - source.width / 2)
            top = round(piece["y"] * height - source.height / 2)
            composite.paste(source.convert("RGB"), (left, top), alpha)
            occupied.paste(Image.new("L", alpha.size, 255), (left, top), alpha)
        draw = ImageDraw.Draw(occupied)
        for stroke in item["penStrokes"]:
            points = [(round(x * width), round(y * height)) for x, y in stroke["points"]]
            fill = 0 if stroke["mode"] == "erase" else 255
            stroke_width = max(1, round(stroke["radius"] * width * 2))
            if len(points) == 1:
                x, y = points[0]; radius = stroke_width // 2
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
            else:
                draw.line(points, fill=fill, width=stroke_width, joint="curve")
        rgb = np.asarray(composite, dtype=np.uint8)
        mask = np.asarray(occupied, dtype=np.uint8) > 12
        sources.append(SourceImage(item["name"], rgb))
        try:
            mode = LayerMode(item["mode"])
        except ValueError as exc:
            raise ProjectValidationError(f"unknown layer mode: {item['mode']}") from exc
        material = Material(item["minimumMm"], item["maximumMm"], item["attenuation"],
                            item["gamma"], item["invert"])
        layers.append(Layer(item["name"], mask, mode, material, item["visible"], layer_index))
    for index in range(len(layers) - 1):
        layers[index].fuse_with_next = document["layers"][index + 1]["linkedToPrevious"]
    project = Project(sources[0].image, layers, width_mm=document["canvas"]["widthMm"],
                      mesh_resolution_mm=document["canvas"]["meshResolutionMm"], sources=sources)
    return project


class LightboxWebHandler(BaseHTTPRequestHandler):
    server_version = "LightboxWeb/0.1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(200, {"status": "ok", "service": "lightbox-web"})
            return
        if path.startswith("/api/sources/"):
            self._stored_asset(self.server.source_images, path.removeprefix("/api/sources/"), "image/*")
            return
        if path.startswith("/api/masks/"):
            self._stored_asset(self.server.piece_masks, path.removeprefix("/api/masks/"), "image/png")
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents or not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/sources/"):
            self._upload_source(path.removeprefix("/api/sources/"))
            return
        if path.startswith("/api/masks/"):
            self._upload_mask(path.removeprefix("/api/masks/"))
            return
        if path in ("/api/projects/preview", "/api/projects/export"):
            self._render_project(path.endswith("export"))
            return
        if path == "/api/projects/save":
            self._save_web_project()
            return
        if path == "/api/projects/open":
            self._open_web_project()
            return
        if path != "/api/projects/validate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ProjectValidationError("project document is too large")
            project = validate_web_project(json.loads(self.rfile.read(length)))
        except (json.JSONDecodeError, ProjectValidationError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "project": project})

    def _upload_source(self, source_id: str) -> None:
        try:
            if not source_id or "/" in source_id:
                raise ProjectValidationError("invalid source ID")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 15_000_000:
                raise ProjectValidationError("source image must be between 1 byte and 15 MB")
            body = self.rfile.read(length)
            with Image.open(BytesIO(body)) as image:
                image.verify()
            with Image.open(BytesIO(body)) as image:
                width, height = image.size
                image_format = image.format
            if width * height > 24_000_000:
                raise ProjectValidationError("source image exceeds the 24 megapixel limit")
            self.server.source_images[source_id] = body
        except (OSError, ValueError, ProjectValidationError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "sourceId": source_id, "width": width,
                         "height": height, "format": image_format})

    def _upload_mask(self, piece_id: str) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not piece_id or not 0 < length <= 5_000_000:
                raise ProjectValidationError("invalid piece mask")
            body = self.rfile.read(length)
            with Image.open(BytesIO(body)) as image:
                image.verify()
            self.server.piece_masks[piece_id] = body
        except (OSError, ValueError, ProjectValidationError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "pieceId": piece_id})

    def _stored_asset(self, storage: dict[str, bytes], asset_id: str, content_type: str) -> None:
        body = storage.get(asset_id)
        if body is None:
            self.send_error(404)
            return
        if content_type == "image/*":
            try:
                with Image.open(BytesIO(body)) as image:
                    content_type = Image.MIME.get(image.format, "application/octet-stream")
            except OSError:
                content_type = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _save_web_project(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            document = validate_web_project(json.loads(self.rfile.read(length)))
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("project.json", json.dumps(document, separators=(",", ":")))
                for source in document["sources"]:
                    raw = self.server.source_images.get(source["id"])
                    if raw is None:
                        raise ProjectValidationError(f"source image {source['id']} has not been uploaded")
                    bundle.writestr(f"sources/{source['id']}", raw)
                for layer in document["layers"]:
                    for piece in layer["pieces"]:
                        raw = self.server.piece_masks.get(piece["id"])
                        if raw is not None:
                            bundle.writestr(f"masks/{piece['id']}.png", raw)
            body = buffer.getvalue()
        except (json.JSONDecodeError, ProjectValidationError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="lightbox-project.lbox"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _open_web_project(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 60_000_000:
                raise ProjectValidationError("project archive must be between 1 byte and 60 MB")
            with zipfile.ZipFile(BytesIO(self.rfile.read(length))) as bundle:
                document = validate_web_project(json.loads(bundle.read("project.json")))
                loaded_sources = {source["id"]: bundle.read(f"sources/{source['id']}") for source in document["sources"]}
                loaded_masks = {}
                for layer in document["layers"]:
                    for piece in layer["pieces"]:
                        name = f"masks/{piece['id']}.png"
                        if name in bundle.namelist():
                            loaded_masks[piece["id"]] = bundle.read(name)
            self.server.source_images.update(loaded_sources)
            self.server.piece_masks.update(loaded_masks)
        except (KeyError, OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile, ProjectValidationError) as exc:
            self._json(400, {"ok": False, "error": f"invalid project archive: {exc}"})
            return
        self._json(200, {"ok": True, "project": document})

    def _render_project(self, archive: bool) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            document = json.loads(self.rfile.read(length))
            project = web_project_to_project(document, self.server.source_images,
                                             self.server.piece_masks)
            if archive:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "lightbox"
                    export(project, output)
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
                        for path in output.iterdir():
                            bundle.write(path, path.name)
                    body = buffer.getvalue()
                content_type = "application/zip"
                filename = "lightbox-export.zip"
            else:
                buffer = BytesIO()
                Image.fromarray(preview_u8(project)).save(buffer, "PNG")
                body = buffer.getvalue()
                content_type = "image/png"
                filename = None
        except (json.JSONDecodeError, ProjectValidationError, ValueError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"Lightbox web editor: http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), LightboxWebHandler)
    server.source_images = {}
    server.piece_masks = {}
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Lightbox web editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
