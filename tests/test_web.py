import json
import io
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from lightbox.web import ProjectValidationError, WEB_ROOT, validate_web_project, web_project_to_project


def sample_project():
    return {
        "canvas": {"widthMm": 180, "heightMm": 120},
        "sources": [{"id": "source-a", "name": "portrait.png"}],
        "layers": [
            {"id": "front", "name": "Front", "pieces": [
                {"id": "piece-a", "sourceId": "source-a", "x": .4, "y": .6,
                 "width": .5, "height": .7, "rotation": 12, "opacity": .8}
            ]},
            {"id": "back", "name": "Back", "linkedToPrevious": True, "pieces": []},
        ],
    }


def test_web_project_schema_preserves_multi_image_layer_transforms_and_links():
    result = validate_web_project(sample_project())
    piece = result["layers"][0]["pieces"][0]
    assert result["version"] == 1
    assert (piece["x"], piece["y"], piece["rotation"]) == (.4, .6, 12.0)
    assert result["layers"][1]["linkedToPrevious"] is True


def test_web_project_schema_rejects_pieces_with_unknown_sources():
    payload = sample_project()
    payload["layers"][0]["pieces"][0]["sourceId"] = "missing"
    with pytest.raises(ProjectValidationError, match="known source"):
        validate_web_project(payload)


def test_web_project_schema_rejects_unsafe_dimensions():
    payload = sample_project()
    payload["canvas"]["widthMm"] = 5000
    with pytest.raises(ProjectValidationError, match="canvas.widthMm"):
        validate_web_project(payload)


def test_web_editor_assets_exist_and_call_python_validation_api():
    assert (WEB_ROOT / "index.html").is_file()
    assert "/api/projects/validate" in (WEB_ROOT / "app.js").read_text()
    assert "Backlit" in (WEB_ROOT / "index.html").read_text()


def test_browser_project_operations_with_node():
    if not shutil.which("node"):
        pytest.skip("Node is required for browser model tests")
    test_file = Path(__file__).parent / "web" / "editor-core.test.mjs"
    result = subprocess.run(["node", "--test", str(test_file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_upload_accepts_real_images_and_rejects_arbitrary_bytes():
    from http.server import ThreadingHTTPServer
    from lightbox.web import LightboxWebHandler
    import threading
    import urllib.error
    import urllib.request

    server = ThreadingHTTPServer(("127.0.0.1", 0), LightboxWebHandler)
    server.source_images = {}
    server.piece_masks = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        image_bytes = io.BytesIO()
        Image.new("RGB", (12, 8), "red").save(image_bytes, format="PNG")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/sources/source-a",
            data=image_bytes.getvalue(), method="POST", headers={"Content-Type": "image/png"})
        result = json.loads(urllib.request.urlopen(request).read())
        assert result["ok"] and (result["width"], result["height"]) == (12, 8)
        assert server.source_images["source-a"].startswith(b"\x89PNG")

        bad_request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/sources/bad",
            data=b"not an image", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad_request)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_web_document_builds_real_python_project_from_transformed_pieces():
    payload = sample_project()
    image_bytes = io.BytesIO()
    Image.new("RGBA", (20, 10), (20, 40, 80, 255)).save(image_bytes, format="PNG")
    result = web_project_to_project(payload, {"source-a": image_bytes.getvalue()}, {}, max_pixels=80)
    assert len(result.layers) == 2
    assert result.layers[0].mask.any()
    assert result.layers[0].rotation_degrees == 0
    assert result.layers[0].fuse_with_next is True
    assert result.width_mm == 180
