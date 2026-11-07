from __future__ import annotations

import threading
import zipfile
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from lightbox.web import LightboxWebHandler


def test_browser_composes_two_images_transforms_links_backlights_and_syncs(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    first = tmp_path / "red.png"
    second = tmp_path / "blue.png"
    Image.new("RGBA", (80, 60), (230, 50, 50, 220)).save(first)
    Image.new("RGBA", (60, 80), (40, 90, 230, 200)).save(second)

    server = ThreadingHTTPServer(("127.0.0.1", 0), LightboxWebHandler)
    server.source_images = {}
    server.piece_masks = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")

            upload = page.locator("#image-input")
            upload.set_input_files(str(first))
            page.locator("#sources .source").wait_for()
            assert page.locator("#stage .piece").count() == 1
            assert "1 piece" in page.locator(".layer-card").nth(0).inner_text()

            page.locator(".layer-card").nth(1).click()
            upload.set_input_files(str(second))
            page.wait_for_function("document.querySelectorAll('#stage .piece').length === 2")
            assert page.locator("#sources .source").count() == 2
            assert len(server.source_images) == 2
            assert len(server.piece_masks) == 2

            selected = page.locator("#stage .piece.selected")
            before = selected.bounding_box()
            page.mouse.move(before["x"] + before["width"] / 2, before["y"] + before["height"] / 2)
            page.mouse.down()
            page.mouse.move(before["x"] + before["width"] / 2 + 70, before["y"] + before["height"] / 2 + 35)
            page.mouse.up()
            after = selected.bounding_box()
            assert after["x"] > before["x"] + 40

            page.locator("input[data-field=rotation]").fill("32")
            assert "rotate(32deg)" in selected.get_attribute("style")

            piece_id = selected.get_attribute("data-piece-id")
            original_mask = server.piece_masks[piece_id]
            page.get_by_role("button", name="Brush").click()
            page.locator("#paint-mode").select_option("erase")
            box = selected.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2 + 15, box["y"] + box["height"] / 2)
            page.mouse.up()
            page.wait_for_function("document.querySelector('#sync-status').textContent !== 'Syncing…'")
            page.wait_for_timeout(500)
            assert server.piece_masks[piece_id] != original_mask, (errors, page.locator("#sync-status").inner_text(), page.locator("#stage").get_attribute("class"))
            page.locator("#undo").click(); page.wait_for_timeout(150)
            assert server.piece_masks[piece_id] == original_mask
            page.locator("#redo").click(); page.wait_for_timeout(150)
            assert server.piece_masks[piece_id] != original_mask

            before_wand = server.piece_masks[piece_id]
            page.get_by_role("button", name="Transform").click()
            page.get_by_role("button", name="Selected layer").click()
            page.locator(f'.piece[data-piece-id="{piece_id}"]').click()
            selected = page.locator("#stage .piece.selected")
            page.get_by_role("button", name="Wand").click()
            box = selected.bounding_box()
            wand_visual_before = selected.screenshot()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_timeout(250)
            assert server.piece_masks[piece_id] != before_wand
            assert selected.screenshot() != wand_visual_before
            page.locator("#undo").click(); page.wait_for_timeout(150)

            page.get_by_role("button", name="Transform").click()
            page.locator(f'.piece[data-piece-id="{piece_id}"]').click()
            selected = page.locator("#stage .piece.selected")
            page.get_by_role("button", name="Crop").click()
            box = selected.bounding_box()
            before_crop = server.piece_masks[piece_id]
            page.mouse.move(box["x"] + box["width"] / 2 - 20, box["y"] + box["height"] / 2 - 20); page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2 + 20, box["y"] + box["height"] / 2 + 20); page.mouse.up()
            page.wait_for_timeout(250)
            assert server.piece_masks[piece_id] != before_crop
            page.locator("#undo").click(); page.wait_for_timeout(150)

            page.locator("#layer-mode").select_option("silhouette")
            assert "silhouette" in page.locator(".layer-card.active").inner_text()

            page.get_by_role("button", name="Transform").click()
            page.locator(f'.piece[data-piece-id="{piece_id}"]').click()
            page.locator("#duplicate-piece").click()
            assert page.locator("#stage .piece").count() == 2
            page.locator("#piece-layer").select_option("layer-1")
            page.locator("#delete-piece").click()
            assert page.locator("#stage .piece").count() == 1

            page.get_by_role("button", name="Pen").click()
            stage = page.locator("#stage").bounding_box()
            page.mouse.move(stage["x"] + 40, stage["y"] + 40)
            page.mouse.down(); page.mouse.move(stage["x"] + 120, stage["y"] + 80); page.mouse.up()
            assert page.locator("#pen-preview polyline").count() == 1

            page.get_by_role("button", name="Backlit").click()
            assert "backlit" in page.locator("#stage").get_attribute("class")
            page.get_by_text("Authoritative Python preview").wait_for()
            assert "authoritative" in page.locator("#stage").get_attribute("class")

            page.locator(".link").first.click()
            assert "on" in page.locator(".link").first.get_attribute("class")

            page.get_by_role("button", name="Sync with Python").click()
            page.get_by_text("Python project in sync").wait_for()

            with page.expect_download() as download_info:
                page.get_by_role("button", name="Export STL / 3MF").click()
            download = download_info.value
            target = tmp_path / download.suggested_filename
            download.save_as(target)
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
            assert "model.3mf" in names and "layer_1.stl" in names and "backlit_preview.png" in names

            with page.expect_download() as project_download_info:
                page.get_by_role("button", name="Save project").click()
            project_archive = tmp_path / "roundtrip.lbox"
            project_download_info.value.save_as(project_archive)
            with zipfile.ZipFile(project_archive) as archive:
                assert "project.json" in archive.namelist()
                assert len([name for name in archive.namelist() if name.startswith("sources/")]) == 2

            page.reload()
            assert page.locator("#stage .piece").count() == 0
            page.locator("#open-project").set_input_files(str(project_archive))
            page.get_by_text("Project opened").wait_for()
            assert page.locator("#sources .source").count() == 2
            assert page.locator("#stage .piece").count() == 2
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
