import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_main_window_has_no_pin_object_to_layer_feature():
    source = (Path(__file__).parents[1] / "src/lightbox/app.py").read_text()
    tree = ast.parse(source)
    main_window = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    method_names = {
        node.name for node in main_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    attributes = {
        node.attr for node in ast.walk(main_window)
        if isinstance(node, ast.Attribute)
    }

    assert "Pin object to layer" not in source
    assert "pin_object" not in method_names
    assert {"pin_button", "pinning", "ai_points"}.isdisjoint(attributes)


def test_layer_settings_ui_has_apply_all_checkbox_and_layer_title():
    source = (Path(__file__).parents[1] / "src/lightbox/app.py").read_text()
    assert "▸  LAYER SETTINGS" in source
    assert 'QCheckBox("Apply settings to all layers")' in source
    assert "self.apply_settings_to_all.setChecked(True)" in source
    assert "ALL LAYER SETTINGS" not in source


def test_brush_and_pen_size_control_updates_canvas_cursor_circle():
    source = (Path(__file__).parents[1] / "src/lightbox/app.py").read_text()
    tree = ast.parse(source)
    canvas = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Canvas")
    main_window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    canvas_methods = {node.name: node for node in canvas.body if isinstance(node, ast.FunctionDef)}
    tool_value_changed = next(node for node in main_window.body if isinstance(node, ast.FunctionDef) and node.name == "tool_value_changed")

    assert {"set_brush_radius", "_apply_tool_cursor"} <= canvas_methods.keys()
    assert "self.tool_mode in ('brush', 'pen')" in ast.unparse(canvas_methods["_apply_tool_cursor"])
    assert "self.tool_box.currentText() in ('Brush', 'Pen')" in ast.unparse(tool_value_changed)
    assert "self.canvas.set_brush_radius(value)" in ast.unparse(tool_value_changed)


def test_add_layer_action_allows_choosing_any_other_layer():
    source = (Path(__file__).parents[1] / "src/lightbox/app.py").read_text()
    tree = ast.parse(source)
    main_window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_window.body if isinstance(node, ast.FunctionDef) and node.name == "add_selected_layer")
    implementation = ast.unparse(method)

    assert 'QPushButton("Add layer…")' in source
    assert "if i != target" in implementation
    assert "QInputDialog.getItem" in implementation
    assert "self.project.add_layer(target, source)" in implementation
    assert "add_previous_layer" not in source


def test_full_mask_action_selects_the_entire_current_layer():
    source = (Path(__file__).parents[1] / "src/lightbox/app.py").read_text()
    tree = ast.parse(source)
    main_window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_window.body if isinstance(node, ast.FunctionDef) and node.name == "full_mask")
    implementation = ast.unparse(method)

    assert 'QPushButton("Full mask"' in source
    assert "self.full_mask_button.clicked.connect(self.full_mask)" in source
    assert "self.record_history()" in implementation
    assert "self.project.fill_layer_mask(li)" in implementation
    assert "self.view_box.setCurrentText" in implementation
    assert "Selected mask" in implementation


def test_main_image_and_illustration_share_a_resizable_horizontal_splitter():
    pytest.importorskip("PySide6")
    script = textwrap.dedent(
        """
        import os
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from lightbox.app import MainWindow

        app = QApplication([])
        MainWindow._restore_session = lambda self: None
        window = MainWindow()
        window.show()
        app.processEvents()
        splitter = window.preview_splitter
        assert splitter.orientation() == Qt.Horizontal
        assert splitter.count() == 2
        assert splitter.widget(0).isAncestorOf(window.stage_title)
        assert splitter.widget(0).isAncestorOf(window.canvas_stack)
        assert splitter.widget(1).isAncestorOf(window.illustration)
        assert not splitter.childrenCollapsible()
        before = splitter.sizes()
        splitter.setSizes([before[0] // 2, before[1] + before[0] // 2])
        app.processEvents()
        after = splitter.sizes()
        assert after[0] < before[0]
        assert after[1] > before[1]
        os._exit(0)
        """
    )
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-c", script], env=environment)
    assert result.returncode == 0
