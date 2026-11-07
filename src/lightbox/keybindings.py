from __future__ import annotations

import json
from pathlib import Path


COMMANDS = (
    ("history.undo", "Undo"),
    ("history.redo", "Redo"),
    ("tool.brush", "Select Brush tool"),
    ("tool.wand", "Select Magic Wand tool"),
    ("tool.transform", "Select Transform tool"),
    ("tool.pen", "Select output Pen tool"),
    ("view.image", "View: Image"),
    ("view.segmentations", "View: Segmentations"),
    ("view.selected_segment", "View: Selected segment"),
    ("view.layer_segments", "View: Layer segments"),
    ("view.connectors", "View: Connector view"),
    ("view.depth", "View: Depth map"),
    ("view.selected_mask", "View: Selected mask"),
    ("view.backlit", "View: Backlit preview"),
    ("segment.add", "Add selected segment to layer"),
    ("segment.erase", "Erase selected segment from layer"),
)

DEFAULT_BINDINGS = {"history.undo": "Ctrl+Z", "history.redo": "Ctrl+Shift+Z"}

COMMAND_IDS = frozenset(command_id for command_id, _label in COMMANDS)
EMPTY_BINDING_FILE = {"version": 1, "bindings": {}}

TOOLS = {"tool.brush": "Brush", "tool.wand": "Magic wand", "tool.transform": "Transform", "tool.pen": "Pen"}
VIEWS = {
    "view.image": "Image", "view.segmentations": "Segmentations",
    "view.selected_segment": "Selected segment", "view.layer_segments": "Layer segments",
    "view.connectors": "Connector view", "view.depth": "Depth map",
    "view.selected_mask": "Selected mask", "view.backlit": "Backlit preview",
}


def dispatch_keybinding(command_id, select_tool, select_view, apply_segment, undo=None, redo=None):
    """Route a stored command to the editor without depending on Qt."""
    if command_id == "history.undo" and undo:
        undo()
    elif command_id == "history.redo" and redo:
        redo()
    elif command_id in TOOLS:
        select_tool(TOOLS[command_id])
    elif command_id in VIEWS:
        select_view(VIEWS[command_id])
    elif command_id == "segment.add":
        apply_segment("add")
    elif command_id == "segment.erase":
        apply_segment("erase")


class KeybindingStore:
    """Persist user shortcuts separately from project/session data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            self.save({})
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            bindings = payload.get("bindings", {})
            if not isinstance(bindings, dict):
                return {}
            return {
                command_id: shortcut
                for command_id, shortcut in bindings.items()
                if command_id in COMMAND_IDS and isinstance(shortcut, str) and shortcut.strip()
            }
        except (OSError, json.JSONDecodeError, AttributeError):
            return {}

    def save(self, bindings: dict[str, str]) -> None:
        clean = {
            command_id: shortcut.strip()
            for command_id, shortcut in bindings.items()
            if command_id in COMMAND_IDS and isinstance(shortcut, str) and shortcut.strip()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": 1, "bindings": clean}, indent=2) + "\n",
            encoding="utf-8",
        )
