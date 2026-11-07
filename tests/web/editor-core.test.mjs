import test from "node:test";
import assert from "node:assert/strict";
import { addLayer, addPiece, addSource, createProject, moveLayer, removeLayer, toggleLink, updateLayer, updatePiece } from "../../web/editor-core.js";

test("multiple image pieces retain independent transforms", () => {
  let project = createProject();
  project = addSource(project, { id: "one", name: "one.png" });
  project = addSource(project, { id: "two", name: "two.png" });
  project = addPiece(project, "layer-1", { id: "p1", sourceId: "one" });
  project = addPiece(project, "layer-1", { id: "p2", sourceId: "two" });
  project = updatePiece(project, "p1", { x: .2, rotation: 35 });
  assert.equal(project.layers[0].pieces[0].x, .2);
  assert.equal(project.layers[0].pieces[0].rotation, 35);
  assert.equal(project.layers[0].pieces[1].x, .5);
});

test("piece sizes are bounded away from zero", () => {
  let project = addSource(createProject(), { id: "one", name: "one.png" });
  project = addPiece(project, "layer-1", { id: "p1", sourceId: "one" });
  project = updatePiece(project, "p1", { width: -2, height: 0 });
  assert.equal(project.layers[0].pieces[0].width, .01);
  assert.equal(project.layers[0].pieces[0].height, .01);
});

test("only non-front layers can link to their predecessor", () => {
  let project = createProject();
  project = toggleLink(project, "layer-1");
  assert.equal(project.layers[0].linkedToPrevious, false);
  project = toggleLink(project, "layer-2");
  assert.equal(project.layers[1].linkedToPrevious, true);
});

test("layers can be added, reordered, hidden, and removed", () => {
  let project = addLayer(createProject(), "Glow");
  const id = project.layers[3].id;
  project = moveLayer(project, id, -1);
  assert.equal(project.layers[2].name, "Glow");
  project = updateLayer(project, id, { visible: false, mode: "silhouette" });
  assert.equal(project.layers[2].visible, false);
  assert.equal(project.layers[2].mode, "silhouette");
  project = removeLayer(project, id);
  assert.equal(project.layers.length, 3);
});
