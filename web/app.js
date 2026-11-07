import { addLayer, addPiece, addSource, createProject, findPiece, moveLayer, removeLayer, toggleLink, updateLayer, updatePiece } from "./editor-core.js";

let project = createProject();
let selectedLayerId = project.layers[0].id;
let selectedPieceId = null;
let currentView = "stack";
let currentTool = "transform";
const imageUrls = new Map();
const sourceImages = new Map();
const pieceMasks = new Map();
const undoStack = [];
const redoStack = [];
const stage = document.querySelector("#stage");
const uid = (prefix) => `${prefix}-${crypto.randomUUID()}`;
const selected = () => selectedPieceId ? findPiece(project, selectedPieceId) : null;
const activeLayer = () => project.layers.find((layer) => layer.id === selectedLayerId);

function cloneProject(value = project) { return structuredClone(value); }
function snapshot() { return { project: cloneProject(), masks: [...pieceMasks].map(([id, canvas]) => [id, canvas.width, canvas.height, canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height)]) }; }
function restoreSnapshot(state) { project = state.project; pieceMasks.clear(); for (const [id, width, height, data] of state.masks) { const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height; canvas.getContext("2d").putImageData(data, 0, 0); pieceMasks.set(id, canvas); } }
function checkpoint() { undoStack.push(snapshot()); if (undoStack.length > 40) undoStack.shift(); redoStack.length = 0; }
function setProject(next, remember = true) { if (remember) checkpoint(); project = next; stage.classList.remove("authoritative"); render(); scheduleSync(); }

function render() {
  const active = selected();
  stage.classList.toggle("backlit", currentView === "backlit");
  stage.classList.toggle("brush-cursor", currentTool !== "transform");
  stage.style.aspectRatio = `${project.canvas.widthMm}/${project.canvas.heightMm}`;
  stage.querySelectorAll(".piece").forEach((node) => node.remove());
  const visibleLayers = currentView === "layer" ? project.layers.filter((layer) => layer.id === selectedLayerId) : project.layers.filter((layer) => layer.visible);
  for (const layer of [...visibleLayers].reverse()) for (const piece of layer.pieces) stage.append(pieceNode(piece, piece.id === selectedPieceId));
  const selectedNode = selectedPieceId && stage.querySelector(`.piece[data-piece-id="${CSS.escape(selectedPieceId)}"]`);
  if (selectedNode) stage.append(selectedNode);
  stage.querySelector(".empty").style.display = project.sources.length ? "none" : "grid";
  renderPenStrokes(visibleLayers); renderLayers(); renderInspector(active);
  document.querySelector("#canvas-size").textContent = `${project.canvas.widthMm} × ${project.canvas.heightMm} mm`;
  document.querySelector("#undo").disabled = !undoStack.length; document.querySelector("#redo").disabled = !redoStack.length;
}

function pieceNode(piece, isSelected) {
  const node = document.createElement("div"); node.className = `piece${isSelected ? " selected" : ""}`; node.dataset.pieceId = piece.id;
  Object.assign(node.style, { left: `${(piece.x - piece.width / 2) * 100}%`, top: `${(piece.y - piece.height / 2) * 100}%`, width: `${piece.width * 100}%`, height: `${piece.height * 100}%`, transform: `rotate(${piece.rotation}deg)`, opacity: piece.opacity });
  const image = document.createElement("img"); image.src = imageUrls.get(piece.sourceId); image.alt = "";
  const mask = pieceMasks.get(piece.id); if (mask) { const url = mask.toDataURL("image/png"); image.style.maskImage = `url(${url})`; image.style.webkitMaskImage = `url(${url})`; image.style.maskSize = image.style.webkitMaskSize = "100% 100%"; }
  node.append(image); const handle = document.createElement("span"); handle.className = "handle"; node.append(handle);
  node.addEventListener("pointerdown", (event) => beginPieceAction(event, piece, event.target === handle));
  node.addEventListener("click", (event) => { event.stopPropagation(); selectedPieceId = piece.id; render(); }); return node;
}

function beginPieceAction(event, piece, onHandle) {
  event.preventDefault(); event.stopPropagation(); selectedPieceId = piece.id;
  if (currentTool === "transform") beginTransform(event, piece, onHandle ? "resize" : "move");
  else if (currentTool === "brush") beginBrush(event, piece);
  else if (currentTool === "wand") applyWand(event, piece);
  else if (currentTool === "crop") beginCrop(event, piece);
}

function beginTransform(event, piece, mode) {
  checkpoint(); const start = { x: event.clientX, y: event.clientY, piece: { ...piece } }; const rect = stage.getBoundingClientRect();
  const move = (next) => { const dx = (next.clientX - start.x) / rect.width, dy = (next.clientY - start.y) / rect.height; project = updatePiece(project, piece.id, mode === "move" ? { x: start.piece.x + dx, y: start.piece.y + dy } : { width: start.piece.width + dx * 2, height: start.piece.height + dy * 2 }); render(); };
  trackPointer(move, scheduleSync); render();
}

function trackPointer(move, finish) { const done = (event) => { window.removeEventListener("pointermove", move); window.removeEventListener("mousemove", move); window.removeEventListener("pointerup", done, true); window.removeEventListener("mouseup", done, true); window.removeEventListener("pointercancel", done, true); finish?.(event); }; window.addEventListener("pointermove", move); window.addEventListener("mousemove", move); window.addEventListener("pointerup", done, true); window.addEventListener("mouseup", done, true); window.addEventListener("pointercancel", done, true); }

function piecePoint(event, piece) {
  const rect = stage.getBoundingClientRect(); const dx = event.clientX - (rect.left + piece.x * rect.width); const dy = event.clientY - (rect.top + piece.y * rect.height); const angle = -piece.rotation * Math.PI / 180;
  return { x: (dx * Math.cos(angle) - dy * Math.sin(angle)) / (piece.width * rect.width) + .5, y: (dx * Math.sin(angle) + dy * Math.cos(angle)) / (piece.height * rect.height) + .5 };
}

function ensureMask(piece) {
  if (pieceMasks.has(piece.id)) return pieceMasks.get(piece.id); const source = sourceImages.get(piece.sourceId); const canvas = document.createElement("canvas"); canvas.width = source.naturalWidth; canvas.height = source.naturalHeight;
  const context = canvas.getContext("2d"); context.drawImage(source, 0, 0); context.globalCompositeOperation = "source-in"; context.fillStyle = "white"; context.fillRect(0, 0, canvas.width, canvas.height); context.globalCompositeOperation = "source-over"; pieceMasks.set(piece.id, canvas); return canvas;
}

function paintMask(piece, point) { const canvas = ensureMask(piece); const context = canvas.getContext("2d"); const radius = Number(document.querySelector("#brush-size").value) / 2; const erase = document.querySelector("#paint-mode").value === "erase"; context.globalCompositeOperation = erase ? "destination-out" : "source-over"; context.fillStyle = "white"; context.beginPath(); context.arc(point.x * canvas.width, point.y * canvas.height, radius, 0, Math.PI * 2); context.fill(); context.globalCompositeOperation = "source-over"; }
function beginBrush(event, piece) { checkpoint(); status("Painting mask…"); paintMask(piece, piecePoint(event, piece)); uploadMask(piece.id).then(() => status("Mask synced", "good")).catch(showError); render(); const move = (next) => { paintMask(piece, piecePoint(next, piece)); uploadMask(piece.id).catch(showError); render(); }; trackPointer(move, () => uploadMask(piece.id).then(() => status("Mask synced", "good")).catch(showError)); }

function applyWand(event, piece) {
  checkpoint(); const source = sourceImages.get(piece.sourceId); const point = piecePoint(event, piece); const width = source.naturalWidth, height = source.naturalHeight; const sourceCanvas = document.createElement("canvas"); sourceCanvas.width = width; sourceCanvas.height = height; const sourceContext = sourceCanvas.getContext("2d"); sourceContext.drawImage(source, 0, 0); const pixels = sourceContext.getImageData(0, 0, width, height).data;
  const sx = Math.max(0, Math.min(width - 1, Math.floor(point.x * width))), sy = Math.max(0, Math.min(height - 1, Math.floor(point.y * height))); const seed = (sy * width + sx) * 4; const tolerance = 2 + Number(document.querySelector("#wand-sensitivity").value) * 2; const seen = new Uint8Array(width * height); const queue = [sy * width + sx]; seen[queue[0]] = 1; const selectedPixels = [];
  while (queue.length) { const index = queue.pop(); selectedPixels.push(index); const x = index % width; for (const neighbor of [index - 1, index + 1, index - width, index + width]) { if (neighbor < 0 || neighbor >= width * height || seen[neighbor] || (neighbor === index - 1 && x === 0) || (neighbor === index + 1 && x === width - 1)) continue; const offset = neighbor * 4; const distance = Math.hypot(pixels[offset] - pixels[seed], pixels[offset + 1] - pixels[seed + 1], pixels[offset + 2] - pixels[seed + 2]); if (distance <= tolerance) { seen[neighbor] = 1; queue.push(neighbor); } } }
  const mask = ensureMask(piece); const context = mask.getContext("2d"); const image = context.getImageData(0, 0, width, height); const erase = document.querySelector("#paint-mode").value === "erase"; for (const index of selectedPixels) { const offset = index * 4; image.data[offset] = image.data[offset + 1] = image.data[offset + 2] = erase ? 0 : 255; image.data[offset + 3] = erase ? 0 : 255; } context.putImageData(image, 0, 0); render(); uploadMask(piece.id);
}

function beginCrop(event, piece) {
  checkpoint(); const start = piecePoint(event, piece); const canvas = ensureMask(piece); const original = document.createElement("canvas"); original.width = canvas.width; original.height = canvas.height; original.getContext("2d").drawImage(canvas, 0, 0);
  const apply = (next) => { const end = piecePoint(next, piece); const x = Math.min(start.x, end.x) * canvas.width, y = Math.min(start.y, end.y) * canvas.height, width = Math.abs(end.x - start.x) * canvas.width, height = Math.abs(end.y - start.y) * canvas.height; const context = canvas.getContext("2d"); context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(original, x, y, width, height, x, y, width, height); uploadMask(piece.id).catch(showError); render(); };
  trackPointer(apply, apply);
}

function beginPen(event) {
  if (currentTool !== "pen") return; event.preventDefault(); checkpoint(); const rect = stage.getBoundingClientRect(); const point = (next) => [(next.clientX - rect.left) / rect.width, (next.clientY - rect.top) / rect.height]; const stroke = { mode: document.querySelector("#paint-mode").value, radius: Number(document.querySelector("#brush-size").value) / rect.width, points: [point(event)] }; project = updateLayer(project, selectedLayerId, { penStrokes: [...activeLayer().penStrokes, stroke] }); render();
  const move = (next) => { stroke.points.push(point(next)); project = updateLayer(project, selectedLayerId, { penStrokes: [...activeLayer().penStrokes.slice(0, -1), stroke] }); render(); }; trackPointer(move, scheduleSync);
}

function renderPenStrokes(layers) { const svg = document.querySelector("#pen-preview"); svg.replaceChildren(); for (const layer of layers) for (const stroke of layer.penStrokes) { const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); path.setAttribute("points", stroke.points.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ")); path.setAttribute("fill", "none"); path.setAttribute("stroke", stroke.mode === "erase" ? "#f4f2ed" : "#26222d"); path.setAttribute("stroke-width", stroke.radius * 2000); path.setAttribute("stroke-linecap", "round"); path.setAttribute("stroke-linejoin", "round"); svg.append(path); } }

function renderLayers() {
  const strip = document.querySelector("#layers"); strip.replaceChildren(); project.layers.forEach((layer, index) => { if (index) { const link = document.createElement("button"); link.className = `link${layer.linkedToPrevious ? " on" : ""}`; link.textContent = "↔"; link.onclick = () => setProject(toggleLink(project, layer.id)); strip.append(link); }
    const card = document.createElement("div"); card.className = `layer-card${layer.id === selectedLayerId ? " active" : ""}`; card.innerHTML = `<b>${layer.visible ? "◉" : "○"} ${layer.name}</b><span>${layer.pieces.length} pieces · ${layer.mode}</span><div class="layer-actions"><button data-action="up">↑</button><button data-action="down">↓</button><button data-action="visible">◉</button><button data-action="delete">×</button></div>`;
    card.onclick = (event) => { const action = event.target.dataset.action; if (!action) { selectedLayerId = layer.id; render(); return; } event.stopPropagation(); if (action === "up") setProject(moveLayer(project, layer.id, -1)); if (action === "down") setProject(moveLayer(project, layer.id, 1)); if (action === "visible") setProject(updateLayer(project, layer.id, { visible: !layer.visible })); if (action === "delete") { const next = removeLayer(project, layer.id); selectedLayerId = next.layers[0].id; selectedPieceId = null; setProject(next); } }; strip.append(card); });
  const add = document.createElement("button"); add.className = "add-layer"; add.textContent = "+ Layer"; add.onclick = () => { const next = addLayer(project); selectedLayerId = next.layers.at(-1).id; setProject(next); }; strip.append(add);
}

function renderInspector(active) {
  const controls = document.querySelector("#transform-controls"); controls.classList.toggle("disabled", !active); document.querySelector("#selection-title").textContent = active ? active.layer.name : "No piece selected";
  const layer = activeLayer(); document.querySelector("#layer-mode").value = layer.mode; document.querySelector("#minimum-mm").value = layer.minimumMm; document.querySelector("#maximum-mm").value = layer.maximumMm; document.querySelector("#layer-gamma").value = layer.gamma; document.querySelector("#layer-attenuation").value = layer.attenuation; document.querySelector("#layer-invert").checked = layer.invert;
  const layerSelect = document.querySelector("#piece-layer"); layerSelect.replaceChildren(...project.layers.map((item) => { const option = document.createElement("option"); option.value = item.id; option.textContent = item.name; return option; }));
  if (!active) return; controls.querySelectorAll("input[data-field]").forEach((input) => { input.value = active.piece[input.dataset.field]; }); layerSelect.value = active.layer.id; document.querySelector("#rotation-value").textContent = `${Math.round(active.piece.rotation)}°`;
}

document.querySelector("#image-input").addEventListener("change", async (event) => { try { for (const file of event.target.files) { checkpoint(); const source = { id: uid("source"), name: file.name }; const url = URL.createObjectURL(file); imageUrls.set(source.id, url); const image = new Image(); image.src = url; await image.decode(); sourceImages.set(source.id, image); project = addSource(project, source); const piece = { id: uid("piece"), sourceId: source.id, width: .5, height: .5 }; project = addPiece(project, selectedLayerId, piece); selectedPieceId = piece.id; ensureMask(piece); await Promise.all([uploadSource(source.id, file), uploadMask(piece.id)]); } renderSources(); render(); scheduleSync(); } catch (error) { showError(error); } finally { event.target.value = ""; } });
async function uploadSource(sourceId, file) { const response = await fetch(`/api/sources/${encodeURIComponent(sourceId)}`, { method: "POST", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file }); const result = await response.json(); if (!response.ok) throw new Error(result.error || "Image upload failed"); }
async function uploadMask(pieceId) { const canvas = pieceMasks.get(pieceId); if (!canvas) return; const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png")); const response = await fetch(`/api/masks/${encodeURIComponent(pieceId)}`, { method: "POST", headers: { "Content-Type": "image/png" }, body: blob }); if (!response.ok) throw new Error((await response.json()).error); }
async function syncMasks() { await Promise.all([...pieceMasks.keys()].map(uploadMask)); }
function renderSources() { const list = document.querySelector("#sources"); list.replaceChildren(); for (const source of project.sources) { const item = document.createElement("button"); item.className = "source"; item.innerHTML = `<img alt=""><span></span>`; item.querySelector("img").src = imageUrls.get(source.id); item.querySelector("span").textContent = source.name; item.onclick = () => { checkpoint(); const piece = { id: uid("piece"), sourceId: source.id }; project = addPiece(project, selectedLayerId, piece); selectedPieceId = piece.id; ensureMask(piece); uploadMask(piece.id); render(); scheduleSync(); }; list.append(item); } }

document.querySelector("#transform-controls").addEventListener("input", (event) => { if (!event.target.dataset.field || !selectedPieceId) return; setProject(updatePiece(project, selectedPieceId, { [event.target.dataset.field]: Number(event.target.value) })); });
document.querySelectorAll(".view-switch button").forEach((button) => button.onclick = async () => { currentView = button.dataset.view; document.querySelectorAll(".view-switch button").forEach((item) => item.classList.toggle("active", item === button)); render(); if (currentView === "backlit") await requestPreview(); });
document.querySelectorAll(".tool-palette button").forEach((button) => button.onclick = () => { currentTool = button.dataset.tool; document.querySelectorAll(".tool-palette button").forEach((item) => item.classList.toggle("active", item === button)); const help = { transform: "Drag to move; use the corner to resize.", brush: "Paint the selected image mask.", wand: "Click a connected color region.", crop: "Drag the area to keep.", pen: "Draw material directly on the selected layer." }; document.querySelector("#tool-help").textContent = help[currentTool]; render(); });
stage.addEventListener("pointerdown", beginPen, true); stage.addEventListener("click", () => { if (currentTool === "transform") { selectedPieceId = null; render(); } });
for (const [id, field] of [["width-mm", "widthMm"], ["height-mm", "heightMm"]]) document.querySelector(`#${id}`).addEventListener("change", (event) => setProject({ ...project, canvas: { ...project.canvas, [field]: Number(event.target.value) } }));
document.querySelector("#piece-layer").onchange = (event) => { const active = selected(); if (!active) return; checkpoint(); const piece = active.piece; project = { ...project, layers: project.layers.map((layer) => layer.id === active.layer.id ? { ...layer, pieces: layer.pieces.filter((item) => item.id !== piece.id) } : layer.id === event.target.value ? { ...layer, pieces: [...layer.pieces, piece] } : layer) }; selectedLayerId = event.target.value; render(); scheduleSync(); };
document.querySelector("#duplicate-piece").onclick = () => { const active = selected(); if (!active) return; checkpoint(); const copy = { ...active.piece, id: uid("piece"), x: active.piece.x + .03, y: active.piece.y + .03 }; project = addPiece(project, active.layer.id, copy); const sourceMask = pieceMasks.get(active.piece.id); if (sourceMask) { const canvas = document.createElement("canvas"); canvas.width = sourceMask.width; canvas.height = sourceMask.height; canvas.getContext("2d").drawImage(sourceMask, 0, 0); pieceMasks.set(copy.id, canvas); uploadMask(copy.id); } selectedPieceId = copy.id; render(); scheduleSync(); };
document.querySelector("#delete-piece").onclick = () => { const active = selected(); if (!active) return; checkpoint(); project = { ...project, layers: project.layers.map((layer) => ({ ...layer, pieces: layer.pieces.filter((piece) => piece.id !== active.piece.id) })) }; pieceMasks.delete(active.piece.id); selectedPieceId = null; render(); scheduleSync(); };
for (const [id, field, checkbox] of [["layer-mode", "mode"], ["minimum-mm", "minimumMm"], ["maximum-mm", "maximumMm"], ["layer-gamma", "gamma"], ["layer-attenuation", "attenuation"], ["layer-invert", "invert", true]]) document.querySelector(`#${id}`).addEventListener("change", (event) => setProject(updateLayer(project, selectedLayerId, { [field]: checkbox ? event.target.checked : event.target.type === "number" ? Number(event.target.value) : event.target.value })));
document.querySelector("#undo").onclick = () => { if (!undoStack.length) return; redoStack.push(snapshot()); restoreSnapshot(undoStack.pop()); selectedPieceId = null; render(); syncMasks().then(scheduleSync).catch(showError); }; document.querySelector("#redo").onclick = () => { if (!redoStack.length) return; undoStack.push(snapshot()); restoreSnapshot(redoStack.pop()); selectedPieceId = null; render(); syncMasks().then(scheduleSync).catch(showError); };

let syncTimer; function scheduleSync() { clearTimeout(syncTimer); syncTimer = setTimeout(syncProject, 700); }
function status(text, kind = "") { const node = document.querySelector("#sync-status"); node.textContent = text; node.className = `status ${kind}`; } function showError(error) { status(error.message || String(error), "bad"); }
async function syncProject() { status("Syncing…"); try { const response = await fetch("/api/projects/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(project) }); const result = await response.json(); if (!response.ok) throw new Error(result.error); project = result.project; status("Python project in sync", "good"); render(); return true; } catch (error) { showError(error); return false; } }
async function requestPreview() { try { status("Rendering Python preview…"); const response = await fetch("/api/projects/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(project) }); if (!response.ok) throw new Error((await response.json()).error); const image = document.querySelector("#python-preview"); if (image.src.startsWith("blob:")) URL.revokeObjectURL(image.src); image.src = URL.createObjectURL(await response.blob()); stage.classList.add("authoritative"); status("Authoritative Python preview", "good"); } catch (error) { showError(error); } }
async function exportProject() { try { status("Generating STL / 3MF…"); const response = await fetch("/api/projects/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(project) }); if (!response.ok) throw new Error((await response.json()).error); const url = URL.createObjectURL(await response.blob()); const link = document.createElement("a"); link.href = url; link.download = "lightbox-export.zip"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); status("Export downloaded", "good"); } catch (error) { showError(error); } }
async function saveProject() { try { await syncMasks(); const response = await fetch("/api/projects/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(project) }); if (!response.ok) throw new Error((await response.json()).error); const url = URL.createObjectURL(await response.blob()); const link = document.createElement("a"); link.href = url; link.download = "lightbox-project.lbox"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); status("Project saved", "good"); } catch (error) { showError(error); } }
async function openProject(file) { try { status("Opening project…"); const response = await fetch("/api/projects/open", { method: "POST", headers: { "Content-Type": "application/zip" }, body: file }); const result = await response.json(); if (!response.ok) throw new Error(result.error); project = result.project; selectedLayerId = project.layers[0].id; selectedPieceId = null; imageUrls.clear(); sourceImages.clear(); pieceMasks.clear(); for (const source of project.sources) { const blob = await (await fetch(`/api/sources/${encodeURIComponent(source.id)}`)).blob(); const url = URL.createObjectURL(blob); imageUrls.set(source.id, url); const image = new Image(); image.src = url; await image.decode(); sourceImages.set(source.id, image); } for (const layer of project.layers) for (const piece of layer.pieces) { const response = await fetch(`/api/masks/${encodeURIComponent(piece.id)}`); if (response.ok) { const bitmap = await createImageBitmap(await response.blob()); const canvas = document.createElement("canvas"); canvas.width = bitmap.width; canvas.height = bitmap.height; canvas.getContext("2d").drawImage(bitmap, 0, 0); pieceMasks.set(piece.id, canvas); } } undoStack.length = 0; redoStack.length = 0; renderSources(); render(); status("Project opened", "good"); } catch (error) { showError(error); } }
document.querySelector("#validate").onclick = syncProject; document.querySelector("#export").onclick = exportProject; document.querySelector("#save-project").onclick = saveProject; document.querySelector("#open-project").onchange = (event) => { if (event.target.files[0]) openProject(event.target.files[0]); event.target.value = ""; }; render();
