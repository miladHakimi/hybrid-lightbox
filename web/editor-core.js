export function createProject() {
  return {
    version: 1,
    canvas: { widthMm: 180, heightMm: 120 },
    sources: [],
    layers: [0, 1, 2].map((index) => ({
      id: `layer-${index + 1}`,
      name: ["Foreground", "Middle", "Background"][index],
      linkedToPrevious: false,
      visible: true,
      mode: "hybrid",
      minimumMm: .4,
      maximumMm: 1.6,
      attenuation: .75,
      gamma: 1,
      invert: false,
      penStrokes: [],
      pieces: [],
    })),
  };
}

export function addSource(project, source) {
  if (project.sources.some((item) => item.id === source.id)) throw new Error("Duplicate source ID");
  return { ...project, sources: [...project.sources, { id: source.id, name: source.name }] };
}

export function addPiece(project, layerId, piece) {
  if (!project.sources.some((source) => source.id === piece.sourceId)) throw new Error("Unknown source");
  return {
    ...project,
    layers: project.layers.map((layer) => layer.id === layerId
      ? { ...layer, pieces: [...layer.pieces, { x: .5, y: .5, width: .5, height: .5, rotation: 0, opacity: 1, ...piece }] }
      : layer),
  };
}

export function updatePiece(project, pieceId, patch) {
  const bounded = { ...patch };
  if ("width" in bounded) bounded.width = Math.max(.01, Math.min(5, Number(bounded.width)));
  if ("height" in bounded) bounded.height = Math.max(.01, Math.min(5, Number(bounded.height)));
  if ("opacity" in bounded) bounded.opacity = Math.max(.05, Math.min(1, Number(bounded.opacity)));
  return {
    ...project,
    layers: project.layers.map((layer) => ({
      ...layer,
      pieces: layer.pieces.map((piece) => piece.id === pieceId ? { ...piece, ...bounded } : piece),
    })),
  };
}

export function toggleLink(project, layerId) {
  return {
    ...project,
    layers: project.layers.map((layer, index) => layer.id === layerId && index > 0
      ? { ...layer, linkedToPrevious: !layer.linkedToPrevious }
      : layer),
  };
}

export function findPiece(project, pieceId) {
  for (const layer of project.layers) {
    const piece = layer.pieces.find((item) => item.id === pieceId);
    if (piece) return { layer, piece };
  }
  return null;
}

export function addLayer(project, name = `Layer ${project.layers.length + 1}`) {
  const id = `layer-${globalThis.crypto?.randomUUID?.() || Date.now()}`;
  return { ...project, layers: [...project.layers, {
    id, name, linkedToPrevious: false, visible: true, mode: "hybrid",
    minimumMm: .4, maximumMm: 1.6, attenuation: .75, gamma: 1, invert: false,
    penStrokes: [], pieces: [],
  }] };
}

export function removeLayer(project, layerId) {
  if (project.layers.length <= 1) throw new Error("A project needs at least one layer");
  const layers = project.layers.filter((layer) => layer.id !== layerId)
    .map((layer, index) => index === 0 ? { ...layer, linkedToPrevious: false } : layer);
  return { ...project, layers };
}

export function moveLayer(project, layerId, direction) {
  const index = project.layers.findIndex((layer) => layer.id === layerId);
  const destination = index + direction;
  if (index < 0 || destination < 0 || destination >= project.layers.length) return project;
  const layers = [...project.layers];
  [layers[index], layers[destination]] = [layers[destination], layers[index]];
  return { ...project, layers: layers.map((layer, position) => position === 0 ? { ...layer, linkedToPrevious: false } : layer) };
}

export function updateLayer(project, layerId, patch) {
  return { ...project, layers: project.layers.map((layer) => layer.id === layerId ? { ...layer, ...patch } : layer) };
}
