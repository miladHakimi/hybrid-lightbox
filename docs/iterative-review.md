# Iterative Lightbox Review Process

This process is used for every customer image submitted for lightbox conversion.
It is an agent-orchestrated workflow around `lightbox-cli`; the CLI generates and
validates the artifacts, while a separate review agent makes the design decision.

## Loop

1. Create `round_1` with `lightbox-cli generate`.
2. Give the source image and the complete round directory to an independent
   reviewer agent. Do not give the reviewer the generation agent's reasoning or
   suggested verdict.
3. The reviewer returns one verdict: `APPROVE` or `REVISE`, plus findings in the
   schema below.
4. On `REVISE`, adjust generation parameters or layer assignments, regenerate
   every artifact into the next round directory, and request a fresh review.
5. Stop immediately on `APPROVE`. Otherwise stop after review round 5 and return
   the best round with all unresolved findings clearly identified.

Never overwrite a prior round. Each round must remain available for comparison.

## Required artifacts per round

- `layer_*.stl`
- `layer_*_mask.png`
- `layer_*_structure.png`
- `backlit_preview.png`
- `project.hbox`
- `report.json`
- `review.json` after independent review

## Reviewer rubric

The independent reviewer is deliberately strict. Approval requires all checks
below to pass.

### Print integrity

- Every STL is watertight.
- Every physical layer has exactly one connected component after structural
  bridging; there are no loose or floating parts.
- Bridges and minimum features are printable at the intended scale.
- Small decorative islands are removed, enlarged, or attached without visibly
  damaging the composition.

### Semantic layer integrity

- A visually coherent object is assigned to one physical layer. An object must
  not be split across layers merely because its highlights and shadows differ.
- Necessary internal holes may remain, but the object's printable material must
  form a solid, connected structure.
- Repeated elements may occupy different layers only when they are distinct
  objects in the source image.

### Registration and alignment

- All layers have identical physical width, height, origin, orientation, and
  outer-frame geometry.
- Layer masks and structures align with the source and with one another.
- No crop, resize, or aspect-ratio drift occurs between layers.

### Optical result

- The backlit preview preserves the important silhouette, focal subject, and
  readable depth order.
- Foreground layers do not unintentionally block essential rear-layer detail.
- Structural bridges are hidden behind foreground material where practical.
- Tonal separation remains useful when printed and illuminated.

## Reviewer output

`review.json` uses this shape:

```json
{
  "round": 1,
  "verdict": "REVISE",
  "summary": "Short independent assessment",
  "checks": {
    "watertight": "PASS",
    "single_piece_per_layer": "PASS",
    "semantic_object_integrity": "FAIL",
    "registration": "PASS",
    "printability": "WARN",
    "optical_quality": "FAIL"
  },
  "findings": [
    {
      "severity": "BLOCKER",
      "artifact": "layer_2_mask.png",
      "region": "plain-language location",
      "problem": "what is wrong",
      "required_change": "specific correction",
      "evidence": "observable reason"
    }
  ],
  "approved": false
}
```

Any `BLOCKER` or `FAIL` forces `REVISE`. Approval is permitted with minor
non-printability-neutral warnings only when the reviewer explains why they do
not affect the finished lightbox.

## Generation response to feedback

The generation agent must map every reviewer finding to a concrete change, such
as layer count, AI segmentation points or quality, minimum segment area, feature
and hole filtering, hidden bridges versus backing, frame dimensions, smoothing,
mesh resolution, occlusion optimization, or manual mask/layer reassignment.

The next round must regenerate the STL files, masks, structures, preview,
project, and report together. Reusing stale artifacts from an earlier round is
not allowed.

## Delivery

The final package contains:

- the accepted/best round in `final/`;
- all numbered rounds and reviewer reports in `rounds/`;
- a short iteration summary listing changes and verdicts;
- a ZIP archive containing the full package.

