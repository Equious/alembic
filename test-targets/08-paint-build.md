# 08 — Paint & Build-Mode

## Scope

User-driven cell creation paths: the Paint tool, Prefab tool (Beaker, Box,
Battery), Wire tool, Vacuum and Pipet tools. These are the primary
entrypoints the player uses to influence the sim.

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Paint tool

- **Paint produces cells with consistent Kind/Element pairing.** Every
  cell written by `paint` has a `Kind` matching the element's default
  kind (e.g., painting Water produces Kind::Liquid cells, painting Iron
  produces Kind::Gravel cells). No mismatched pairings.

- **Frozen flag respected.** When `frozen=true` is passed, painted cells
  have `frozen=true`; when `false`, they don't. The flag is not silently
  flipped by downstream passes within the same tick.

- **Brush radius 0 paints exactly the center cell.** Not a ring, not
  nothing.

- **Brush radius N paints a disk, not a square.** Cells at Chebyshev
  distance ≤ N but Euclidean distance > N are NOT painted (or vice
  versa, whichever the intended metric is — pick one and assert it).

### Prefab tool

- **Beaker produces a valid container.** Every prefab-spawned Beaker:
  has a closed bottom and side walls of the chosen material, an open top,
  and the interior is Empty cells. No overlapping kinds, no frozen/loose
  mismatch on the walls.

- **Box produces a sealed container.** Closed on all four sides,
  interior Empty. Walls all frozen.

- **Battery produces a valid circuit element.** Battery cells of the
  chosen material are placed with documented layout (anode/cathode
  sides). Energized flood-fill from a Battery picks it up.

- **Prefabs respect size sliders.** A beaker at size N has the expected
  interior dimensions; wire at thickness T produces a T-wide band (not
  T+1, not 0).

### Wire tool

- **Wire thickness accuracy.** Painting a wire from point A to B with
  thickness T produces a connected band exactly T cells wide along its
  perpendicular axis.

- **Wire material matches selection.** Cells painted are the selected
  wire material (copper, etc.), not the generic Element::Metal.

### Pressure inheritance

- **Build-mode solids inherit replaced-cell pressure.** Painting a solid
  where a pressurized gas existed produces a cell whose pressure equals
  the prior cell's pressure, not 0.

  *Why:* Without this, fresh walls register artificial pressure gaps
  with neighbors and spontaneously crumble mid-stroke (known failure
  mode).

- **Paint-pressure fresh-vs-overpaint.** See `03-pressure-model.md` —
  these invariants are shared with the pressure model. Reference here,
  don't duplicate.

### Vacuum & Pipet

- **Vacuum tool moves cells, doesn't duplicate.** After one vacuum
  operation, the net atom count is conserved (atoms pulled from region
  A equal atoms appearing in region B, within per-frame probability
  tolerance).

- **Pipet captures a valid element for later painting.** The pipet's
  stored element is a valid `Element` discriminant, even when pipet-ing
  from a derived-compound cell.

## Known regressions

- **Wall crumble mid-stroke.** Historic bug where build-mode solids
  didn't inherit pressure, causing immediate spontaneous burst. Fixed;
  regression-test via the "Build-mode solids inherit" invariant above.

## Out of scope

- UI event handling (button clicks, dropdown open/close) — not part of
  the sim; tested only via integration tests that use the sim APIs
  directly.
