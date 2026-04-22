# 03 — Pressure Model

## Scope

The pressure, buoyancy, and atmospheric-gradient rules the user has
validated through multiple iterations. These are the most regression-prone
invariants in the sim because they interact with almost every other pass
(thermal, reactions, gas expansion, wall burst).

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Empty = vacuum

- **Empty cells act as vacuum, not air.** Spawning gas cells adjacent to
  Empty causes gas to expand into Empty with high per-frame probability
  (~50% per adjacent Empty).

- **Sealed boxes fill with gas.** Paint a closed frozen wall box containing
  a small gas source. After enough ticks, every interior cell contains gas
  (or is reachable from gas); no permanent Empty pockets persist at the top.

  *Why:* The user previously tried Empty=air-mass-29 and heavy gases
  stratified with permanent air gaps. Vacuum interpretation is the one
  that works.

### Buoyancy is a global force

- **Gas rise/sink is global, not neighbor-based.** An H₂ cell in an
  otherwise empty sealed chamber still drifts upward. An O₂ cell still
  drifts downward. Stratification emerges from differing rise/sink rates,
  not from neighbor mass comparison.

### Pressure conservation (sealed vessel retention)

- **Gas cells blend UP only toward thermal+hydrostatic target.** If
  current pressure > target, cell stays above target. If current < target,
  cell approaches target. Never the reverse.

- **Sealed gas-filled boxes retain overpressure.** Paint a closed frozen
  box, fill with high-pressure gas, tick many frames. Mean interior
  pressure does not decay below the thermal target.

- **Non-gas cells (Empty, solid, liquid) DO blend down.** Leftover pressure
  in those cells decays toward the altitude baseline so spikes don't
  persist forever.

### Paint pressure

- **Fresh-spawned gas uses formation pressure only.** Paint one fresh
  gas cell; its pressure matches the element's `formation_pressure`
  (20–30 for atomic gases). No +400 boost on first spawn.

- **Overpaint stacks +400/frame, capped.** Painting the same element on
  the same cell for consecutive frames adds ~400 pressure per frame, up
  to i16 max.

  *Why:* Lets players pressurize sealed containers by holding the button
  without heavy gases geysering out of nearby openings during routine
  fill.

- **Build-mode solids inherit replaced-cell pressure.** Paint a solid
  where a pressurized gas/liquid existed; the new solid takes the same
  pressure, not 0.

  *Why:* Prevents fresh walls from registering artificial pressure gaps
  with neighbors (which would trigger spurious wall-burst crumble
  mid-stroke).

### Play-space boundaries

- **Left/right edges are open.** Pressure diffuses out to P=0 at
  horizontal edges. Gas cells touching horizontal edges have ~5% per-frame
  dissipation.

- **Top/bottom edges are sealed.** Sky/ground walls retain pressure
  normally.

- **Without horizontal openness, play space accumulates indefinitely.**
  (This is a property of the model, not a test — it's why the openness
  exists. If someone proposes closing horizontal edges, this memory
  should block it.)

### Atmospheric altitude gradient

- **Empty cells contribute small hydrostatic weight.** Each Empty cell's
  contribution to column integration is ~`AMBIENT_AIR.molar_mass × 0.02`.
  Empty cells near the floor have higher `target_p` than empty cells
  near the ceiling even when nothing is painted.

- **Frozen walls break the hydrostatic column.** When column integration
  hits any frozen cell (any Kind — Solid, Gravel, Powder, etc.), `col_p`
  resets to 0 and the wall's own target is skipped. Tall iron columns
  must not self-saturate and burst.

  *Why:* Kind-specific checks (e.g., `Kind::Solid` only) previously missed
  iron (Kind::Gravel, density 79 → cell_weight 39.5), causing tall iron
  columns to self-pressurize to 4000 and spurious-burst. The check must
  be `frozen`, not kind-based.

### Wall burst physics

- **Burst is unit-failure per same-element column.** The innermost frozen
  cell of a wall probes outward through cells of the SAME ELEMENT to
  measure thickness T. Threshold is `2500 + 350 × (T-1)`. Pressure gaps
  below threshold do nothing. At or above threshold, the ENTIRE column
  fails in one frame: outermost cell first, inner cells teleport outward.

- **Blocked cells shatter to Empty.** Cells that can't move (backed by
  material) convert to Empty so gas gets an escape path.

- **Composite walls fail per element, not per composite.** A glass window
  embedded in an iron wall fails on the glass's own thickness, not the
  composite's.

- **Non-frozen solids pressure-shove.** Loose painted solids call
  `try_pressure_shove` (same path as liquids/powders) — they're pushed by
  a blast gradient, not standing rigid.

## Known regressions

- **Iron column self-burst.** Historic bug where tall iron columns
  accumulated hydrostatic pressure and burst themselves. Fixed by making
  the column-break check `frozen`, not `Kind::Solid`. Any test covering
  the column break must pass for ALL frozen kinds.

- **Geyser-on-fill.** Historic bug where painting gas into a container
  caused geysers because overpaint-pressure applied to first spawn.
  Fixed by separating formation pressure (fresh) from overpaint (+400).

## Out of scope

- Thermal coupling into pressure target — covered in `04-thermal.md`.
- Shockwave pressure injection — covered in `07-shockwaves.md`.
