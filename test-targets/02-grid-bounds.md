# 02 — Grid & Bounds Safety

## Scope

All index math stays inside the grid. External inputs (paint center, brush
radius, shockwave origin) can be pathological; the sim must clamp or reject
them without panicking.

## Prerequisites

`00-headless-harness.md`.

## Invariants

- **Paint clamps to grid.** Calling `paint(cx, cy, radius, ...)` with any
  `cx`, `cy`, `radius` (including negative, zero, and values far outside
  the grid) must not panic and must only modify cells where
  `0 ≤ x < WIDTH` and `0 ≤ y < HEIGHT`.

- **Prefab spawning clamps to grid.** Beaker, Box, and Battery prefabs
  placed at any center and any size must not write outside the grid.

- **Wire painting clamps to grid.** Wire tool with any endpoints and any
  thickness must not write outside the grid.

- **Shockwaves handle off-grid origins.** Spawning a shockwave at
  `(cx, cy)` outside the grid, or with radius that propagates past the
  edge, must tick through without panicking. Cells that would be affected
  outside the grid are silently skipped.

- **Internal scratch buffers are grid-sized.** `temp_scratch`,
  `pressure_scratch`, `support_scratch`, `vacuum_moved`, `wind_exposed`,
  `energized`, `cathode_mask`, `anode_mask`, `u_component_size`,
  `u_burst_committed`, `u_component_cx`, `u_component_cy`,
  `u_central_blast_fired` — all must be exactly `W * H` in length after
  `World::new()` and stay that way across ticks.

- **History ring buffer bounded.** `history` never grows past
  `HISTORY_CAPACITY` regardless of how many frames are ticked. `history_write`
  and `history_count` stay within valid ranges.

- **Shockwave vec bounded.** The active shockwave count never exceeds a
  documented ceiling (see `07-shockwaves.md` for the specific bound).

## Why

Every index into a `Vec<Cell>` is a potential OOB panic. The sim has many
code paths that compute offsets (`y * W + x`, neighbor offsets, shockwave
radius expansion, wall-burst probes). Fuzzing these with extreme inputs is
exactly the kind of thing Cygent is well-suited for.

## Out of scope

- Physical plausibility of what happens when you paint pathologically —
  only that the sim doesn't crash.
