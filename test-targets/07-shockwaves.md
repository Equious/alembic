# 07 — Shockwaves

## Scope

The impulsive pressure fronts spawned by gunpowder ignition, U criticality,
and highly exothermic reactions. These are the sim's most visually
dramatic primitive and also a common source of "lag / too many at once"
issues.

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Bounds & validity

- **Yield clamped.** `spawn_shockwave` and `spawn_shockwave_capped` must
  never write a `Shockwave` whose `yield_p` is NaN, infinite, or outside
  the documented clamp.

- **Radius grows monotonically until retirement.** Each tick a
  shockwave's `radius` strictly increases until the wave's effective
  magnitude falls below its retire threshold. Then it's removed from
  `shockwaves`.

- **Origin coords finite.** `cx`, `cy` are finite floats for every active
  shockwave.

### Count bounds

- **Active shockwave count has a ceiling.** Document the maximum number of
  concurrent shockwaves the sim permits. Under the worst-case detonation
  (e.g., full 5000-atom U criticality or dense gunpowder pile), the
  active count never exceeds this ceiling.

  *Why:* The user observed a regression where "a bunch more shockwaves
  (to the degree that it lags things out)" appeared after a change.
  Hard cap prevents lag-inducing cascades.

- **Shockwaves retire predictably.** A shockwave spawned with yield Y
  retires within a deterministic number of frames given no new spawns.
  Test: spawn one shockwave in an empty world, tick until `shockwaves`
  is empty, assert that count was reasonable (< 200 frames for a typical
  yield).

### Behavior

- **Leading edge affects pressure ahead.** A shockwave radiating into a
  pressurized gas region increases pressure in the cells the leading
  edge crosses (or triggers wall burst if it hits a frozen wall above
  threshold).

- **Falloff with distance.** The effective magnitude at radius `r` is
  `yield_p / (1 + r/r0)^2`. Tests can assert the max pressure spike
  from a single shockwave decreases with distance from origin.

- **Shockwaves do not re-trigger themselves.** A shockwave's own pressure
  injection must not cause the cells it crosses to spawn new shockwaves
  in a runaway cascade. (Gunpowder chain reactions are legit — those
  arise from the ignited cell's own detonation logic, not from the
  shockwave pressure directly.)

## Known regressions

- **Cascade lag-out.** See "too many shockwaves" above. The hard count
  cap and/or the "one central blast per component" flag in U
  criticality both mitigate this. Regression tests around U criticality
  (see `06-nuclear-criticality.md`) should catch reintroductions.

## Out of scope

- U criticality central-blast logic specifically — covered in
  `06-nuclear-criticality.md`.
- Wall-burst interaction — covered in `03-pressure-model.md`.
