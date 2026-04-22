# 04 — Thermal Model

## Scope

Temperature bounds, phase transitions, diffusion, and latent-heat behavior.

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Bounds

- **Temperature is bounded.** Every `cell.temp` stays within a documented
  range (currently `i16` storage → `-32_768..=32_767` hard limit; practical
  clamp should be tighter, e.g., `-500..=10_000°C`). Assert no cell drifts
  outside this range under any paint/reaction sequence.

- **Ambient offset bounded.** `World.ambient_offset` stays in a documented
  range. No UI or paint action may push it past its clamp.

### Phase transitions

- **Each element with Phase data transitions at its threshold.** For every
  element that defines a `Phase`, heat a cell above the threshold and tick
  until it transitions. Assert the resulting cell matches `Phase.target`.

- **Cooling reverses transitions where applicable.** Steam below 100°C
  condenses to water. Molten iron below its solidification point refreezes.
  Elements without a cooling phase-return are noted and excluded.

- **No spontaneous transitions below threshold.** A cell held below its
  phase threshold for many frames never transitions.

### Latent heat

- **Latent-heat budget limits transition rate.** During a phase transition,
  a cell's temperature change per frame respects the `latent` value. A
  high-latent transition (e.g., water→steam) does not complete in a single
  frame regardless of how much heat is applied.

  *Why:* Without a latent cap, extreme heat sources would teleport past
  phase points and never produce the intermediate-temperature "boiling"
  state the sim relies on.

### Diffusion

- **Heat diffusion converges.** Two adjacent cells at different temperatures
  equalize toward a common value over N ticks, monotonically (no
  oscillation, no blow-up).

- **Diffusion preserves total energy approximately.** Summed temperature
  across a closed insulated region drifts only within a documented
  tolerance per tick.

### Coupling

- **Thermal target feeds into pressure target.** Gas cells respond to
  temperature: hotter → higher target pressure. The exact formula is
  implementation-detail; the property is monotonicity — `target_p` at
  higher `temp` ≥ `target_p` at lower `temp` (all else equal).

## Known regressions

- (Placeholder — add entries here for specific thermal bugs we fix.)

## Out of scope

- Combustion ignition — covered in `05-reactions.md`.
- Joule heating from current flow — covered in `09-electrical.md`.
