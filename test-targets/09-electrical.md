# 09 — Electrical / Circuits

## Scope

The energized-cell flood-fill, battery voltage, galvanic EMF generation,
electrolysis (cathode/anode masking, metal plating), and Joule heating.

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Energized flood-fill

- **Energized bitmap is correct.** After a tick, a cell is marked
  `energized` iff it is reachable from any Battery via a path of
  conductors (plus one hop into a noble-gas cell for glow). Test with
  known circuit topologies:
  - Isolated conductor (no battery) → no cells energized.
  - Battery + connected wire → all wire cells energized.
  - Battery + wire broken by an insulator gap → only cells on the
    battery side energized.

- **Noble-gas glow hop.** A noble-gas cell adjacent to an energized
  conductor is itself energized. A noble-gas cell one further hop away
  is NOT (single-hop only).

- **Bitmap clears when no battery exists.** Paint a conductor network
  with no battery; energized bitmap is all-false.

### Battery voltage

- **Battery voltage propagates to `active_emf`.** With any battery in
  the scene, `active_emf == battery_voltage`.

- **No battery → fall back to galvanic.** With no battery but a galvanic
  pair in contact with brine, `active_emf == galvanic_voltage`.

- **No battery, no galvanic → zero.** `active_emf == 0.0`.

### Galvanic

- **EMF derived from EN gap.** `galvanic_voltage` is computed from the
  electronegativity difference between the cathode and anode metals in
  contact with brine. Higher EN gap → higher voltage. Two identical
  metals in brine → zero galvanic voltage.

- **Galvanic cathode/anode elements identified correctly.** In a
  Zn/Cu/brine cell, `galvanic_cathode_el == Some(Element::Cu)` (higher
  EN) and `galvanic_anode_el == Some(Element::Zn)`.

### Electrolysis

- **Cathode and anode masks reflect circuit topology.** A metal cell
  touching brine that's electrically connected (via non-brine conductors)
  to the battery's positive side is in `anode_mask`; negative side is
  in `cathode_mask`. A metal cell not in brine contact, or not
  electrically connected, is in neither.

- **Plating deposits on cathode.** Running electrolysis in a saturated
  salt solution for N frames produces new metal cells on the cathode
  side. The new metal is the cation species from the dissolved salt.

- **Adherent plating retains frozen state.** A fraction of plated cells
  (configured by `ADHERENT_FRAC`) are frozen; others are loose. Assert
  both populations exist after many plating events.

### Joule heating

- **Heating ∝ V² × resistance.** Doubling `battery_voltage` roughly
  quadruples the heat-per-frame applied to resistive elements in the
  circuit, within implementation tolerance.

- **Heating bounded.** Per-cell Joule heat addition per frame has a
  documented ceiling. Fuzz with extreme voltages and assert no cell
  `temp` jumps past the ceiling in a single frame.

- **Non-conductors don't heat.** An insulator cell adjacent to an
  energized wire does not receive Joule heating (conductive heat
  transfer via `04-thermal.md` is fine, but not direct Joule).

## Known regressions

- (Placeholder.)

## Out of scope

- Electrolysis-driven reaction byproducts (e.g., H₂ + O₂ from water
  electrolysis) — overlaps with `05-reactions.md`; the byproduct
  invariants live there, the mask-correctness invariants live here.
