# 09 — Electrical / Circuits

## Scope

The energized-cell flood-fill, battery voltage, galvanic EMF generation,
electrolysis (cathode/anode masking, metal plating), and Joule heating.

## Prerequisites

`00-headless-harness.md`.

## Invariants

### Energized flood-fill

- **Energized bitmap requires closed-circuit reachability.** A cell is
  `energized` iff it is reachable from BOTH the positive and negative
  terminals of an active battery via paths of conductors. Reachability
  from only one terminal (an open circuit) does NOT energize the cell.

  *Why:* Kyle's tightening. A wire touching only the + terminal carries
  no current in reality; marking it energized would corrupt Joule
  heating, electrolysis masks, and the noble-gas glow hop downstream.

- **Noble-gas glow is a one-hop extension of energized conductors.** A
  noble-gas cell adjacent to an energized conductor is itself energized.
  A noble-gas cell one further hop away is NOT (single-hop only).

- **Topology tests.** Test with known circuit shapes:
  - Isolated conductor (no battery) → no cells energized.
  - Battery with only one terminal connected to a wire → no cells
    energized (open circuit).
  - Battery with both terminals bridged by a complete wire loop → all
    wire cells on the loop energized.
  - Loop broken by an insulator gap → no cells energized on either side.

- **Bitmap clears when no battery exists.** Paint a conductor network
  with no battery; energized bitmap is all-false.

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

- **Galvanic loop validity requires an external conductor path.**
  `galvanic_voltage` stays at 0 unless a dry (non-electrolyte) conductor
  path closes the loop between cathode and anode. Two metals sitting in
  brine with no external wire connection produce no EMF.

  *Why:* Kyle's catch. Without this gate, any two dissimilar metals
  dropped in brine would generate phantom voltage, including pairs the
  player never intended as a cell.

- **Galvanic cathode/anode elements identified correctly.** In a
  Zn/Cu/brine cell with an external wire, `galvanic_cathode_el ==
  Some(Element::Cu)` (higher EN) and `galvanic_anode_el ==
  Some(Element::Zn)`.

### Electrolysis

- **Cathode and anode masks reflect circuit topology.** A metal cell
  touching brine that's electrically connected (via non-brine conductors)
  to the battery's positive side is in `anode_mask`; negative side is
  in `cathode_mask`. A metal cell not in brine contact, or not
  electrically connected, is in neither.

- **Electrode role exclusivity.** No cell is simultaneously in both
  `cathode_mask` and `anode_mask`. Role assignment is consistent per
  mode (electrolytic mode uses battery polarity; galvanic mode uses EN
  ordering); modes don't overlap their assignments for the same cell.

- **Electrolysis gated on `active_emf > 0` AND ≥1 energized cell.**
  Electrolysis and Joule heating do nothing when either condition is
  false. Painting electrodes in brine with no battery and no galvanic
  loop produces zero plating and zero Joule heat.

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
