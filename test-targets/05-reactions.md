# 05 — Reactions & Emergence

## Scope

The emergent donor/acceptor reaction engine, derived-compound creation,
and the principle that data-driven chemistry matching real behavior is a
feature to preserve, not a bug to sanitize.

## Prerequisites

`00-headless-harness.md`.

## Core principle (LOAD-BEARING)

**When the atomic data's emergent behavior accidentally matches real
chemistry, preserve it. Bespoke compounds layer on top — they do not
replace the emergent byproducts.**

Example: Ca + H₂O produces H₂ gas + heat via the emergent engine because
Ca (E=1.00, v=2) matches the reactive-metal predicate against water's
O-surface signature. Real chemistry: Ca + 2H₂O → Ca(OH)₂ + H₂. The H₂
production is the emergent win.

When Ca(OH)₂ is added as a derived compound, the reaction MUST still
produce H₂ gas as a visible byproduct — the Ca may transmute to Ca(OH)₂,
but a third output cell (H₂ molecule into an adjacent empty) must spawn
so the fizzing-hydrogen behavior is preserved.

## Invariants

### Structural correctness

- **Reactions produce valid element indices.** For every reaction that
  triggers in fuzz testing, assert output cells have a valid
  `Element` discriminant and a valid `Kind` for that element. No garbage
  enum values, no Kind/Element mismatches.

- **Derived-compound indices are valid.** When a reaction produces a
  derived compound, the `derived_id` must correspond to an existing
  entry in the compound table. Reading back `derived_physics_of(idx)`,
  `derived_color_of(idx)`, `derived_formula_of(idx)` must succeed.

- **No reaction output outside the grid.** Reactions may spawn byproducts
  into neighbor cells; those neighbors must be in-bounds. Reactions at
  grid edges that would spawn off-grid silently drop the byproduct (or
  spawn in an in-bounds alternative neighbor).

### Emergence preservation

- **Ca + H₂O produces H₂ byproduct.** Paint a Ca cell adjacent to water,
  tick until reaction triggers. Assert at least one neighbor cell becomes
  an H₂ gas cell (not just that Ca transmuted).

- **Na + H₂O produces H₂ byproduct.** Same property for sodium.

- **Fe + O + H₂O produces rust AND (possibly) H₂.** Rust formation must
  not erase a real H₂ byproduct if the chemistry says it's there.

- **Derived-compound reactions preserve their emergent byproducts.** For
  every derived compound that represents a real multi-output reaction
  (Ca + 2H₂O → Ca(OH)₂ + H₂, etc.), the test suite asserts the byproduct
  still spawns after the derived compound is added.

### Conservation (soft)

- **Atom count bounded across ticks.** Sum of atom counts (per element)
  across all cells, across a tick with no external paint, should not
  drift beyond a documented tolerance. Reactions can merge, split,
  transmute — but gross drift indicates a conservation bug.

  *Why:* Not strict mass conservation (fire consumes things, etc.), but
  no reaction should silently double or erase atoms.

- **No reaction spawns atoms from nothing.** A reaction in a region with
  zero reactants must not produce cells. Fuzz: paint a single reactive
  cell surrounded by Empty; assert no reaction output spawns beyond
  plausible decay/evaporation.

### Thresholds

- **Ignition thresholds respected.** Flammable elements (wood, gunpowder,
  etc.) do not ignite below their `ignite_above` temperature regardless
  of ambient oxygen or neighbors.

- **Explosives detonate only above ignition.** Gunpowder cells below
  ignition temperature never spawn shockwaves spontaneously.

## Known regressions

- (Placeholder — track specific reaction bugs here.)

## Out of scope

- Nuclear transmutation (U → Pb under criticality) — covered in
  `06-nuclear-criticality.md`.
- Electrolysis-driven reactions — covered in `09-electrical.md`.
- Exotic reactions (Al+Ga embrittlement, Hg amalgams, thermite, aqua
  regia) — parked for a future phase, see memory
  `project_exotic_reactions_phase`.
