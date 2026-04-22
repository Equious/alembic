# Test Targets

This directory specifies **what** to test about Alembic's simulation — not how.
Each file defines a set of invariants or behavioral properties grouped by
feature or phase. They are intended to be consumed by Cygent (or any other
agent) as generation targets for Rust integration tests under `tests/`.

## Philosophy

- **Describe invariants, not implementations.** "Gas cells in a sealed box
  retain overpressure" — the agent decides how to paint the box, what gas to
  use, how many frames to tick, and what numeric tolerance counts as "retain."
- **Include the *why* where it matters.** Edge-case judgement requires the
  agent to know *why* the invariant exists. If an invariant encodes a
  regression (past bug), say so.
- **Prefer properties to specific scenarios.** "No tick panics on any
  sequence of random paint inputs" over "tick 50 frames with brush size 5."
- **One feature per file.** Keep scope tight so a single PR can add tests
  for one target file without sprawling.

## File format

Each target file has:

1. **Scope** — one paragraph on what feature/area this covers.
2. **Prerequisites** — what the harness must expose for these tests to be
   writable (usually a reference to `00-headless-harness.md`).
3. **Invariants** — bullet list. Each bullet leads with the property, then a
   short *Why* when the motivation isn't self-evident.
4. **Known regressions** — optional. Bugs already observed that tests must
   catch if reintroduced.
5. **Out of scope** — optional. Things that look like they belong here but
   don't, with a pointer to the right target.

## Ordering

The numeric prefix is rough dependency / priority order:

- `00` — headless harness (foundation; everything else depends on it)
- `01–02` — cross-cutting robustness / bounds safety
- `03–09` — per-feature invariants
- `10+` — non-simulation (supply chain, build hygiene)

New targets get the next free number. Files can be added, split, or merged
freely as the project grows.

## For the agent

When generating tests for a target file:

- Place tests under `tests/<target-name>.rs` (one file per target).
- Use the headless harness API — do not open a macroquad window.
- Seed the RNG explicitly so tests are reproducible.
- Prefer many small focused tests over one large scenario test.
- If an invariant is ambiguous, add a comment flagging it for review rather
  than guessing.
