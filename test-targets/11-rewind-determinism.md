# 11 — Rewind & Determinism

## Scope

The ring-buffer timeline (`history`, `history_write`, `history_count`,
`rewind_offset`), simulation determinism under a fixed RNG seed, and the
purity of read-only queries. A single determinism test catches a huge
class of regressions, so this file is high-leverage — prioritize it.

## Prerequisites

`00-headless-harness.md`. Tests here also need the harness to expose the
history API entry points (take snapshot, rewind by N, query
`history_count` / `history_write` / `rewind_offset`).

## Invariants

### Ring buffer bounds

- **`history_count <= HISTORY_CAPACITY`.** Never grows past the capacity
  regardless of how many frames are ticked.

- **`history_write` wraps cleanly modulo capacity.** After every
  snapshot, `history_write < HISTORY_CAPACITY`. Never out of range.

- **`rewind_offset` bounded.** Always `0 <= rewind_offset <= history_count`.

### Snapshot semantics

- **Taking a snapshot resets `rewind_offset` to 0.** Any new snapshot
  (user-triggered or automatic) restores the "live" pointer — you can't
  be rewound and also taking new snapshots simultaneously.

- **Snapshot capture is a full deep copy.** Mutating `cells` after
  snapshot does not alter the snapshot's contents.

### Rewind round-trip fidelity

- **Forward N, rewind N → byte-identical grid.** Seed the RNG, snapshot
  the cells buffer, tick forward N frames with deterministic inputs,
  rewind N frames, assert the resulting cells buffer is bytewise equal
  to the original snapshot.

  *Why:* This single test catches any silent state mutation in the tick
  loop, any RNG desync, any scratch-buffer cross-frame leak, and any
  snapshot-copy bug. Kyle's point: huge regression surface for one test.

- **Rewind is idempotent at the boundary.** Rewinding when
  `rewind_offset == history_count` is a no-op, not a panic or a wrap.

### Determinism under fixed seed

- **Same seed + same initial grid + same inputs → byte-identical output.**
  Run the sim twice with identical setup (`rand::srand(S)`, `World::new()`,
  same paint sequence, same per-tick wind vectors, N steps). Assert
  `a.cells == b.cells` bytewise at the end.

  *Why:* If this ever fails, something is reading uninitialized memory,
  leaking state across `World::new()` calls, depending on wall-clock
  time, or pulling from an unseeded RNG. Any of those is a real bug.

- **No hidden global state across worlds.** Two `World::new()` calls in
  the same process produce equivalent fresh worlds. Seeding between them
  is the only thing that may affect subsequent tick outputs.

### Purity of queries

- **Read-only helpers do not mutate state.** Functions that are by name
  or docs declared query-only (`in_bounds`, component labelers used in
  query-only contexts, cell-property getters) must not write to `cells`,
  scratch buffers, or any other simulation state.

  *Why:* A query that silently mutates is a debugging nightmare and
  breaks rewind fidelity. Easier to assert now than hunt later.

- **Idempotence of component labelers.** Calling a component-labeling
  pass twice in a row (without a tick between) produces identical
  outputs on the second call.

## Known regressions

- (Placeholder — this file is new; no regressions recorded yet.)

## Out of scope

- Reproducibility across different Rust versions or platforms — in
  scope only within a single CI configuration. If we ever promise
  cross-platform determinism, this belongs in a dedicated target.
- Save file determinism — no persistence exists yet; revisit when save
  files land.
