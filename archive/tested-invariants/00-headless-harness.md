# 00 — Headless Harness

## Scope

The testing foundation: a way to construct, drive, and inspect the simulation
without opening a graphics window. Every other target file depends on this
existing.

## Required surface

The harness must expose enough of the simulation that tests can:

- **Construct** a fresh `World` without initializing any macroquad window,
  font loading, or GPU context. (Current `World::new()` is already free of
  those — the blocker is visibility, not behavior.)
- **Tick** the simulation one frame at a time, passing a wind vector
  (`(0.0, 0.0)` is the common default). Ticks must be deterministic given
  the same RNG seed and input sequence.
- **Paint** cells via the same entrypoints the game uses (`paint`, prefab
  spawning, wire painting), so tests exercise real code paths.
- **Read** cell state at any `(x, y)` — element, kind, temperature, pressure,
  frozen flag, derived-compound index. Whatever fields are load-bearing for
  an invariant must be readable from outside the sim.
- **Enumerate** world-level state: shockwave count, active EMF, ambient
  oxygen, ambient offset, etc.

## Determinism

- The harness must provide a **seedable RNG**. The sim currently uses
  `macroquad::rand::gen_range`, which has a global seed (`rand::srand(...)`)
  — tests must set this at the start of each test so fuzz runs are
  reproducible.
- No test may depend on wall-clock time. The sim exposes no `get_time()`
  today; keep it that way.
- If determinism ever leaks (e.g., `HashMap` iteration order, parallel work),
  document it here and provide a deterministic alternative for tests.

## Not required (yet)

- Graphics. Tests never render.
- Audio. No audio exists.
- Input events. Tests drive the sim by calling `paint`/`tick` directly, not
  by simulating mouse/keyboard events.

## Deliverables

1. `src/lib.rs` exposing the public surface above. `src/main.rs` becomes a
   thin wrapper that calls into the library and runs the macroquad loop.
2. `Cargo.toml` declares both `[lib]` and `[[bin]]` targets.
3. A minimal `tests/smoke.rs` that:
   - Constructs a `World`
   - Ticks it 100 frames with zero wind
   - Asserts no panic, no NaN, no OOB — proving the harness works end to end.

## Why this exists

Without the lib split, Cygent cannot write integration tests at all for a
binary-only crate. The smoke test proves the harness is viable before any
more ambitious invariants are written against it.
