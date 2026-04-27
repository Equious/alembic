# 01 — Tick Robustness

## Scope

Cross-cutting "the simulation does not crash" invariants. These are the
cheapest-to-write and highest-value tests for a Steam release: a panic in
release mode = crash to desktop = refund. None of these depend on specific
feature semantics — they apply to the whole tick.

## Prerequisites

`00-headless-harness.md`.

## Invariants

- **No panic under arbitrary paint input.** Construct a fresh `World`. For
  N random frames, paint a random element at a random `(x, y)` with a random
  brush radius, random frozen flag, random tool mode. Tick after each paint.
  Assert no panic, no `unwrap`/`expect` failure, no arithmetic overflow in
  release or debug.

  *Why:* This is the baseline robustness property. Any bug that crashes on
  a specific input sequence surfaces here given enough frames.

- **No NaN or ±infinity in any `f32` field, ever.** After each tick, scan
  all cells and all global sim state (EMF, ambient oxygen, gravity, ambient
  offset, shockwave coordinates/radii/yields). Assert every float is finite.

  *Why:* A NaN in a pressure or coordinate field silently corrupts all
  subsequent math. Catching it at the source is much cheaper than debugging
  its downstream effects.

- **Tick completes in bounded time.** No tick may loop forever. Assert every
  tick of a reasonable-sized world returns in under (say) 1 second on CI
  hardware, even under adversarial paint patterns. Flood-fill, wall-burst
  probes, connected-component passes, and shockwave propagation must all
  terminate.

  *Why:* Guards against infinite loops in iterative passes — e.g., a wall
  probe that fails to advance, a flood-fill that revisits cells.

- **Empty world is stable.** Construct a `World`, never paint anything, tick
  for M frames. Assert every cell remains `Cell::EMPTY` and no shockwaves
  spawn. The atmospheric pressure gradient may populate `target_p` but must
  not spawn fluid or trigger reactions.

- **Repeated construction doesn't leak or diverge.** `World::new()` called
  many times in one test produces identical initial state.

- **`FLAG_UPDATED` bit prevents double-processing per frame.** A cell that
  has been consumed by one pass in a given tick (motion, chemistry, etc.)
  must not be re-processed by a later pass in the same tick. The
  `FLAG_UPDATED` bit is cleared at tick boundaries and set as each pass
  consumes a cell.

  *Why:* Double-processing causes order-dependent artifacts (a cell
  "moves then reacts" or "reacts then moves" in the same frame), which
  compounds across features and is hard to debug. Kyle flagged this —
  reading the code surfaces it even though it's invisible from our
  tuning conversations.

## Known regressions

- (Placeholder — add entries here as specific crash bugs are found and
  fixed, so we never lose coverage for them.)

## Out of scope

- Correctness of specific physics (pressure, thermal, reactions) — covered
  in their own target files.
- Performance tuning — "tick finishes in under 1s" is a liveness check, not
  a perf benchmark.
