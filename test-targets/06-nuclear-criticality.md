# 06 — Nuclear Criticality

## Scope

Uranium pile-size thresholds, mid-pile popping behavior, full criticality
detonation, and post-detonation ambient state. This is the phase most
recently tuned — these invariants freeze the current behavior as the
target.

## Prerequisites

`00-headless-harness.md`.

## Thresholds (currently tuned)

- Stable pile: < 1500 U atoms (connected component).
- Reactive pile: 1500 – 5000 atoms. Mid-pile pops, localized shockwaves,
  transmutation to Pb at increasing rate with size.
- Critical: ≥ 5000 atoms. Full detonation within bounded frames, with
  a single large central blast and a strong surface-layer pressure wave.

If these thresholds ever change via explicit tuning, update this file
and the tests together — don't let the tests drift silently.

## Invariants

### Stability below 1500

- **Piles of 1000 U atoms remain stable indefinitely.** Paint a ~1000-atom
  U pile in vacuum (no ambient heat, no neighbors), tick 500+ frames.
  Assert no shockwaves spawned, no transmutation to Pb, no explosion.

### Proportional reactivity 1500–5000

- **Shockwave rate scales with pile size.** Paint piles of (say) 2000,
  3500, 4800 atoms. Count shockwaves spawned per 100 frames. Assert
  monotonic increase with pile size.

- **Mid-pile pops show heat glow.** A popping U cell's `temp` rises at
  the moment of pop (visible as the glow the user specifically asked to
  preserve). Assert at each pop event the originating cell's temp is
  elevated relative to its pre-pop value.

- **Piles in this range do not instantly fully detonate.** A 3000-atom
  pile tick'd for many frames produces distributed popping, not a single
  giant blast.

### Critical ≥ 5000

- **Full detonation within bounded frames.** Paint a 5000+ atom U pile.
  Assert a central-blast shockwave fires within N frames (document N
  from tuning — probably <30 frames).

- **Central blast fires exactly once per component.** The component flood-
  fills `u_central_blast_fired`; no cell in the same component fires a
  second central blast.

- **Pressure wave damages a glass box around the pile.** Wrap a 5000+
  U pile in a frozen glass box. Trigger criticality. Assert the glass
  wall bursts (at least one wall goes to Empty or shattered-glass cells).

  *Why:* Explicit user requirement — "this glass box should be obliterated
  when uranium goes critical in it."

- **Criticality does not vaporize everything nearby.** Surrounding trees
  ignite; surrounding far-away fragile materials survive. There's a
  tuning sweet-spot the user approved — this is a regression-catch, not
  a perf target.

### Post-detonation ambient

- **Clearing the grid returns the ambient state to baseline.** After a
  detonation, call the "clear everything" path (shift+C equivalent).
  Assert ambient temperature, ambient oxygen, active shockwaves, and
  any per-cell residual state reset to the `World::new()` baseline.

- **A fresh U pile after clear does not prime-detonate.** Spawn a
  B10-sized (small) U pile immediately after clearing. Assert it does
  not detonate within N frames — it should remain stable like any fresh
  small pile.

  *Why:* KNOWN BUG the user observed — "next time I spawn U at brush
  size B30, it just blows immediately - well under what's expected for
  criticality, and shift+c would have ensured the play space was cleaned."
  This test is specifically a regression for that bug.

## Known regressions

- **Post-detonation priming.** See "A fresh U pile after clear does not
  prime-detonate" above. Not fully root-caused as of the last tuning
  session — likely ambient temperature, latent heat field, or
  `u_component_size` not clearing. This test will fail until the root
  cause is fixed.

- **Simultaneous-pile flash.** Earlier bug where reaching threshold made
  every U cell flash at once instead of producing the slow shockwave-
  bubble effect. Fixed by preferring per-cell transmutation chance over
  global component-flash. Any regression that reintroduces the
  "everything flashes simultaneously" failure mode should fail the
  reactivity-scales-with-size tests.

## Out of scope

- Radium behavior — open question at last check ("should radium be
  showing any popping/shockwaves at any size pile?"). Add a separate
  target file when that's answered.
- Fission product chemistry — out of scope for the current phase.
