# Pressure & Explosions Phase — Plan (deferred)

Paused during the elements/compounds refactor. Pick this up once the
periodic-table foundation is in place — pressure will interact with
atomic-level gas behavior (H₂, O₂, CO₂, etc.) in much nicer ways once
we're not working with bespoke "Steam" abstractions.

## Core idea
Add a scalar **pressure field** per cell, alongside temp and moisture.
Pressure drives gas dispersion, explosions, wind, and deflection of
airborne matter — all through one mechanism.

## New per-cell state
- `Cell.pressure: i16` — overpressure above atmospheric. +N = compressed,
  -N = vacuum-like, 0 = ambient.
- Adds ~2 bytes per cell. History buffer grows proportionally.

## New profile: `PressureProfile`
```
permeability       — how easily pressure conducts through this cell
compliance         — how much this cell is pushed by a pressure gradient
formation_pressure — one-shot pressure injected at phase change (water→steam → +80)
explosive          — Option<(yield, radius)>: if ignited, detonates
```

## New pass: `pressure()` runs after `thermal()`
1. **Diffusion** — pressure equalizes with neighbors, bottlenecked by min permeability.
2. **Motion from gradient** — fluid cells bias movement toward lower pressure
   (this is what makes steam actually disperse instead of clumping).
3. **Decay** — overpressure bleeds toward atmospheric so spikes don't persist.

## Behavioral payoffs (all emergent from the same field)
- Steam disperses naturally; sealed chambers pressurize.
- Wind unified: global wind becomes a pressure-gradient bias. One mechanism, not two.
- Falling deflection: airborne powders/gravels follow horizontal gradient.
- Explosions: `explosive` cells detonate with 1/dist² falloff + heat + scatter;
  pressure diffusion handles the shockwave for free.

## Gunpowder element
- Kind::Powder, dark gray, density 15
- ignite_above: 200, explosive: (yield=500, radius=8)
- Chain reactions for free (gunpowder ignites neighboring gunpowder).

## Rollout — three sub-phases
- **2A Foundation**: field, diffusion, decay, steam formation pressure, gas gradient motion
- **2B Explosions**: Gunpowder, detonate(), shockwave displacement
- **2C Wind unification**: replace global wind with pressure bias, add falling-body deflection

Estimated ~400 lines total across the three PRs.
