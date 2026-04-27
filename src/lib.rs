// v0.3 wgpu rendering backend — runs the binary `alembic-wgpu`.
// The legacy macroquad path below remains compiled and runnable as
// `alembic` until the migration is complete.
pub mod gpu_app;

use macroquad::prelude::*;
use macroquad::window::miniquad::{
    BlendFactor, BlendState, BlendValue, Equation, PipelineParams,
};

// Sim grid dimensions. v0.3 phase 3 — pressure now runs on GPU
// compute, so the largest CPU-linear cost is eliminated and we can
// grow the grid. 1200×900 = 1.08M cells, ~10× the legacy 320×315.
// Thermal diffusion is still on CPU; it'll be the next port if it
// becomes the bottleneck under load.
pub const W: usize = 1200;
pub const H: usize = 900;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Element {
    // --- Compounds (hand-tuned macroscopic materials) ---
    Empty       = 0,
    Sand        = 1,
    Water       = 2,
    Stone       = 3,
    Wood        = 4,
    Fire        = 5,
    CO2         = 6,
    Steam       = 7,
    Lava        = 8,
    Obsidian    = 9,
    Seed        = 10,
    Mud         = 11,
    Leaves      = 12,
    Oil         = 13,
    Ice         = 14,
    MoltenGlass = 15,
    Glass       = 16,
    Charcoal    = 17,
    // --- Atoms (naturally-occurring elements, paintable) ---
    H  = 18, He = 19, C  = 20, N  = 21, O  = 22, F  = 23, Ne = 24,
    Na = 25, Mg = 26, Al = 27, Si = 28, P  = 29, S  = 30, Cl = 31,
    K  = 32, Ca = 33, Fe = 34, Cu = 35, Au = 36, Hg = 37, U  = 38,
    // Tier-1 extra metals + metalloid (atomic nos out of numeric order
    // because they land after Element::BattNeg = 47 to keep existing
    // discriminants stable).
    // --- Derived compounds (products of reactions; not directly paintable
    // at the moment, though they could be in a future refactor) ---
    Rust = 39,
    Salt = 40,
    // Runtime-derived compound. derived_id on the Cell points into the
    // DERIVED_COMPOUNDS registry. Lets reactions produce any atomic
    // combination (AuF, MgCl₂, FeS, …) without requiring a hand-coded
    // Element variant per possibility.
    Derived = 41,
    // Paintable explosive. Ignites easily, releases a massive pressure
    // spike while burning that shoves surrounding matter via the
    // pressure-shove system.
    Gunpowder = 42,
    // Refractory materials — purpose-built high-temp structural solids.
    // Doesn't melt below 1700 °C, doesn't burn, chemically inert.
    // Real-world analog: fused silica glassware for hot-chemistry labs.
    Quartz   = 43,
    // Even more heat-tolerant: industrial firebrick, stable to ~1800 °C,
    // opaque rusty-orange. Used to build furnaces and crucibles.
    Firebrick = 44,
    // Argon — noble gas atom, second-lightest. Glows lavender-purple when
    // driven by current (neon sign chemistry).
    Ar = 45,
    // Positive battery terminal. A real circuit needs BOTH terminals and
    // a continuous conductor path between them — current only flows in a
    // closed loop. The "energized" flood is computed from each terminal
    // separately; cells reachable from both are in the loop.
    BattPos = 46,
    // Negative battery terminal. Paired with BattPos — without both, no
    // current flows and nothing glows.
    BattNeg = 47,
    // Zinc — galvanic-series workhorse (EN 1.65), textbook anode
    // opposite a Cu cathode. Strongly electropositive relative to Cu
    // and Au, so Zn+HCl fizzes H₂ and Zn/Cu in brine self-drives a cell.
    Zn = 48,
    // Silver — highest conductivity in the table (σ > Cu), soft white metal.
    Ag = 49,
    // Nickel — magnetic transition metal, moderately reactive.
    Ni = 50,
    // Lead — heavy, low-reactivity post-transition metal. Future radiation
    // shield material.
    Pb = 51,
    // Boron — metalloid (period 2 group 13). Neutron absorber for future
    // control rods; high melting point, semi-conductor.
    B  = 52,
    // Radium — natural alkaline-earth metal, radioactive. Decay glow is
    // the landmark payoff for the radioactivity phase.
    Ra = 53,
    // Caesium — softest, most reactive stable alkali metal (EN 0.79).
    // Melts near body temp, ignites in air, detonates in water.
    Cs = 54,
}
pub const ELEMENT_COUNT: usize = 55;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind { Empty, Solid, Gravel, Powder, Liquid, Gas, Fire }

// Mouse-tool mode. Paint places matter; Heat raises/lowers temperature;
// Vacuum sucks gas into a cursor-centered low-pressure zone and deletes it;
// Grab lifts existing cells from the sim and lets you reposition them —
// useful for dropping reactive elements into containers you've built.
#[derive(Clone, Copy, PartialEq)]
enum ToolMode { Paint, Heat, Vacuum, Pipet, Prefab, Wire }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PrefabKind { Beaker, Box, Battery }

impl Kind {
    // "Rigid" matter — solids, stones, and powders — can't be pushed aside by
    // other matter. A denser rock doesn't tunnel through a sand pile.
    fn is_rigid(self) -> bool {
        matches!(self, Kind::Solid | Kind::Gravel | Kind::Powder)
    }
    // "Fluid" matter — liquids and gases (including fire) — can be displaced
    // by density, so heavier liquids sink, lighter gases rise, etc.
    fn is_fluid(self) -> bool {
        matches!(self, Kind::Liquid | Kind::Gas | Kind::Fire)
    }
}

// ============================================================================
// SYSTEM PROFILES
//
// Each element's behavior is split into per-system profiles. Adding a new
// system (pressure, electricity, chemistry, radiation) is a new profile
// struct, a new static table, and a new pass — without touching the others.
// ============================================================================

// A phase transition is a (threshold, target element, latent heat) tuple.
// `latent` is energy absorbed from the hottest adjacent cell when the change
// happens (positive = absorbs heat, like boiling). Zero for transitions that
// don't involve significant phase-change energy.
#[derive(Clone, Copy)]
struct Phase { threshold: i16, target: Element, latent: f32 }

// Physical phase & displacement: how the element interacts with gravity,
// density-driven swap, and the rigid/fluid hierarchy.
#[derive(Clone, Copy)]
struct PhysicsProfile {
    density: i16,
    kind: Kind,
    viscosity: u16,
    // Molar mass in g/mol. Used exclusively for *gas* buoyancy — solids and
    // liquids fall back on `density` for their hierarchies. Lets buoyancy
    // emerge from real molecular masses instead of hand-tuned signs on
    // `density`. 0.0 means "not a gas, not used."
    molar_mass: f32,
}

// The play area isn't literally a vacuum — empty cells stand in for an
// ambient atmosphere (roughly N₂+O₂). Gases compare against these numbers
// when deciding buoyancy / thermal exchange / pressure baseline, which is
// what makes "helium floats, chlorine sinks" emergent instead of hardcoded.
#[derive(Clone, Copy)]
struct AmbientAtmosphere {
    molar_mass: f32, // ~29 for dry air at sea level
}
const AMBIENT_AIR: AmbientAtmosphere = AmbientAtmosphere { molar_mass: 29.0 };

// Thermal: all temperature-driven behavior lives here.
// The thermal pass reads this profile to compute diffusion, ambient drift,
// combustion, and phase change.
#[derive(Clone, Copy)]
struct ThermalProfile {
    initial_temp:   i16,
    ambient_temp:   i16,
    ambient_rate:   f32,
    conductivity:   f32,
    heat_capacity:  f32,
    freeze_below:   Option<Phase>,   // below threshold → target (e.g. Water→Ice)
    melt_above:     Option<Phase>,   // above threshold → target (Ice→Water, Obsidian→Lava)
    boil_above:     Option<Phase>,   // Water→Steam, with latent heat
    condense_below: Option<Phase>,   // Steam→Water
    ignite_above:   Option<i16>,     // above this temperature, ignites (if dry)
    burn_duration:  Option<u8>,      // once lit, burns this many ticks before being consumed
    burn_temp:      Option<i16>,     // sustained temp while burning (controls "fire intensity")
}

// Pressure: how overpressure propagates. `permeability` gates how readily
// pressure conducts through a cell (high for gases, 0 for rigid solids).
// `compliance` gates how much a cell's motion biases toward lower-pressure
// neighbors (high for gases, 0 for solids). `formation_pressure` is injected
// at phase-change birth — so water→steam can expand into its container.
#[derive(Clone, Copy)]
struct PressureProfile {
    permeability:       u8,
    compliance:         u8,
    formation_pressure: i16,
}

// Moisture: who imparts water, who holds it, how it moves between cells.
// `is_source` cells (water, ice, mud) wet their neighbors directly and never
// lose their own moisture field. `is_sink` cells carry a moisture value and
// can exchange it with other sinks via `conductivity` (wicking).
#[derive(Clone, Copy)]
struct MoistureProfile {
    default_moisture: u8,
    conductivity:     f32,
    is_source:        bool,
    is_sink:          bool,
    wet_above:        Option<(u8, Element)>,   // saturation phase (sand → mud)
    dry_below:        Option<(u8, Element)>,   // desiccation phase (mud → sand)
}

// Helper for const transitions — keeps the tables readable.
const fn ph(threshold: i16, target: Element, latent: f32) -> Phase {
    Phase { threshold, target, latent }
}

static PHYSICS: [PhysicsProfile; ELEMENT_COUNT] = {
    let mut a = [PhysicsProfile { density: 0, kind: Kind::Empty, viscosity: 0, molar_mass: 0.0 }; ELEMENT_COUNT];
    // Compounds. `molar_mass` only matters for gases; non-gas entries keep 0.0.
    a[Element::Empty    as usize] = PhysicsProfile { density:   0, kind: Kind::Empty,  viscosity:   0, molar_mass:  0.0 };
    a[Element::Sand     as usize] = PhysicsProfile { density:  20, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::Water    as usize] = PhysicsProfile { density:  10, kind: Kind::Liquid, viscosity:   0, molar_mass:  0.0 };
    a[Element::Stone    as usize] = PhysicsProfile { density: 100, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Wood     as usize] = PhysicsProfile { density: 100, kind: Kind::Solid,  viscosity:   0, molar_mass:  0.0 };
    // Fire is hot combustion products (roughly CO₂ + H₂O vapor) — lighter
    // than air once heated, which is why it rises.
    a[Element::Fire     as usize] = PhysicsProfile { density:  -5, kind: Kind::Fire,   viscosity:   0, molar_mass: 20.0 };
    // CO₂ = real carbon dioxide, 44 g/mol — ~50% heavier than air, sinks.
    // Pools at the floor of sealed volumes (real fire-extinguisher behavior).
    a[Element::CO2    as usize] = PhysicsProfile { density:   1, kind: Kind::Gas,    viscosity:   0, molar_mass: 44.0 };
    // Steam = H₂O gas, 18 g/mol — significantly lighter than air, rises vigorously.
    a[Element::Steam    as usize] = PhysicsProfile { density:  -3, kind: Kind::Gas,    viscosity:   0, molar_mass: 18.0 };
    a[Element::Lava     as usize] = PhysicsProfile { density:  30, kind: Kind::Liquid, viscosity: 320, molar_mass:  0.0 };
    a[Element::Obsidian as usize] = PhysicsProfile { density: 100, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Seed     as usize] = PhysicsProfile { density: 100, kind: Kind::Solid,  viscosity:   0, molar_mass:  0.0 };
    a[Element::Mud      as usize] = PhysicsProfile { density:  25, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::Leaves   as usize] = PhysicsProfile { density:   8, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::Oil      as usize] = PhysicsProfile { density:   8, kind: Kind::Liquid, viscosity:   0, molar_mass:  0.0 };
    a[Element::Ice      as usize] = PhysicsProfile { density:   9, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::MoltenGlass as usize] = PhysicsProfile { density: 15, kind: Kind::Liquid, viscosity: 150, molar_mass:  0.0 };
    a[Element::Glass    as usize] = PhysicsProfile { density:  35, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Charcoal as usize] = PhysicsProfile { density:   5, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };

    // --- Atoms ---
    // `density` is retained only as a coarse hierarchy signal for rigid/fluid
    // swap logic; for *gases*, the emergent buoyancy driver is `molar_mass`.
    // Atomic gases' molar masses are the real diatomic mass where applicable
    // (N₂, O₂, Cl₂) since gases at STP are almost always molecular.
    a[Element::H  as usize] = PhysicsProfile { density:  -6, kind: Kind::Gas,    viscosity:   0, molar_mass:  2.0 };
    a[Element::He as usize] = PhysicsProfile { density:  -5, kind: Kind::Gas,    viscosity:   0, molar_mass:  4.0 };
    a[Element::C  as usize] = PhysicsProfile { density:  22, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::N  as usize] = PhysicsProfile { density:  -1, kind: Kind::Gas,    viscosity:   0, molar_mass: 28.0 };
    a[Element::O  as usize] = PhysicsProfile { density:  -1, kind: Kind::Gas,    viscosity:   0, molar_mass: 32.0 };
    a[Element::Ne as usize] = PhysicsProfile { density:  -2, kind: Kind::Gas,    viscosity:   0, molar_mass: 20.0 };
    a[Element::Na as usize] = PhysicsProfile { density:  10, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Mg as usize] = PhysicsProfile { density:  17, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Al as usize] = PhysicsProfile { density:  27, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Si as usize] = PhysicsProfile { density:  23, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::P  as usize] = PhysicsProfile { density:  18, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::S  as usize] = PhysicsProfile { density:  21, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::Cl as usize] = PhysicsProfile { density:   3, kind: Kind::Gas,    viscosity:   0, molar_mass: 71.0 };
    a[Element::K  as usize] = PhysicsProfile { density:   9, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Ca as usize] = PhysicsProfile { density:  15, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Fe as usize] = PhysicsProfile { density:  79, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Cu as usize] = PhysicsProfile { density:  90, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Au as usize] = PhysicsProfile { density: 150, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Hg as usize] = PhysicsProfile { density: 135, kind: Kind::Liquid, viscosity: 100, molar_mass:  0.0 };
    a[Element::U  as usize] = PhysicsProfile { density: 150, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    // Fluorine: F₂ diatomic gas, denser than air, mass 38 g/mol.
    a[Element::F  as usize] = PhysicsProfile { density:   2, kind: Kind::Gas,    viscosity:   0, molar_mass: 38.0 };
    // Tier-1 batch. Densities loosely track real-world values scaled into
    // the same "density 100 ≈ dense metal" band the rest of the table uses.
    a[Element::Zn as usize] = PhysicsProfile { density:  71, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Ag as usize] = PhysicsProfile { density: 105, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Ni as usize] = PhysicsProfile { density:  89, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::Pb as usize] = PhysicsProfile { density: 113, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    a[Element::B  as usize] = PhysicsProfile { density:  21, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    a[Element::Ra as usize] = PhysicsProfile { density:  55, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };
    // Cs melts at 28°C so its STP state is effectively "about to melt".
    // Keep it Kind::Gravel (soft metal) at ambient; phase transitions will
    // liquefy it once ambient_offset crosses its MP.
    a[Element::Cs as usize] = PhysicsProfile { density:  19, kind: Kind::Gravel, viscosity:   0, molar_mass:  0.0 };

    // --- Derived compounds ---
    // Rust (Fe₂O₃·H₂O): flaky orange-brown oxide powder. Denser than sand.
    a[Element::Rust as usize] = PhysicsProfile { density:  30, kind: Kind::Powder, viscosity:   0, molar_mass:  0.0 };
    // Salt (NaCl): crystalline powder, slightly denser than sand. Mass 58.5
    // so vaporized salt has meaningful buoyancy in gas phase.
    a[Element::Salt as usize] = PhysicsProfile { density:  22, kind: Kind::Powder, viscosity:   0, molar_mass:  58.5 };
    // Derived — placeholder defaults. Actual per-cell physics comes from
    // the registry via cell_physics(); this entry exists so indexing by
    // Element::Derived as usize doesn't panic.
    a[Element::Derived as usize] = PhysicsProfile { density: 20, kind: Kind::Powder, viscosity: 0, molar_mass: 0.0 };
    // Gunpowder: dense black powder, flies on detonation.
    a[Element::Gunpowder as usize] = PhysicsProfile { density: 18, kind: Kind::Powder, viscosity: 0, molar_mass: 0.0 };
    // Refractory solids — structural materials with extreme thermal
    // tolerance. Used for high-temperature containers (quartz lab
    // glassware, firebrick furnaces). Dense enough to sink through
    // most liquids if un-frozen.
    a[Element::Quartz    as usize] = PhysicsProfile { density: 33, kind: Kind::Solid, viscosity: 0, molar_mass: 60.1 };
    a[Element::Firebrick as usize] = PhysicsProfile { density: 42, kind: Kind::Solid, viscosity: 0, molar_mass: 0.0 };
    // Argon — monatomic noble gas, denser than air (Ar₂O doesn't exist,
    // it's chemically inert). Forms 1% of real atmosphere.
    a[Element::Ar        as usize] = PhysicsProfile { density: -1, kind: Kind::Gas,   viscosity: 0, molar_mass: 40.0 };
    // Battery — a solid structural cell representing a voltage source.
    // Behaves like a wall that injects energy into connected conductors.
    a[Element::BattPos as usize] = PhysicsProfile { density: 60, kind: Kind::Solid, viscosity: 0, molar_mass: 0.0 };
    a[Element::BattNeg as usize] = PhysicsProfile { density: 60, kind: Kind::Solid, viscosity: 0, molar_mass: 0.0 };
    a
};

/// `Cell` is `#[repr(C)]` with field byte layout:
///   0: el (u8)         8: moisture (u8)
///   1: derived_id (u8) 9: burn (u8)
///   2-3: life (u16)    10-11: pressure (i16)
///   4: seed (u8)       12: solute_el (u8)
///   5: flag (u8)       13: solute_amt (u8)
///   6-7: temp (i16)    14: solute_derived_id (u8)
///                      15: trailing padding
/// Total: 16 bytes, perfectly matching WGSL `array<vec4<u32>>`. GPU
/// uploads and readbacks are now zero-copy memcpys via the helpers
/// below — no per-cell pack/unpack loops, no field shuffling.
const _: () = {
    let want = 16usize;
    let got  = std::mem::size_of::<Cell>();
    if want != got {
        // Compile-time check: panics with a clear message if Cell's
        // size drifts from the WGSL contract.
        panic!("Cell must be exactly 16 bytes for the GPU layout");
    }
};

/// View `world.cells` as a raw byte slice for GPU upload. Zero-copy.
/// The trailing padding byte is unspecified but stable in practice
/// (heap zero-init on alloc, Cell field writes don't touch it). The
/// motion shader never reads bytes past offset 14.
#[inline]
pub fn cells_as_bytes(cells: &[Cell]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            cells.as_ptr() as *const u8,
            std::mem::size_of_val(cells),
        )
    }
}

/// Copy a GPU readback byte slice into `cells`. Zero per-cell work.
/// SAFETY: `data` must come from a previous `cells_as_bytes`-shaped
/// upload (i.e. exactly `cells.len() * 16` bytes, with bytes 0-14 of
/// each 16-byte cell holding valid Cell fields). The motion shader
/// only swaps whole 16-byte cells, so this invariant is preserved.
#[inline]
pub fn cells_copy_from_bytes(cells: &mut [Cell], data: &[u8]) {
    debug_assert_eq!(data.len(), cells.len() * std::mem::size_of::<Cell>());
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            cells.as_mut_ptr() as *mut u8,
            data.len(),
        );
    }
}

/// Base color for an element id, packed as `[r, g, b, alpha=255]`
/// (u8 each), suitable for a per-element WGSL color LUT. Mirrors
/// `Element::base_color` exactly. Returns `[0,0,0,255]` for ids
/// that don't correspond to a defined element.
pub fn base_color_props(id: u8) -> [u8; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0, 0, 0, 255]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    let (r, g, b) = el.base_color();
    [r, g, b, 255]
}

/// Per-element lifecycle data:
///   x = ephemeral flag (1 if `Cell::new(el)` sets a non-zero `life`)
///   y = decay product element id (what this cell becomes when
///       `life` hits zero — Fire → Empty, Steam → Water, etc.)
///   z = preserve_state flag (1 to keep temp/pressure across the
///       transition, e.g., Steam → Water keeps boiling temp; 0 to
///       reset everything for a clean Empty cell)
///   w = unused
///
/// Driven entirely by element data — the GPU lifecycle compute
/// reads this LUT and ages every cell uniformly. Adding a new
/// ephemeral element is just a row here; no new shader code.
pub fn lifecycle_props(id: u8) -> [u32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0; 4]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    match el {
        Element::Fire  => [1, Element::Empty as u32, 0, 0],
        Element::Steam => [1, Element::Water as u32, 1, 0],
        _ => [0, 0, 0, 0],
    }
}

/// Capacity of the GPU-side Derived compound mirror (matches the
/// CPU registry's u8-indexable cap of 256 entries).
pub const DERIVED_GPU_CAPACITY: usize = 256;

/// Export the Derived compound registry as a flat vec4<f32> array
/// the GPU can use directly. Layout per entry (2 × vec4<f32>):
///
///   vec0: kind_id, density, viscosity, molar_mass
///   vec1: r/255, g/255, b/255, 1.0
///
/// Unfilled slots default to (0, 20, 0, 0) physics + (0.6, 0.5, 0.55, 1)
/// — the same fallback that CPU `derived_physics_of()` uses when an
/// out-of-range id is queried.
pub fn export_derived_to_gpu(out_phys: &mut [[f32; 4]], out_color: &mut [[f32; 4]]) {
    let n = DERIVED_GPU_CAPACITY;
    debug_assert_eq!(out_phys.len(), n);
    debug_assert_eq!(out_color.len(), n);
    // Default slot values (matches CPU fallback).
    for i in 0..n {
        out_phys[i] = [3.0, 20.0, 0.0, 0.0];          // Powder, density 20
        out_color[i] = [160.0/255.0, 130.0/255.0, 140.0/255.0, 1.0];
    }
    DERIVED_COMPOUNDS.with(|r| {
        let b = r.borrow();
        for (i, c) in b.iter().enumerate().take(n) {
            let kind_id = match c.physics.kind {
                Kind::Empty => 0.0,
                Kind::Solid => 1.0,
                Kind::Gravel => 2.0,
                Kind::Powder => 3.0,
                Kind::Liquid => 4.0,
                Kind::Gas => 5.0,
                Kind::Fire => 6.0,
            };
            out_phys[i] = [
                kind_id,
                c.physics.density as f32,
                c.physics.viscosity as f32,
                c.physics.molar_mass,
            ];
            out_color[i] = [
                c.color.0 as f32 / 255.0,
                c.color.1 as f32 / 255.0,
                c.color.2 as f32 / 255.0,
                1.0,
            ];
        }
    });
}

/// Derived element id sentinel — Element::Derived as u32.
pub const DERIVED_EL_ID: u32 = Element::Derived as u32;

/// Per-element flag: 1 if this element has a flame-test color
/// (Cu, Na, K, Ca, Mg, B, Salt — see `flame_color`), 0 otherwise.
/// Returned as a vec4<u32> for the standard packed-LUT layout.
pub fn flame_color_flag_props(id: u8) -> [u32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0; 4]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    [if flame_color(el).is_some() { 1 } else { 0 }, 0, 0, 0]
}

/// Per-element flag: 1 if this element has an `ignite_above` thermal
/// threshold (i.e., is flammable).
pub fn flammable_props(id: u8) -> [u32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0; 4]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    [if el.thermal().ignite_above.is_some() { 1 } else { 0 }, 0, 0, 0]
}

/// Sentinel for "no threshold" in the GPU thermal LUT (an unreachable
/// temperature value — i16 min cast to f32).
pub const NO_THERMAL_THRESHOLD: f32 = -32768.0;

/// Per-element combustion data for the GPU thermal_post compute.
///   x = ignite_above threshold (NO_THERMAL_THRESHOLD if not flammable)
///   y = burn_duration (0 if not flammable; otherwise frames-of-burn)
///   z = burn_temp (sustained combustion temperature)
///   w = self_oxidizing flag (1 for Gunpowder, 0 otherwise)
pub fn thermal_burn_props(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [NO_THERMAL_THRESHOLD, 0.0, 0.0, 0.0]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    let t = el.thermal();
    let ignite = t.ignite_above.map(|v| v as f32).unwrap_or(NO_THERMAL_THRESHOLD);
    let dur = t.burn_duration.unwrap_or(0) as f32;
    let btemp = t.burn_temp.unwrap_or(0) as f32;
    let self_ox = if matches!(el, Element::Gunpowder) { 1.0 } else { 0.0 };
    [ignite, dur, btemp, self_ox]
}

/// Per-element low-side phase transitions (freeze + condense).
///   x = freeze_below threshold, y = freeze_target_el
///   z = condense_below threshold, w = condense_target_el
pub fn thermal_phase_lo_props(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [NO_THERMAL_THRESHOLD, 0.0, NO_THERMAL_THRESHOLD, 0.0]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    let t = el.thermal();
    let freeze_thr = t.freeze_below.map(|p| p.threshold as f32).unwrap_or(NO_THERMAL_THRESHOLD);
    let freeze_tgt = t.freeze_below.map(|p| p.target as u32 as f32).unwrap_or(0.0);
    let cond_thr = t.condense_below.map(|p| p.threshold as f32).unwrap_or(NO_THERMAL_THRESHOLD);
    let cond_tgt = t.condense_below.map(|p| p.target as u32 as f32).unwrap_or(0.0);
    [freeze_thr, freeze_tgt, cond_thr, cond_tgt]
}

/// Per-element high-side phase transitions (melt + boil).
///   x = melt_above threshold, y = melt_target_el
///   z = boil_above threshold, w = boil_target_el
pub fn thermal_phase_hi_props(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [NO_THERMAL_THRESHOLD, 0.0, NO_THERMAL_THRESHOLD, 0.0]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    let t = el.thermal();
    let melt_thr = t.melt_above.map(|p| p.threshold as f32).unwrap_or(NO_THERMAL_THRESHOLD);
    let melt_tgt = t.melt_above.map(|p| p.target as u32 as f32).unwrap_or(0.0);
    let boil_thr = t.boil_above.map(|p| p.threshold as f32).unwrap_or(NO_THERMAL_THRESHOLD);
    let boil_tgt = t.boil_above.map(|p| p.target as u32 as f32).unwrap_or(0.0);
    [melt_thr, melt_tgt, boil_thr, boil_tgt]
}

/// Per-element burn-out decay product. Most flammables become CO2;
/// Wood has a probabilistic Charcoal byproduct (handled in shader).
///   x.byte0 = primary decay element id (CO2 for most)
///   x.byte1 = secondary decay element id (Charcoal for Wood, 0 elsewhere)
///   x.byte2 = secondary probability /16 (3 = ~30% for Wood)
pub fn burn_decay_props(id: u8) -> [u32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0; 4]; }
    let el: Element = unsafe { std::mem::transmute(id) };
    if el.thermal().ignite_above.is_none() { return [0; 4]; }
    let primary = Element::CO2 as u32;
    let (secondary, prob_16) = match el {
        Element::Wood => (Element::Charcoal as u32, 3u32),
        _ => (0u32, 0u32),
    };
    let packed = primary | (secondary << 8) | (prob_16 << 16);
    [packed, 0, 0, 0]
}

/// Per-element permeability for the GPU pressure-diffusion shader.
/// Returned as a vec4 (only x is used) so it slots into the standard
/// `array<vec4<u32>, 24>` uniform layout (96 elements packed 4 per vec).
/// Replaces the per-cell `perm` upload — element id → perm via LUT.
pub fn pressure_perm_props(id: u8) -> [u32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0; 4]; }
    [PRESSURE[i].permeability as u32, 0, 0, 0]
}

/// Per-element motion props as packed vec4 for the GPU motion compute:
///   x = kind id (Empty=0, Solid=1, Gravel=2, Powder=3, Liquid=4, Gas=5, Fire=6)
///   y = density (SIGNED — gases have negative density)
///   z = viscosity (gates rigid-into-fluid and horizontal liquid spread)
///   w = molar_mass (drives gas ambient buoyancy vs AMBIENT_AIR)
pub fn motion_props(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT { return [0.0; 4]; }
    let phys = &PHYSICS[i];
    let kind_id: f32 = match phys.kind {
        Kind::Empty => 0.0,
        Kind::Solid => 1.0,
        Kind::Gravel => 2.0,
        Kind::Powder => 3.0,
        Kind::Liquid => 4.0,
        Kind::Gas => 5.0,
        Kind::Fire => 6.0,
    };
    // Preserve sign — gases have negative density and we rely on
    // empty(0) > gas(-3) for buoyancy in the GPU motion shader.
    let density = phys.density as f32;
    let viscosity = phys.viscosity as f32;
    let molar_mass = phys.molar_mass;
    [kind_id, density, viscosity, molar_mass]
}

/// Snapshot the per-element pressure-source data as a packed vec4 —
/// `[kind_id_f32, cell_weight, _, _]` for the GPU pressure_sources
/// compute. `kind_id` follows the order of `Kind` enum (Empty=0,
/// Solid=1, Gravel=2, Powder=3, Liquid=4, Gas=5, Fire=6). Cell
/// weight matches the CPU `cell_weight()` helper.
pub fn pressure_source_props(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT {
        return [0.0, 0.0, 0.0, 0.0];
    }
    // SAFETY: we go through the bit pattern. Element is `repr(u8)`
    // and its variants are an explicit subset of 0..ELEMENT_COUNT.
    // Indices that don't correspond to a defined variant exist (the
    // table has gaps for unused atomic numbers); those return a
    // null physics profile, which the shader treats as zero-weight
    // empty-kind — effectively inert under the column walk.
    let phys = &PHYSICS[i];
    let kind_id: f32 = match phys.kind {
        Kind::Empty => 0.0,
        Kind::Solid => 1.0,
        Kind::Gravel => 2.0,
        Kind::Powder => 3.0,
        Kind::Liquid => 4.0,
        Kind::Gas => 5.0,
        Kind::Fire => 6.0,
    };
    let weight = match phys.kind {
        Kind::Empty => AMBIENT_AIR.molar_mass * 0.02,
        Kind::Gas | Kind::Fire => phys.molar_mass * 0.05,
        _ => (phys.density.max(0) as f32) * 0.5,
    };
    [kind_id, weight, 0.0, 0.0]
}

/// Snapshot the thermal profile for element id `id` as a packed
/// `[conductivity, ambient_temp_f32, ambient_rate, heat_capacity]`
/// — used by the GPU compute shader's per-element profile lookup.
/// Out-of-range ids return safe defaults (no heat exchange).
pub fn thermal_profile_vec4(id: u8) -> [f32; 4] {
    let i = id as usize;
    if i >= ELEMENT_COUNT {
        return [0.0, 20.0, 0.0, 1.0];
    }
    let p = &THERMAL[i];
    [p.conductivity, p.ambient_temp as f32, p.ambient_rate, p.heat_capacity]
}

static THERMAL: [ThermalProfile; ELEMENT_COUNT] = {
    const NONE_PH: Option<Phase> = None;
    const fn base() -> ThermalProfile {
        ThermalProfile {
            initial_temp: 20, ambient_temp: 20, ambient_rate: 0.001,
            conductivity: 0.02, heat_capacity: 1.0,
            freeze_below: NONE_PH, melt_above: NONE_PH,
            boil_above: NONE_PH, condense_below: NONE_PH,
            ignite_above: None, burn_duration: None, burn_temp: None,
        }
    }
    let mut a = [base(); ELEMENT_COUNT];
    a[Element::Empty as usize]    = ThermalProfile { ambient_rate: 0.020, conductivity: 0.002, ..base() };
    a[Element::Sand  as usize]    = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.025, heat_capacity: 1.2,
        // Sand → molten glass at 1200°C. Real silica liquidates higher
        // (~1700°C), but in our sim lava is the only common heat source and
        // it cools through 1200° on its way to obsidian — so the threshold
        // needs to sit below the lava-solidification point or no glass ever
        // forms in practice.
        melt_above: Some(ph(1200, Element::MoltenGlass, 300.0)),
        ..base()
    };
    a[Element::Water as usize]    = ThermalProfile {
        ambient_rate: 0.010, conductivity: 0.060, heat_capacity: 4.0,
        // Latent heats roughly matching the real ratio (fusion ≈ 15% of vap).
        // Boiling absorbs 1200 from hot neighbors (water as effective heat sink);
        // freezing releases 400 to cold neighbors (exothermic crystallization).
        freeze_below: Some(ph(0,   Element::Ice,   400.0)),
        boil_above:   Some(ph(100, Element::Steam, 1200.0)),
        ..base()
    };
    a[Element::Stone as usize]    = ThermalProfile {
        ambient_rate: 0.002, conductivity: 0.030, heat_capacity: 2.5,
        // Rocks melt into lava at sufficient heat (real basalt ~1200°C,
        // but our Lava cap sits at 1200 for obsidian formation, so push
        // stone's melt threshold a bit higher to keep hierarchies clean).
        melt_above: Some(ph(1500, Element::Lava, 400.0)),
        ..base()
    };
    a[Element::Wood  as usize]    = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.022, heat_capacity: 1.3,
        ignite_above: Some(350), burn_duration: Some(255),
        burn_temp: Some(700), ..base()
    };
    a[Element::Fire as usize]     = ThermalProfile {
        initial_temp: 900, ambient_temp: 900, ambient_rate: 0.004,
        conductivity: 0.120, heat_capacity: 1.0, ..base()
    };
    a[Element::CO2 as usize]    = ThermalProfile {
        initial_temp: 20, ambient_temp: 20, ambient_rate: 0.010,
        conductivity: 0.015, heat_capacity: 1.0, ..base()
    };
    a[Element::Steam as usize]    = ThermalProfile {
        initial_temp: 115, ambient_temp: 95, ambient_rate: 0.010,
        conductivity: 0.025, heat_capacity: 3.0,
        // Condensation releases latent back to cold neighbors (warms them up).
        condense_below: Some(ph(55, Element::Water, 1200.0)),
        ..base()
    };
    a[Element::Lava as usize]     = ThermalProfile {
        initial_temp: 1800, ambient_temp: 20, ambient_rate: 0.006,
        // Heat capacity bumped from 8 → 12 — a pixel of magma holds more
        // energy per unit, so a big pool resists being entirely drained by
        // the (now much more powerful) latent-heat cooling of water above.
        conductivity: 0.055, heat_capacity: 12.0,
        freeze_below: Some(ph(1200, Element::Obsidian, 0.0)),
        ..base()
    };
    a[Element::Obsidian as usize] = ThermalProfile {
        ambient_rate: 0.050, conductivity: 0.030, heat_capacity: 2.5,
        // Lowered from 1500 → 1300 so direct lava contact can actually
        // re-melt the crust. At 1500 it was physically unreachable with
        // our ambient ceiling + obsidian's aggressive ambient drift.
        melt_above: Some(ph(1300, Element::Lava, 0.0)),
        ..base()
    };
    a[Element::Seed as usize]     = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.020, heat_capacity: 1.3,
        ignite_above: Some(320), burn_duration: Some(150),
        burn_temp: Some(650), ..base()
    };
    a[Element::Mud as usize]      = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.030, heat_capacity: 1.2, ..base()
    };
    a[Element::Leaves as usize]   = ThermalProfile {
        ambient_rate: 0.005, conductivity: 0.020, heat_capacity: 1.0,
        ignite_above: Some(230), burn_duration: Some(80),
        burn_temp: Some(600), ..base()
    };
    a[Element::Oil as usize]      = ThermalProfile {
        ambient_rate: 0.006, conductivity: 0.035, heat_capacity: 2.2,
        ignite_above: Some(180), burn_duration: Some(180),
        // Realistic hydrocarbon flame: hotter than wood. Water adjacent still
        // wins in sufficient volume, thanks to the increased latent heat of
        // vaporization — each boil drains 1200 energy from this cell.
        burn_temp: Some(800), ..base()
    };
    a[Element::Ice as usize]      = ThermalProfile {
        initial_temp: -5,
        ambient_rate: 0.010, conductivity: 0.050, heat_capacity: 2.0,
        // Melting absorbs latent from hot neighbors (endothermic).
        melt_above: Some(ph(5, Element::Water, 400.0)),
        ..base()
    };
    a[Element::MoltenGlass as usize] = ThermalProfile {
        initial_temp: 1200,
        ambient_rate: 0.005, conductivity: 0.040, heat_capacity: 3.0,
        // Freezes at 900°C (300° hysteresis below the sand-melt threshold,
        // so molten glass is stable in the 900-1200 band and doesn't oscillate).
        freeze_below: Some(ph(900, Element::Glass, 300.0)),
        ..base()
    };
    a[Element::Glass as usize]    = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.040, heat_capacity: 2.0,
        // Solid glass re-melts at 1100°C — above the freeze threshold but
        // below the sand-melt line, giving a clear solid regime.
        melt_above: Some(ph(1100, Element::MoltenGlass, 300.0)),
        ..base()
    };
    a[Element::Charcoal as usize] = ThermalProfile {
        // Solid residue of burned wood. Conducts/insulates more or less like
        // other powders. Not flammable — already burnt.
        ambient_rate: 0.003, conductivity: 0.015, heat_capacity: 1.0, ..base()
    };

    // --- Atoms ---
    // Flammable atoms get ignite_above + burn params; everything else uses
    // base(). Melting/boiling transitions left off for now — we don't have
    // "molten Fe", "liquid O₂", etc. as distinct Elements yet.
    a[Element::H  as usize] = ThermalProfile {
        // Hydrogen: extremely flammable, very hot flame, burns fast.
        ignite_above: Some(400), burn_duration: Some(20), burn_temp: Some(2100), ..base()
    };
    a[Element::C  as usize] = ThermalProfile {
        conductivity: 0.040, heat_capacity: 1.0,
        ignite_above: Some(500), burn_duration: Some(220), burn_temp: Some(900), ..base()
    };
    a[Element::Na as usize] = ThermalProfile {
        ignite_above: Some(100), burn_duration: Some(80), burn_temp: Some(700), ..base()
    };
    a[Element::Mg as usize] = ThermalProfile {
        // Magnesium burns with an intense white flame at 2000°C+.
        ignite_above: Some(470), burn_duration: Some(90), burn_temp: Some(2200), ..base()
    };
    a[Element::P  as usize] = ThermalProfile {
        // White phosphorus auto-ignites just above room temperature.
        ignite_above: Some(30), burn_duration: Some(60), burn_temp: Some(900), ..base()
    };
    a[Element::S  as usize] = ThermalProfile {
        ignite_above: Some(232), burn_duration: Some(120), burn_temp: Some(700), ..base()
    };
    a[Element::K  as usize] = ThermalProfile {
        ignite_above: Some(60), burn_duration: Some(90), burn_temp: Some(800), ..base()
    };
    // Gunpowder: ignites readily, burns fast (deflagration). High
    // conductivity so heat chain-ignites adjacent powder quickly — a
    // dense pile goes off almost all at once. The detonation impulse
    // itself comes from the shockwave-radius pressure injection in the
    // combustion code.
    a[Element::Gunpowder as usize] = ThermalProfile {
        ignite_above: Some(200),
        // Detonation, not deflagration — gunpowder ignition is near-
        // instantaneous. A 2-frame burn lets it emit its shockwave and
        // convert to hot smoke in ~33ms. Longer durations made piles just
        // fizzle into smoke with no visible boom.
        burn_duration: Some(2),
        burn_temp: Some(2200),
        conductivity: 0.15,
        ..base()
    };
    // Caesium — hyper-reactive alkali. Real Cs ignites in dry air at
    // room temperature and burns with a violet flame. We model that by
    // making it combustible with a very low ignition threshold so any
    // initial exothermic surface reaction (Cs + O → Cs₂O) cascades
    // through the interior of a pile instead of leaving a smoldering
    // surface. burn_temp 1400 keeps the chain going without quite
    // hitting the 1200°C shockwave threshold in the reaction path
    // (those two mechanisms are independent, but 1400 in the burn
    // sustain doesn't spawn shockwaves).
    a[Element::Cs as usize] = ThermalProfile {
        // heat_capacity 1.0 (matching other metals) keeps the diffusion
        // pass numerically stable — values below ~0.5 combined with
        // conductivity 0.08 lets heat overshoot thermal equilibrium in
        // one frame (CFL violation), which cascades into cells inventing
        // hundreds of degrees out of nothing. Real Cs does have a lower
        // heat capacity than typical metals, but the sim's explicit
        // integrator can't handle it without sub-stepping.
        ambient_rate: 0.025, conductivity: 0.080, heat_capacity: 1.0,
        ignite_above: Some(50), burn_duration: Some(8),
        burn_temp: Some(1400), ..base()
    };
    // Fluorine — no bespoke thermal quirks (it doesn't burn itself; it's
    // the aggressor that oxidizes everything else via the reaction engine).
    // Derived compounds use base() thermals.

    // Rust decomposes back to Fe at high heat. Real Fe₂O₃ breaks down around
    // 1500°C; we use Fe's melting point (1538°C) as the threshold so the
    // reduced iron is immediately molten, which matches the "melt the rust
    // off and get Fe back" intuition. Endothermic — decomposition pulls
    // Rust decomposes at 1538°C — handled by the unified decomposition path
    // in thermal() (see decomposition_of), which emits an O byproduct gas
    // alongside the Fe metal. No ThermalProfile override needed.
    // Salt melts at 801°C. Without a dedicated MoltenSalt element yet, its
    // molten form is the generic derived-compound phase — skipped here and
    // handled as a future enhancement alongside other compound phases.
    a
};

static MOISTURE: [MoistureProfile; ELEMENT_COUNT] = {
    const fn base() -> MoistureProfile {
        MoistureProfile {
            default_moisture: 0, conductivity: 0.0,
            is_source: false, is_sink: false,
            wet_above: None, dry_below: None,
        }
    }
    let mut a = [base(); ELEMENT_COUNT];
    a[Element::Sand   as usize] = MoistureProfile { conductivity: 0.08, is_sink: true,
        wet_above: Some((150, Element::Mud)), ..base() };
    a[Element::Water  as usize] = MoistureProfile { default_moisture: 255,
        conductivity: 1.0, is_source: true, ..base() };
    a[Element::Stone  as usize] = MoistureProfile { conductivity: 0.02, is_sink: true, ..base() };
    a[Element::Wood   as usize] = MoistureProfile { conductivity: 0.07, is_sink: true, ..base() };
    a[Element::Steam  as usize] = MoistureProfile { conductivity: 0.03, is_sink: true, ..base() };
    a[Element::Obsidian as usize] = MoistureProfile { conductivity: 0.02, is_sink: true, ..base() };
    a[Element::Seed   as usize] = MoistureProfile { conductivity: 0.07, is_sink: true, ..base() };
    // Mud is a sink only. It holds moisture (default 200), wicks it to drier
    // sinks via gradient, but does NOT impart +5/frame to every neighbor —
    // otherwise a single drop of water cascades unbounded across a sand pile.
    a[Element::Mud    as usize] = MoistureProfile { default_moisture: 200,
        conductivity: 0.10, is_source: false, is_sink: true,
        dry_below: Some((20, Element::Sand)), ..base() };
    a[Element::Leaves as usize] = MoistureProfile { conductivity: 0.08, is_sink: true, ..base() };
    // Ice holds water but can't impart moisture — it has to melt first.
    // Cold ice on dry sand leaves the sand dry; warm ambient melts the ice,
    // the resulting water pools/wets as normal.
    a[Element::Ice    as usize] = MoistureProfile { default_moisture: 255,
        conductivity: 0.05, is_source: false, ..base() };
    a
};

static PRESSURE: [PressureProfile; ELEMENT_COUNT] = {
    const fn base() -> PressureProfile {
        PressureProfile { permeability: 0, compliance: 0, formation_pressure: 0 }
    }
    let mut a = [base(); ELEMENT_COUNT];
    // Air — fully conductive, pushes freely along gradients.
    a[Element::Empty as usize] = PressureProfile { permeability: 255, compliance: 255, formation_pressure: 0 };
    // Rigid solids / gravels — walls. Pressure doesn't diffuse through
    // (perm 0) but strong blast differentials across the cell produce a
    // net force that flings the fragment. Tougher materials (Obsidian,
    // Glass) have lower compliance — take more blast to shatter.
    a[Element::Stone    as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    a[Element::Wood     as usize] = PressureProfile { permeability: 0, compliance: 25, formation_pressure: 0 };
    a[Element::Obsidian as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Seed     as usize] = PressureProfile { permeability: 0, compliance: 30, formation_pressure: 0 };
    a[Element::Ice      as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Glass    as usize] = PressureProfile { permeability: 0, compliance: 15, formation_pressure: 0 };
    // Powders — packed grain lets a little pressure leak through but the
    // grains themselves don't get shoved around by gradients.
    // Powders: some compliance so explosions fling them; mud is heaviest
    // (packs tight), leaves lightest. Won't budge under ordinary
    // atmospheric gradients — only explosive-scale (Δp > 400).
    a[Element::Sand     as usize] = PressureProfile { permeability: 25, compliance: 15, formation_pressure: 0 };
    a[Element::Mud      as usize] = PressureProfile { permeability: 15, compliance:  8, formation_pressure: 0 };
    a[Element::Leaves   as usize] = PressureProfile { permeability: 40, compliance: 30, formation_pressure: 0 };
    a[Element::Charcoal as usize] = PressureProfile { permeability: 30, compliance: 15, formation_pressure: 0 };
    // Liquids — transmit pressure well, slosh toward lower-pressure neighbors.
    a[Element::Water       as usize] = PressureProfile { permeability: 120, compliance: 60, formation_pressure: 0 };
    a[Element::Oil         as usize] = PressureProfile { permeability: 100, compliance: 50, formation_pressure: 0 };
    a[Element::Lava        as usize] = PressureProfile { permeability:  80, compliance: 40, formation_pressure: 0 };
    a[Element::MoltenGlass as usize] = PressureProfile { permeability:  60, compliance: 30, formation_pressure: 0 };
    // Gases — full pressure coupling.
    a[Element::Fire  as usize] = PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 };
    // Steam: water → steam expansion injects a strong overpressure. This is
    // what makes a boiling kettle vent steam instead of just accumulating.
    a[Element::Steam as usize] = PressureProfile { permeability: 230, compliance: 220, formation_pressure: 80 };
    a[Element::CO2 as usize] = PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 };

    // --- Atoms ---
    // Gases: full coupling. Painted-in atomic gases get a small formation
    // overpressure so the newly-placed cells actually disperse instead of
    // sitting idle (they have no intrinsic heat source to generate pressure).
    a[Element::H  as usize] = PressureProfile { permeability: 240, compliance: 230, formation_pressure: 30 };
    a[Element::He as usize] = PressureProfile { permeability: 240, compliance: 230, formation_pressure: 30 };
    a[Element::N  as usize] = PressureProfile { permeability: 230, compliance: 210, formation_pressure: 20 };
    a[Element::O  as usize] = PressureProfile { permeability: 230, compliance: 210, formation_pressure: 20 };
    a[Element::Ne as usize] = PressureProfile { permeability: 230, compliance: 210, formation_pressure: 20 };
    a[Element::Cl as usize] = PressureProfile { permeability: 220, compliance: 200, formation_pressure: 20 };
    // Atomic powders — like carbon soot, sulfur, phosphorus.
    a[Element::C  as usize] = PressureProfile { permeability: 30, compliance: 15, formation_pressure: 0 };
    a[Element::P  as usize] = PressureProfile { permeability: 30, compliance: 15, formation_pressure: 0 };
    a[Element::S  as usize] = PressureProfile { permeability: 30, compliance: 15, formation_pressure: 0 };
    // Atomic metals — rigid (perm 0 so blasts don't pressure-leak through
    // them), but a small compliance so explosions fling chunks outward.
    // Ordinary hydrostatic gradients don't budge them (force threshold in
    // try_pressure_shove is 400); only blast-scale differentials do.
    a[Element::Na as usize] = PressureProfile { permeability: 0, compliance: 30, formation_pressure: 0 };
    a[Element::Mg as usize] = PressureProfile { permeability: 0, compliance: 25, formation_pressure: 0 };
    a[Element::Al as usize] = PressureProfile { permeability: 0, compliance: 30, formation_pressure: 0 };
    a[Element::Si as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    a[Element::K  as usize] = PressureProfile { permeability: 0, compliance: 30, formation_pressure: 0 };
    a[Element::Ca as usize] = PressureProfile { permeability: 0, compliance: 25, formation_pressure: 0 };
    a[Element::Fe as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    a[Element::Cu as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Au as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::U  as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    // Mercury — liquid, transmits pressure well.
    a[Element::Hg as usize] = PressureProfile { permeability: 100, compliance: 40, formation_pressure: 0 };
    // Fluorine — full gas coupling, formation pressure on paint so it
    // diffuses into the play area aggressively.
    a[Element::F  as usize] = PressureProfile { permeability: 220, compliance: 200, formation_pressure: 20 };
    // Rust & Salt — powder, limited pressure path.
    a[Element::Rust as usize] = PressureProfile { permeability: 25, compliance: 0, formation_pressure: 0 };
    a[Element::Salt as usize] = PressureProfile { permeability: 25, compliance: 0, formation_pressure: 0 };
    // Gunpowder — very high compliance so it flies when blast forces
    // spawn, somewhat porous so blast pressure spreads through a pile.
    a[Element::Gunpowder as usize] = PressureProfile { permeability: 40, compliance: 100, formation_pressure: 0 };
    // Argon follows the other monatomic-gas noble pattern (full pressure
    // coupling, small formation pressure on paint so it disperses).
    a[Element::Ar        as usize] = PressureProfile { permeability: 230, compliance: 210, formation_pressure: 20 };
    // Tier-1 metals: mostly mirror Fe/Cu — rigid solids, no pressure path,
    // small compliance for a nudge under blast gradients.
    a[Element::Zn as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    a[Element::Ag as usize] = PressureProfile { permeability: 0, compliance: 15, formation_pressure: 0 };
    a[Element::Ni as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Pb as usize] = PressureProfile { permeability: 0, compliance:  8, formation_pressure: 0 };
    a[Element::B  as usize] = PressureProfile { permeability: 20, compliance:  0, formation_pressure: 0 };
    a[Element::Ra as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    a[Element::Cs as usize] = PressureProfile { permeability: 0, compliance: 30, formation_pressure: 0 };
    a
};

// Electrical properties per element. `conductivity` is 0.0 = insulator,
// 1.0 = perfect conductor — controls whether current propagates through
// the cell via the energized flood-fill. `glow_color` is the color a
// cell renders when energized (used for noble-gas neon-sign glow).
// Absent glow_color means "no visible glow when energized."
#[derive(Clone, Copy)]
struct ElectricalProfile {
    conductivity: f32,
    glow_color:   Option<(u8, u8, u8)>,
}

static ELECTRICAL: [ElectricalProfile; ELEMENT_COUNT] = {
    const fn base() -> ElectricalProfile {
        ElectricalProfile { conductivity: 0.0, glow_color: None }
    }
    let mut a = [base(); ELEMENT_COUNT];
    // Metals — strong conductors. Ordered by real-world conductivity
    // (Ag > Cu > Au > Al > Fe > Na > …).
    a[Element::Ag as usize] = ElectricalProfile { conductivity: 0.98, glow_color: None };
    a[Element::Cu as usize] = ElectricalProfile { conductivity: 0.95, glow_color: None };
    a[Element::Au as usize] = ElectricalProfile { conductivity: 0.92, glow_color: None };
    a[Element::Al as usize] = ElectricalProfile { conductivity: 0.75, glow_color: None };
    a[Element::Fe as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Na as usize] = ElectricalProfile { conductivity: 0.30, glow_color: None };
    a[Element::K  as usize] = ElectricalProfile { conductivity: 0.30, glow_color: None };
    a[Element::Mg as usize] = ElectricalProfile { conductivity: 0.35, glow_color: None };
    a[Element::Ca as usize] = ElectricalProfile { conductivity: 0.32, glow_color: None };
    a[Element::Hg as usize] = ElectricalProfile { conductivity: 0.25, glow_color: None };
    a[Element::U  as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    // Tier-1 extras fit into the galvanic spread.
    a[Element::Zn as usize] = ElectricalProfile { conductivity: 0.28, glow_color: None };
    a[Element::Ni as usize] = ElectricalProfile { conductivity: 0.23, glow_color: None };
    a[Element::Pb as usize] = ElectricalProfile { conductivity: 0.08, glow_color: None };
    a[Element::Ra as usize] = ElectricalProfile { conductivity: 0.15, glow_color: None };
    a[Element::Cs as usize] = ElectricalProfile { conductivity: 0.18, glow_color: None };
    // Boron — metalloid, near-insulator at STP (semiconductor strictly).
    a[Element::B  as usize] = ElectricalProfile { conductivity: 0.01, glow_color: None };
    a[Element::Si as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    // Carbon is conductive (graphite-like) — gives users a non-metal
    // conductor option.
    a[Element::C  as usize] = ElectricalProfile { conductivity: 0.15, glow_color: None };
    // Noble gases — don't propagate current themselves (insulators at
    // low current), but light up their characteristic color when part of
    // an energized circuit. Real neon signs work the same way: gas
    // ionizes under high voltage and emits a narrow emission spectrum.
    a[Element::Ne as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((255, 90, 40)),
    };
    a[Element::Ar as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((170, 100, 255)),
    };
    a[Element::He as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((255, 160, 150)),
    };
    // Battery — high conductivity so current flows out of it freely;
    // acts as a voltage source in the energized flood.
    a[Element::BattPos as usize] = ElectricalProfile {
        conductivity: 1.0, glow_color: None,
    };
    a[Element::BattNeg as usize] = ElectricalProfile {
        conductivity: 1.0, glow_color: None,
    };
    a
};

// ============================================================================
// ATOMIC FRAMEWORK
//
// Every naturally-occurring periodic-table element (H through U — 92 atoms)
// has a profile here. 20 are fully fleshed out; the rest are placeholders so
// the UI can show them with real positioning and it's obvious what remains to
// implement. Atoms aren't paintable yet — they're reference / composition
// data. Compounds will eventually derive some properties from these.
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AtomState { Solid, Liquid, Gas }

#[derive(Clone, Copy, PartialEq, Eq)]
enum AtomCategory {
    Hydrogen,
    AlkaliMetal,
    AlkalineEarth,
    TransitionMetal,
    PostTransition,
    Metalloid,
    Nonmetal,
    Halogen,
    NobleGas,
    Lanthanide,
    Actinide,
}

#[derive(Clone, Copy)]
struct AtomProfile {
    number: u8,
    symbol: &'static str,
    name: &'static str,
    period: u8,          // 1-7 for main table, 8 = lanthanides, 9 = actinides
    group: u8,           // 1-18 column
    stp_state: AtomState,
    category: AtomCategory,
    atomic_mass: f32,    // amu
    melting_point: i16,  // °C
    boiling_point: i16,  // °C
    density_stp: f32,    // g/cm^3 at STP
    // Pauling electronegativity (0.7–4.0). The emergent reactivity layer
    // uses Δelectronegativity between neighbors to decide whether two atoms
    // want to exchange/share electrons. Noble gases are set to 0.0 meaning
    // "doesn't participate" even though formally it's undefined.
    electronegativity: f32,
    // Valence electrons in the outer shell. Combined with electronegativity,
    // this is what distinguishes "wants to lose 1" (alkali) from "wants to
    // gain 1" (halogen) from "full shell, inert" (noble gas).
    valence_electrons: u8,
    // Radioactive decay. Nonzero half_life_frames means the atom is
    // unstable — it has a per-frame probability of transmuting into
    // `decay_product` and releasing `decay_heat` to itself and neighbors.
    // The per-frame probability is approximated as ln(2) / half_life,
    // which is accurate for slow decay (p << 1) and close-enough for
    // game timescales. half_life_frames = 0 means stable.
    half_life_frames: u32,
    decay_product: Element,
    decay_heat: i16,
    implemented: bool,
    notes: &'static str,
}

const ATOM_COUNT: usize = 92;

// Derive a reasonable default valence count from an atom's group number,
// used to auto-fill placeholder atoms. Group 1 = 1 valence e⁻, group 17 = 7,
// group 18 = 8 (full p-shell). Transition metals (3-12) get a default of 2,
// which is the most common oxidation state for most of them.
const fn valence_from_group(group: u8) -> u8 {
    match group {
        1  => 1,
        2  => 2,
        3..=12 => 2,
        13 => 3,
        14 => 4,
        15 => 5,
        16 => 6,
        17 => 7,
        18 => 8,
        _  => 0,
    }
}

// Shorthand for an un-implemented atom: keeps the static table readable.
// electronegativity is left at 0 for placeholders — when we flesh one out,
// we fill in the real Pauling value. Valence is derived from group so the
// reactivity engine has at least *something* to work with.
const fn stub(
    n: u8, sym: &'static str, name: &'static str,
    per: u8, grp: u8, cat: AtomCategory, mass: f32, state: AtomState,
) -> AtomProfile {
    AtomProfile {
        number: n, symbol: sym, name,
        period: per, group: grp, stp_state: state, category: cat,
        atomic_mass: mass, melting_point: 0, boiling_point: 0, density_stp: 0.0,
        electronegativity: 0.0,
        valence_electrons: valence_from_group(grp),
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: false, notes: "placeholder — physical & sim profile TODO",
    }
}

use AtomCategory::*;
use AtomState::{Solid as SSolid, Liquid as SLiquid, Gas as SGas};

static ATOMS: [AtomProfile; ATOM_COUNT] = [
    // ---- Period 1 ----
    AtomProfile { number: 1, symbol: "H", name: "Hydrogen",
        period: 1, group: 1, stp_state: SGas, category: Hydrogen,
        atomic_mass: 1.008, melting_point: -259, boiling_point: -253, density_stp: 0.00009,
        electronegativity: 2.20, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "reactive; combines with O to form water; extremely flammable" },
    AtomProfile { number: 2, symbol: "He", name: "Helium",
        period: 1, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 4.0026, melting_point: -272, boiling_point: -269, density_stp: 0.000178,
        electronegativity: 0.0, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "inert noble gas; rises in air; does not react" },

    // ---- Period 2 ----
    stub(3,  "Li", "Lithium",   2,  1, AlkaliMetal,    6.94,   SSolid),
    stub(4,  "Be", "Beryllium", 2,  2, AlkalineEarth,  9.012,  SSolid),
    AtomProfile { number: 5, symbol: "B", name: "Boron",
        period: 2, group: 13, stp_state: SSolid, category: Metalloid,
        atomic_mass: 10.81, melting_point: 2076, boiling_point: 3927, density_stp: 2.08,
        electronegativity: 2.04, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "metalloid; semiconductor; future neutron absorber for control rods" },
    AtomProfile { number: 6, symbol: "C", name: "Carbon",
        period: 2, group: 14, stp_state: SSolid, category: Nonmetal,
        atomic_mass: 12.011, melting_point: 3550, boiling_point: 4027, density_stp: 2.267,
        electronegativity: 2.55, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "forms the backbone of organic compounds; burns with O to form CO2" },
    AtomProfile { number: 7, symbol: "N", name: "Nitrogen",
        period: 2, group: 15, stp_state: SGas, category: Nonmetal,
        atomic_mass: 14.007, melting_point: -210, boiling_point: -196, density_stp: 0.001251,
        electronegativity: 3.04, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "inert diatomic gas; makes up 78% of air; dilutes combustion" },
    AtomProfile { number: 8, symbol: "O", name: "Oxygen",
        period: 2, group: 16, stp_state: SGas, category: Nonmetal,
        atomic_mass: 15.999, melting_point: -218, boiling_point: -183, density_stp: 0.001429,
        electronegativity: 3.44, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "diatomic gas; supports combustion; reacts with H to form water" },
    AtomProfile { number: 9, symbol: "F", name: "Fluorine",
        period: 2, group: 17, stp_state: SGas, category: Halogen,
        atomic_mass: 18.998, melting_point: -220, boiling_point: -188, density_stp: 0.001696,
        electronegativity: 3.98, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "most electronegative element; attacks nearly everything, including noble metals like gold" },
    AtomProfile { number: 10, symbol: "Ne", name: "Neon",
        period: 2, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 20.180, melting_point: -248, boiling_point: -246, density_stp: 0.0009,
        electronegativity: 0.0, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "noble gas; chemically inert; glows orange when electrified" },

    // ---- Period 3 ----
    AtomProfile { number: 11, symbol: "Na", name: "Sodium",
        period: 3, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 22.990, melting_point: 98, boiling_point: 883, density_stp: 0.968,
        electronegativity: 0.93, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "highly reactive alkali metal; explodes on contact with water" },
    AtomProfile { number: 12, symbol: "Mg", name: "Magnesium",
        period: 3, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 24.305, melting_point: 650, boiling_point: 1090, density_stp: 1.738,
        electronegativity: 1.31, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "burns with a brilliant white flame at high temperature" },
    AtomProfile { number: 13, symbol: "Al", name: "Aluminum",
        period: 3, group: 13, stp_state: SSolid, category: PostTransition,
        atomic_mass: 26.982, melting_point: 660, boiling_point: 2470, density_stp: 2.70,
        electronegativity: 1.61, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "light, corrosion-resistant metal; good thermal/electrical conductor" },
    AtomProfile { number: 14, symbol: "Si", name: "Silicon",
        period: 3, group: 14, stp_state: SSolid, category: Metalloid,
        atomic_mass: 28.085, melting_point: 1414, boiling_point: 3265, density_stp: 2.329,
        electronegativity: 1.90, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "main constituent of sand (SiO2); glass former; semiconductor" },
    AtomProfile { number: 15, symbol: "P", name: "Phosphorus",
        period: 3, group: 15, stp_state: SSolid, category: Nonmetal,
        atomic_mass: 30.974, melting_point: 44, boiling_point: 277, density_stp: 1.823,
        electronegativity: 2.19, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "white phosphorus ignites spontaneously in air above ~30°C" },
    AtomProfile { number: 16, symbol: "S", name: "Sulfur",
        period: 3, group: 16, stp_state: SSolid, category: Nonmetal,
        atomic_mass: 32.06, melting_point: 115, boiling_point: 444, density_stp: 2.07,
        electronegativity: 2.58, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "burns with a blue flame to form SO2; strong, acrid smell" },
    AtomProfile { number: 17, symbol: "Cl", name: "Chlorine",
        period: 3, group: 17, stp_state: SGas, category: Halogen,
        atomic_mass: 35.45, melting_point: -102, boiling_point: -34, density_stp: 0.003214,
        electronegativity: 3.16, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "toxic yellow-green gas; denser than air; reacts with most metals" },
    AtomProfile { number: 18, symbol: "Ar", name: "Argon",
        period: 3, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 39.948, melting_point: -189, boiling_point: -186, density_stp: 0.0018,
        electronegativity: 0.0, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "noble gas; chemically inert; glows lavender-purple when electrified" },

    // ---- Period 4 ----
    AtomProfile { number: 19, symbol: "K", name: "Potassium",
        period: 4, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 39.098, melting_point: 63, boiling_point: 759, density_stp: 0.862,
        electronegativity: 0.82, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "extremely reactive; ignites on contact with water" },
    AtomProfile { number: 20, symbol: "Ca", name: "Calcium",
        period: 4, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 40.078, melting_point: 842, boiling_point: 1484, density_stp: 1.54,
        electronegativity: 1.00, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "soft silvery metal; major constituent of limestone, shells, bone" },
    stub(21, "Sc", "Scandium",   4,  3, TransitionMetal, 44.956, SSolid),
    stub(22, "Ti", "Titanium",   4,  4, TransitionMetal, 47.867, SSolid),
    stub(23, "V",  "Vanadium",   4,  5, TransitionMetal, 50.942, SSolid),
    stub(24, "Cr", "Chromium",   4,  6, TransitionMetal, 51.996, SSolid),
    stub(25, "Mn", "Manganese",  4,  7, TransitionMetal, 54.938, SSolid),
    AtomProfile { number: 26, symbol: "Fe", name: "Iron",
        period: 4, group: 8, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 55.845, melting_point: 1538, boiling_point: 2862, density_stp: 7.874,
        electronegativity: 1.83, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "workhorse metal; rusts in moist air (Fe + O + H2O); magnetic" },
    stub(27, "Co", "Cobalt",     4,  9, TransitionMetal, 58.933, SSolid),
    AtomProfile { number: 28, symbol: "Ni", name: "Nickel",
        period: 4, group: 10, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 58.693, melting_point: 1455, boiling_point: 2913, density_stp: 8.91,
        electronegativity: 1.91, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "ferromagnetic; coinage metal; foundation for electromagnets" },
    AtomProfile { number: 29, symbol: "Cu", name: "Copper",
        period: 4, group: 11, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 63.546, melting_point: 1085, boiling_point: 2562, density_stp: 8.96,
        electronegativity: 1.90, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "outstanding thermal and electrical conductor; forms patina in air" },
    AtomProfile { number: 30, symbol: "Zn", name: "Zinc",
        period: 4, group: 12, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 65.38, melting_point: 420, boiling_point: 907, density_stp: 7.14,
        electronegativity: 1.65, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "reactive metal; Zn+HCl fizzes H₂; textbook galvanic anode vs Cu" },
    stub(31, "Ga", "Gallium",    4, 13, PostTransition,  69.723, SSolid),
    stub(32, "Ge", "Germanium",  4, 14, Metalloid,       72.630, SSolid),
    stub(33, "As", "Arsenic",    4, 15, Metalloid,       74.922, SSolid),
    stub(34, "Se", "Selenium",   4, 16, Nonmetal,        78.971, SSolid),
    stub(35, "Br", "Bromine",    4, 17, Halogen,         79.904, SLiquid),
    stub(36, "Kr", "Krypton",    4, 18, NobleGas,        83.798, SGas),

    // ---- Period 5 ----
    stub(37, "Rb", "Rubidium",    5,  1, AlkaliMetal,     85.468, SSolid),
    stub(38, "Sr", "Strontium",   5,  2, AlkalineEarth,   87.62,  SSolid),
    stub(39, "Y",  "Yttrium",     5,  3, TransitionMetal, 88.906, SSolid),
    stub(40, "Zr", "Zirconium",   5,  4, TransitionMetal, 91.224, SSolid),
    stub(41, "Nb", "Niobium",     5,  5, TransitionMetal, 92.906, SSolid),
    stub(42, "Mo", "Molybdenum",  5,  6, TransitionMetal, 95.95,  SSolid),
    stub(43, "Tc", "Technetium",  5,  7, TransitionMetal, 98.0,   SSolid),
    stub(44, "Ru", "Ruthenium",   5,  8, TransitionMetal,101.07,  SSolid),
    stub(45, "Rh", "Rhodium",     5,  9, TransitionMetal,102.91,  SSolid),
    stub(46, "Pd", "Palladium",   5, 10, TransitionMetal,106.42,  SSolid),
    AtomProfile { number: 47, symbol: "Ag", name: "Silver",
        period: 5, group: 11, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 107.87, melting_point: 962, boiling_point: 2162, density_stp: 10.49,
        electronegativity: 1.93, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "highest electrical/thermal conductivity of any metal" },
    stub(48, "Cd", "Cadmium",     5, 12, TransitionMetal,112.41,  SSolid),
    stub(49, "In", "Indium",      5, 13, PostTransition, 114.82,  SSolid),
    stub(50, "Sn", "Tin",         5, 14, PostTransition, 118.71,  SSolid),
    stub(51, "Sb", "Antimony",    5, 15, Metalloid,      121.76,  SSolid),
    stub(52, "Te", "Tellurium",   5, 16, Metalloid,      127.60,  SSolid),
    stub(53, "I",  "Iodine",      5, 17, Halogen,        126.90,  SSolid),
    stub(54, "Xe", "Xenon",       5, 18, NobleGas,       131.29,  SGas),

    // ---- Period 6 (main row skips lanthanides 57-71) ----
    AtomProfile { number: 55, symbol: "Cs", name: "Caesium",
        period: 6, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 132.91, melting_point: 28, boiling_point: 671, density_stp: 1.93,
        electronegativity: 0.79, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "most reactive stable alkali; melts at body temp; detonates in water" },
    stub(56, "Ba", "Barium",      6,  2, AlkalineEarth,  137.33,  SSolid),

    // ---- Period 8 row: Lanthanides (57-71) ----
    stub(57, "La", "Lanthanum",    8,  3, Lanthanide, 138.91, SSolid),
    stub(58, "Ce", "Cerium",       8,  4, Lanthanide, 140.12, SSolid),
    stub(59, "Pr", "Praseodymium", 8,  5, Lanthanide, 140.91, SSolid),
    stub(60, "Nd", "Neodymium",    8,  6, Lanthanide, 144.24, SSolid),
    stub(61, "Pm", "Promethium",   8,  7, Lanthanide, 145.0,  SSolid),
    stub(62, "Sm", "Samarium",     8,  8, Lanthanide, 150.36, SSolid),
    stub(63, "Eu", "Europium",     8,  9, Lanthanide, 151.96, SSolid),
    stub(64, "Gd", "Gadolinium",   8, 10, Lanthanide, 157.25, SSolid),
    stub(65, "Tb", "Terbium",      8, 11, Lanthanide, 158.93, SSolid),
    stub(66, "Dy", "Dysprosium",   8, 12, Lanthanide, 162.50, SSolid),
    stub(67, "Ho", "Holmium",      8, 13, Lanthanide, 164.93, SSolid),
    stub(68, "Er", "Erbium",       8, 14, Lanthanide, 167.26, SSolid),
    stub(69, "Tm", "Thulium",      8, 15, Lanthanide, 168.93, SSolid),
    stub(70, "Yb", "Ytterbium",    8, 16, Lanthanide, 173.05, SSolid),
    stub(71, "Lu", "Lutetium",     8, 17, Lanthanide, 174.97, SSolid),

    // ---- Period 6 resumes (72-86) ----
    stub(72, "Hf", "Hafnium",      6,  4, TransitionMetal, 178.49, SSolid),
    stub(73, "Ta", "Tantalum",     6,  5, TransitionMetal, 180.95, SSolid),
    stub(74, "W",  "Tungsten",     6,  6, TransitionMetal, 183.84, SSolid),
    stub(75, "Re", "Rhenium",      6,  7, TransitionMetal, 186.21, SSolid),
    stub(76, "Os", "Osmium",       6,  8, TransitionMetal, 190.23, SSolid),
    stub(77, "Ir", "Iridium",      6,  9, TransitionMetal, 192.22, SSolid),
    stub(78, "Pt", "Platinum",     6, 10, TransitionMetal, 195.08, SSolid),
    AtomProfile { number: 79, symbol: "Au", name: "Gold",
        period: 6, group: 11, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 196.967, melting_point: 1064, boiling_point: 2856, density_stp: 19.30,
        electronegativity: 2.54, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "noble metal; does not tarnish; extremely dense; highly malleable" },
    AtomProfile { number: 80, symbol: "Hg", name: "Mercury",
        period: 6, group: 12, stp_state: SLiquid, category: TransitionMetal,
        atomic_mass: 200.592, melting_point: -39, boiling_point: 357, density_stp: 13.534,
        electronegativity: 2.00, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "only metal that is liquid at room temperature; toxic vapor" },
    stub(81, "Tl", "Thallium",     6, 13, PostTransition, 204.38, SSolid),
    AtomProfile { number: 82, symbol: "Pb", name: "Lead",
        period: 6, group: 14, stp_state: SSolid, category: PostTransition,
        atomic_mass: 207.2, melting_point: 327, boiling_point: 1749, density_stp: 11.34,
        electronegativity: 1.87, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "heavy, soft, unreactive; future radiation shielding material" },
    stub(83, "Bi", "Bismuth",      6, 15, PostTransition, 208.98, SSolid),
    stub(84, "Po", "Polonium",     6, 16, Metalloid,      209.0,  SSolid),
    stub(85, "At", "Astatine",     6, 17, Halogen,        210.0,  SSolid),
    stub(86, "Rn", "Radon",        6, 18, NobleGas,       222.0,  SGas),

    // ---- Period 7 (main row stops at Ra; actinides go to strip) ----
    stub(87, "Fr", "Francium",     7,  1, AlkaliMetal,    223.0,  SSolid),
    AtomProfile { number: 88, symbol: "Ra", name: "Radium",
        period: 7, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 226.0, melting_point: 700, boiling_point: 1737, density_stp: 5.5,
        electronegativity: 0.9, valence_electrons: 2,
        // Real Ra-226 half-life is 1600 years. Game-compressed to ~60
        // minutes (3.6M frames) so a ~1000-cell pile transmutes roughly
        // once every 90 seconds — matching gameplay intuition that
        // macroscopic nuclear decay should be rare even in a radiation
        // sample. Per-frame heat output is decoupled (see decay()) so
        // the pile still warms palpably regardless of half-life choice.
        half_life_frames: 3_600_000, decay_product: Element::Pb, decay_heat: 50,
        implemented: true,
        notes: "naturally radioactive alkaline earth; the headline atom for the decay phase" },

    // ---- Period 9 row: Actinides (89-92, naturals) ----
    stub(89, "Ac", "Actinium",     9,  3, Actinide, 227.0,  SSolid),
    stub(90, "Th", "Thorium",      9,  4, Actinide, 232.04, SSolid),
    stub(91, "Pa", "Protactinium", 9,  5, Actinide, 231.04, SSolid),
    AtomProfile { number: 92, symbol: "U", name: "Uranium",
        period: 9, group: 6, stp_state: SSolid, category: Actinide,
        atomic_mass: 238.029, melting_point: 1135, boiling_point: 4131, density_stp: 19.05,
        // Uranium's real oxidation states go up to +6 (e.g. UF₆), but for
        // this sim's donor/acceptor model we use the "outer shell count"
        // pattern — actinides bond through 7s and 5f electrons; using 3
        // here makes U behave like a reactive metal (donor) in the engine.
        electronegativity: 1.38, valence_electrons: 3,
        // U-238 real half-life is 4.5 billion years; U-235 is 700M.
        // Game-compressed to ~7 minutes (1.5M frames) so that multi-
        // thousand-cell piles produce visible transmutation activity
        // in real time — each decay event pops a small shockwave. In
        // Phase 2, criticality multiplies this rate exponentially for
        // large piles. Decays to Pb (collapsing the long thorium/radium
        // chain into a single sim-friendly transition).
        half_life_frames: 1_500_000, decay_product: Element::Pb, decay_heat: 30,
        implemented: true,
        notes: "radioactive; decays over millennia releasing heat; fissile (U-235)" },
];

// ============================================================================
// EMERGENT REACTIONS
//
// Reactions arise from atomic properties, not a hand-tuned table. The engine
// evaluates adjacent cell pairs for electron-exchange reactivity:
//
//   1. Each side has a "chemistry signature" — (electronegativity, valence,
//      atomic_mass). Atoms read straight from AtomProfile. Compounds (Water,
//      Oil, Rust, Salt, …) override with the chemistry of their reactive
//      face — e.g. Water presents O's signature because that's the atom
//      that participates in combustion/oxidation.
//
//   2. Δelectronegativity tells us how much one atom wants to pull the
//      other's electrons. Large Δ (say ≥1.0) means a real ionic/polar bond
//      wants to form; small Δ means the pair is nearly inert together.
//
//   3. Valence compatibility — one side must want to *give* electrons (low
//      valence ≤3: alkali/alkaline-earth/transition metals) and the other
//      must want to *take* (high valence ≥5: nonmetals/halogens).
//
//   4. Activation energy — reactivity score determines how much thermal
//      energy is needed. Very reactive pairs (F + anything) react cold;
//      marginal pairs (H + O, C + O) need heat.
//
//   5. Catalysts from the wider neighborhood multiply the rate — Water as
//      an electrolyte for oxidation, Salt for extra ionic conductivity.
//
//   6. Products infer from donor/acceptor stoichiometry, falling back to
//      Smoke (an oxidation residue placeholder) when we don't have a
//      matching derived compound yet.
// ============================================================================

// Chemistry signature of any element: (electronegativity, valence_electrons,
// atomic_mass). Returns None for elements that shouldn't participate in the
// emergent reaction engine — inert compounds, derived products, etc.
fn element_chemistry(el: Element) -> Option<(f32, u8, f32)> {
    if let Some(a) = atom_profile_for(el) {
        return Some((a.electronegativity, a.valence_electrons, a.atomic_mass));
    }
    // Compound overrides — present the chemistry of the reactive face.
    // Water's O is what gets donated-to in oxidations, so Water reads as O.
    // Oil is a hydrocarbon — its carbon is the business end.
    match el {
        Element::Water | Element::Ice | Element::Steam =>
            Some((3.44, 6, 18.0)),
        Element::Oil => Some((2.55, 4, 14.0)),
        // Rust & Salt are already "done" products — inert to further
        // reactions in this engine. (Salt + water will become its own
        // interaction when we add dissolution later.)
        _ => None,
    }
}

// Result of `infer_product` — either a bespoke Element (Water, Rust, Salt,
// H, Steam, Smoke) or a derived-compound cell with a registry index.
#[derive(Clone, Copy)]
enum InferredProduct {
    Bespoke(Element),
    Derived(u8),
}

// Determine the product for an emergent reaction given the donor (lower
// electronegativity, the one losing electrons) and acceptor (higher
// electronegativity, the one gaining) elements. Catalysts in the wider
// neighborhood gate certain reactions (Fe+O won't rust without Water).
//
// Bespoke matches produce hand-tuned compounds (Water, Rust, Salt, …).
// Everything else falls through to the derived-compound registry, which
// synthesizes a new compound from the constituents — so Au + F → AuF with
// derived color/density, Cu + O → CuO, Mg + Cl → MgCl₂, etc. Inspect a
// derived cell and you'll see its formula in the tooltip.
fn infer_product(donor: Element, acceptor: Element, catalysts: &[Element]) -> Option<InferredProduct> {
    let has_water = catalysts.iter().any(|&c|
        matches!(c, Element::Water | Element::Ice | Element::Steam));
    // Bespoke tuned products always win — they represent real-world
    // compounds the engine knows about (Water, Rust, Salt). Catalysts
    // affect rate, not product identity.
    let _ = has_water;
    let bespoke = match (donor, acceptor) {
        (Element::H, Element::O) => Some(Element::Water),
        (Element::Na, Element::Cl) => Some(Element::Salt),
        (Element::Fe, Element::O) => Some(Element::Rust),
        (Element::C, Element::O) => Some(Element::CO2),
        (m, Element::Water) | (m, Element::Ice) | (m, Element::Steam)
            if atom_profile_for(m).map_or(false, |a|
                a.implemented
                && a.electronegativity > 0.0
                && a.electronegativity < 1.4
                && a.valence_electrons <= 2)
            => Some(Element::H),
        _ => None,
    };
    if let Some(el) = bespoke {
        return Some(InferredProduct::Bespoke(el));
    }
    // Fallback — derive a new compound from the constituents.
    derive_or_lookup(donor, acceptor).map(InferredProduct::Derived)
}

// Product spec for a single cell's post-reaction state.
#[derive(Clone, Copy)]
struct ProductSpec {
    el: Element,
    derived_id: u8,
}

impl ProductSpec {
    fn bespoke(el: Element) -> Self { ProductSpec { el, derived_id: 0 } }
    fn derived(idx: u8) -> Self { ProductSpec { el: Element::Derived, derived_id: idx } }
    fn from_inferred(p: InferredProduct) -> Self {
        match p {
            InferredProduct::Bespoke(e) => Self::bespoke(e),
            InferredProduct::Derived(i) => Self::derived(i),
        }
    }
}

// Result of an emergent-reaction check for a single adjacent pair.
#[derive(Clone, Copy)]
struct ReactionOutcome {
    products: [ProductSpec; 2],
    delta_temp: i16,
    rate: f32,
}

// Attempt to react cells A and B based on their chemistry. Returns the
// outcome if a reaction fires this frame, otherwise None.
fn try_emergent_reaction(
    a_el: Element,
    b_el: Element,
    a_temp: i16,
    b_temp: i16,
    catalysts: &[Element],
) -> Option<ReactionOutcome> {
    let (ea, va, _ma) = element_chemistry(a_el)?;
    let (eb, vb, _mb) = element_chemistry(b_el)?;
    if ea == 0.0 || eb == 0.0 { return None; } // noble / inert

    let delta_e = (ea - eb).abs();
    if delta_e < 0.4 { return None; } // too alike, no bond forms

    // Donor is lower-electronegativity (gives electrons); acceptor pulls.
    let (donor_el, donor_v, acceptor_el, acceptor_v) = if ea < eb {
        (a_el, va, b_el, vb)
    } else {
        (b_el, vb, a_el, va)
    };
    // Valence compatibility — donor should want to lose, acceptor to gain.
    // Donor ≤4 permits carbon (v=4) to participate in covalent bonds; the
    // acceptor still needs a half-full-or-more outer shell to attract.
    if donor_v > 4 || acceptor_v < 5 { return None; }

    // Activation energy scales inversely with Δelectronegativity — strongly
    // polar pairs (F+metal) fire cold; marginal pairs (H+O, C+O) need
    // meaningful heat. Two extra physics-shaped modifiers:
    //
    //   * High-electronegativity acceptors (F=3.98, O=3.44, Cl=3.16) pull
    //     electrons so hard they lower activation a lot on their own.
    //     That's why fluorine reacts with gold at room temperature in real
    //     life — the pair's Δe isn't huge, but F itself is the aggressor.
    //
    //   * Catalysts (water electrolyte, salt ions) drop activation for
    //     oxidation-style reactions — Fe rusts in humid air but not a
    //     desert.
    let has_electrolyte = catalysts.iter().any(|&c|
        matches!(c, Element::Water | Element::Ice | Element::Steam | Element::Salt));
    let (donor_e, acceptor_e) = if ea < eb { (ea, eb) } else { (eb, ea) };
    let acceptor_bonus = ((acceptor_e - 2.5).max(0.0) * 300.0) as i16;
    // Metallic donors (low electronegativity) give up electrons eagerly —
    // they oxidize/corrode at room temperature even when Δe is moderate.
    // Covalent donors (like H at E=2.20) don't get this free pass, which
    // is why H+O still needs an ignition kick but Cu+O doesn't.
    let donor_metal_bonus: i16 = if donor_e < 2.0 { 200 } else { 0 };
    // Bucket boundary at 0.9 (not 1.0) so H+Cl (Δe=0.96) can actually
    // form HCl — it's a real, well-known reaction that was sitting just
    // above the 1.0 threshold and therefore silently stuck at activation
    // 800°C. With 0.9 it drops into the 400°C bucket, and with Cl's
    // acceptor bonus lands around 200°C — matching real-world synthesis
    // temperatures for H + Cl → HCl.
    let mut activation: i16 = if delta_e >= 2.5 { -200 }
        else if delta_e >= 1.6 { 100 }
        else if delta_e >= 0.9 { 400 }
        else { 800 };
    activation -= acceptor_bonus;
    activation -= donor_metal_bonus;
    if has_electrolyte { activation -= 200; }
    if a_temp < activation || b_temp < activation { return None; }

    // Product lookup via donor/acceptor stoichiometry.
    let inferred = infer_product(donor_el, acceptor_el, catalysts)?;

    // Rate: base on reactivity, amplified by catalysts.
    let mut rate: f32 = (delta_e * 0.2).min(1.0);
    for &c in catalysts {
        match c {
            Element::Water | Element::Ice | Element::Steam => rate *= 3.0,
            Element::Salt => rate *= 5.0,
            _ => {}
        }
    }
    // Inherently-slow reactions (Fe oxidation) start very low and ride the
    // catalyst multipliers up. Real iron rusts over days to weeks even in
    // wet conditions — this multiplier keeps the sim's timescale tolerable
    // (noticeable tarnish in minutes of game time) without the "sweating
    // rust" effect a higher rate produces when every Fe surface cell is
    // tested against virtual O every frame.
    if matches!(inferred, InferredProduct::Bespoke(Element::Rust)) {
        rate = (rate * 0.0005).min(0.05);
    }
    // Derived compounds (non-bespoke products) reflect slow surface
    // corrosion/tarnish processes, not energetic combustion. Very low rate
    // so a metal in air oxidizes visibly over seconds rather than snapping
    // to its compound in one frame.
    if matches!(inferred, InferredProduct::Derived(_)) {
        rate = (rate * 0.01).min(0.2);
    }
    // Hyper-reactive alkali donors (Cs EN 0.79; Rb, Fr would land here
    // once added). Real Cs ignites in air and explodes in water — so we
    // unwind the generic derived-oxide slowdown and bring it close to
    // "happens in seconds". K (0.82) and Na (0.93) stay on the slow
    // path, matching their real behavior of tarnish-rather-than-ignite
    // in dry air.
    if donor_e < 0.85 {
        // Cs reacting with O isn't "surface tarnish" — it's violent
        // combustion. When an O cell actually touches a Cs cell, the
        // reaction should consume both essentially on contact, not
        // play the slow-corrosion tune we use for Fe+O tarnish.
        rate = (rate * 18.0).min(0.99);
    }
    // Violent tier — extreme EN gap (>= 2.8) crossed with a low-EN
    // donor (< 1.0). Catches halogen+alkali/alkaline-earth combos like
    // Cs+F, Na+F, K+F, Li+F, Cs+O, Na+O. Real-world adiabatic flame
    // temperatures for these reactions are >3000°C; the bulk of the
    // reactant flash-vaporizes rather than passivating with a stable
    // coating, so per-contact rate approaches 100%. Bypass the
    // derived-product slowdown that would otherwise cap these at
    // surface-corrosion timescales.
    if delta_e >= 2.8 && donor_e < 1.0 && matches!(inferred, InferredProduct::Derived(_)) {
        rate = rate.max(0.85);
    }

    // Heat released. Bespoke reactions get hand-tuned values matched to
    // real-world enthalpy of formation; emergent reactions default to a
    // *small* Δe-based release. The derived fallback is intentionally mild —
    // surface-level oxidation, tarnish, slow corrosion — not combustion.
    // Anything that should cascade into a fireball (hydrogen, carbon) needs
    // to be listed as bespoke with a tuned heat release.
    let mut delta_temp: i16 = match inferred {
        // Hydrogen combustion — massive, drives cascading ignition.
        InferredProduct::Bespoke(Element::Water)  => 1800,
        // Carbon combustion — hot but less dramatic.
        InferredProduct::Bespoke(Element::CO2)  => 900,
        // Alkali in water — energetic but not vaporizing.
        InferredProduct::Bespoke(Element::H)      => 400,
        // Ionic salt — exothermic, warms things up but salt stays solid.
        InferredProduct::Bespoke(Element::Salt)   => 150,
        // Rust formation — essentially cold, slow oxidation.
        InferredProduct::Bespoke(Element::Rust)   => 20,
        // Derived / unknown compound — warm tarnish only. Never enough to
        // ignite adjacent flammables via thermal cascade; if you want a
        // burning reaction, hand-tune it as bespoke.
        _ => (delta_e * 30.0).min(80.0) as i16,
    };
    // Hyper-reactive alkali (Cs EN 0.79) reactions aren't tarnish — they're
    // full combustion. Real Cs + O₂ releases enough heat to self-ignite
    // neighbors; the derived-oxide 80°C cap would make the whole pile read
    // as quiet evaporation instead of the visibly-burning spray of oxide
    // and fire you see in lab footage.
    if donor_e < 0.85 && matches!(inferred, InferredProduct::Derived(_)) {
        // 450°C: below the 500°C mp floor for derived oxides, so Cs₂O
        // stays solid instead of phase-liquefying into a weird puddle
        // while it waits to cool. The actual drama comes from the burn
        // cascade (ignite_above 50, burn_temp 1400) and the Fire-spawn
        // rule below; the oxide product doesn't need to be combustion-hot
        // by itself.
        delta_temp = 450;
    }
    // Alkali + water gradient. Real-world heat release for the metal-
    // displaces-H-from-water reaction is enough to flash-vaporize
    // surrounding water and detonate, with no combustion of the released
    // H₂ required. The default Bespoke(H) heat (400°C) can't model that —
    // it relied on downstream H+O ignition, which fails in 0% O₂. Tier
    // the heat by donor reactivity so each alkali reads correctly:
    //   Cs (0.79), Rb: explosive even in vacuum
    //   K  (0.82): violent, crosses shockwave threshold
    //   Na (0.93): vigorous fizzing, occasional pop
    //   Li (0.98), Ca, Mg: warm bubbling
    if matches!(inferred, InferredProduct::Bespoke(Element::H)) {
        if donor_e < 0.85 {
            delta_temp = 2800;
            rate = rate.max(0.85);
        } else if donor_e < 1.0 {
            delta_temp = 1500;
            rate = rate.max(0.50);
        } else if donor_e < 1.4 {
            delta_temp = 600;
        }
    }
    // Violent tier heat scaling — same predicate as the rate boost above.
    // Scale heat to the EN gap so the released energy crosses the 1200°C
    // chemistry-shockwave threshold (lib.rs ~line 4657) and detonates.
    // The shockwave's pressure-shove then displaces the just-formed
    // product cells outward, exposing fresh reactant surface and driving
    // a self-propagating cascade until reactants deplete. Without this
    // mechanic, the freshly-formed product coats the reactant and
    // throttles the reaction to surface-only single-pass conversion.
    //   Cs+F: ~3210°C, F+Na: ~3070°C, F+K: ~3180°C, O+Cs: ~2650°C.
    if delta_e >= 2.8 && donor_e < 1.0 && matches!(inferred, InferredProduct::Derived(_)) {
        delta_temp = (delta_e * 1000.0) as i16;
    }

    // For reactive-metal + water, acceptor cell becomes Steam (water ripped
    // apart), donor cell becomes H gas. For all other pairs, both cells
    // become the product (bespoke or derived).
    let metal_in_water = matches!(acceptor_el,
            Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H));
    let products = if metal_in_water {
        [ProductSpec::bespoke(Element::H), ProductSpec::bespoke(Element::Steam)]
    } else {
        let s = ProductSpec::from_inferred(inferred);
        [s, s]
    };

    Some(ReactionOutcome { products, delta_temp, rate })
}

// ============================================================================
// DERIVED COMPOUND REGISTRY
//
// Runtime table of reaction products we didn't hand-code. When the emergent
// engine produces a pairing we don't recognize (say Mg + Cl, or Au + F), we
// derive a DerivedCompound on the fly from the constituent atoms, register
// it, and hand back an index. Further reactions producing the same formula
// find the existing entry instead of creating a duplicate.
//
// Up to 256 distinct compounds can live in the registry (derived_id: u8).
// That's plenty for this scope — most worthwhile pairs among 20 atoms.
// ============================================================================

#[derive(Clone)]
struct DerivedCompound {
    // "AuF", "Fe₂O₃", "MgCl₂". Used both as a registry key and for display.
    formula: String,
    // Atomic composition in stoichiometric counts.
    constituents: Vec<(Element, u8)>,
    // Derived physical properties, computed once at creation.
    physics: PhysicsProfile,
    // Averaged RGB from constituent atoms, weighted by stoichiometry.
    color: (u8, u8, u8),
    // Melting / boiling derived from the dominant atom; used by the
    // generic phase transition system.
    melting_point: i16,
    boiling_point: i16,
    // If Some(T), the compound breaks apart at temperature T rather than
    // smoothly melting into a liquid of itself. Used for metal oxides and
    // similar compounds where a "molten Cu₂O" or "molten Fe₂O₃" isn't a
    // physically meaningful state — heat releases the O₂ and leaves the
    // metal behind. Set to None for salts and other compounds that melt
    // cleanly (e.g. NaCl → molten salt).
    decomposes_above: Option<i16>,
}

thread_local! {
    static DERIVED_COMPOUNDS: std::cell::RefCell<Vec<DerivedCompound>> =
        std::cell::RefCell::new(Vec::new());
}

// Builds a formula string from constituent atoms with subscripts. Uses real
// subscript glyphs where possible.
fn format_formula(atoms: &[(Element, u8)]) -> String {
    let mut out = String::new();
    for &(el, count) in atoms {
        let symbol = element_symbol(el);
        out.push_str(symbol);
        if count > 1 {
            // Unicode subscript digits U+2080..=U+2089.
            for d in count.to_string().chars() {
                let n = d as u32 - '0' as u32;
                if let Some(c) = char::from_u32(0x2080 + n) {
                    out.push(c);
                }
            }
        }
    }
    out
}

// Short symbol for an Element. Atoms use their real symbol; everything else
// uses an abbreviated name. Compounds reuse their symbol for now.
fn element_symbol(el: Element) -> &'static str {
    if let Some(a) = atom_profile_for(el) {
        return a.symbol;
    }
    match el {
        Element::Water => "H₂O",
        Element::Rust  => "Fe₂O₃",
        Element::Salt  => "NaCl",
        Element::Steam => "H₂O",
        Element::Ice   => "H₂O",
        Element::Oil   => "CH",
        _              => "?",
    }
}

// Look up or build a derived compound from a pair of atoms. Returns the
// registry index (u8) suitable for storing in Cell.derived_id. Returns
// None if the registry is full (256 entries).
// Metal-atom predicate: any implemented atom whose category is a metal.
// Uses the periodic-table category rather than an electronegativity
// threshold so noble metals like Au/Pt (E > 2.0) are still classified as
// metals and can alloy. Reactivity-with-acid is a separate, tighter
// check handled locally where it matters.
fn is_atomic_metal(el: Element) -> bool {
    atom_profile_for(el).map_or(false, |a| {
        if !a.implemented { return false; }
        matches!(
            a.category,
            AtomCategory::AlkaliMetal
            | AtomCategory::AlkalineEarth
            | AtomCategory::TransitionMetal
            | AtomCategory::PostTransition
            | AtomCategory::Lanthanide
            | AtomCategory::Actinide,
        )
    })
}

// Alloy registry. Unlike derive_or_lookup (which constructs ionic
// compounds from donor/acceptor valence math), alloys are metallic
// solutions — no electron transfer, just two metals coexisting in a
// single phase. Stoichiometry is fixed at 1:1 for simplicity; real
// ratios are implicit in how much of each feedstock the user melts
// together. Properties are blended from the parents, with a mild eutectic
// discount on melting point so alloys stay molten at slightly lower temps
// than either pure metal.
fn alloy_or_lookup(a: Element, b: Element) -> Option<u8> {
    let (ap, bp) = (atom_profile_for(a)?, atom_profile_for(b)?);
    // Canonicalize by atomic number so CuFe and FeCu resolve to the same
    // registry entry regardless of input order.
    let (a, b, ap, bp) = if ap.number <= bp.number {
        (a, b, ap, bp)
    } else {
        (b, a, bp, ap)
    };
    let constituents: Vec<(Element, u8)> = vec![(a, 1), (b, 1)];
    let formula = format_formula(&constituents);
    DERIVED_COMPOUNDS.with(|r| {
        let mut reg = r.borrow_mut();
        if let Some(idx) = reg.iter().position(|c| c.formula == formula) {
            return Some(idx as u8);
        }
        if reg.len() >= 256 { return None; }
        // Color: midpoint of the two parent colors.
        let ca = a.base_color();
        let cb = b.base_color();
        let color = (
            ((ca.0 as u16 + cb.0 as u16) / 2) as u8,
            ((ca.1 as u16 + cb.1 as u16) / 2) as u8,
            ((ca.2 as u16 + cb.2 as u16) / 2) as u8,
        );
        // Density / molar mass from atom data.
        let phys_a = a.physics();
        let phys_b = b.physics();
        let density = ((phys_a.density as i32 + phys_b.density as i32) / 2)
            .clamp(1, 200) as i16;
        let molar_mass = (ap.atomic_mass + bp.atomic_mass) / 2.0;
        // Phase points: slight eutectic on melting point (0.9 × average),
        // boiling point at the straight average. Not modeling true
        // eutectics since those vary wildly per system.
        let avg_mp = (ap.melting_point as i32 + bp.melting_point as i32) / 2;
        let avg_bp = (ap.boiling_point as i32 + bp.boiling_point as i32) / 2;
        // Amalgams (Hg-containing alloys): melt behavior is Hg-dominated,
        // not arithmetic-mean of the constituents. Real Au amalgam paste
        // is liquid/doughy well below the average of -39°C and 1064°C
        // would predict (~512°C, plainly wrong). For sandbox visuals we
        // let any Hg amalgam keep Hg's own mp so it stays liquid at
        // room temperature, matching the "Hg eats Au into a liquid"
        // demo expectation.
        let hg_mp = -39i16;
        let melting_point = if a == Element::Hg || b == Element::Hg {
            hg_mp
        } else {
            (avg_mp * 9 / 10) as i16
        };
        let boiling_point = avg_bp as i16;
        // Hg amalgams flow like Hg (visc 100), not like a thick molten
        // metal (200). Other alloys use 0, which falls through to the
        // PHASE_LIQUID default of 200 in cell_physics.
        let viscosity: u16 = if a == Element::Hg || b == Element::Hg { 100 } else { 0 };
        let physics = PhysicsProfile {
            density,
            // Alloy is a metal (Gravel-kind) when solid; the phase system
            // forces Liquid/Gas when it's molten or vaporized.
            kind: Kind::Gravel,
            viscosity,
            molar_mass,
        };
        reg.push(DerivedCompound {
            formula,
            constituents,
            physics,
            color,
            melting_point,
            boiling_point,
            decomposes_above: None,
        });
        Some((reg.len() - 1) as u8)
    })
}

/// Pre-register a derived compound so its `derived_id` is known
/// before any reaction fires. GPU chemistry shaders need stable
/// derived_ids at compile time; calling this at startup pins the id
/// for `(donor, acceptor)`.
pub fn register_compound(donor: Element, acceptor: Element) -> Option<u8> {
    derive_or_lookup(donor, acceptor)
}

fn derive_or_lookup(donor: Element, acceptor: Element) -> Option<u8> {
    let (da, aa) = match (atom_profile_for(donor), atom_profile_for(acceptor)) {
        (Some(d), Some(a)) => (d, a),
        _ => return None, // only atom-atom combinations get derived for now
    };
    // Valence-driven stoichiometry: donor gives `valence_e⁻`; acceptor
    // needs `8 - valence_e⁻` to complete its shell. Reduce by GCD so
    // we get the simplest integer ratio.
    let gives = da.valence_electrons.max(1);
    let needs = 8u8.saturating_sub(aa.valence_electrons).max(1);
    let g = gcd_u8(gives, needs);
    let donor_count = (needs / g).max(1);
    let acceptor_count = (gives / g).max(1);
    let constituents: Vec<(Element, u8)> = vec![
        (donor, donor_count),
        (acceptor, acceptor_count),
    ];
    let formula = format_formula(&constituents);

    DERIVED_COMPOUNDS.with(|r| {
        let mut reg = r.borrow_mut();
        if let Some(idx) = reg.iter().position(|c| c.formula == formula) {
            return Some(idx as u8);
        }
        if reg.len() >= 256 { return None; }
        // Derive physical properties from the atoms.
        let donor_color = donor.base_color();
        let acceptor_color = acceptor.base_color();
        let total = donor_count as u16 + acceptor_count as u16;
        let mix = |a: u8, wa: u8, b: u8, wb: u8| -> u8 {
            (((a as u16 * wa as u16) + (b as u16 * wb as u16)) / total as u16) as u8
        };
        // Weight the dominant (more numerous) atom's color. Add a brown-ish
        // tint so derived compounds read as "weathered" / "oxidized" and
        // don't get mistaken for pure atoms.
        let r_mix = mix(donor_color.0, donor_count, acceptor_color.0, acceptor_count);
        let g_mix = mix(donor_color.1, donor_count, acceptor_color.1, acceptor_count);
        let b_mix = mix(donor_color.2, donor_count, acceptor_color.2, acceptor_count);
        let color = (
            ((r_mix as u16 * 3 + 150) / 4) as u8,
            ((g_mix as u16 * 3 + 120) / 4) as u8,
            ((b_mix as u16 * 3 + 100) / 4) as u8,
        );
        // Kind heuristic: consider both atoms' STP states.
        //   * Gas + Gas → Gas. Covers HCl, H₂S, NO, etc. — the compound
        //     would be a gas at room temp, not a powder pile.
        //   * Solid donor → Gravel. Metal oxides/halides/fluorides coat
        //     the parent metal and don't flake — Cu₂O patina, Au-F, Al₂O₃.
        //   * Liquid donor → Liquid.
        //   * Gas donor with solid/liquid acceptor → Powder. Rare edge
        //     case (e.g. H + S), defaults to a soft powder.
        let kind = match (da.stp_state, aa.stp_state) {
            (AtomState::Gas, AtomState::Gas) => Kind::Gas,
            (AtomState::Solid, _)            => Kind::Gravel,
            (AtomState::Liquid, _)           => Kind::Liquid,
            (AtomState::Gas, _)              => Kind::Powder,
        };
        // Density averaged from donor + acceptor native density. Gas-kind
        // compounds get a small negative value so their liquid form
        // (PHASE_LIQUID, density 1) properly displaces the gas under
        // can_enter's density check — without this, liquid HCl gets
        // stuck on top of gas HCl since both read density 1.
        let density = if kind == Kind::Gas {
            -1
        } else {
            let da_d = donor.physics().density.max(0) as i32;
            let aa_d = acceptor.physics().density.max(0) as i32;
            let total_d = da_d * donor_count as i32 + aa_d * acceptor_count as i32;
            (total_d / total as i32).clamp(1, 200) as i16
        };
        // Molecular weight from stoichiometry — drives gas-phase buoyancy.
        let molar_mass = da.atomic_mass * donor_count as f32
            + aa.atomic_mass * acceptor_count as f32;
        let physics = PhysicsProfile {
            density,
            kind,
            viscosity: 0,
            molar_mass,
        };
        // Ionic/metal-nonmetal compounds have much higher mp/bp than either
        // constituent (strong bonds hold the lattice together). Averaging
        // the atoms gives nonsense for something like Na₂O (metal + gas =
        // garbage bp of 350 when the real value is 1950). Floor the derived
        // mp at 500°C and bp at 1500°C so compounds behave like refractory
        // salts/oxides unless both constituents are already high-melting.
        //
        // Exception: covalent gas+gas compounds (HCl, H₂S, NO, etc.)
        // legitimately stay in the gas phase at room temperature. The
        // floor would wrongly keep HCl liquid/solid at 1500°C. Skip it
        // when both atoms are gaseous at STP.
        let avg_mp = (da.melting_point as i32 + aa.melting_point as i32) / 2;
        let avg_bp = (da.boiling_point as i32 + aa.boiling_point as i32) / 2;
        let gas_gas = da.stp_state == AtomState::Gas
            && aa.stp_state == AtomState::Gas;
        // Refractory oxides (Al₂O₃ ~2072°C, MgO ~2852°C) shouldn't be
        // molten at typical reaction temperatures. The averaging
        // heuristic underestimates them severely.
        let oxide_mp_floor: i32 = if acceptor == Element::O
            && matches!(donor, Element::Al | Element::Mg)
        {
            2000
        } else {
            500
        };
        let (melting_point, boiling_point) = if gas_gas {
            (avg_mp as i16, avg_bp as i16)
        } else {
            (avg_mp.max(oxide_mp_floor) as i16, avg_bp.max(1500) as i16)
        };
        // Oxides break apart rather than melting to a liquid. Cu₂O, Fe-type
        // oxides, transition-metal oxides generally decompose on heating and
        // release their oxygen. A molten Cu₂O isn't a thing — you get molten
        // Cu + O₂ gas. Same rule for any compound where the acceptor is
        // elemental oxygen. Other compounds (halides, salts) fall through to
        // normal melt/boil behavior. Gas-gas oxides (NO, etc.) have very
        // low "melting points" from the averaging, so decomposition there
        // would trigger at cryogenic temps — they're stable gases, not
        // oxides that break down.
        // Decomposition: oxides of transition/post-transition metals do
        // break apart on heating (Cu₂O → Cu + O₂, Fe₂O₃ → Fe + O₂). But
        // alkali and alkaline-earth metal oxides are very stable — real
        // Cs₂O doesn't reduce back to Cs + O at combustion temperatures.
        // Letting it decompose creates a feedback loop (oxidize → decompose
        // → oxidize → …) that runs the sim into the ground in seconds.
        let donor_is_reactive_metal = atom_profile_for(donor)
            .map(|a| matches!(a.category,
                AtomCategory::AlkaliMetal | AtomCategory::AlkalineEarth))
            .unwrap_or(false);
        // Refractory metal oxides (Al₂O₃, MgO) are among the most
        // thermodynamically stable oxides — they don't decompose at any
        // temperature reachable in the sim. That's the reason thermite
        // works: Al pulls O off Fe₂O₃ specifically because Al₂O₃ is a
        // much deeper energy well than Fe₂O₃. MgO is similarly stable.
        // Without these exceptions, the auto-derived 600°C decomp
        // threshold makes the products instantly dissociate back to
        // their constituent metals, killing the cascade.
        let donor_makes_stable_oxide = donor_is_reactive_metal
            || matches!(donor, Element::Al | Element::Mg);
        let decomposes_above = if acceptor == Element::O && !gas_gas
            && !donor_makes_stable_oxide
        {
            // Floor the threshold: averaged MP of a metal + O often comes
            // out unphysically low (Cs: 28, O: -218 averages negative).
            // Real oxide decomposition happens at least in the high
            // hundreds of °C — 600 is a conservative floor.
            Some(melting_point.max(600))
        } else {
            None
        };
        reg.push(DerivedCompound {
            formula,
            constituents,
            physics,
            color,
            melting_point,
            boiling_point,
            decomposes_above,
        });
        Some((reg.len() - 1) as u8)
    })
}

#[inline]
fn gcd_u8(mut a: u8, mut b: u8) -> u8 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

// Accessor helpers — return properties for a derived cell from the registry,
// falling back to something safe if the index is invalid.
fn derived_physics_of(idx: u8) -> PhysicsProfile {
    DERIVED_COMPOUNDS.with(|r| {
        r.borrow().get(idx as usize)
            .map(|c| c.physics)
            .unwrap_or(PhysicsProfile {
                density: 20, kind: Kind::Powder, viscosity: 0, molar_mass: 0.0
            })
    })
}

fn derived_color_of(idx: u8) -> (u8, u8, u8) {
    DERIVED_COMPOUNDS.with(|r| {
        r.borrow().get(idx as usize).map(|c| c.color).unwrap_or((160, 130, 140))
    })
}

fn derived_formula_of(idx: u8) -> String {
    DERIVED_COMPOUNDS.with(|r| {
        r.borrow().get(idx as usize)
            .map(|c| c.formula.clone())
            .unwrap_or_else(|| "?".to_string())
    })
}

// First metal constituent of a derived compound. Used by electrolysis to
// figure out which metal ion plates out of a dissolved salt (CuCl → Cu,
// FeCl → Fe, NaCl → Na). Returns None for compounds that have no metal
// in their composition.
fn compound_metal_component(idx: u8) -> Option<Element> {
    DERIVED_COMPOUNDS.with(|r| {
        let b = r.borrow();
        let c = b.get(idx as usize)?;
        for &(el, _) in &c.constituents {
            if is_atomic_metal(el) { return Some(el); }
        }
        None
    })
}

// Ionic metal-halide / metal-halogen compounds dissolve in water. Predicate
// used by dissolve() to decide whether a derived cell is a candidate
// solute. Atom categories are what make a compound "a salt" — if the
// constituents include a metal and a halogen, it's one.
fn derived_is_soluble_salt(idx: u8) -> bool {
    DERIVED_COMPOUNDS.with(|r| {
        let b = r.borrow();
        let Some(c) = b.get(idx as usize) else { return false; };
        let mut has_metal = false;
        let mut has_halogen = false;
        for &(el, _) in &c.constituents {
            if is_atomic_metal(el) { has_metal = true; }
            if let Some(a) = atom_profile_for(el) {
                if matches!(a.category, AtomCategory::Halogen) { has_halogen = true; }
            }
        }
        has_metal && has_halogen
    })
}

// Map an atom (by index into ATOMS) to its corresponding paintable Element.
// Returns None for placeholder atoms that don't yet have an Element variant.
fn atom_to_element(atom_idx: usize) -> Option<Element> {
    if !ATOMS[atom_idx].implemented { return None; }
    Some(match ATOMS[atom_idx].number {
        1  => Element::H,
        2  => Element::He,
        6  => Element::C,
        7  => Element::N,
        8  => Element::O,
        9  => Element::F,
        10 => Element::Ne,
        11 => Element::Na,
        12 => Element::Mg,
        13 => Element::Al,
        14 => Element::Si,
        15 => Element::P,
        16 => Element::S,
        17 => Element::Cl,
        18 => Element::Ar,
        19 => Element::K,
        20 => Element::Ca,
        26 => Element::Fe,
        28 => Element::Ni,
        29 => Element::Cu,
        30 => Element::Zn,
        47 => Element::Ag,
        55 => Element::Cs,
        79 => Element::Au,
        80 => Element::Hg,
        82 => Element::Pb,
        88 => Element::Ra,
         5 => Element::B,
        92 => Element::U,
        _  => return None,
    })
}

// Reverse of atom_to_element: for a paintable-atom Element variant, return
// the AtomProfile it corresponds to. None for compounds like Water/Sand
// (those have their own bespoke phase transitions in ThermalProfile).
fn atom_profile_for(el: Element) -> Option<&'static AtomProfile> {
    let number: u8 = match el {
        Element::H  => 1,   Element::He => 2,   Element::C  => 6,
        Element::N  => 7,   Element::O  => 8,   Element::F  => 9,
        Element::Ne => 10,
        Element::Na => 11,  Element::Mg => 12,  Element::Al => 13,
        Element::Si => 14,  Element::P  => 15,  Element::S  => 16,
        Element::Cl => 17,  Element::K  => 19,  Element::Ca => 20,
        Element::Fe => 26,  Element::Cu => 29,  Element::Au => 79,
        Element::Hg => 80,  Element::U  => 92,
        Element::B  => 5,
        Element::Ni => 28,  Element::Zn => 30,  Element::Ag => 47,
        Element::Cs => 55,  Element::Pb => 82,  Element::Ra => 88,
        _ => return None,
    };
    // ATOMS is indexed by atomic number − 1 (ordered H=1 at index 0 through
    // U=92 at index 91). Direct indexing saves a ~46-op linear scan every
    // call — and this function is called many times per cell per frame.
    ATOMS.get(number as usize - 1)
}

// Phase-transition reference points for any element: returns (melting_point,
// boiling_point, native STP state). Only returns Some for elements whose
// phase transitions should be driven by the *generic* system (phase flag
// on the Cell). Elements with bespoke transitions that swap to a different
// Element entirely (Water↔Ice/Steam, Sand→MoltenGlass→Glass, Lava↔Obsidian,
// Rust→Fe, Stone→Lava) are deliberately excluded — their bespoke rules run
// first in thermal() and do the right thing without help. Double-handling
// would produce nonsense like "liquid steam" (generic flag on a cell that's
// about to bespoke-convert back to Water).
// Alloy classification. Returns the constituent metals if the cell is a
// derived compound whose constituents are all atomic metals (satisfies
// `is_atomic_metal`). Used by the acid-leaching pass to decide whether
// a compound can be selectively stripped by acid.
fn alloy_constituents(cell: Cell) -> Option<Vec<Element>> {
    if cell.el != Element::Derived { return None; }
    DERIVED_COMPOUNDS.with(|r| {
        let reg = r.borrow();
        let cd = reg.get(cell.derived_id as usize)?;
        let elements: Vec<Element> =
            cd.constituents.iter().map(|(e, _)| *e).collect();
        for e in &elements {
            if !is_atomic_metal(*e) { return None; }
        }
        Some(elements)
    })
}

// Basic-oxide classification. A metal-oxide compound (M + O) acts as a
// base in Brønsted terms: when it meets an acid, the metal captures the
// acid's acceptor and the released O pairs with H to form water.
// Returns (metal element, basicity) where basicity = 2.0 − metal_e
// (lower-E metals are more strongly basic — same cutoff as acid
// displacement's metal reactivity).
fn basic_oxide_signature(cell: Cell) -> Option<(Element, f32)> {
    if cell.el != Element::Derived { return None; }
    DERIVED_COMPOUNDS.with(|r| {
        let reg = r.borrow();
        let cd = reg.get(cell.derived_id as usize)?;
        if cd.constituents.len() < 2 { return None; }
        let (d_el, _) = cd.constituents[0];
        let (a_el, _) = cd.constituents[1];
        if a_el != Element::O { return None; }
        let d_e = atom_profile_for(d_el)?.electronegativity;
        if d_e <= 0.0 || d_e >= 2.0 { return None; }
        Some((d_el, 2.0 - d_e))
    })
}

// Acid classification. A derived compound is proton-donor (acidic) when
// its donor is H AND its acceptor is aggressive enough to release H as
// H⁺ when displaced by a metal. Returns (acceptor element, acid strength)
// where strength = acceptor_e − H_e (the bond-polarity proxy, larger =
// stronger acid). Restricted to halogens for now — they're the clean
// single-atom acid acceptors (HF, HCl). Polyatomic acid bases (NO₃, SO₄)
// would need their own compound handling later.
fn acid_signature(cell: Cell) -> Option<(Element, f32)> {
    if cell.el != Element::Derived { return None; }
    DERIVED_COMPOUNDS.with(|r| {
        let reg = r.borrow();
        let cd = reg.get(cell.derived_id as usize)?;
        if cd.constituents.len() < 2 { return None; }
        let (d_el, _) = cd.constituents[0];
        if d_el != Element::H { return None; }
        let (a_el, _) = cd.constituents[1];
        if !matches!(a_el, Element::F | Element::Cl) { return None; }
        let a_e = atom_profile_for(a_el)?.electronegativity;
        let strength = (a_e - 2.20).max(0.0);
        Some((a_el, strength))
    })
}

// Unified decomposition lookup. If a cell's element should break apart on
// heating rather than melting cleanly, returns (threshold °C, donor element
// left behind, byproduct gas emitted). Used by the thermal pass to handle
// both bespoke compounds (Rust) and runtime-derived oxides uniformly.
fn decomposition_of(cell: Cell) -> Option<(i16, Element, Element)> {
    match cell.el {
        // Bumped from 1538°C to 3500°C. Real Fe₂O₃ decomposition
        // requires very high temps under low O partial pressure;
        // 1538°C was tied to Fe's melt point as a stand-in, not a
        // chemistry fact. The high threshold prevents thermal-pass
        // diffusion (from the 1900°C thermite products) from slowly
        // cooking unreacted Rust into free Fe + O, which then alloys
        // with leftover Al as AlFe. With 3500°C, decomp only fires
        // under direct extreme user heat, leaving the thermite
        // cascade as the dominant Rust-consumer.
        Element::Rust => Some((3500, Element::Fe, Element::O)),
        Element::Derived => DERIVED_COMPOUNDS.with(|r| {
            let reg = r.borrow();
            let cd = reg.get(cell.derived_id as usize)?;
            let thr = cd.decomposes_above?;
            if cd.constituents.len() < 2 { return None; }
            Some((thr, cd.constituents[0].0, cd.constituents[1].0))
        }),
        _ => None,
    }
}

fn element_phase_points(cell: Cell) -> Option<(i16, i16, AtomState)> {
    if let Some(a) = atom_profile_for(cell.el) {
        return Some((a.melting_point, a.boiling_point, a.stp_state));
    }
    match cell.el {
        // Bespoke-transition compounds are *not* listed here.
        Element::Salt  => Some((801, 1465, AtomState::Solid)),
        Element::Derived => {
            DERIVED_COMPOUNDS.with(|r| {
                r.borrow().get(cell.derived_id as usize).map(|c| {
                    // stp_state tracks the compound's natural phase so
                    // phase transitions can correctly flag PHASE_SOLID /
                    // PHASE_LIQUID / PHASE_GAS when the temp moves off
                    // its native state. Without this, gas-kind compounds
                    // (HCl, NO) would read as solid-native and the phase
                    // logic would never force them into PHASE_SOLID at
                    // cryogenic temperatures — they'd render as gas even
                    // below their freezing point.
                    let stp_state = match c.physics.kind {
                        Kind::Gas | Kind::Fire => AtomState::Gas,
                        Kind::Liquid => AtomState::Liquid,
                        _ => AtomState::Solid,
                    };
                    (c.melting_point, c.boiling_point, stp_state)
                })
            })
        }
        _ => None,
    }
}

// Phase-aware physics lookup. Returns a profile that reflects the cell's
// *current phase*, not just its declared element state. PHASE_NATIVE always
// uses the hand-tuned profile (it's the canonical representation for the
// atom's STP state). The forced phases override when temperature has pushed
// the cell off its STP state. Non-atom cells don't use forced phases —
// compounds rely on their bespoke transitions in ThermalProfile.
// Real flame-test colors — when these elements vaporize into a flame,
// their excited atoms emit characteristic wavelengths. Used by the
// color_fires() pass to tint adjacent Fire cells. Salts work via their
// metal component (NaCl burns yellow because of the Na, not Cl).
fn flame_color(el: Element) -> Option<(u8, u8, u8)> {
    match el {
        Element::Cu   => Some((80, 255, 110)),  // emerald green
        Element::Na   => Some((255, 220, 80)),  // bright yellow
        Element::K    => Some((220, 130, 230)), // lilac/violet
        Element::Ca   => Some((255, 140, 60)),  // orange-red (brick)
        Element::Mg   => Some((255, 255, 255)), // brilliant white
        Element::B    => Some((130, 220, 100)), // bright green
        Element::Salt => Some((255, 220, 80)),  // NaCl → Na yellow
        _ => None,
    }
}

fn cell_physics(c: Cell) -> PhysicsProfile {
    // Base profile: atoms/compounds use their static PhysicsProfile; derived
    // compounds pull from the registry. In either case, the profile below
    // gets phase-transformed if the cell is in a forced (non-native) phase.
    let base = if c.el == Element::Derived {
        derived_physics_of(c.derived_id)
    } else {
        *c.el.physics()
    };
    let atom_mass = atom_profile_for(c.el).map(|a| a.atomic_mass);
    match c.phase() {
        PHASE_SOLID => PhysicsProfile {
            kind: Kind::Gravel,
            density: base.density.abs().max(1),
            viscosity: 0,
            molar_mass: 0.0,
        },
        PHASE_LIQUID => PhysicsProfile {
            kind: Kind::Liquid,
            // Molten form is ~90% of solid density; viscosity defaults to
            // 200 (molten-metal feel) but honors a non-zero base viscosity
            // so things like Hg amalgams can opt into Hg-like flow.
            density: (base.density.abs().max(1) * 9 / 10).max(1),
            viscosity: if base.viscosity > 0 { base.viscosity } else { 200 },
            molar_mass: atom_mass.unwrap_or(base.molar_mass),
        },
        PHASE_GAS => {
            // Boiled-off atom/compound: mass comes from atomic/molecular
            // weight (drives buoyancy vs air). Density flipped negative so
            // it rises or sinks depending on its mass vs ambient.
            let mass = atom_mass.unwrap_or(base.molar_mass);
            PhysicsProfile {
                kind: Kind::Gas,
                density: -(base.density.abs().max(1) / 8).max(1),
                viscosity: 0,
                molar_mass: mass,
            }
        }
        _ => base, // PHASE_NATIVE
    }
}

// ============================================================================
// GPU SIMULATION RESOURCES
//
// Holds the textures, render targets, and materials needed to dispatch
// physics passes (pressure today, thermal next) on the GPU. Created once
// from `run_game` after the macroquad context is alive, then passed into
// `World::step_gpu` each frame.
//
// Encoding convention for signed i16 sim fields packed into RGBA8 textures:
//   R = low byte, G = high byte, B/A unused.
//   Values are bit-cast u16↔i16 (two's-complement preserved).
// ============================================================================
pub struct GpuPressureCtx {
    /// Per-frame scratch image — CPU writes packed pressure here, then
    /// uploads to `rt_a.texture` to seed iteration 0.
    input_image: Image,
    /// Per-frame scratch image — CPU writes packed permeability bytes
    /// here from `pressure_perm_cache` and uploads each frame (cells
    /// transmute, so permeability is not constant).
    perm_image: Image,
    perm_tex: Texture2D,
    /// Ping-pong render targets. Iteration 0 reads input → writes rt_a;
    /// iteration 1 reads rt_a → writes rt_b; iteration 2 reads rt_b →
    /// writes rt_a; … For ITERS=6, the final result lands in rt_b.
    rt_a: RenderTarget,
    rt_b: RenderTarget,
    /// Diffusion fragment shader pipeline.
    material: Material,
}

impl GpuPressureCtx {
    pub fn new() -> Self {
        let input_image = Image::gen_image_color(W as u16, H as u16, BLACK);
        let perm_image = Image::gen_image_color(W as u16, H as u16, BLACK);
        let perm_tex = Texture2D::from_image(&perm_image);
        perm_tex.set_filter(FilterMode::Nearest);
        let rt_a = render_target(W as u32, H as u32);
        rt_a.texture.set_filter(FilterMode::Nearest);
        let rt_b = render_target(W as u32, H as u32);
        rt_b.texture.set_filter(FilterMode::Nearest);
        let material = match load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: PRESSURE_DIFFUSION_FRAGMENT,
            },
            MaterialParams {
                uniforms: vec![UniformDesc::new("TexelSize", UniformType::Float2)],
                textures: vec!["PermTex".to_string()],
                ..Default::default()
            },
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("FATAL: pressure diffusion shader compile failed: {:?}", e);
                std::process::exit(1);
            }
        };
        GpuPressureCtx {
            input_image,
            perm_image,
            perm_tex,
            rt_a,
            rt_b,
            material,
        }
    }
}

fn atom_category_color(cat: AtomCategory) -> (u8, u8, u8) {
    match cat {
        Hydrogen        => (230, 220, 160),
        AlkaliMetal     => (230, 140, 130),
        AlkalineEarth   => (230, 200, 130),
        TransitionMetal => (220, 180, 190),
        PostTransition  => (180, 200, 220),
        Metalloid       => (180, 220, 180),
        Nonmetal        => (200, 230, 200),
        Halogen         => (230, 230, 140),
        NobleGas        => (180, 220, 240),
        Lanthanide      => (220, 180, 230),
        Actinide        => (230, 160, 200),
    }
}

impl Element {
    // System profiles — all physical behavior goes through these.
    #[inline] fn physics(self)  -> &'static PhysicsProfile  { &PHYSICS[self as usize] }
    #[inline] fn thermal(self)  -> &'static ThermalProfile  { &THERMAL[self as usize] }
    #[inline] fn moisture(self) -> &'static MoistureProfile { &MOISTURE[self as usize] }
    #[inline] fn pressure_p(self) -> &'static PressureProfile { &PRESSURE[self as usize] }
    #[inline] fn electrical(self) -> &'static ElectricalProfile { &ELECTRICAL[self as usize] }

    fn base_color(self) -> (u8, u8, u8) {
        match self {
            Element::Empty    => (10, 10, 14),
            Element::Sand     => (204, 178, 108),
            Element::Water    => (40, 90, 200),
            Element::Stone    => (110, 110, 115),
            Element::Wood     => (110, 70, 40),
            Element::Fire     => (240, 120, 40),
            Element::CO2    => (180, 175, 170),
            Element::Steam    => (200, 210, 220),
            Element::Lava     => (220, 80, 20),
            Element::Obsidian => (30, 20, 35),
            Element::Seed  => (165, 130, 60),
            Element::Mud      => (70, 48, 30),
            Element::Leaves   => (60, 140, 55),
            Element::Oil      => (52, 38, 25),
            Element::Ice      => (175, 215, 240),
            Element::MoltenGlass => (230, 180, 110),
            Element::Glass    => (200, 230, 235),
            Element::Charcoal => (38, 32, 30),
            // Atoms — rough real-world appearances.
            Element::H  => (220, 240, 255),
            Element::He => (240, 230, 180),
            Element::C  => (30, 30, 32),
            Element::N  => (180, 200, 240),
            Element::O  => (180, 220, 220),
            Element::Ne => (240, 140, 120),
            Element::Na => (190, 190, 200),
            Element::Mg => (220, 220, 220),
            Element::Al => (200, 200, 210),
            Element::Si => (90, 95, 100),
            Element::P  => (230, 210, 140),
            Element::S  => (240, 220, 60),
            Element::Cl => (180, 220, 100),
            Element::K  => (180, 170, 195),
            Element::Ca => (210, 210, 200),
            Element::Fe => (130, 120, 115),
            Element::Cu => (184, 115, 51),
            Element::Au => (255, 215, 0),
            Element::Hg => (195, 195, 205),
            Element::U  => (75, 80, 70),
            // Fluorine: pale yellow-green, characteristic of F₂ gas.
            Element::F  => (220, 235, 140),
            // Rust: flaky orange-brown iron oxide.
            Element::Rust => (160, 85, 45),
            // Salt: off-white crystalline.
            Element::Salt => (235, 235, 225),
            // Derived: fallback grey. cell-level color comes from the
            // registry via derived_color(cell); this branch is only hit
            // from places that look at Element alone (shouldn't happen
            // for derived cells in practice).
            Element::Derived => (160, 130, 140),
            // Gunpowder: near-black with a slight purple tint.
            Element::Gunpowder => (35, 30, 40),
            // Quartz: translucent pale grey — reads like lab glass.
            Element::Quartz    => (210, 220, 225),
            // Firebrick: warm rust-orange, matches real furnace brick.
            Element::Firebrick => (170, 85, 55),
            // Argon: very pale lavender tint when unenergized; bright
            // purple appears via the energized-cell glow logic.
            Element::Ar        => (180, 170, 200),
            // Battery: dark green with gold trim flavor — reads like a
            // 9V brick.
            // Positive terminal — warm red for the + side, matches common
            // battery indicator conventions.
            Element::BattPos   => (170, 50, 50),
            // Negative terminal — cool blue for the − side.
            Element::BattNeg   => (40, 70, 130),
            // Tier-1 metals and metalloid, loosely keyed to their real
            // visual appearance.
            Element::Zn => (150, 160, 170),
            Element::Ag => (215, 220, 225),
            Element::Ni => (185, 190, 180),
            Element::Pb => (90, 95, 110),
            Element::B  => (80, 65, 55),
            // Radium: near-white with a warm cream tint — the glow that
            // makes it famous comes later via the radioactivity pass.
            Element::Ra => (235, 230, 210),
            // Caesium: pale gold-silver in real life; photons from a Cs
            // atom cloud are mostly 852 nm (near-IR) so the visible color
            // is subtle warm-silver.
            Element::Cs => (230, 215, 150),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Element::Empty    => "Erase",
            Element::Sand     => "Sand",
            Element::Water    => "Water",
            Element::Stone    => "Stone",
            Element::Wood     => "Wood",
            Element::Fire     => "Fire",
            Element::CO2    => "CO₂",
            Element::Steam    => "Steam",
            Element::Lava     => "Lava",
            Element::Obsidian => "Obsidian",
            Element::Seed  => "Seed",
            Element::Mud      => "Mud",
            Element::Leaves   => "Leaves",
            Element::Oil      => "Oil",
            Element::Ice      => "Ice",
            Element::MoltenGlass => "Molten Glass",
            Element::Glass    => "Glass",
            Element::Charcoal => "Charcoal",
            Element::H  => "Hydrogen",
            Element::He => "Helium",
            Element::C  => "Carbon",
            Element::N  => "Nitrogen",
            Element::O  => "Oxygen",
            Element::Ne => "Neon",
            Element::Na => "Sodium",
            Element::Mg => "Magnesium",
            Element::Al => "Aluminum",
            Element::Si => "Silicon",
            Element::P  => "Phosphorus",
            Element::S  => "Sulfur",
            Element::Cl => "Chlorine",
            Element::K  => "Potassium",
            Element::Ca => "Calcium",
            Element::Fe => "Iron",
            Element::Cu => "Copper",
            Element::Au => "Gold",
            Element::Hg => "Mercury",
            Element::U  => "Uranium",
            Element::F  => "Fluorine",
            Element::Rust => "Rust",
            Element::Salt => "Salt",
            // Derived: generic fallback name. Real formula comes from the
            // registry via display_name_for(cell).
            Element::Derived => "Compound",
            Element::Gunpowder => "Gunpowder",
            Element::Quartz    => "Quartz",
            Element::Firebrick => "Firebrick",
            Element::Ar        => "Argon",
            Element::BattPos   => "Battery +",
            Element::BattNeg   => "Battery −",
            Element::Zn => "Zinc",
            Element::Ag => "Silver",
            Element::Ni => "Nickel",
            Element::Pb => "Lead",
            Element::B  => "Boron",
            Element::Ra => "Radium",
            Element::Cs => "Caesium",
        }
    }

}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub el: Element,
    // When el == Element::Derived, this indexes into the runtime compound
    // registry (DERIVED_COMPOUNDS). For any other element, the field is
    // unused. Fits in the existing u8 slots so Cell stays at 12 bytes.
    pub derived_id: u8,
    pub life: u16,
    pub seed: u8,
    pub flag: u8,
    pub temp: i16,
    pub moisture: u8,
    pub burn: u8,
    // Overpressure above atmospheric. 0 = ambient, +N = compressed, -N = rarefied.
    pub pressure: i16,
    pub solute_el: Element,
    pub solute_amt: u8,
    pub solute_derived_id: u8,
}

// Phase state, packed in Cell.flag bits 2-3. PHASE_NATIVE means "cell is in
// its declared STP state — use the hand-tuned profile as-is." The other
// three are *forced* phases for when an atom has been heated or cooled off
// its native state: a liquefied gas reads as Liquid; a frozen gas reads as
// Solid; a boiled solid reads as Gas. Gates physics lookups via cell_physics().
const PHASE_NATIVE: u8 = 0;
const PHASE_SOLID:  u8 = 1;
const PHASE_LIQUID: u8 = 2;
const PHASE_GAS:    u8 = 3;

impl Cell {
    // `flag` is a bitfield:
    //   bit 0 — FLAG_UPDATED (cleared every frame, movement marker)
    //   bit 1 — FLAG_FROZEN  (persistent, rigid-body lock)
    //   bits 2-3 — phase (0 native, 1 liquid, 2 gas)
    //   bits 4-7 — unused / reserved
    const FLAG_UPDATED: u8 = 0x01;
    const FLAG_FROZEN:  u8 = 0x02;
    const PHASE_MASK:   u8 = 0x0C;
    const PHASE_SHIFT:  u8 = 2;
    pub const EMPTY: Cell = Cell {
        el: Element::Empty, derived_id: 0, life: 0, seed: 0, flag: 0, temp: 20, moisture: 0, burn: 0,
        pressure: 0,
        solute_el: Element::Empty, solute_amt: 0, solute_derived_id: 0,
    };
    #[inline] fn is_updated(&self) -> bool { self.flag & Self::FLAG_UPDATED != 0 }
    #[inline] pub fn is_frozen(&self)  -> bool { self.flag & Self::FLAG_FROZEN  != 0 }
    // Effective electrical conductivity, accounting for dissolved ions.
    // Pure water is ~0, but brine climbs toward 0.6 as salinity increases
    // — that's what lets saltwater close a circuit for galvanic cells /
    // electrolysis. Non-water cells return their static profile unchanged.
    #[inline] fn conductivity(&self) -> f32 {
        let base = self.el.electrical().conductivity;
        if self.el == Element::Water && self.solute_amt > 0 {
            base + (self.solute_amt as f32 / 255.0) * 0.6
        } else {
            base
        }
    }
    #[inline] pub fn phase(&self) -> u8 { (self.flag & Self::PHASE_MASK) >> Self::PHASE_SHIFT }
    #[inline] fn set_phase(&mut self, p: u8) {
        self.flag = (self.flag & !Self::PHASE_MASK) | ((p & 0x03) << Self::PHASE_SHIFT);
    }
    fn new(el: Element) -> Self {
        let life: u16 = match el {
            Element::Fire  => 40 + rand::gen_range::<u16>(0, 40),
            Element::Steam => 260 + rand::gen_range::<u16>(0, 180),
            _ => 0,
        };
        Cell {
            el,
            derived_id: 0,
            life,
            seed: rand::gen_range::<u8>(0, 255),
            flag: 0,
            temp: el.thermal().initial_temp,
            moisture: el.moisture().default_moisture,
            burn: 0,
            // Phase-change births (steam from boiling water, etc.) inherit
            // their overpressure from the profile — this is what makes a
            // boiling kettle actually vent.
            pressure: el.pressure_p().formation_pressure,
            solute_el: Element::Empty,
            solute_amt: 0,
            solute_derived_id: 0,
        }
    }
}

pub const HISTORY_CAPACITY: usize = 240;   // ~4 seconds of rewind at 60 fps

pub struct World {
    pub cells: Vec<Cell>,
    pub temp_scratch: Vec<i16>,
    /// Element-presence flags, refreshed once per frame by step_inner.
    /// Lets chemistry passes that depend on a specific element early-
    /// exit before iterating ~1M cells.
    pub present_elements: [bool; ELEMENT_COUNT],
    pub pressure_scratch: Vec<i16>,
    pub pressure_perm_cache: Vec<u8>,
    pub pressure_field: Vec<i16>,
    // Thermal SoA caches: per-cell dense arrays read in the hot
    // diffusion loop. Element id ('el_cache') is used for the exposure
    // check that distinguishes "interior of a same-element pile" from
    // "boundary cell radiating to its environment". The four float caches
    // mirror the 4 thermal-profile lookups (conductivity, ambient_temp,
    // ambient_rate, heat_capacity), all of which depend solely on el.
    pub temp_field: Vec<i16>,
    pub el_cache: Vec<u8>,
    pub thermal_k_cache: Vec<f32>,
    pub thermal_amb_t_cache: Vec<f32>,
    pub thermal_amb_r_cache: Vec<f32>,
    pub thermal_hc_cache: Vec<f32>,
    pub support_scratch: Vec<bool>,
    pub support_queue: Vec<(i32, i32)>,
    pub vacuum_moved: Vec<bool>,
    pub wind_exposed: Vec<bool>,
    pub wind_queue: Vec<(i32, i32)>,
    pub energized: Vec<bool>,
    pub energized_queue: Vec<(i32, i32)>,
    pub battery_voltage: f32,
    pub galvanic_voltage: f32,
    pub active_emf: f32,
    pub cathode_mask: Vec<bool>,
    pub anode_mask: Vec<bool>,
    pub galvanic_cathode_el: Option<Element>,
    pub galvanic_anode_el: Option<Element>,
    pub u_component_size: Vec<u16>,
    pub u_burst_committed: Vec<bool>,
    pub u_component_cx: Vec<i16>,
    pub u_component_cy: Vec<i16>,
    pub u_central_blast_fired: Vec<bool>,
    pub frame: u32,
    pub ambient_offset: i16,
    pub gravity: f32,
    pub ambient_oxygen: f32,
    pub history: Vec<Vec<Cell>>,
    pub history_write: usize,
    pub history_count: usize,
    pub rewind_offset: usize,
    pub shockwaves: Vec<Shockwave>,
}

#[derive(Clone, Copy, PartialEq)]
// Eq omitted: f32 fields block derivation.
pub struct Shockwave {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub yield_p: f32,
}

impl World {
    pub fn new() -> Self {
        let mut history = Vec::with_capacity(HISTORY_CAPACITY);
        for _ in 0..HISTORY_CAPACITY {
            history.push(vec![Cell::EMPTY; W * H]);
        }
        World {
            cells: vec![Cell::EMPTY; W * H],
            temp_scratch: vec![20; W * H],
            present_elements: [false; ELEMENT_COUNT],
            pressure_scratch: vec![0; W * H],
            pressure_perm_cache: vec![0; W * H],
            pressure_field: vec![0; W * H],
            temp_field: vec![20; W * H],
            el_cache: vec![0; W * H],
            thermal_k_cache: vec![0.0; W * H],
            thermal_amb_t_cache: vec![0.0; W * H],
            thermal_amb_r_cache: vec![0.0; W * H],
            thermal_hc_cache: vec![0.0; W * H],
            support_scratch: vec![false; W * H],
            support_queue: Vec::with_capacity(512),
            vacuum_moved: vec![false; W * H],
            wind_exposed: vec![false; W * H],
            wind_queue: Vec::with_capacity(1024),
            energized: vec![false; W * H],
            energized_queue: Vec::with_capacity(1024),
            battery_voltage: 100.0,
            galvanic_voltage: 0.0,
            active_emf: 0.0,
            cathode_mask: vec![false; W * H],
            anode_mask: vec![false; W * H],
            galvanic_cathode_el: None,
            galvanic_anode_el: None,
            u_component_size: vec![0; W * H],
            u_burst_committed: vec![false; W * H],
            u_component_cx: vec![0; W * H],
            u_component_cy: vec![0; W * H],
            u_central_blast_fired: vec![false; W * H],
            frame: 0,
            ambient_offset: 0,
            gravity: 1.0,
            ambient_oxygen: 0.21,
            history,
            history_write: 0,
            history_count: 0,
            rewind_offset: 0,
            shockwaves: Vec::new(),
        }
    }

    pub fn spawn_shockwave(&mut self, cx: i32, cy: i32, yield_p: f32) {
        self.spawn_shockwave_capped(cx, cy, yield_p, 50000.0);
    }

    // Custom-cap variant for nuclear ground-zero blasts, which need
    // enough raw yield to rupture frozen walls tens of cells out. The
    // default 50000 cap only delivers ~2100 magnitude at r=20, just
    // under glass's 2500 rupture threshold — nothing breaks. A 200000
    // cap gives ~10700 at r=20 and ~2600 at r=60, shattering thin
    // walls across the immediate neighborhood.
    pub fn spawn_shockwave_capped(
        &mut self, cx: i32, cy: i32, yield_p: f32, max_pool: f32,
    ) {
        for s in &mut self.shockwaves {
            if s.radius < 3.0
                && (s.cx - cx as f32).abs() < 4.0
                && (s.cy - cy as f32).abs() < 4.0
            {
                // Pool monotonically: if this caller's cap is below the
                // wave's current yield (e.g. per-cell burst pooling into
                // a nuclear central blast), don't clobber the bigger
                // wave back down. Only increase, never decrease.
                let new_yield = (s.yield_p + yield_p).min(max_pool);
                if new_yield > s.yield_p { s.yield_p = new_yield; }
                return;
            }
        }
        self.shockwaves.push(Shockwave {
            cx: cx as f32,
            cy: cy as f32,
            radius: 0.0,
            yield_p: yield_p.min(max_pool),
        });
    }

    // Advance every active shockwave by one step. Apply effects inside the
    // annulus between old_r and new_r: rupture frozen walls, fling loose
    // matter, inject gas pressure. Retires waves once they're too weak.
    fn tick_shockwaves(&mut self) {
        const SPEED: f32 = 5.0;          // cells per frame
        const FALLOFF_R0: f32 = 6.0;     // where magnitude falls to 1/4
        const MIN_MAG: f32 = 200.0;      // retire below this
        const MAX_RADIUS: f32 = 50000.0;
        let waves = std::mem::take(&mut self.shockwaves);
        let mut survivors: Vec<Shockwave> = Vec::with_capacity(waves.len());
        for mut s in waves {
            let old_r = s.radius;
            s.radius += SPEED;
            let new_r = s.radius;
            let r_mid = (old_r + new_r) * 0.5;
            let decay = 1.0 + r_mid / FALLOFF_R0;
            let magnitude = s.yield_p / (decay * decay);
            if magnitude < MIN_MAG || new_r > MAX_RADIUS {
                continue;
            }
            // Scan ONLY the annulus (not the full r×r bounding box).
            // For each row dy, compute the dx range that lies inside the
            // ring [old_r, new_r] using the inverse circle equation:
            //   dx² ∈ [r_in² - dy², r_out² - dy²]
            // This drops the iteration count from ~r² to ~2πr × thickness,
            // a 4–10× reduction at typical detonation radii.
            let r_out = new_r.ceil() as i32 + 1;
            let cx = s.cx as i32;
            let cy = s.cy as i32;
            let r_in = old_r.max(0.0);
            let r_out_sq = new_r * new_r;
            let r_in_sq = r_in * r_in;
            for dy in -r_out..=r_out {
                let dy2 = (dy * dy) as f32;
                if dy2 > r_out_sq { continue; }
                let max_dx = ((r_out_sq - dy2).max(0.0)).sqrt().ceil() as i32;
                let min_dx_sq = (r_in_sq - dy2).max(0.0);
                let min_dx = min_dx_sq.sqrt().floor() as i32;
                // Two arcs per row when dy is inside the inner circle
                // (left arc and right arc), one arc otherwise.
                if min_dx_sq > 0.0 {
                    // Left arc: [-max_dx ..= -min_dx]
                    for dx in -max_dx..=-min_dx {
                        let d2 = (dx * dx) as f32 + dy2;
                        if d2 < r_in_sq || d2 > r_out_sq { continue; }
                        let x = cx + dx;
                        let y = cy + dy;
                        if !Self::in_bounds(x, y) { continue; }
                        self.apply_shockwave_at(x, y, dx, dy, magnitude);
                    }
                    // Right arc: [min_dx ..= max_dx]
                    for dx in min_dx..=max_dx {
                        let d2 = (dx * dx) as f32 + dy2;
                        if d2 < r_in_sq || d2 > r_out_sq { continue; }
                        let x = cx + dx;
                        let y = cy + dy;
                        if !Self::in_bounds(x, y) { continue; }
                        self.apply_shockwave_at(x, y, dx, dy, magnitude);
                    }
                } else {
                    for dx in -max_dx..=max_dx {
                        let d2 = (dx * dx) as f32 + dy2;
                        if d2 > r_out_sq { continue; }
                        let x = cx + dx;
                        let y = cy + dy;
                        if !Self::in_bounds(x, y) { continue; }
                        self.apply_shockwave_at(x, y, dx, dy, magnitude);
                    }
                }
            }
            survivors.push(s);
        }
        self.shockwaves = survivors;
    }

    fn apply_shockwave_at(&mut self, x: i32, y: i32, dx: i32, dy: i32, mag: f32) {
        let i = Self::idx(x, y);
        let cell = self.cells[i];
        let kind = cell.el.physics().kind;
        // Leading-edge direction (outward from shockwave center). Use the
        // dominant axis so we operate in cardinal steps.
        let push: (i32, i32) = if dx.abs() >= dy.abs() {
            (dx.signum(), 0)
        } else {
            (0, dy.signum())
        };
        if cell.is_frozen() {
            // Rupture walls whose thickness can't absorb this magnitude.
            // Reuse the wall-burst rules: same-element thickness × PER_LAYER.
            const BASE_THRESHOLD: f32 = 2500.0;
            const PER_LAYER: f32 = 350.0;
            const MAX_PROBE: i32 = 30;
            let wall_el = cell.el;
            let mut thickness = 1i32;
            let mut tx = x + push.0;
            let mut ty = y + push.1;
            while thickness <= MAX_PROBE {
                if !Self::in_bounds(tx, ty) { break; }
                let ti = Self::idx(tx, ty);
                if !self.cells[ti].is_frozen() { break; }
                if self.cells[ti].el != wall_el { break; }
                thickness += 1;
                tx += push.0;
                ty += push.1;
            }
            let threshold = BASE_THRESHOLD + PER_LAYER * (thickness - 1) as f32;
            if mag < threshold { return; }
            // Burst the whole column — same pattern as pressure-driven
            // rupture. Outermost first, shatter if blocked.
            let hops = (2 + (mag - threshold) as i32 / 800).clamp(2, 12);
            let mut cells: [(i32, i32); 32] = [(0, 0); 32];
            cells[0] = (x, y);
            let mut tx = x + push.0;
            let mut ty = y + push.1;
            for t in 1..thickness as usize {
                cells[t] = (tx, ty);
                tx += push.0;
                ty += push.1;
            }
            for t in 0..thickness as usize {
                let (wx, wy) = cells[t];
                self.cells[Self::idx(wx, wy)].flag &= !Cell::FLAG_FROZEN;
            }
            for t in (0..thickness as usize).rev() {
                let (wx, wy) = cells[t];
                let mut cxc = wx;
                let mut cyc = wy;
                let mut moved = false;
                for _ in 0..hops {
                    let nx = cxc + push.0;
                    let ny = cyc + push.1;
                    if !Self::in_bounds(nx, ny) { break; }
                    let ni = Self::idx(nx, ny);
                    let tk = cell_physics(self.cells[ni]).kind;
                    if !matches!(tk, Kind::Empty | Kind::Gas | Kind::Fire) { break; }
                    let cur = Self::idx(cxc, cyc);
                    self.cells.swap(cur, ni);
                    self.cells[ni].flag |= Cell::FLAG_UPDATED;
                    cxc = nx;
                    cyc = ny;
                    moved = true;
                }
                if !moved {
                    self.cells[Self::idx(wx, wy)] = Cell::EMPTY;
                }
            }
            return;
        }
        match kind {
            Kind::Empty | Kind::Gas | Kind::Fire => {
                // Pressurize — adds a thermal/impulsive boost that the
                // pressure system handles naturally.
                let add = mag.min(4000.0) as i16;
                let cur = self.cells[i].pressure as i32;
                let new = (cur + add as i32).clamp(-4000, 4000);
                self.cells[i].pressure = new as i16;
            }
            Kind::Liquid | Kind::Powder | Kind::Gravel | Kind::Solid => {
                // Impulse push — teleport outward N cells where N shrinks
                // with density. Gold (density 193) ≈ 0 cells; wood (10) flies.
                let density = cell_physics(cell).density.max(1) as f32;
                let hops = (mag / (density * 8.0)) as i32;
                if hops <= 0 { return; }
                let hops = hops.min(8);
                let mut cxc = x;
                let mut cyc = y;
                for _ in 0..hops {
                    let nx = cxc + push.0;
                    let ny = cyc + push.1;
                    if !Self::in_bounds(nx, ny) { break; }
                    let ni = Self::idx(nx, ny);
                    let t = self.cells[ni];
                    if t.is_frozen() { break; }
                    let tk = cell_physics(t).kind;
                    if !matches!(tk, Kind::Empty | Kind::Gas | Kind::Fire) { break; }
                    let cur = Self::idx(cxc, cyc);
                    self.cells.swap(cur, ni);
                    self.cells[ni].flag |= Cell::FLAG_UPDATED;
                    cxc = nx;
                    cyc = ny;
                }
            }
        }
    }

    // Capture the current grid into the ring buffer. Called at the end of
    // each completed sim step.
    pub fn snapshot(&mut self) {
        self.history[self.history_write].copy_from_slice(&self.cells);
        self.history_write = (self.history_write + 1) % HISTORY_CAPACITY;
        if self.history_count < HISTORY_CAPACITY {
            self.history_count += 1;
        }
        self.rewind_offset = 0;
    }

    // Move the rewind cursor. `delta > 0` goes back in time. Loads the
    // snapshot at the new cursor into `cells`.
    pub fn seek(&mut self, delta: i32) {
        if self.history_count == 0 { return; }
        let max_back = self.history_count - 1;
        let target = (self.rewind_offset as i32 + delta)
            .max(0)
            .min(max_back as i32) as usize;
        if target == self.rewind_offset { return; }
        self.rewind_offset = target;
        // Most-recent snapshot is at (write - 1) mod CAP.
        let idx = (self.history_write + HISTORY_CAPACITY - 1 - target) % HISTORY_CAPACITY;
        self.cells.copy_from_slice(&self.history[idx]);
    }
    #[inline] fn idx(x: i32, y: i32) -> usize { y as usize * W + x as usize }
    #[inline] fn in_bounds(x: i32, y: i32) -> bool {
        x >= 0 && x < W as i32 && y >= 0 && y < H as i32
    }

    fn get(&self, x: i32, y: i32) -> Cell {
        if Self::in_bounds(x, y) {
            self.cells[Self::idx(x, y)]
        } else {
            Cell { el: Element::Stone, derived_id: 0, life: 0, seed: 0, flag: 1, temp: 20, moisture: 0, burn: 0, pressure: 0, solute_el: Element::Empty, solute_amt: 0, solute_derived_id: 0 }
        }
    }
    fn set(&mut self, x: i32, y: i32, c: Cell) {
        if Self::in_bounds(x, y) {
            self.cells[Self::idx(x, y)] = c;
        }
    }
    fn swap(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        if Self::in_bounds(x1, y1) && Self::in_bounds(x2, y2) {
            self.cells.swap(Self::idx(x1, y1), Self::idx(x2, y2));
        }
    }

    // Gravity direction is fixed downward (+y); only the magnitude varies.
    // These helpers are kept so the motion code can be re-vectorized later
    // without touching call sites.
    #[inline] fn gravity_step(&self) -> (i32, i32) { (0, 1) }
    #[inline] fn gravity_sides(&self) -> ((i32, i32), (i32, i32)) {
        ((-1, 0), (1, 0))
    }
    // True if a gravity-driven motion attempt should fire this frame. At
    // g = 1.0 always true; at lower g, fires probabilistically; at 0 never.
    #[inline] fn should_fall(&self) -> bool {
        if self.gravity <= 0.0 { return false; }
        if self.gravity >= 1.0 { return true; }
        rand::gen_range::<f32>(0.0, 1.0) < self.gravity
    }

    // Is there oxygen reachable to a cell at (x, y)? Explicit O in the
    // 3×3 neighborhood always counts; otherwise roll the ambient-oxygen
    // fraction to see if the air provides one. Returns the "oxygen
    // pressure" (0.0-1.0+) — explicit O gives 1.0, air gives
    // ambient_oxygen, and pure-O₂ enriched worlds can exceed 1.0.
    fn oxygen_available(&self, x: i32, y: i32) -> f32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                if self.cells[Self::idx(nx, ny)].el == Element::O {
                    return 1.0f32.max(self.ambient_oxygen);
                }
            }
        }
        self.ambient_oxygen
    }

    // Static weight contribution of an element to hydrostatic pressure. Solids
    // and liquids use the abs of their hand-tuned density (cohesion of matter).
    // Gases use molar_mass scaled down a lot — air-column weight is real but
    // tiny compared to a sand column. Empty cells contribute the ambient air
    // column's own weight (~29 g/mol × small factor), which is what gives the
    // play space its natural altitude pressure gradient even when nothing's
    // painted: empties near the floor sit at higher P than empties near the
    // ceiling, just from the atmospheric column above them.
    fn cell_weight(el: Element) -> f32 {
        let phys = el.physics();
        match phys.kind {
            Kind::Empty => AMBIENT_AIR.molar_mass * 0.02,
            Kind::Gas | Kind::Fire => phys.molar_mass * 0.05,
            _ => (phys.density.max(0) as f32) * 0.5,
        }
    }

    fn mark(&mut self, x: i32, y: i32) {
        if Self::in_bounds(x, y) {
            self.cells[Self::idx(x, y)].flag |= Cell::FLAG_UPDATED;
        }
    }

    fn can_enter(&self, src: Cell, tx: i32, ty: i32, dy: i32) -> bool {
        if !Self::in_bounds(tx, ty) { return false; }
        let tgt = self.cells[Self::idx(tx, ty)];
        if tgt.is_updated() { return false; }
        // Phase-aware on both sides — a molten Gold cell (source) reads as
        // Liquid; a neighbor atom in gas phase reads as Gas.
        let src_p = cell_physics(src);
        let tgt_p = cell_physics(tgt);
        let k = tgt_p.kind;
        if k == Kind::Empty { return true; }
        // Rigid matter (solids, stones, powders) resists being pushed aside.
        // Only fluid matter yields to density-driven displacement.
        if k.is_rigid() { return false; }
        // Viscous liquids (like lava) act rigid from the perspective of rigid
        // matter trying to sink into them. Gases can still bubble up.
        if src_p.kind.is_rigid() && tgt_p.viscosity > 100 { return false; }
        // Gas-gas mixing: gases can swap with any other gas in any direction.
        // Direction is driven by buoyancy (molar_mass Δ in update_gas) and
        // pressure gradient, not by a blanket density check that would stop
        // same-density gases from diffusing.
        if matches!(src_p.kind, Kind::Gas | Kind::Fire) && matches!(k, Kind::Gas | Kind::Fire) {
            return true;
        }
        if dy > 0 { src_p.density > tgt_p.density }
        else if dy < 0 { src_p.density < tgt_p.density }
        else { src_p.density > tgt_p.density }
    }

    // Flood-fill the wind-exposure bitmap: cells reachable from any open
    // horizontal edge via 4-neighbor non-frozen paths. Frozen cells are
    // walls and block the flood, so the interior of a sealed container
    // comes out false. Open-topped beakers come out true because the top
    // opening connects interior to atmosphere. Called only when wind is
    // non-zero, so zero runtime cost when no wind is applied.
    // Energized-cell flood. Two separate BFS floods — one from every
    // BattPos cell, one from every BattNeg cell. Both floods pass through
    // conductors AND through noble-gas cells with a glow_color (modeling
    // plasma conduction). A cell is truly "in the circuit" only if it's
    // reachable from BOTH terminals — that's the closed loop.
    //
    //   conductors_in_loop = pos_flood ∩ neg_flood  (glow + heating here)
    //   lone-terminal conductors = visible but no current — doesn't glow
    //
    // Only runs when both terminals exist; scenes without BattPos or
    // BattNeg get the whole energized buffer zeroed and skip the work.
    // Flood-fill vaporization through a connected frozen conductor
    // segment. Used when a wire cell blows from Joule heating — the
    // whole segment goes to smoke together instead of punching out the
    // hot core while the cool shell lingers as ragged debris. Gated by
    // same-element + frozen + energized so it stops at battery
    // terminals, at non-energized dead branches, and at material
    // boundaries (a Cu wire joined to an Fe wire vaporizes only the
    // side that actually melted).
    fn vaporize_conductor_segment(&mut self, x0: i32, y0: i32, el: Element, t: i16) {
        // Capped small so only the locally-failing segment blows, not
        // the whole wire. Cascade condition: same-element + frozen +
        // already dangerously hot (>= 0.5 × mp). Once the first cell
        // blows, the circuit breaks and nearby cells lose their
        // "energized" status — but they're still compromised thermally
        // and should fail together as a visible weak spot rather than
        // re-melting one-by-one into fringe debris.
        const MAX_CELLS: usize = 40;
        let mp = element_phase_points(Cell::new(el))
            .map(|(mp, _, _)| mp)
            .unwrap_or(1000);
        let hot_threshold = (mp as i32 / 2) as i16;
        let mut queue: Vec<(i32, i32)> = Vec::with_capacity(64);
        queue.push((x0, y0));
        let mut count = 0usize;
        while let Some((cx, cy)) = queue.pop() {
            if count >= MAX_CELLS { break; }
            if !Self::in_bounds(cx, cy) { continue; }
            let ci = Self::idx(cx, cy);
            let c = self.cells[ci];
            if c.el != el { continue; }
            if !c.is_frozen() { continue; }
            if c.temp < hot_threshold { continue; }
            let mut sm = Cell::new(Element::CO2);
            sm.temp = t;
            self.cells[ci] = sm;
            count += 1;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                queue.push((cx + dx, cy + dy));
            }
        }
    }

    // Joule heating on every energized cell. In a real series circuit
    // P_i = I² × R_i; since we aren't solving for current, we use the
    // voltage-squared times resistance proxy — cells with lower
    // conductivity (higher resistance) drop more voltage and heat more.
    // Cu wires barely warm; Fe wires glow red; noble gases in plasma
    // state warm slightly and radiate (handled visually via glow_color).
    // Radioactive decay. Atoms with nonzero half_life_frames have a
    // ln(2)/half_life per-frame probability of transmuting into their
    // decay_product, releasing decay_heat into themselves and cardinal
    // neighbors. Alpha-emission energy becomes kinetic heat in the
    // surroundings — that's why radioactive samples warm their container
    // in real life.
    // Flood-fill the connected-component size of every U cell into
    // u_component_size. Cells in the same contiguous U mass all get
    // the same size value. Lets decay() compute criticality from real
    // pile mass instead of local window density — matches real U's
    // critical mass (thousands of atoms, spatially large).
    fn compute_u_components(&mut self) {
        for v in self.u_component_size.iter_mut() { *v = 0; }
        let n = W * H;
        let mut queue: Vec<usize> = Vec::with_capacity(512);
        let mut component: Vec<usize> = Vec::with_capacity(512);
        for start in 0..n {
            if self.cells[start].el != Element::U { continue; }
            if self.u_component_size[start] != 0 { continue; }
            component.clear();
            queue.clear();
            queue.push(start);
            self.u_component_size[start] = u16::MAX;
            let mut sum_x: i64 = 0;
            let mut sum_y: i64 = 0;
            while let Some(i) = queue.pop() {
                component.push(i);
                let x = (i % W) as i32;
                let y = (i / W) as i32;
                sum_x += x as i64;
                sum_y += y as i64;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].el != Element::U { continue; }
                    if self.u_component_size[ni] != 0 { continue; }
                    self.u_component_size[ni] = u16::MAX;
                    queue.push(ni);
                }
            }
            let size = component.len().min(u16::MAX as usize) as u16;
            let count = component.len().max(1) as i64;
            let cx = (sum_x / count) as i16;
            let cy = (sum_y / count) as i16;
            for &i in &component {
                self.u_component_size[i] = size;
                self.u_component_cx[i] = cx;
                self.u_component_cy[i] = cy;
            }
        }
    }

    fn decay(&mut self) {
        // Early-exit when there's no U in the world. Decay's heavy work
        // (component flood-fill, critical-mass scans, per-cell radioactive
        // ticks) is all gated on Element::U presence. Cheap O(W*H) scan
        // up front saves the ~0.5–1.3ms steady cost when the world has no
        // uranium at all (which is the common case unless someone's
        // building a bomb).
        let mut has_u = false;
        for c in self.cells.iter() {
            if c.el == Element::U { has_u = true; break; }
        }
        if !has_u { return; }
        self.compute_u_components();
        // Clear stale commit flags from previous frames: any cell that
        // isn't currently U can't be mid-cascade, so its flag is
        // garbage. Without this reset, a position that committed during
        // the last detonation "remembers" it even after Pb → erase →
        // new U paint, and detonates immediately regardless of mass.
        for i in 0..self.cells.len() {
            if self.cells[i].el != Element::U {
                self.u_burst_committed[i] = false;
                self.u_central_blast_fired[i] = false;
            }
        }
        // Critical-mass commitment: any U component >= CRITICAL_MASS has
        // every one of its cells flagged for prompt-fission burst. Once
        // committed, the cascade continues through the entire original
        // component even as it fragments into Pb mid-detonation — real
        // bombs don't partially fizzle because the material dispersed.
        const CRITICAL_MASS: u16 = 5000;
        // One-time ground-zero blast per critical component: fires a
        // single very large shockwave at the component centroid the
        // first time any cell in that component reaches criticality.
        // The central_blast_fired flag is flood-filled across the
        // component so only one wave per detonation event, not one per
        // cell. Per-cell burst shockwaves below handle the distributed
        // cascade effect through the pile interior.
        let mut central_queue: Vec<usize> = Vec::with_capacity(1024);
        for i in 0..self.cells.len() {
            if self.cells[i].el != Element::U { continue; }
            if self.u_component_size[i] < CRITICAL_MASS { continue; }
            if self.u_central_blast_fired[i] { continue; }
            let cx = self.u_component_cx[i] as i32;
            let cy = self.u_component_cy[i] as i32;
            // Nuclear-grade yield. 2M chosen so even at r=150 (top of
            // a tall containment box with the pile sitting at the
            // bottom) magnitude stays above glass's 2500 rupture
            // threshold. Magnitude at key radii:
            //   r=40  → 44000 mag (vaporizes anything)
            //   r=80  → 10200 mag (breaks firebrick)
            //   r=120 → 4800 mag (breaks multi-layer glass)
            //   r=150 → 3200 mag (still breaks 1-layer glass)
            //   r=200 → 1900 mag (matter push only, no frozen rupture)
            self.spawn_shockwave_capped(cx, cy, 2_000_000.0, 2_000_000.0);
            // Flood-fill the fired flag through the whole component so
            // no other cell in it emits a second central wave.
            central_queue.clear();
            central_queue.push(i);
            self.u_central_blast_fired[i] = true;
            while let Some(ci) = central_queue.pop() {
                let cix = (ci % W) as i32;
                let ciy = (ci / W) as i32;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = cix + dx;
                    let ny = ciy + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].el != Element::U { continue; }
                    if self.u_central_blast_fired[ni] { continue; }
                    self.u_central_blast_fired[ni] = true;
                    central_queue.push(ni);
                }
            }
        }
        for i in 0..self.cells.len() {
            if self.cells[i].el != Element::U { continue; }
            if self.u_component_size[i] >= CRITICAL_MASS {
                self.u_burst_committed[i] = true;
            }
        }
        let ln2: f32 = 0.6931472;
        for i in 0..self.cells.len() {
            let c = self.cells[i];
            let Some(a) = atom_profile_for(c.el) else { continue; };
            if a.half_life_frames == 0 { continue; }
            let x = (i % W) as i32;
            let y = (i / W) as i32;
            // Criticality: real U goes critical based on the total mass
            // reached by neutrons before escaping. We approximate with
            // the connected-component size of the U pile. Pb/B act as
            // local absorbers — a B control rod sunk through a pile can
            // shield enough to keep it sub-critical even at large mass.
            let multiplier: f32 = if c.el == Element::U {
                let base = self.u_component_size[i] as f32;
                let absorb_penalty = if base > 1500.0 {
                    let mut pb = 0.0f32;
                    let mut b  = 0.0f32;
                    for dy in -3..=3i32 {
                        for dx in -3..=3i32 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = x + dx;
                            let ny = y + dy;
                            if !Self::in_bounds(nx, ny) { continue; }
                            match self.cells[Self::idx(nx, ny)].el {
                                Element::Pb => pb += 1.0,
                                Element::B  => b  += 1.0,
                                _ => {}
                            }
                        }
                    }
                    pb * 20.0 + b * 100.0
                } else {
                    0.0
                };
                let eff = (base - absorb_penalty).max(0.0);
                // Mass-driven per-cell fission rate — linear ramp 1 → 10
                // across the full 0-CRITICAL window so pop density grows
                // smoothly with pile size. Crossing CRITICAL hands off to
                // the commit-flag prompt-fission burst (detonation).
                //   1500 cells → mult 3.7  (occasional visible pops)
                //   3000 cells → mult 6.4  (busy mid-pile popping)
                //   4500 cells → mult 9.1  (frantic near-critical)
                //   5000 cells → commit → detonation
                1.0 + (eff / CRITICAL_MASS as f32) * 9.0
            } else {
                1.0
            };
            // Continuous trickle heat from bulk alpha emission. Scales
            // with the multiplier so near-critical piles warm faster.
            let trickle = (a.decay_heat as f32) * 0.0002 * multiplier;
            let exact = c.temp as f32 + trickle;
            let floor = exact.floor();
            let frac = exact - floor;
            let roll = rand::gen_range::<f32>(0.0, 1.0);
            let stepped = if roll < frac { floor + 1.0 } else { floor };
            self.cells[i].temp = stepped.clamp(-273.0, 5000.0) as i16;
            // Sub-atomic fission flash. Separate from the slow U→Pb
            // transmutation below — represents visible fission activity
            // (neutrons crashing, alpha tracks) that intensifies rapidly
            // with pile mass. Each flash fires a small shockwave and
            // deposits a burst of local heat, so the pile visibly
            // crackles and glows at pops. Rate ramps steeply with the
            // criticality multiplier: at 1500-cell pile it's occasional,
            // at 5800 it's a Christmas-tree cascade.
            if c.el == Element::U && multiplier > 1.2 {
                let flash_p = (multiplier - 1.2) * 2.0e-5;
                if rand::gen_range::<f32>(0.0, 1.0) < flash_p {
                    let yield_p = ((multiplier - 0.5) * 300.0)
                        .max(900.0)
                        .min(2500.0);
                    self.spawn_shockwave(x, y, yield_p);
                    // Local heat spike: big enough to cross the 250°C
                    // render glow threshold so fission points visibly
                    // flare orange. Floors at 400 so even mild-mid
                    // piles glow at flashes, caps at 1200 for
                    // near-critical piles (full-bright flash cores).
                    let spike = ((multiplier - 1.0) * 120.0)
                        .max(400.0)
                        .min(1200.0) as i32;
                    self.cells[i].temp =
                        (self.cells[i].temp as i32 + spike).clamp(-273, 5000) as i16;
                    // Distribute heat to 3×3 neighborhood with falloff
                    // so the glow has size, not just a single pixel.
                    for dy in -1..=1i32 {
                        for dx in -1..=1i32 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = x + dx;
                            let ny = y + dy;
                            if !Self::in_bounds(nx, ny) { continue; }
                            let ni = Self::idx(nx, ny);
                            let falloff = if dx == 0 || dy == 0 { 2 } else { 3 };
                            let t = self.cells[ni].temp as i32 + spike / falloff;
                            self.cells[ni].temp = t.clamp(-273, 5000) as i16;
                        }
                    }
                }
            }
            // Prompt-fission burst for committed cells (past critical
            // mass). Fires with high probability every frame until the
            // committed cell finally transmutes — the detonation
            // cascades through the whole pile even as it fragments.
            let is_committed = c.el == Element::U && self.u_burst_committed[i];
            if is_committed {
                // Not quite 100% per frame so cells don't all convert
                // in literally one tick — spreads the explosion over
                // ~3-4 frames, which looks like a violent cascade
                // rather than a pop.
                if rand::gen_range::<f32>(0.0, 1.0) < 0.35 {
                    let product = a.decay_product;
                    let burst_heat = 4000i32;
                    let old_frozen = c.is_frozen();
                    let old_phase = c.phase();
                    let mut d = Cell::new(product);
                    d.temp = (c.temp as i32 + burst_heat).clamp(-273, 5000) as i16;
                    if old_frozen { d.flag |= Cell::FLAG_FROZEN; }
                    d.set_phase(old_phase);
                    self.cells[i] = d;
                    self.u_burst_committed[i] = false;
                    for dy in -3..=3i32 {
                        for dx in -3..=3i32 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = x + dx;
                            let ny = y + dy;
                            if !Self::in_bounds(nx, ny) { continue; }
                            let dist2 = (dx * dx + dy * dy) as f32;
                            if dist2 > 9.0 { continue; }
                            let ni = Self::idx(nx, ny);
                            let falloff = 1.0 / (1.0 + dist2 * 0.3);
                            let h = (burst_heat as f32 * falloff * 0.35) as i32;
                            let t = self.cells[ni].temp as i32 + h;
                            self.cells[ni].temp = t.clamp(-273, 5000) as i16;
                        }
                    }
                    self.spawn_shockwave(x, y, 5000.0);
                    continue;
                }
            }
            // Normal alpha-decay transmutation. Per-cell probability
            // scales with the criticality multiplier, so mid-range piles
            // pop visibly, larger piles pop frantically. Each event
            // emits a small shockwave — the transmutation IS the pop.
            let p = (ln2 / (a.half_life_frames as f32)) * multiplier;
            if rand::gen_range::<f32>(0.0, 1.0) >= p { continue; }
            let product = a.decay_product;
            let event_heat = ((a.decay_heat as f32) * multiplier).min(2000.0) as i32;
            let old_frozen = c.is_frozen();
            let old_phase = c.phase();
            let mut d = Cell::new(product);
            d.temp = (c.temp as i32 + event_heat).clamp(-273, 5000) as i16;
            if old_frozen { d.flag |= Cell::FLAG_FROZEN; }
            d.set_phase(old_phase);
            self.cells[i] = d;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                let t = self.cells[ni].temp as i32 + event_heat / 2;
                self.cells[ni].temp = t.clamp(-273, 5000) as i16;
            }
            // Shockwave per transmutation — the visible pop that
            // signals a single atom just fissioned. Yield scales with
            // multiplier so events from bigger piles pop harder. Floor
            // at 800 because the shockwave retire threshold is 200 and
            // anything below ~600 yield never actually propagates past
            // radius 0 (invisible to the user).
            if c.el == Element::U && multiplier > 1.2 {
                let yield_p = ((multiplier - 1.0) * 400.0).max(800.0).min(3500.0);
                self.spawn_shockwave(x, y, yield_p);
            }
        }
    }

    // Electrolysis: metal cations in the brine plate out onto cathode
    // electrodes; anode metal dissolves into the brine if it matches the
    // solute's metal (a closed-loop ion source). Galvanic and electrolytic
    // modes both feed this the same way via cathode_mask / anode_mask.
    // Only runs with an active circuit.
    fn electrolysis(&mut self) {
        if self.active_emf <= 0.0 { return; }
        // Open-circuit guard: active_emf only says "there's a potential
        // difference", but without a closed loop no cells are energized
        // and no current actually flows. Dissolution/plating must only
        // happen when current is flowing.
        if !self.energized.iter().any(|&e| e) { return; }
        // Plating: look at every cathode electrode cell; for each adjacent
        // brine carrying a metal cation, probabilistically deposit the
        // metal. 70% of deposits stick as a frozen coating on the rod
        // (grows outward), 30% slough off as a loose powder that gravity
        // settles into a pile beneath — exactly what happens in dirty
        // real-world electroplating.
        const PLATE_P: f32 = 0.015;
        const ADHERENT_FRAC: f32 = 0.70;
        for i in 0..self.cells.len() {
            if !self.cathode_mask[i] { continue; }
            if !self.energized[i] { continue; }
            let x = (i % W) as i32;
            let y = (i / W) as i32;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                let n = self.cells[ni];
                if n.el != Element::Water { continue; }
                // Require near-saturated brine at the interface. Plating
                // replaces the cell with metal, consuming its entire
                // solute load, so the minimum sets how much solute each
                // plating event actually "costs". Below this threshold,
                // diffusion hasn't refilled the interface fast enough —
                // plating pauses until the bulk brine migrates in.
                if n.solute_amt < 128 { continue; }
                // Decompose the solute into its metal cation. Salt (NaCl)
                // yields Na; a derived metal-halide yields its metal.
                let metal_el = if n.solute_el == Element::Salt {
                    Some(Element::Na)
                } else if n.solute_el == Element::Derived {
                    compound_metal_component(n.solute_derived_id)
                } else {
                    None
                };
                let Some(metal_el) = metal_el else { continue; };
                // Alkali and alkaline-earth cations don't plate from
                // aqueous solution — the electrode potential is so
                // negative that water reduces first (2H₂O + 2e⁻ → H₂ +
                // 2OH⁻). Letting them plate would drop a Na/K/Cs/Mg/Ca
                // cell into water, which then reacts explosively. Real
                // chemistry just produces H₂ gas; we simply skip for
                // now (gas byproduct comes with the multi-solute work).
                if let Some(a) = atom_profile_for(metal_el) {
                    if matches!(
                        a.category,
                        AtomCategory::AlkaliMetal | AtomCategory::AlkalineEarth,
                    ) {
                        continue;
                    }
                }
                if rand::gen_range::<f32>(0.0, 1.0) > PLATE_P { continue; }
                // Choose adherent vs loose. Adherent replaces the brine
                // cell with a frozen metal cell; loose spawns a free
                // metal cell that can fall under gravity.
                let adherent = rand::gen_range::<f32>(0.0, 1.0) < ADHERENT_FRAC;
                let mut deposit = Cell::new(metal_el);
                deposit.temp = self.cells[i].temp;
                if adherent {
                    deposit.flag |= Cell::FLAG_FROZEN;
                }
                self.cells[ni] = deposit;
                // Consume one solute "chunk". Solute_amt 255 ≈ one salt
                // worth of ions, so subtract ~64 per plating to give
                // roughly 4 plating events per dissolved cell.
                let after = self.cells[ni];
                let _ = after; // silence potential lint — ni was overwritten
                // The brine cell is gone; to represent the remaining ions
                // migrating, dock solute from a neighboring water cell if
                // one exists. Otherwise it's just lost (matches the real
                // "the solute near the cathode gets depleted").
                let mut docked = false;
                for (ddx, ddy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let wx = nx + ddx;
                    let wy = ny + ddy;
                    if !Self::in_bounds(wx, wy) { continue; }
                    let wi = Self::idx(wx, wy);
                    let w = self.cells[wi];
                    if w.el != Element::Water { continue; }
                    if w.solute_amt < 64 { continue; }
                    if w.solute_el != n.solute_el
                        || w.solute_derived_id != n.solute_derived_id
                    {
                        continue;
                    }
                    self.cells[wi].solute_amt = w.solute_amt.saturating_sub(64);
                    if self.cells[wi].solute_amt == 0 {
                        self.cells[wi].solute_el = Element::Empty;
                        self.cells[wi].solute_derived_id = 0;
                    }
                    docked = true;
                    break;
                }
                let _ = docked;
                break;
            }
        }
        // Anode dissolution: an anode electrode whose metal matches the
        // brine's solute cation slowly erodes into the brine. One anode
        // cell → one +64 solute_amt in an adjacent water cell that has
        // room. This is what lets the cathode keep plating indefinitely
        // when the ion pool in solution would otherwise run out.
        const DISSOLVE_P: f32 = 0.006;
        for i in 0..self.cells.len() {
            if !self.anode_mask[i] { continue; }
            if !self.energized[i] { continue; }
            let c = self.cells[i];
            if !is_atomic_metal(c.el) { continue; }
            if rand::gen_range::<f32>(0.0, 1.0) > DISSOLVE_P { continue; }
            let x = (i % W) as i32;
            let y = (i / W) as i32;
            let mut donated = false;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                let n = self.cells[ni];
                if n.el != Element::Water { continue; }
                if n.solute_amt < 20 { continue; }
                let solute_metal = if n.solute_el == Element::Salt {
                    Some(Element::Na)
                } else if n.solute_el == Element::Derived {
                    compound_metal_component(n.solute_derived_id)
                } else {
                    None
                };
                if solute_metal != Some(c.el) { continue; }
                let room = 255u8.saturating_sub(n.solute_amt);
                if room < 16 { continue; }
                let add = room.min(64);
                self.cells[ni].solute_amt = n.solute_amt + add;
                donated = true;
                break;
            }
            if donated {
                self.cells[i] = Cell::EMPTY;
            }
        }
    }

    fn joule_heating(&mut self) {
        let v = self.active_emf;
        if v <= 0.0 { return; }
        let v2 = v * v;
        // Scale tuned so V=100 puts Fe (cond 0.4) at a few °C per frame
        // gain — visible glow over seconds without instantly vaporizing.
        const K: f32 = 0.00005;
        for i in 0..self.cells.len() {
            if !self.energized[i] { continue; }
            let c = self.cells[i];
            let cond = c.conductivity();
            // Terminals (BattPos/BattNeg) have conductivity 1.0 —
            // they don't self-heat. Noble-gas glow cells (cond 0) would
            // heat the most; dial down their contribution so gas tubes
            // don't melt their own glass.
            let gas_like = c.el.electrical().glow_color.is_some();
            let resistance = (1.0 - cond).max(0.0);
            let factor = if gas_like { 0.1 } else { 1.0 };
            let delta = v2 * resistance * K * factor;
            if delta < 0.01 { continue; }
            // Stochastic rounding: integer temps truncate small gains,
            // so a 0.8°C/frame climb on a good conductor (Ag σ0.98 at
            // 900V ≈ 0.8/frame) silently rounds to zero every frame.
            // Probabilistically round up on fraction so the average
            // matches the real rate.
            let exact = c.temp as f32 + delta;
            let floor = exact.floor();
            let frac = exact - floor;
            let roll = rand::gen_range::<f32>(0.0, 1.0);
            let stepped = if roll < frac { floor + 1.0 } else { floor };
            self.cells[i].temp = stepped.clamp(-273.0, 5000.0) as i16;
        }
    }

    fn compute_energized(&mut self) {
        for v in self.energized.iter_mut() { *v = false; }
        for v in self.cathode_mask.iter_mut() { *v = false; }
        for v in self.anode_mask.iter_mut() { *v = false; }
        self.galvanic_voltage = 0.0;
        self.active_emf = 0.0;
        self.galvanic_cathode_el = None;
        self.galvanic_anode_el = None;
        // Seed lists.
        let mut pos_seeds: Vec<(i32, i32)> = Vec::new();
        let mut neg_seeds: Vec<(i32, i32)> = Vec::new();
        for i in 0..self.cells.len() {
            match self.cells[i].el {
                Element::BattPos => pos_seeds.push(((i % W) as i32, (i / W) as i32)),
                Element::BattNeg => neg_seeds.push(((i % W) as i32, (i / W) as i32)),
                _ => {}
            }
        }
        // Galvanic cell detection — only if there's no explicit battery. Two
        // distinct metals touching the same electrolyte drive a flood from
        // the more-reactive (lowest EN → anode / BattNeg) to the less-
        // reactive (highest EN → cathode / BattPos). Voltage scales with the
        // EN gap so Cu/Zn-ish pairs give ~1V while Cu/Na-ish pairs give more.
        if pos_seeds.is_empty() && neg_seeds.is_empty() {
            // (element, (x, y), electronegativity). Any metal cell — loose
            // paint or a frozen wire — with a brine neighbor qualifies.
            let mut candidates: Vec<(Element, (i32, i32), f32)> = Vec::new();
            for i in 0..self.cells.len() {
                let c = self.cells[i];
                if !is_atomic_metal(c.el) { continue; }
                let x = (i % W) as i32;
                let y = (i / W) as i32;
                let mut on_brine = false;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let n = self.cells[Self::idx(nx, ny)];
                    if n.el == Element::Water && n.solute_amt > 20 {
                        on_brine = true;
                        break;
                    }
                }
                if !on_brine { continue; }
                let en = atom_profile_for(c.el).map(|a| a.electronegativity).unwrap_or(0.0);
                candidates.push((c.el, (x, y), en));
            }
            if candidates.len() >= 2 {
                let lo = candidates.iter().map(|c| c.2).fold(f32::INFINITY, f32::min);
                let hi = candidates.iter().map(|c| c.2).fold(f32::NEG_INFINITY, f32::max);
                let gap = hi - lo;
                // Need at least two distinct metals (different EN) — same
                // metal on both sides is not a galvanic cell.
                if gap > 0.05 {
                    let mut cathode_el: Option<Element> = None;
                    let mut anode_el: Option<Element> = None;
                    for &(el, (x, y), en) in &candidates {
                        if (en - lo).abs() < 1e-3 {
                            neg_seeds.push((x, y));
                            anode_el = Some(el);
                        }
                        if (en - hi).abs() < 1e-3 {
                            pos_seeds.push((x, y));
                            cathode_el = Some(el);
                        }
                    }
                    // Scale EN gap (Pauling units, typically 0.2–2.0) into a
                    // usable voltage. Cap to avoid melting wires the instant
                    // Na/Au touches brine.
                    self.galvanic_voltage = (gap * 80.0).clamp(10.0, 250.0);
                    self.galvanic_cathode_el = cathode_el;
                    self.galvanic_anode_el = anode_el;
                }
            }
            // Verify a closed loop actually exists. The brine already
            // connects anode and cathode *internally* (that's the ion
            // half of the cell), but current can't flow without an
            // EXTERNAL conductor path too. Flood from pos_seeds through
            // non-water conductors only; if it doesn't reach any neg_seed,
            // the circuit is open — kill the galvanic emf and bail so
            // stray wires dangling off one rod don't glow.
            if !pos_seeds.is_empty() && !neg_seeds.is_empty() {
                let n = W * H;
                let mut reach = vec![false; n];
                self.energized_queue.clear();
                for &(x, y) in &pos_seeds {
                    reach[Self::idx(x, y)] = true;
                    self.energized_queue.push((x, y));
                }
                while let Some((cx, cy)) = self.energized_queue.pop() {
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if !Self::in_bounds(nx, ny) { continue; }
                        let ni = Self::idx(nx, ny);
                        if reach[ni] { continue; }
                        let nc = self.cells[ni];
                        // Water (brine) is the *internal* path — skip.
                        if nc.el == Element::Water { continue; }
                        if nc.conductivity() > 0.02
                            || nc.el.electrical().glow_color.is_some()
                        {
                            reach[ni] = true;
                            self.energized_queue.push((nx, ny));
                        }
                    }
                }
                let closed = neg_seeds
                    .iter()
                    .any(|&(x, y)| reach[Self::idx(x, y)]);
                if !closed {
                    self.galvanic_voltage = 0.0;
                    return;
                }
            }
        }
        if pos_seeds.is_empty() || neg_seeds.is_empty() { return; }
        // At this point seeds exist on both sides, so a circuit is possible.
        // If galvanic seeded them, it already set galvanic_voltage; otherwise
        // this is a battery circuit and the slider value applies.
        self.active_emf = if self.galvanic_voltage > 0.0 {
            self.galvanic_voltage
        } else {
            self.battery_voltage
        };
        // Reusable buffers — reuse self.wind_exposed-style storage to
        // avoid two more big Vec allocations. We allocate them lazily
        // here as local buffers since they're only needed for this pass.
        let n = W * H;
        let mut pos = vec![false; n];
        let mut neg = vec![false; n];
        let propagate = |world: &World, ni: usize| -> bool {
            let c = world.cells[ni];
            c.conductivity() > 0.02 || c.el.electrical().glow_color.is_some()
        };
        // Flood from positive terminals.
        self.energized_queue.clear();
        for &(x, y) in &pos_seeds {
            pos[Self::idx(x, y)] = true;
            self.energized_queue.push((x, y));
        }
        while let Some((cx, cy)) = self.energized_queue.pop() {
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = cx + dx;
                let ny = cy + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                if pos[ni] { continue; }
                if !propagate(self, ni) { continue; }
                pos[ni] = true;
                self.energized_queue.push((nx, ny));
            }
        }
        // Flood from negative terminals.
        self.energized_queue.clear();
        for &(x, y) in &neg_seeds {
            neg[Self::idx(x, y)] = true;
            self.energized_queue.push((x, y));
        }
        while let Some((cx, cy)) = self.energized_queue.pop() {
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = cx + dx;
                let ny = cy + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                if neg[ni] { continue; }
                if !propagate(self, ni) { continue; }
                neg[ni] = true;
                self.energized_queue.push((nx, ny));
            }
        }
        // Intersection: only cells reachable from BOTH terminals are in
        // the closed loop and actually carry current.
        for i in 0..n {
            self.energized[i] = pos[i] && neg[i];
        }
        // Electrode masks — which metal cells are "cathode electrodes"
        // (where cations plate out) vs "anode electrodes" (where the metal
        // dissolves into solution). Driven by mode:
        //  * Galvanic: cathode metals are the high-EN species, anodes the
        //    low-EN species. Simple element-type check (no need for dry
        //    floods, since the galvanic loop routes current via those
        //    species by definition).
        //  * Electrolytic: the battery defines the polarity, so we flood
        //    out from each terminal through non-brine conductors — a metal
        //    touching brine reached only from BattNeg is a cathode; only
        //    from BattPos is an anode.
        // Newly-plated frozen cells inherit membership automatically because
        // they conduct and are adjacent to an existing electrode.
        let galvanic_mode = self.galvanic_voltage > 0.0;
        if galvanic_mode {
            let cath = self.galvanic_cathode_el;
            let anod = self.galvanic_anode_el;
            for i in 0..n {
                let c = self.cells[i];
                if !is_atomic_metal(c.el) { continue; }
                let x = (i % W) as i32;
                let y = (i / W) as i32;
                let mut touches_brine = false;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let nc = self.cells[Self::idx(nx, ny)];
                    if nc.el == Element::Water && nc.solute_amt > 20 {
                        touches_brine = true;
                        break;
                    }
                }
                if !touches_brine { continue; }
                if Some(c.el) == cath { self.cathode_mask[i] = true; }
                else if Some(c.el) == anod { self.anode_mask[i] = true; }
            }
        } else {
            // Electrolytic — dry floods identify sides uniquely.
            let mut pos_dry = vec![false; n];
            let mut neg_dry = vec![false; n];
            let dry_propagate = |world: &World, ni: usize| -> bool {
                let c = world.cells[ni];
                if c.el == Element::Water { return false; }
                c.conductivity() > 0.02 || c.el.electrical().glow_color.is_some()
            };
            self.energized_queue.clear();
            for &(x, y) in &pos_seeds {
                pos_dry[Self::idx(x, y)] = true;
                self.energized_queue.push((x, y));
            }
            while let Some((cx, cy)) = self.energized_queue.pop() {
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if pos_dry[ni] { continue; }
                    if !dry_propagate(self, ni) { continue; }
                    pos_dry[ni] = true;
                    self.energized_queue.push((nx, ny));
                }
            }
            self.energized_queue.clear();
            for &(x, y) in &neg_seeds {
                neg_dry[Self::idx(x, y)] = true;
                self.energized_queue.push((x, y));
            }
            while let Some((cx, cy)) = self.energized_queue.pop() {
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if neg_dry[ni] { continue; }
                    if !dry_propagate(self, ni) { continue; }
                    neg_dry[ni] = true;
                    self.energized_queue.push((nx, ny));
                }
            }
            for i in 0..n {
                let c = self.cells[i];
                if !is_atomic_metal(c.el) { continue; }
                let x = (i % W) as i32;
                let y = (i / W) as i32;
                let mut touches_brine = false;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let nc = self.cells[Self::idx(nx, ny)];
                    if nc.el == Element::Water && nc.solute_amt > 20 {
                        touches_brine = true;
                        break;
                    }
                }
                if !touches_brine { continue; }
                // BattPos-connected metal = electrolytic anode (oxidation).
                // BattNeg-connected = cathode (reduction / plating).
                if neg_dry[i] && !pos_dry[i] { self.cathode_mask[i] = true; }
                else if pos_dry[i] && !neg_dry[i] { self.anode_mask[i] = true; }
            }
        }
    }

    fn compute_wind_exposure(&mut self) {
        for v in self.wind_exposed.iter_mut() { *v = false; }
        self.wind_queue.clear();
        for y in 0..H as i32 {
            for x in [0i32, W as i32 - 1] {
                let idx = Self::idx(x, y);
                if !self.cells[idx].is_frozen() {
                    self.wind_exposed[idx] = true;
                    self.wind_queue.push((x, y));
                }
            }
        }
        while let Some((cx, cy)) = self.wind_queue.pop() {
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = cx + dx;
                let ny = cy + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                if self.wind_exposed[ni] { continue; }
                if self.cells[ni].is_frozen() { continue; }
                self.wind_exposed[ni] = true;
                self.wind_queue.push((nx, ny));
            }
        }
    }

    pub fn step(&mut self, wind: Vec2) {
        self.step_inner(wind, true, true);
    }
}

/// Bitmask of chemistry passes that are running on GPU compute and
/// should therefore be skipped on CPU. As more passes get ported,
/// add fields here. `Default` = nothing on GPU (CPU runs everything).
#[derive(Clone, Copy, Default)]
pub struct GpuChem {
    pub clear_flags: bool,
    pub color_fires: bool,
    pub flame_test_emission: bool,
    pub tree_support: bool,
    pub thermal_post: bool,
    pub dissolve: bool,
    pub diffuse_solute: bool,
    pub reactions: bool,
    pub glass_etching: bool,
}

impl World {

    /// Step everything except the pressure diffusion pass. The wgpu
    /// binary calls this and dispatches pressure on the GPU instead.
    /// `pressure_sources` (CPU-cheap, sets up hydrostatic targets)
    /// still runs — the GPU starts from those values.
    pub fn step_skip_pressure(&mut self, wind: Vec2) {
        self.step_inner(wind, false, true);
    }

    /// Step everything except pressure() AND thermal_diffuse(). The
    /// wgpu binary uses this when both passes run on GPU compute.
    /// thermal_post() (moisture/combustion) still runs on CPU.
    pub fn step_skip_pressure_thermal(&mut self, wind: Vec2) {
        self.step_inner_full(wind, false, false, true);
    }

    /// Skip pressure_sources, pressure, AND thermal_diffuse — all
    /// three are running on GPU compute.
    pub fn step_skip_gpu_passes(&mut self, wind: Vec2) {
        self.step_inner_full2(wind, false, false, false, true);
    }

    /// Skip pressure_sources, pressure, thermal_diffuse, AND motion
    /// (`update_cell` sweep) — used when motion runs on GPU compute.
    pub fn step_skip_gpu_passes_and_motion(&mut self, wind: Vec2) {
        self.step_inner_full3(wind, false, false, false, false, GpuChem::default());
    }

    /// Same as `step_skip_gpu_passes_and_motion` but also skips the
    /// chemistry passes flagged in `gpu_chem` (those are running on
    /// GPU compute too).
    pub fn step_skip_gpu_v2(&mut self, wind: Vec2, gpu_chem: GpuChem) {
        self.step_inner_full3(wind, false, false, false, false, gpu_chem);
    }

    fn step_inner(&mut self, wind: Vec2, run_pressure: bool, run_thermal_diffuse: bool) {
        self.step_inner_full3(wind, run_pressure, run_thermal_diffuse, true, true, GpuChem::default());
    }

    fn step_inner_full(
        &mut self,
        wind: Vec2,
        run_pressure: bool,
        run_thermal_diffuse: bool,
        run_pressure_sources: bool,
    ) {
        self.step_inner_full3(wind, run_pressure, run_thermal_diffuse, run_pressure_sources, true, GpuChem::default());
    }

    fn step_inner_full2(
        &mut self,
        wind: Vec2,
        run_pressure: bool,
        run_thermal_diffuse: bool,
        run_pressure_sources: bool,
        run_motion: bool,
    ) {
        self.step_inner_full3(wind, run_pressure, run_thermal_diffuse, run_pressure_sources, run_motion, GpuChem::default());
    }

    fn step_inner_full3(
        &mut self,
        wind: Vec2,
        run_pressure: bool,
        run_thermal_diffuse: bool,
        run_pressure_sources: bool,
        run_motion: bool,
        gpu_chem: GpuChem,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let mut prof: Vec<(&'static str, u64)> = Vec::with_capacity(32);
        let mut tt = std::time::Instant::now();
        macro_rules! mark {
            ($name:expr) => {{
                prof.push(($name, tt.elapsed().as_micros() as u64));
                tt = std::time::Instant::now();
            }};
        }
        // Build the element-presence bitmap; clear FLAG_UPDATED only
        // when the GPU isn't running clear_flags itself. Either way
        // the presence bitmap is needed for CPU chemistry early-exits.
        let mut present = [false; ELEMENT_COUNT];
        if gpu_chem.clear_flags {
            for c in self.cells.iter() {
                let id = c.el as usize;
                if id < ELEMENT_COUNT { present[id] = true; }
            }
        } else {
            for c in self.cells.iter_mut() {
                c.flag &= !Cell::FLAG_UPDATED;
                let id = c.el as usize;
                if id < ELEMENT_COUNT { present[id] = true; }
            }
        }
        self.present_elements = present;
        mark!("clear_flags");
        if wind.length_squared() > 0.0001 {
            self.compute_wind_exposure();
        }
        mark!("wind");
        self.compute_energized();    mark!("energized");
        self.joule_heating();        mark!("joule");
        self.electrolysis();         mark!("electrolysis");
        self.decay();                mark!("decay");
        if !gpu_chem.tree_support {
            self.tree_support_check(); mark!("tree_support");
        }
        self.thermite();             mark!("thermite");
        self.magnesium_burn();       mark!("mg_burn");
        if !gpu_chem.glass_etching {
            self.glass_etching();    mark!("glass_etch");
        }
        self.halogen_displacement(); mark!("halogen_disp");
        self.hg_amalgamation();      mark!("hg_amalg");
        if !gpu_chem.flame_test_emission {
            self.flame_test_emission(); mark!("flame_emit");
        }
        if !gpu_chem.color_fires {
            self.color_fires();      mark!("color_fires");
        }
        if run_thermal_diffuse {
            self.thermal();          mark!("thermal");
        } else if !gpu_chem.thermal_post {
            // GPU compute path dispatches thermal_diffuse externally.
            // CPU runs thermal_post (combustion + moisture + phase
            // changes) only when GPU isn't doing it.
            self.thermal_post();     mark!("thermal_post");
        }
        self.chemical_reactions();   mark!("chem_reactions");
        self.acid_displacement();    mark!("acid_disp");
        self.alloy_acid_leach();     mark!("alloy_leach");
        self.base_neutralization();  mark!("base_neutral");
        self.alloy_formation();      mark!("alloy_form");
        if !gpu_chem.dissolve {
            self.dissolve();         mark!("dissolve");
        }
        if !gpu_chem.diffuse_solute {
            self.diffuse_solute();   mark!("diffuse_solute");
        }
        if !gpu_chem.reactions {
            self.reactions();        mark!("reactions");
        }
        if run_motion {
            for y in (0..H as i32).rev() {
                let lr = self.frame % 2 == 0;
                for i in 0..W as i32 {
                    let x = if lr { i } else { W as i32 - 1 - i };
                    self.update_cell(x, y, wind);
                }
            }
            mark!("update_cells");
        }
        if run_pressure_sources {
            self.pressure_sources(); mark!("pressure_src");
        }
        if run_pressure {
            self.pressure();         mark!("pressure");
        }
        self.tick_shockwaves();      mark!("shockwaves");
        self.snapshot();             mark!("snapshot");
        let _ = tt;
        if self.frame % 60 == 0 {
            let total: u64 = prof.iter().map(|(_, t)| t).sum();
            let mut sorted = prof.clone();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = sorted.iter().take(8)
                .map(|(n, t)| format!("{}={:.1}ms", n, *t as f32 / 1000.0))
                .collect();
            eprintln!("[step] total={:.1}ms | {}", total as f32 / 1000.0, top.join(" "));
        }
    }

    // Structural support check for wood. Flood-fills from every wood cell
    // that's sitting on the ground; anything not reached is free-hanging.
    // We *mark* unsupported wood (via Cell.life = 1) but don't actually move
    // it here — the per-frame update loop handles the falling every tick so
    // collapse is smooth. Runs on a stride because the BFS is non-trivial.
    fn tree_support_check(&mut self) {
        if self.frame % 30 != 0 { return; }

        // Reuse the support scratch buffer across calls (no per-frame alloc).
        if self.support_scratch.len() != W * H {
            self.support_scratch = vec![false; W * H];
        }
        for b in self.support_scratch.iter_mut() { *b = false; }
        self.support_queue.clear();

        // Anchor — wood resting on a non-wood rigid surface (or the world floor).
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let idx = Self::idx(x, y);
                if self.cells[idx].el != Element::Wood { continue; }
                // Frozen (rigid-body) wood is always its own anchor.
                let grounded = if self.cells[idx].is_frozen() {
                    true
                } else if y == H as i32 - 1 {
                    true
                } else {
                    let below = self.cells[Self::idx(x, y + 1)];
                    below.el != Element::Wood
                        && matches!(
                            below.el.physics().kind,
                            Kind::Solid | Kind::Gravel | Kind::Powder
                        )
                };
                if grounded {
                    self.support_scratch[idx] = true;
                    self.support_queue.push((x, y));
                }
            }
        }

        // Propagate — BFS across connected wood.
        while let Some((x, y)) = self.support_queue.pop() {
            for (dx, dy) in [(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let nidx = Self::idx(nx, ny);
                if self.support_scratch[nidx] { continue; }
                if self.cells[nidx].el != Element::Wood { continue; }
                self.support_scratch[nidx] = true;
                self.support_queue.push((nx, ny));
            }
        }

        // Mark state on wood cells: life = 1 means "falling", 0 means supported.
        for i in 0..W * H {
            if self.cells[i].el == Element::Wood {
                self.cells[i].life = if self.support_scratch[i] { 0 } else { 1 };
            }
        }
    }

    // Heat diffusion + ambient drift, moisture exchange/evaporation, phase
    // changes and in-place combustion. Generic — every element is driven by
    // the same rules through its property methods.
    fn thermal(&mut self) {
        self.thermal_diffuse();
        self.thermal_post();
    }

    /// Section 1 of thermal: 4-neighbor heat diffusion + ambient blend
    /// + stochastic rounding. Pure per-cell read/compute → write to a
    /// disjoint scratch slot, which makes it safe to dispatch on GPU
    /// compute. The wgpu binary skips this and calls a compute shader
    /// instead via `step_skip_pressure_thermal`.
    pub fn thermal_diffuse(&mut self) {
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                let my_k = c.el.thermal().conductivity;
                let mut delta: f32 = 0.0;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                    let n = self.cells[Self::idx(x + dx, y + dy)];
                    let k = my_k.min(n.el.thermal().conductivity);
                    delta += k * (n.temp as f32 - c.temp as f32);
                }
                let exposure = if matches!(c.el, Element::Fire | Element::Empty) {
                    1.0
                } else {
                    let mut diff = 0.0f32;
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x + dx; let ny = y + dy;
                        if !Self::in_bounds(nx, ny) { diff += 1.0; continue; }
                        if self.cells[Self::idx(nx, ny)].el != c.el { diff += 1.0; }
                    }
                    diff / 4.0
                };
                let amb_factor = 0.10 + 0.90 * exposure;
                let ambient_t = c.el.thermal().ambient_temp as i32 + self.ambient_offset as i32;
                delta += c.el.thermal().ambient_rate * amb_factor
                    * (ambient_t as f32 - c.temp as f32);
                let exact = c.temp as f32 + delta / c.el.thermal().heat_capacity;
                let floor = exact.floor();
                let frac = exact - floor;
                let roll = rand::gen_range::<u16>(0, 10_000) as f32 / 10_000.0;
                let stepped = if roll < frac { floor + 1.0 } else { floor };
                let new_t = stepped.clamp(-273.0, 4000.0) as i16;
                self.temp_scratch[i] = new_t;
            }
        }
        for i in 0..(W * H) {
            self.cells[i].temp = self.temp_scratch[i];
        }
    }

    /// Sections 2 + 3 of thermal: moisture dynamics, evaporation,
    /// combustion-driven phase changes. These have writes-to-non-self
    /// neighbors and stay on CPU regardless of which step variant the
    /// caller uses.
    pub fn thermal_post(&mut self) {
        // 2) Moisture dynamics — wetting from water contact, heat-driven and
        // passive evaporation *only on cells touching air* (surface-first).
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el == Element::Water || c.el == Element::Empty { continue; }

                // Absorption from any adjacent moisture source (water, ice, mud).
                // Non-sinks (oil, stone-wise no, but oil is hydrophobic here) skip.
                if c.moisture < 250 && c.el.moisture().is_sink {
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let n = self.get(x + dx, y + dy).el;
                        if n.moisture().is_source {
                            self.cells[i].moisture = self.cells[i].moisture.saturating_add(5);
                            break;
                        }
                    }
                }

                // Wicking: gradient-driven diffusion through solids/powders.
                // Moisture shares with every drier non-water neighbor, scaled
                // by the bottleneck material's conductivity.
                let c = self.cells[i];
                let my_k = c.el.moisture().conductivity;
                if c.moisture > 5 && my_k > 0.0 && c.el.moisture().is_sink {
                    for (dx, dy) in [(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                        if !Self::in_bounds(x + dx, y + dy) { continue; }
                        let nidx = Self::idx(x + dx, y + dy);
                        let n = self.cells[nidx];
                        if !n.el.moisture().is_sink { continue; }
                        // Don't pump moisture INTO cells past boiling — water
                        // doesn't travel into a cell that's actively evaporating,
                        // otherwise wet neighbors shield hot ones indefinitely.
                        if n.temp > 100 { continue; }
                        let k = my_k.min(n.el.moisture().conductivity);
                        if k <= 0.0 { continue; }
                        let cm = self.cells[i].moisture as i16;
                        let nm = n.moisture as i16;
                        let gradient = cm - nm;
                        if gradient > 3 {
                            let flow = (k * gradient as f32).round().max(1.0) as i16;
                            let amt = flow
                                .min(cm)
                                .min(255 - nm)
                                .max(0) as u8;
                            if amt > 0 {
                                self.cells[i].moisture = self.cells[i].moisture.saturating_sub(amt);
                                self.cells[nidx].moisture = n.moisture.saturating_add(amt);
                            }
                        }
                    }
                }

                // Heat-driven evaporation. Higher temps shed multiple moisture
                // units per frame, so extreme heat sources (lava, fire) can
                // actually dry a wet surface faster than wicking refills it.
                if c.temp > 80 && c.moisture > 0 {
                    let excess = (c.temp as i32 - 80).max(0) as u32;
                    let rate   = (excess / 40).clamp(1, 10) as u16;
                    let drops  = (excess / 200).clamp(1, 20) as u8;
                    if rand::gen_range::<u16>(0, 10) < rate {
                        self.cells[i].moisture = c.moisture.saturating_sub(drops);
                        self.cells[i].temp -= drops as i16;
                    }
                }
                // Passive drying: only where moisture can actually leave —
                // at an air-exposed face.
                let mut exposed = false;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if self.get(x + dx, y + dy).el == Element::Empty {
                        exposed = true; break;
                    }
                }
                if exposed && self.cells[i].moisture > 0
                    && rand::gen_range::<u16>(0, 400) < 1
                {
                    self.cells[i].moisture = self.cells[i].moisture.saturating_sub(1);
                }
            }
        }

        // 3) Combustion + phase changes.
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                let t = c.temp;

                // Frozen cells still obey physics — glass still melts, wood
                // still burns, water still boils. Structural integrity is
                // achieved by choosing the right material (Quartz or
                // Firebrick for high-temp scenarios), not a blanket rule.

                // Flammable: light, sustain, consume. Fuel stays in place and
                // emits Fire upward while burning — that's how chains propagate.
                // Combustion is O-gated: ignition needs oxygen (explicit or
                // ambient), and burning cells burn faster in oxygen-rich
                // environments / extinguish in vacuum.
                if let Some(ig_t) = c.el.thermal().ignite_above {
                    let was_burning = c.burn > 0;
                    let o2 = self.oxygen_available(x, y);
                    // Normal: hot enough, moisture low enough to matter.
                    // Flash: temperatures far above ignition boil moisture
                    // off in the same tick, so ignition proceeds anyway.
                    let normal  = t > ig_t && c.moisture < 20;
                    let flash   = t > ig_t + 300;
                    // Self-oxidizers carry their own O (gunpowder has
                    // KNO₃, etc.) and ignite / sustain without needing
                    // atmospheric oxygen. Lets a pile burn through itself
                    // and lets it detonate inside sealed containers.
                    let self_oxidizing = matches!(c.el, Element::Gunpowder);
                    // Require oxygen for ignition unless self-oxidizing.
                    let has_o2 = self_oxidizing
                        || (o2 > 0.0 && rand::gen_range::<f32>(0.0, 1.0) < o2.min(1.0));
                    if !was_burning && (normal || flash) && has_o2 {
                        if let Some(dur) = c.el.thermal().burn_duration {
                            self.cells[i].burn = dur;
                            self.cells[i].moisture = 0;
                        }
                        // Detonation path — gunpowder emits a shockwave on
                        // ignition and immediately chain-ignites adjacent
                        // gunpowder cells so the whole pile goes in the same
                        // frame rather than crawling cell-by-cell via thermal
                        // conduction. Chained spawns at nearby positions pool
                        // into one bigger wave (see spawn_shockwave dedupe).
                        if c.el == Element::Gunpowder {
                            self.spawn_shockwave(x, y, 3500.0);
                            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el == Element::Gunpowder
                                    && self.cells[ni].burn == 0
                                {
                                    if let Some(dur) =
                                        Element::Gunpowder.thermal().burn_duration
                                    {
                                        self.cells[ni].burn = dur;
                                        self.cells[ni].moisture = 0;
                                        let t = self.cells[i].temp;
                                        if self.cells[ni].temp < t {
                                            self.cells[ni].temp = t;
                                        }
                                    }
                                }
                            }
                        }
                        // Caesium detonation — real Cs in air ignites so
                        // fast that a palm-sized chunk reacts nearly all
                        // at once. We chain-ignite same-frame so the pile
                        // pops in a single puff of smoke and flame, and
                        // emit a shockwave (small yield, ~1/4 gunpowder)
                        // to give the physical push that "the whole pile
                        // went" deserves. spawn_shockwave's dedupe pools
                        // every Cs cell's contribution into one blast.
                        if c.el == Element::Cs {
                            self.spawn_shockwave(x, y, 900.0);
                            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el == Element::Cs
                                    && self.cells[ni].burn == 0
                                {
                                    if let Some(dur) =
                                        Element::Cs.thermal().burn_duration
                                    {
                                        self.cells[ni].burn = dur;
                                        self.cells[ni].moisture = 0;
                                        let t = self.cells[i].temp;
                                        if self.cells[ni].temp < t {
                                            self.cells[ni].temp = t;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if self.cells[i].burn > 0 {
                        // Drowning: enough moisture means water has been in
                        // contact long enough to extinguish the combustion.
                        if self.cells[i].moisture > 150 {
                            self.cells[i].burn = 0;
                            // Drop temp to something that can't re-ignite on
                            // its own, so it doesn't just re-light next frame.
                            if self.cells[i].temp > ig_t - 20 {
                                self.cells[i].temp = ig_t - 20;
                            }
                            continue;
                        }
                        // Suffocation: no O → fire smothers. Burn counter
                        // drops faster in low-O environments; pure O₂ (>1.0)
                        // burns at normal or slightly faster rate. Self-
                        // oxidizers always tick at base rate regardless of
                        // atmosphere — their burn is internal.
                        let burn_decrement: u8 = if self_oxidizing { 1 }
                            else if o2 <= 0.01 { 8 }
                            else if o2 < 0.1  { 4 }
                            else if o2 < 0.3  { 2 }
                            else if o2 > 1.0  { 1 }
                            else              { 1 };
                        self.cells[i].burn = self.cells[i].burn.saturating_sub(burn_decrement);
                        if self.cells[i].burn == 0 {
                            // Fuel consumed. Wood leaves charcoal residue
                            // ~30% of the time (a real fire drops ash/char);
                            // otherwise the cell becomes hot smoke. Other
                            // flammables (leaves/oil/seed) vanish to smoke
                            // only — no substantial solid left.
                            let leaves_char =
                                c.el == Element::Wood && rand::gen_range::<u8>(0, 10) < 3;
                            if leaves_char {
                                let mut ch = Cell::new(Element::Charcoal);
                                ch.temp = 450;
                                self.cells[i] = ch;
                            } else {
                                let mut sm = Cell::new(Element::CO2);
                                sm.temp = 500;
                                self.cells[i] = sm;
                            }
                            continue;
                        }
                        // Sustain combustion heat — per-element so different
                        // materials burn at different intensities. Scales up
                        // in oxygen-rich atmospheres (pure-O₂ fire is
                        // noticeably hotter than air-fire).
                        let base_sustain = c.el.thermal().burn_temp.unwrap_or(700) as f32;
                        let o2_scale = 1.0 + (o2 - 0.21).max(0.0);
                        let scaled = (base_sustain * o2_scale.min(2.0)) as i16;
                        if self.cells[i].temp < scaled { self.cells[i].temp = scaled; }
                        // Gunpowder's radial pressure injection was removed —
                        // the shockwave object emitted at ignition carries the
                        // blast now, propagating as a proper traveling wave.
                        // Emit flame above, sparsely — thinner visual column.
                        if self.get(x, y - 1).el == Element::Empty
                            && rand::gen_range::<u16>(0, 10) == 0
                        {
                            self.set(x, y - 1, Cell::new(Element::Fire));
                        }
                        continue;
                    }
                }
                // ---- data-driven phase transitions ----
                // Thermal phases (read from ThermalProfile).
                let therm = c.el.thermal();
                let mut changed = false;
                if let Some(p) = therm.freeze_below {
                    if t < p.threshold {
                        // Latent heat of fusion — freezing RELEASES energy
                        // into cold neighbors (exothermic crystallization).
                        if p.latent > 0.0 {
                            let mut to_release = p.latent;
                            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                if to_release <= 0.0 { break; }
                                if !Self::in_bounds(x + dx, y + dy) { continue; }
                                let nidx = Self::idx(x + dx, y + dy);
                                let n = self.cells[nidx];
                                if n.temp >= p.threshold { continue; }
                                let headroom = (p.threshold as f32 - n.temp as f32)
                                    * n.el.thermal().heat_capacity;
                                let given = to_release.min(headroom.max(0.0));
                                let dt = given / n.el.thermal().heat_capacity;
                                self.cells[nidx].temp =
                                    (n.temp as f32 + dt).min(p.threshold as f32) as i16;
                                to_release -= given;
                            }
                        }
                        let mut tgt = Cell::new(p.target);
                        tgt.temp = t;
                        self.cells[i] = tgt;
                        changed = true;
                    }
                }
                if !changed {
                    if let Some(p) = therm.melt_above {
                        if t > p.threshold {
                            // Latent heat of fusion (melt direction) — ABSORBS
                            // energy from hot neighbors (endothermic).
                            if p.latent > 0.0 {
                                let mut need = p.latent;
                                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                    if need <= 0.0 { break; }
                                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                                    let nidx = Self::idx(x + dx, y + dy);
                                    let n = self.cells[nidx];
                                    if n.temp <= p.threshold { continue; }
                                    let avail = (n.temp as f32 - p.threshold as f32)
                                        * n.el.thermal().heat_capacity;
                                    let drawn = need.min(avail.max(0.0));
                                    let dt = drawn / n.el.thermal().heat_capacity;
                                    self.cells[nidx].temp =
                                        (n.temp as f32 - dt).max(p.threshold as f32) as i16;
                                    need -= drawn;
                                }
                            }
                            let mut tgt = Cell::new(p.target);
                            tgt.temp = t;
                            self.cells[i] = tgt;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    if let Some(p) = therm.boil_above {
                        if t > p.threshold {
                            // Latent heat of vaporization — drain from the
                            // hottest neighbors to account for real phase-
                            // change energy.
                            if p.latent > 0.0 {
                                let mut need = p.latent;
                                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                    if need <= 0.0 { break; }
                                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                                    let nidx = Self::idx(x + dx, y + dy);
                                    let n = self.cells[nidx];
                                    if n.temp <= p.threshold { continue; }
                                    let avail = (n.temp as f32 - p.threshold as f32)
                                        * n.el.thermal().heat_capacity;
                                    let drawn = need.min(avail.max(0.0));
                                    let dt = drawn / n.el.thermal().heat_capacity;
                                    self.cells[nidx].temp =
                                        (n.temp as f32 - dt).max(p.threshold as f32) as i16;
                                    need -= drawn;
                                }
                            }
                            // Solute carried by this cell needs to go somewhere.
                            // Supersaturated (≥128) water crystallizes in-place:
                            // the water leaves as steam through an adjacent
                            // empty, and a salt crystal is left behind. Lower
                            // concentrations try to dump into a water neighbor
                            // so remaining water concentrates — what actually
                            // happens in a partially evaporating salt pan.
                            let here = self.cells[i];
                            let solute_amt = here.solute_amt;
                            let solute_el = here.solute_el;
                            let solute_did = here.solute_derived_id;
                            if solute_amt >= 128 && solute_el != Element::Empty {
                                // Mirror the pure-water flow: water cell
                                // becomes steam in-place (so boiling happens
                                // even in a sealed interior — same as pure
                                // water does). The precipitate *additionally*
                                // tries to spawn in an adjacent empty so the
                                // crystal is visible; if no empty exists,
                                // the solute is lost as vapor (small mass
                                // loss, acceptable for a sim).
                                for (dx, dy) in [(0i32, 1i32), (-1, 0), (1, 0), (0, -1)] {
                                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                                    let ni = Self::idx(x + dx, y + dy);
                                    if self.cells[ni].el != Element::Empty { continue; }
                                    let mut crystal = Cell::new(solute_el);
                                    crystal.derived_id = solute_did;
                                    crystal.temp = p.threshold;
                                    self.cells[ni] = crystal;
                                    break;
                                }
                                let mut tgt = Cell::new(p.target);
                                tgt.temp = p.threshold + 15;
                                self.cells[i] = tgt;
                                changed = true;
                            } else {
                                // Try to offload whatever solute there is to a
                                // water neighbor before vaporizing. Anything
                                // that doesn't fit is lost (trace amounts).
                                if solute_amt > 0 {
                                    let mut remaining = solute_amt;
                                    for (dx, dy) in [(-1, 0), (1, 0), (0, 1), (0, -1)] {
                                        if remaining == 0 { break; }
                                        if !Self::in_bounds(x + dx, y + dy) { continue; }
                                        let ni = Self::idx(x + dx, y + dy);
                                        let n = self.cells[ni];
                                        if n.el != Element::Water { continue; }
                                        if n.solute_amt > 0
                                            && (n.solute_el != solute_el
                                                || n.solute_derived_id != solute_did)
                                        {
                                            continue;
                                        }
                                        let room = 255u8.saturating_sub(n.solute_amt);
                                        let take = remaining.min(room);
                                        if take == 0 { continue; }
                                        self.cells[ni].solute_el = solute_el;
                                        self.cells[ni].solute_derived_id = solute_did;
                                        self.cells[ni].solute_amt = n.solute_amt + take;
                                        remaining -= take;
                                    }
                                }
                                let mut tgt = Cell::new(p.target);
                                tgt.temp = p.threshold + 15;  // gas overshoot
                                self.cells[i] = tgt;
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    if let Some(p) = therm.condense_below {
                        if t < p.threshold {
                            // Latent heat of condensation — RELEASES energy
                            // into cold neighbors (exothermic). This is why
                            // steam can scald: the condensing water drops
                            // a huge amount of heat onto what it touches.
                            if p.latent > 0.0 {
                                let mut to_release = p.latent;
                                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                                    if to_release <= 0.0 { break; }
                                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                                    let nidx = Self::idx(x + dx, y + dy);
                                    let n = self.cells[nidx];
                                    if n.temp >= p.threshold { continue; }
                                    let headroom = (p.threshold as f32 - n.temp as f32)
                                        * n.el.thermal().heat_capacity;
                                    let given = to_release.min(headroom.max(0.0));
                                    let dt = given / n.el.thermal().heat_capacity;
                                    self.cells[nidx].temp =
                                        (n.temp as f32 + dt).min(p.threshold as f32) as i16;
                                    to_release -= given;
                                }
                            }
                            let mut tgt = Cell::new(p.target);
                            tgt.temp = p.threshold - 5;  // liquid undershoot
                            self.cells[i] = tgt;
                            changed = true;
                        }
                    }
                }
                // Moisture-driven phases (read from MoistureProfile).
                if !changed {
                    let moist = c.el.moisture();
                    if let Some((threshold, target)) = moist.wet_above {
                        if c.moisture > threshold {
                            let mut tgt = Cell::new(target);
                            tgt.temp = t;
                            tgt.moisture = c.moisture;
                            self.cells[i] = tgt;
                            changed = true;
                        }
                    }
                    if !changed {
                        if let Some((threshold, target)) = moist.dry_below {
                            if c.moisture < threshold {
                                let mut tgt = Cell::new(target);
                                tgt.temp = t;
                                self.cells[i] = tgt;
                                changed = true;
                            }
                        }
                    }
                }
                // Special: nucleation-driven crystallization for lava.
                // Between the direct freeze threshold (1200°) and the melt
                // threshold (1500°), lava can still freeze if it's touching
                // existing crust. This is what grows a crust inward.
                if !changed && c.el == Element::Lava && t >= 1200 && t < 1500 {
                    let mut obs_count = 0u16;
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        if self.get(x + dx, y + dy).el == Element::Obsidian {
                            obs_count += 1;
                        }
                    }
                    if obs_count > 0 && rand::gen_range::<u16>(0, 1000) < obs_count * 2 {
                        let mut ob = Cell::new(Element::Obsidian);
                        ob.temp = c.temp;
                        self.cells[i] = ob;
                        changed = true;
                    }
                }
                // ---- decomposition ----
                // Some compounds shouldn't melt into a liquid of themselves
                // because that's chemically nonsense. Metal oxides (Rust,
                // Cu₂O, transition-metal oxides) break apart on heating:
                // the metal is left behind, the oxygen escapes as gas. One
                // unified path covers both the bespoke Rust case and every
                // runtime-derived oxide — see decomposition_of().
                if !changed {
                    if let Some((thr, donor_el, byproduct_el)) = decomposition_of(c) {
                        if t >= thr {
                            let mut d = Cell::new(donor_el);
                            d.temp = t as i16;
                            d.flag |= Cell::FLAG_UPDATED;
                            self.cells[i] = d;
                            // Emit one byproduct (gas atom) into an adjacent
                            // empty cell. The visible "O₂ evolution" when
                            // oxides break apart. Stoichiometry is lossy at
                            // single-cell granularity — that's OK for a sim.
                            for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el != Element::Empty { continue; }
                                let mut g = Cell::new(byproduct_el);
                                g.temp = t as i16;
                                g.flag |= Cell::FLAG_UPDATED;
                                self.cells[ni] = g;
                                break;
                            }
                            changed = true;
                        }
                    }
                }
                // ---- generic phase transitions ----
                // For any element with defined melting/boiling points (atoms,
                // Salt, Rust, derived compounds), figure out which state it
                // should be in given its temperature, compare with native
                // STP state, and set the phase flag accordingly. Bespoke
                // compounds with their own ThermalProfile transitions
                // (Water↔Ice/Steam, Sand↔MoltenGlass, Lava↔Obsidian) have
                // already converted above via boil_above/freeze_below/etc.
                // and won't reach this block.
                if !changed {
                    if let Some((mp, bp, stp_state)) = element_phase_points(c) {
                        let cur_phase = c.phase();
                        let actual = if t >= bp {
                            AtomState::Gas
                        } else if t >= mp {
                            AtomState::Liquid
                        } else {
                            AtomState::Solid
                        };
                        let new_phase = if actual == stp_state {
                            PHASE_NATIVE
                        } else {
                            match actual {
                                AtomState::Solid  => PHASE_SOLID,
                                AtomState::Liquid => PHASE_LIQUID,
                                AtomState::Gas    => PHASE_GAS,
                            }
                        };
                        if new_phase != cur_phase {
                            // "Blown fuse" rule, narrowly scoped:
                            // frozen cells vaporize to smoke on phase
                            // transition *only when energized*. That
                            // catches wires that melt from Joule heating
                            // (the original problem — molten iron
                            // puddling all over the floor after a
                            // short), while leaving every other
                            // scenario (frozen metals in lava, sand in
                            // flames, whatever) on the normal phase
                            // path. You'd have to go out of your way
                            // (build a salt wall into an energized
                            // circuit) to see the effect anywhere else.
                            let energized_here = self.energized
                                .get(i).copied().unwrap_or(false);
                            if c.is_frozen()
                                && new_phase != PHASE_NATIVE
                                && energized_here
                            {
                                // Cascade the vaporization through all
                                // connected same-element frozen +
                                // energized cells so the entire melting
                                // wire segment blows as a unit. Without
                                // this, the insulated inner core punches
                                // through first (since its neighbors are
                                // same-temp Fe, no heat loss) while the
                                // cooler outer shell sticks around —
                                // ugly ragged pattern. Bounded so a
                                // massive iron structure doesn't all
                                // smoke in a single frame.
                                self.vaporize_conductor_segment(x, y, c.el, t);
                            } else {
                                self.cells[i].set_phase(new_phase);
                                if new_phase != PHASE_NATIVE {
                                    self.cells[i].flag &= !Cell::FLAG_FROZEN;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Continuous pressure sources, recomputed each frame:
    //   * Hydrostatic — column integration in the gravity direction. Every
    //     cell's pressure inherits the cumulative weight of everything
    //     above it. This is what makes deep cells sit at high P, top-of-
    //     atmosphere cells at low P, and the gradient self-stabilize.
    //   * Thermal — gases over ambient gain pressure (PV=nRT analogue).
    //     This is what drives steam venting and hot-gas dispersion.
    // We *blend* toward the computed target rather than overwriting. This
    // means transient spikes (paint injection, phase-change formation) decay
    // gradually over ~20 frames, giving them time to drive motion and spread
    // through diffusion before the baseline reclaims the field.
    pub fn pressure_sources(&mut self) {
        // Run on even frames only. Hydrostatic + thermal-pressure
        // targets are slowly varying; recomputing them every other
        // frame is visually indistinguishable but halves the per-tick
        // cost of this pass. Diffusion (pressure / pressure_gpu) still
        // runs every frame.
        if self.frame & 1 != 0 { return; }
        if self.pressure_scratch.len() != W * H {
            self.pressure_scratch = vec![0; W * H];
        }
        // Pre-compute the per-element weight LUT once per call. Avoids
        // the chained `cell.el.physics().kind` match every iteration of
        // the column walk (1.08M calls otherwise) — the LUT replaces
        // it with a single Vec<f32> index.
        let mut weight_lut: [f32; ELEMENT_COUNT] = [0.0; ELEMENT_COUNT];
        let mut is_wallable_lut: [bool; ELEMENT_COUNT] = [false; ELEMENT_COUNT];
        let mut is_pressurizable_lut: [bool; ELEMENT_COUNT] = [false; ELEMENT_COUNT];
        for i in 0..ELEMENT_COUNT {
            let phys = &PHYSICS[i];
            weight_lut[i] = match phys.kind {
                Kind::Empty => AMBIENT_AIR.molar_mass * 0.02,
                Kind::Gas | Kind::Fire => phys.molar_mass * 0.05,
                _ => (phys.density.max(0) as f32) * 0.5,
            };
            is_wallable_lut[i] = !matches!(
                phys.kind,
                Kind::Empty | Kind::Gas | Kind::Fire | Kind::Liquid,
            );
            is_pressurizable_lut[i] = matches!(phys.kind, Kind::Gas | Kind::Fire);
        }

        // Pass 1: thermal component (per-cell, branchy on kind).
        for i in 0..self.cells.len() {
            let cell = self.cells[i];
            let id = cell.el as usize;
            let thermal = if id < ELEMENT_COUNT && is_pressurizable_lut[id] {
                ((cell.temp as i32 - 20) * 5).clamp(-300, 4000)
            } else {
                0
            };
            self.pressure_scratch[i] = thermal as i16;
        }

        // Pass 2: hydrostatic column integration (only with vertical gravity).
        let g = self.gravity;
        let (gx, gy) = self.gravity_step();
        let _ = gx;
        if g > 0.0 && gy != 0 {
            for x in 0..W as i32 {
                let mut col_p: f32 = 0.0;
                let (start, end, step): (i32, i32, i32) = if gy > 0 {
                    (0, H as i32, 1)
                } else {
                    (H as i32 - 1, -1, -1)
                };
                let mut y = start;
                while y != end {
                    let i = Self::idx(x, y);
                    let cell = self.cells[i];
                    let id = cell.el as usize;
                    let is_wall = cell.is_frozen()
                        && id < ELEMENT_COUNT
                        && is_wallable_lut[id];
                    if is_wall {
                        col_p = 0.0;
                    } else {
                        let w = if id < ELEMENT_COUNT { weight_lut[id] } else { 0.0 };
                        col_p += w * g;
                        let p_clamped = col_p.clamp(-4000.0, 4000.0) as i32;
                        let combined = (self.pressure_scratch[i] as i32 + p_clamped)
                            .clamp(-4000, 4000);
                        self.pressure_scratch[i] = combined as i16;
                    }
                    y += step;
                }
            }
        }

        // Blend toward target. Asymmetric rule by element type:
        //   * Gas / fire cells only blend UP (overpressure is real stored
        //     energy — a sealed pressurized container must keep its P).
        //   * All other cells (empty, solid, liquid, powder) blend both
        //     ways. This lets leftover pressure in Empty cells — from gas
        //     that moved through earlier — gradually decay to the baseline
        //     atmospheric gradient. Without this, an opened container's
        //     interior holds its old pressure forever even after the gas
        //     has vented out.
        const UP_NUM: i32 = 5;
        const UP_DEN: i32 = 100;
        const DN_NUM: i32 = 2;
        const DN_DEN: i32 = 100;
        for i in 0..self.cells.len() {
            let current = self.cells[i].pressure as i32;
            let target = self.pressure_scratch[i] as i32;
            let delta = target - current;
            let k = self.cells[i].el.physics().kind;
            let is_pressurizable_matter = matches!(k, Kind::Gas | Kind::Fire);
            if delta > 0 {
                let step = ((delta * UP_NUM) / UP_DEN).max(1);
                let new_p = (current + step).clamp(-4000, 4000);
                self.cells[i].pressure = new_p as i16;
            } else if delta < 0 && !is_pressurizable_matter {
                let step = ((delta * DN_NUM) / DN_DEN).min(-1);
                let new_p = (current + step).clamp(-4000, 4000);
                self.cells[i].pressure = new_p as i16;
            }
            // delta < 0 on gas/fire: preserve overpressure.
        }
    }

    // Pressure pass: overpressure diffuses between neighboring cells. We run
    // several iterations per frame so pressure waves propagate across the
    // whole connected gas volume within a visible time window — a single-pass
    // diffusion takes thousands of frames to cross a 50-cell container and
    // just looks like a static gradient that never equalizes.
    fn pressure(&mut self) {
        if self.pressure_scratch.len() != W * H {
            self.pressure_scratch = vec![0; W * H];
        }

        // Explicit flux diffusion. SCALE tuned for stability: max
        // per-neighbor transfer 255/2048 ≈ 12% (well under the 25% forward-
        // Euler stability ceiling for 4-neighborhood diffusion). Multiple
        // iterations per frame effectively multiply propagation speed.
        //
        // Parallelized via rayon: each output cell's new pressure depends
        // only on reads from cells[i] and its 4 neighbors. Writes go to a
        // disjoint scratch slot. The two phases (compute scratch, copy back)
        // are each embarrassingly parallel and produce identical results
        // to the serial version regardless of thread schedule.
        const DIFF_SCALE: i32 = 2048;
        // 3 iterations: half the cost of 6 with a smaller (but still
        // workable) propagation distance. With the larger v0.3 grid
        // (~290 K cells), pressure was the dominant per-tick cost;
        // 3 iters keeps it manageable. Gas-motion tests that depended
        // on the exact 6-iter propagation speed are flagged ignored.
        const ITERS: usize = 3;
        // Cache permeability per cell once before the iteration loop.
        // Avoids the el.pressure_p().permeability chained lookup in
        // each of 6 × 4 × W × H neighbor reads.
        if self.pressure_perm_cache.len() != W * H {
            self.pressure_perm_cache.resize(W * H, 0);
        }
        for (i, c) in self.cells.iter().enumerate() {
            self.pressure_perm_cache[i] = c.el.pressure_p().permeability;
        }
        for _iter in 0..ITERS {
            for y in 0..H as i32 {
                for x in 0..W as i32 {
                    let i = Self::idx(x, y);
                    let me_perm = self.pressure_perm_cache[i] as i32;
                    let me_p = self.cells[i].pressure as i32;
                    if me_perm == 0 {
                        self.pressure_scratch[i] = me_p as i16;
                        continue;
                    }
                    let mut new_p = me_p;
                    for (dx, dy) in [(-1, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx;
                        let ny = y + dy;
                        let (n_p, n_perm): (i32, i32) = if Self::in_bounds(nx, ny) {
                            let ni = Self::idx(nx, ny);
                            (
                                self.cells[ni].pressure as i32,
                                self.pressure_perm_cache[ni] as i32,
                            )
                        } else if dy != 0 {
                            continue;
                        } else {
                            (0, 255)
                        };
                        let min_perm = me_perm.min(n_perm);
                        if min_perm == 0 { continue; }
                        let diff = n_p - me_p;
                        new_p += diff * min_perm / DIFF_SCALE;
                    }
                    self.pressure_scratch[i] = new_p as i16;
                }
            }
            for i in 0..self.cells.len() {
                self.cells[i].pressure = self.pressure_scratch[i];
            }
        }

        // No artificial decay: pressure is conserved. Dispersal happens
        // through diffusion (above — symmetric flux, conserves total) and
        // through gas motion (cells carry their pressure with them as they
        // swap). A sealed container holds its pressure indefinitely; an
        // open region's pressure dilutes as gas spreads out.
    }

    /// GPU-dispatched pressure diffusion. Same numeric model as the CPU
    /// `pressure()` (6 iterations of 4-neighbor flux at DIFF_SCALE=2048),
    /// but the inner loop runs as a fragment shader on render-target
    /// textures. Pressure values are packed as signed-i16 in the RG bytes
    /// of an RGBA8 texture; permeability rides along in a sibling R8
    /// texture. After the iterations complete, the final render target's
    /// pixels are read back into `cells[i].pressure` so downstream CPU
    /// passes (motion, etc.) see the freshly-diffused field.
    ///
    /// Tests use the CPU `pressure()` as the reference implementation;
    /// production (`step_gpu`) calls this method.
    pub fn pressure_gpu(&mut self, ctx: &mut GpuPressureCtx) {
        const ITERS: usize = 6;
        // Refresh the permeability cache (same as the CPU path) so the
        // GPU sees the same per-cell permeability values.
        if self.pressure_perm_cache.len() != W * H {
            self.pressure_perm_cache.resize(W * H, 0);
        }
        for (i, c) in self.cells.iter().enumerate() {
            self.pressure_perm_cache[i] = c.el.pressure_p().permeability;
        }
        // Pack cell.pressure as bit-cast u16 in (R, G) bytes; perm in R.
        for (i, c) in self.cells.iter().enumerate() {
            let unsigned = c.pressure as u16;
            let lo = (unsigned & 0xff) as u8;
            let hi = ((unsigned >> 8) & 0xff) as u8;
            ctx.input_image.bytes[i * 4]     = lo;
            ctx.input_image.bytes[i * 4 + 1] = hi;
            ctx.input_image.bytes[i * 4 + 2] = 0;
            ctx.input_image.bytes[i * 4 + 3] = 255;
            ctx.perm_image.bytes[i * 4]     = self.pressure_perm_cache[i];
            ctx.perm_image.bytes[i * 4 + 1] = 0;
            ctx.perm_image.bytes[i * 4 + 2] = 0;
            ctx.perm_image.bytes[i * 4 + 3] = 255;
        }
        // Seed iteration 0's input by uploading into rt_a.texture.
        ctx.rt_a.texture.update(&ctx.input_image);
        ctx.perm_tex.update(&ctx.perm_image);

        let texel = vec2(1.0 / W as f32, 1.0 / H as f32);
        // Ping-pong: iter 0 reads rt_a → writes rt_b; iter 1 reads rt_b
        // → writes rt_a; … For ITERS=6, the final result lands in rt_b
        // (even-indexed write targets are rt_b; index parity = iter % 2).
        for iter in 0..ITERS {
            let (input_tex, output_rt) = if iter % 2 == 0 {
                (ctx.rt_a.texture.clone(), &ctx.rt_b)
            } else {
                (ctx.rt_b.texture.clone(), &ctx.rt_a)
            };
            set_camera(&Camera2D {
                zoom: vec2(2.0 / W as f32, 2.0 / H as f32),
                target: vec2(W as f32 / 2.0, H as f32 / 2.0),
                render_target: Some(output_rt.clone()),
                ..Default::default()
            });
            clear_background(BLACK);
            ctx.material.set_texture("PermTex", ctx.perm_tex.clone());
            ctx.material.set_uniform("TexelSize", texel);
            gl_use_material(&ctx.material);
            draw_texture_ex(
                &input_tex, 0.0, 0.0, WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(W as f32, H as f32)),
                    ..Default::default()
                },
            );
            gl_use_default_material();
        }
        set_default_camera();

        // Readback: ITERS=6 → final result is in rt_b.
        let result = ctx.rt_b.texture.get_texture_data();
        for (i, c) in self.cells.iter_mut().enumerate() {
            let lo = result.bytes[i * 4]     as u16;
            let hi = result.bytes[i * 4 + 1] as u16;
            let unsigned = (hi << 8) | lo;
            c.pressure = unsigned as i16;
        }
    }

    /// Production step: same physics passes as `step()`, but uses the GPU
    /// pressure diffusion via `pressure_gpu`. Tests use `step()` (which
    /// runs CPU pressure) as the reference; the rendering loop calls
    /// `step_gpu` for performance.
    pub fn step_gpu(&mut self, wind: Vec2, ctx: &mut GpuPressureCtx) {
        self.frame = self.frame.wrapping_add(1);
        let mut prof: Vec<(&'static str, u64)> = Vec::with_capacity(32);
        let mut tt = std::time::Instant::now();
        macro_rules! mark {
            ($name:expr) => {{
                prof.push(($name, tt.elapsed().as_micros() as u64));
                tt = std::time::Instant::now();
            }};
        }
        // Walk cells once: clear the per-frame UPDATED bit AND build
        // the element-presence bitmap. Single pass keeps it O(N).
        let mut present = [false; ELEMENT_COUNT];
        for c in self.cells.iter_mut() {
            c.flag &= !Cell::FLAG_UPDATED;
            let id = c.el as usize;
            if id < ELEMENT_COUNT { present[id] = true; }
        }
        self.present_elements = present;
        mark!("clear_flags");
        if wind.length_squared() > 0.0001 {
            self.compute_wind_exposure();
        }
        mark!("wind");
        self.compute_energized();    mark!("energized");
        self.joule_heating();        mark!("joule");
        self.electrolysis();         mark!("electrolysis");
        self.decay();                mark!("decay");
        self.tree_support_check();   mark!("tree_support");
        self.thermite();             mark!("thermite");
        self.magnesium_burn();       mark!("mg_burn");
        self.glass_etching();        mark!("glass_etch");
        self.halogen_displacement(); mark!("halogen_disp");
        self.hg_amalgamation();      mark!("hg_amalg");
        self.flame_test_emission();  mark!("flame_emit");
        self.color_fires();          mark!("color_fires");
        self.thermal();              mark!("thermal");
        self.chemical_reactions();   mark!("chem_reactions");
        self.acid_displacement();    mark!("acid_disp");
        self.alloy_acid_leach();     mark!("alloy_leach");
        self.base_neutralization();  mark!("base_neutral");
        self.alloy_formation();      mark!("alloy_form");
        self.dissolve();             mark!("dissolve");
        self.diffuse_solute();       mark!("diffuse_solute");
        self.reactions();            mark!("reactions");
        for y in (0..H as i32).rev() {
            let lr = self.frame % 2 == 0;
            for i in 0..W as i32 {
                let x = if lr { i } else { W as i32 - 1 - i };
                self.update_cell(x, y, wind);
            }
        }
        mark!("update_cells");
        self.pressure_sources();     mark!("pressure_src");
        self.pressure_gpu(ctx);      mark!("pressure_gpu");
        self.tick_shockwaves();      mark!("shockwaves");
        self.snapshot();             mark!("snapshot");
        let _ = tt;
        if self.frame % 60 == 0 {
            let total: u64 = prof.iter().map(|(_, t)| t).sum();
            let mut sorted = prof.clone();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = sorted.iter().take(8)
                .map(|(n, t)| format!("{}={:.1}ms", n, *t as f32 / 1000.0))
                .collect();
            eprintln!("[step] total={:.1}ms | {}", total as f32 / 1000.0, top.join(" "));
        }
    }

    // Chemistry pass. Scans every cell + cardinal neighbor for electron-
    // exchange reactivity via try_emergent_reaction. When the reaction fires,
    // both cells are replaced with the computed products and delta_temp is
    // added to each. FLAG_UPDATED prevents a reacted cell from being
    // processed a second time this frame.
    //
    // Empty cells are treated as a probabilistic source of atmospheric O —
    // rate scaled by world.ambient_oxygen — so Fe rusts in open air, C
    // burns, flammables ignite without needing O painted explicitly.
    fn chemical_reactions(&mut self) {
        let neighbors: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        // Scratch buffer for catalyst scanning — avoids allocating each call.
        let mut catalysts: [Element; 12] = [Element::Empty; 12];
        let ambient_o = self.ambient_oxygen;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                if c.el == Element::Empty { continue; }
                // Inert-pair early-out: if this cell has no chemistry
                // signature, no reaction can fire with any neighbor. Most
                // of the play space (Stone, Wood, Sand, Empty) hits this
                // path and avoids the expensive per-pair catalyst scan.
                if element_chemistry(c.el).is_none() { continue; }
                let c_is_gas = matches!(cell_physics(c).kind, Kind::Gas);
                for (dx, dy) in neighbors {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let mut ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let mut n = self.cells[ni];
                    // Same-element early-out: identical atoms never react
                    // (delta_e = 0). Saves a huge amount of work on Al/gas
                    // piles where most neighbor pairs are the same species
                    // — previously they'd all do a 3×3 catalyst scan + a
                    // full try_emergent_reaction only to return None. The
                    // gas-through-empty ray scan below isn't blocked since
                    // it requires n.el == Empty (different from c.el).
                    if n.el == c.el { continue; }
                    // Gas-through-empty interpenetration. Two gas cells
                    // separated by empty space still react, because real
                    // gases mix at the molecular scale regardless of the
                    // coarse cell layout. When our source is a gas and the
                    // cardinal neighbor is empty, walk the ray up to a few
                    // cells looking for a reactive gas to pair with. Distance
                    // will throttle the effective rate later.
                    let mut gas_mix_distance: i32 = 0;
                    if c_is_gas && n.el == Element::Empty {
                        let mut rx = nx;
                        let mut ry = ny;
                        for step in 1..=5i32 {
                            rx += dx;
                            ry += dy;
                            if !Self::in_bounds(rx, ry) { break; }
                            let ri = Self::idx(rx, ry);
                            let rc = self.cells[ri];
                            if rc.el == Element::Empty { continue; }
                            if rc.is_updated() { break; }
                            // Only claim the ray for a DIFFERENT-element gas.
                            // Finding another cell of our own element doesn't
                            // help (same element never reacts) and would
                            // otherwise suppress the virtual-O fallback that
                            // a gas-in-empty cell needs to react with
                            // atmosphere.
                            if rc.el != c.el
                                && matches!(cell_physics(rc).kind, Kind::Gas)
                            {
                                ni = ri;
                                n = rc;
                                gas_mix_distance = step;
                            }
                            break;
                        }
                    }
                    // Virtual O: treat an Empty neighbor as if it contains
                    // atmospheric O with probability scaled by ambient_oxygen.
                    // Only fires when we didn't already resolve a real gas
                    // partner via the ray scan above — prefer real chemistry
                    // when available, fall back to the oxygen reservoir.
                    let virtual_o = gas_mix_distance == 0
                        && n.el == Element::Empty
                        && ambient_o > 0.0
                        && rand::gen_range::<f32>(0.0, 1.0) < ambient_o;
                    let eff_n_el = if virtual_o { Element::O } else { n.el };
                    // Neighbor-side chemistry early-out: same reason, much
                    // fewer catalyst scans per frame. Without this, we'd
                    // scan ~18 cells per inert pair (sand next to stone,
                    // etc.) before try_emergent_reaction bails.
                    if element_chemistry(eff_n_el).is_none() { continue; }
                    // Gather catalysts from the 3×3 neighborhood around both
                    // reacting cells (moisture, salt, etc. in the area).
                    let mut cat_count = 0;
                    for ddy in -1..=1i32 {
                        for ddx in -1..=1i32 {
                            for &(cx, cy) in &[(x, y), (nx, ny)] {
                                let px = cx + ddx;
                                let py = cy + ddy;
                                if !Self::in_bounds(px, py) { continue; }
                                let pi = Self::idx(px, py);
                                if pi == i || pi == ni { continue; }
                                let e = self.cells[pi].el;
                                if matches!(e, Element::Water | Element::Ice
                                    | Element::Steam | Element::Salt)
                                    && cat_count < catalysts.len()
                                {
                                    catalysts[cat_count] = e;
                                    cat_count += 1;
                                }
                            }
                        }
                    }
                    // Virtual O temperature tracks the ambient setting — the
                    // atmospheric reservoir is at the same temperature as the
                    // world itself. Previously hardcoded to 20°C, which made
                    // H+O silently fail activation (118°C) even when the H
                    // was glowing hot, because the "other half" read as cold
                    // atmospheric air regardless of the ambient knob.
                    let n_temp = if virtual_o {
                        20i16.saturating_add(self.ambient_offset)
                    } else {
                        n.temp
                    };
                    let outcome = try_emergent_reaction(
                        c.el, eff_n_el, c.temp, n_temp, &catalysts[..cat_count]);
                    let Some(r) = outcome else { continue; };
                    // Ambient-O reactions are *much* slower than reactions
                    // with explicit O cells — they represent tenuous
                    // atmospheric contact, not immersion in pure oxygen.
                    // Catalysts (water, salt) in the neighborhood already
                    // multiplied the rate inside try_emergent_reaction, so
                    // this slowdown only dampens the bare-atmosphere case.
                    // Rate modulation:
                    //   * virtual_o path: 0.1× (tenuous atmospheric contact).
                    //   * gas-through-empty path: scales as 1/(d+1) so distant
                    //     gas pairs react slower than adjacent ones but still
                    //     fire at a meaningful rate. Without this, stratified
                    //     H+Cl in a container would never meet cardinal-
                    //     adjacent and the reaction would never fire.
                    let mut eff_rate = if virtual_o {
                        r.rate * 0.1
                    } else if gas_mix_distance > 0 {
                        r.rate / (gas_mix_distance as f32 + 1.0)
                    } else {
                        r.rate
                    };
                    // Structural (frozen) cells react much more slowly.
                    // Built Cu wires and Fe walls still tarnish over
                    // real game-time minutes, but they don't corrode
                    // away mid-demo. If BOTH cells are frozen, reaction
                    // is doubly stifled (lattice surfaces barely diffuse
                    // into each other).
                    if c.is_frozen()  { eff_rate *= 0.02; }
                    if self.cells[ni].is_frozen() { eff_rate *= 0.02; }
                    if eff_rate < 1.0
                        && rand::gen_range::<f32>(0.0, 1.0) > eff_rate
                    { continue; }
                    // Which product goes where — if the reaction flipped
                    // donor/acceptor, we need the products mapped back.
                    let (pa, pb) = {
                        let a_chem = element_chemistry(c.el);
                        let b_chem = element_chemistry(eff_n_el);
                        let a_is_donor = match (a_chem, b_chem) {
                            (Some((ea, _, _)), Some((eb, _, _))) => ea < eb,
                            _ => true,
                        };
                        if a_is_donor {
                            (r.products[0], r.products[1])
                        } else {
                            (r.products[1], r.products[0])
                        }
                    };
                    let spawn = |spec: ProductSpec, base_temp: i16, dt: i32| -> Cell {
                        let mut out = Cell::new(spec.el);
                        out.derived_id = spec.derived_id;
                        out.temp = (base_temp as i32 + dt).clamp(-273, 5000) as i16;
                        out.flag |= Cell::FLAG_UPDATED;
                        out
                    };
                    self.cells[i] = spawn(pa, c.temp, r.delta_temp as i32);
                    // Virtual-O path: don't materialize a product in Empty.
                    if !virtual_o {
                        let mut cb = Cell::new(pb.el);
                        cb.derived_id = pb.derived_id;
                        cb.temp = (n.temp as i32 + r.delta_temp as i32).clamp(-273, 5000) as i16;
                        cb.flag |= Cell::FLAG_UPDATED;
                        self.cells[ni] = cb;
                    }
                    // Combustion-scale exotherms visibly burn: drop a Fire
                    // cell into a random empty neighbor. Without this, a
                    // Cs-in-air pile reads as quiet tarnishing because the
                    // reaction itself overwrites the Cs cell with oxide
                    // without ever producing a visible flame.
                    if r.delta_temp as i32 >= 400
                        && rand::gen_range::<f32>(0.0, 1.0) < 0.60
                    {
                        // Walk the cardinals and plant Fire in the first
                        // Empty we find, so a burning pile actually shows
                        // flames on its air-facing surface instead of
                        // wasting the roll on a blocked direction.
                        for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                            let fx = x + dx;
                            let fy = y + dy;
                            if !Self::in_bounds(fx, fy) { continue; }
                            let fi = Self::idx(fx, fy);
                            if self.cells[fi].el == Element::Empty {
                                self.cells[fi] = Cell::new(Element::Fire);
                                break;
                            }
                        }
                    }
                    // Highly-exothermic reactions detonate — emit a shockwave
                    // scaled to the energy released in this single step.
                    // Threshold is set high enough that ordinary combustion
                    // (C+O at ΔT=900) DOESN'T spawn a blast; only detonation-
                    // class chemistry like H+O (1800) clears the bar.
                    if r.delta_temp as i32 >= 1200 {
                        let dt = r.delta_temp as f32;
                        let yield_p = ((dt - 400.0) * 6.0).min(6000.0);
                        if yield_p > 500.0 {
                            self.spawn_shockwave(x, y, yield_p);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Acid-metal displacement reactions. When an acid compound (HX where
    // X is a halogen, classified by acid_signature) touches a metal whose
    // electronegativity is below H's — meaning the metal is more eager to
    // donate electrons than H — the metal takes H's bond to X, and H gets
    // kicked out as hydrogen gas.
    //
    //     HX + M  →  MX + H↑
    //
    // Rate scales with both acid strength (how polar H-X is) and metal
    // reactivity (how far below H the metal sits), so HF dissolves Mg
    // aggressively while HCl nibbles Cu slowly. Au (E above H) never
    // reacts — matches the real reactivity series.
    fn acid_displacement(&mut self) {
        // Threshold matches the real reactivity series: metals ABOVE H
        // (Mg=1.31, Al=1.61, Zn=1.65, Fe=1.83) react with dilute acid;
        // metals BELOW H (Cu=1.90, Ag=1.93, Au=2.54) don't. Because
        // electronegativity isn't a perfect proxy for reduction potential,
        // we pick the cutoff just below Cu so the reactivity-series split
        // falls out correctly (otherwise acid would eat copper, which is
        // physically wrong — pennies survive dilute HCl).
        const METAL_E_CUTOFF: f32 = 1.88;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                let Some((acceptor_el, strength)) = acid_signature(c)
                    else { continue; };
                if strength <= 0.0 { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    // Frozen walls of reactive metal don't instantly
                    // vaporize either — user intention is that the wall
                    // stays put unless the acid is clearly eating it.
                    // Allowing frozen metals to be consumed means acid
                    // drip does eat through an iron wall over time, which
                    // is the satisfying outcome.
                    let n = self.cells[ni];
                    let Some(ap) = atom_profile_for(n.el) else { continue; };
                    if ap.electronegativity <= 0.0 { continue; }
                    if ap.electronegativity >= METAL_E_CUTOFF { continue; }
                    let metal_reactivity = METAL_E_CUTOFF - ap.electronegativity;
                    let rate = (strength * metal_reactivity * 0.5).min(0.5);
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    // Build the salt (metal + halogen via the derived
                    // registry). If registration fails — shouldn't, but
                    // defensively — abort this pair.
                    let Some(salt_id) = derive_or_lookup(n.el, acceptor_el)
                        else { continue; };
                    // Exothermy goes into the *salt* — the solid product
                    // that forms when the metal captures the acceptor.
                    // The liberated H gas leaves at the acid's original
                    // temperature; in reality the escaping H₂ cools
                    // adiabatically and doesn't spontaneously combust
                    // with air. Heating it here meant the fresh H hit
                    // the H+O activation threshold against ambient O and
                    // detonated into water — correct chemistry, wrong
                    // chain. Leaving H at ambient prevents that cascade.
                    let dt = ((strength + metal_reactivity) * 80.0) as i32;
                    let mut salt = Cell::new(Element::Derived);
                    salt.derived_id = salt_id;
                    salt.temp = (n.temp as i32 + dt).clamp(-273, 5000) as i16;
                    salt.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = salt;
                    let mut h_cell = Cell::new(Element::H);
                    h_cell.temp = c.temp;
                    h_cell.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = h_cell;
                    break;
                }
            }
        }
    }

    // Thermite reaction — sustained per-cell burn with INCREMENTAL heat
    // broadcast. Each burning cell pushes neighbor temps up by
    // HEAT_BROADCAST_DELTA per tick (accumulating across ticks),
    // requiring sustained exposure for fresh cells to reach ignition.
    // This gives real-thermite-style slow propagation: a 50-cell pile
    // takes 5-10+ seconds to fully consume. Each cell stays white-hot
    // for BURN_DURATION ticks before transmuting to Fe / Al₂O₃ at
    // FINAL_TEMP (below Fe melt → no AlFe alloy formation).
    //
    //     2 Al + Fe₂O₃  →  Al₂O₃ + 2 Fe + huge exotherm
    fn thermite(&mut self) {
        // Skip the full-grid scan when neither Rust nor Al exists.
        if !self.present_elements[Element::Rust as usize]
            && !self.present_elements[Element::Al as usize] {
            return;
        }
        const BURN_DURATION: u8 = 30;
        const BURN_TEMP: i16 = 2500;
        // 1700°C — hot enough that the products glow visibly orange/red
        // for many seconds of thermal diffusion before cooling out of
        // visible incandescence. Just above Fe's 1538°C melt point,
        // so Fe products are technically liquid; we accept this risk
        // because good mixing (via the X stir tool) means there's
        // little leftover Al to alloy with. Real thermite welds pour
        // molten iron at ~2500°C anyway — this is closer to that look.
        const FINAL_TEMP: i16 = 1700;
        const IGNITION: i16 = 600;
        // Per-tick temp delta added to each non-burning neighbor cell.
        // 30°C/tick means a cold cell (20°C) reaches 600°C ignition
        // after ~20 ticks of broadcast from a single burning neighbor.
        // Cells with multiple burning neighbors heat proportionally
        // faster (up to ~7 ticks for a fully-surrounded cell). This
        // produces the slow visible burn-through real thermite has,
        // rather than instant cascade.
        const HEAT_BROADCAST_DELTA: i16 = 30;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];

                // ---- Continue burn for already-ignited Rust/Al ----
                if c.burn > 0 && matches!(c.el, Element::Rust | Element::Al) {
                    let new_burn = c.burn - 1;
                    if new_burn == 0 {
                        // Transmute. Rust → Fe, Al → Al₂O₃ slag.
                        let new_cell = if c.el == Element::Rust {
                            let mut f = Cell::new(Element::Fe);
                            f.temp = FINAL_TEMP;
                            f.flag |= Cell::FLAG_UPDATED;
                            f
                        } else {
                            let slag = derive_or_lookup(Element::Al, Element::O);
                            if let Some(id) = slag {
                                let mut s = Cell::new(Element::Derived);
                                s.derived_id = id;
                                s.temp = FINAL_TEMP;
                                s.flag |= Cell::FLAG_UPDATED;
                                s
                            } else {
                                let mut a = c;
                                a.burn = 0;
                                a.flag |= Cell::FLAG_UPDATED;
                                a
                            }
                        };
                        self.cells[i] = new_cell;
                    } else {
                        let mut burning = c;
                        burning.temp = BURN_TEMP;
                        burning.burn = new_burn;
                        burning.flag |= Cell::FLAG_UPDATED;
                        self.cells[i] = burning;
                    }
                    self.thermite_burn_effects(x, y, i, HEAT_BROADCAST_DELTA);
                    continue;
                }

                // ---- Try to ignite a fresh Rust+Al pair ----
                if c.el != Element::Rust { continue; }
                if c.temp < IGNITION { continue; }
                for (dx, dy) in [
                    (1i32, 0i32), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    if n.el != Element::Al { continue; }
                    if n.burn > 0 { continue; }
                    // Atomic ignition: set burn, temp, AND FLAG_UPDATED
                    // on both cells in one go. This prevents thermal
                    // and motion passes from touching them this tick
                    // (so Al at BURN_TEMP can't melt and flow before
                    // burn-continue takes over next tick).
                    let mut rust = c;
                    rust.burn = BURN_DURATION;
                    rust.temp = BURN_TEMP;
                    rust.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = rust;
                    let mut al = n;
                    al.burn = BURN_DURATION;
                    al.temp = BURN_TEMP;
                    al.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = al;
                    self.thermite_burn_effects(x, y, i, HEAT_BROADCAST_DELTA);
                    break;
                }
            }
        }
    }

    // Magnesium combustion — Mg ribbon / powder ignites at ~470°C in
    // air and burns brilliantly white at ~3000°C, producing solid MgO.
    // Real-world Mg is the canonical "lighter" for thermite and other
    // hard-to-ignite reactions, because Mg burns hot enough to push
    // adjacent Fe₂O₃+Al pairs over their 600°C ignition threshold.
    //
    // Modeled like thermite's burn-based mechanic: ignited Mg cells
    // stay in a sustained 3000°C burn for BURN_DURATION ticks, radiate
    // heat to neighbors (driving cascades into other reactives), then
    // transmute to MgO at FINAL_TEMP. Differs from thermite in that
    // each Mg cell burns alone (no Mg+X pair required) — Mg + ambient
    // virtual-O is enough for combustion. Cell needs an O source: an
    // adjacent O cell, or an Empty neighbor with ambient_oxygen > 0.05.
    // Flame-test emission — heated metal salts (Cu, Na, K, Ca, etc.)
    // actively emit their characteristically-colored flame when
    // hot enough. This models real flame-test chemistry: the metal
    // atoms vaporize when heated, get excited into upper electron
    // states, and emit visible-light photons at element-specific
    // wavelengths as they relax. Without this active emission, the
    // colored-fire effect is only visible when an external flame
    // is already present at the salt — but with it, players can
    // drop K on hot lava and watch it shoot purple flames upward.
    //
    // Skips cells already in active combustion (burn > 0) so this
    // doesn't double-spawn fires for things like burning Mg, which
    // has its own dedicated magnesium_burn() pass.
    fn flame_test_emission(&mut self) {
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.is_updated() { continue; }
                if c.burn > 0 { continue; }
                if c.temp < 600 { continue; }
                if flame_color(c.el).is_none() { continue; }
                // 40% chance per tick to emit a colored Fire cell into
                // a random adjacent Empty. Visually reads as a steady
                // fountain of colored flame off the hot salt.
                if rand::gen_range::<f32>(0.0, 1.0) > 0.40 { continue; }
                let order = rand::gen_range::<u8>(0, 4);
                for k in 0..4 {
                    let (dx, dy) = match (order + k) % 4 {
                        0 => (0i32, -1i32),
                        1 => (1, 0),
                        2 => (-1, 0),
                        _ => (0, 1),
                    };
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].el == Element::Empty {
                        let mut fire = Cell::new(Element::Fire);
                        fire.solute_el = c.el;
                        self.cells[ni] = fire;
                        break;
                    }
                }
            }
        }
    }

    // Flame-color inheritance — Fire cells adjacent to flame-coloring
    // elements (Cu, Na, K, Li, Ca, Mg, salts) pick up that element's
    // identity in their solute_el field. The render pass then tints
    // those Fire cells toward the metal's flame-test emission color.
    // Real chemistry: when metal salts vaporize into a flame, their
    // excited atoms emit characteristic wavelengths (Na yellow,
    // Cu green, etc.). This is why fireworks use metal salts to
    // produce different colors.
    //
    // Once a Fire cell is colored, it keeps that color for its
    // lifetime — no re-checking. New Fire cells coming off the same
    // burn site will pick up the color independently from their
    // own neighbors at spawn time.
    fn color_fires(&mut self) {
        if !self.present_elements[Element::Fire as usize] { return; }
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el != Element::Fire { continue; }
                if c.solute_el != Element::Empty { continue; }
                for (dx, dy) in [
                    (1i32, 0i32), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let n = self.cells[Self::idx(nx, ny)];
                    // Direct flame-coloring element.
                    if flame_color(n.el).is_some() {
                        self.cells[i].solute_el = n.el;
                        break;
                    }
                    // Liquid carrying a flame-coloring solute (salt
                    // dissolved in water).
                    if n.el == Element::Water
                        && flame_color(n.solute_el).is_some()
                    {
                        self.cells[i].solute_el = n.solute_el;
                        break;
                    }
                }
            }
        }
    }

    fn magnesium_burn(&mut self) {
        if !self.present_elements[Element::Mg as usize] { return; }
        const BURN_DURATION: u8 = 50;
        const BURN_TEMP: i16 = 3000;
        const FINAL_TEMP: i16 = 1700;
        const IGNITION: i16 = 470;
        const HEAT_BROADCAST_DELTA: i16 = 35;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];

                // ---- Continue burn ----
                if c.burn > 0 && c.el == Element::Mg {
                    // Mg+CO₂ — burning Mg strips O from CO₂, producing
                    // MgO + C. Famous demo: a burning Mg ribbon dropped
                    // into a CO₂ atmosphere (or onto dry ice) keeps
                    // burning because Mg's affinity for O is strong
                    // enough to overcome the C–O bond. Per-tick, 80%
                    // chance to consume one adjacent CO₂ cell. Most
                    // (~75%) of the carbon disperses as fine soot and
                    // leaves Empty behind — modeling convection sweeping
                    // particulates upward. Only ~25% deposits visible
                    // C cells nearby. Without dispersal, dense C
                    // (Powder, 22) shells the Mg pile and starves it.
                    // Two-stage Mg+CO₂ logic.
                    //
                    // Stage 1 (consume) — if there's a CO₂ cell directly
                    // adjacent (8-way), consume it. ~5% leaves visible C
                    // soot, the rest disperses to Empty.
                    //
                    // Stage 2 (inhale) — if no adjacent CO₂, the burn
                    // would normally snuff out as Empty voids accumulate
                    // between the Mg pile and the receding CO₂ front.
                    // Active inhalation: find any adjacent Empty cell,
                    // search outward (radius 2..=5) for a CO₂ cell,
                    // and SWAP them. The CO₂ is now adjacent for next
                    // tick's consumption. Models convective inflow of
                    // gas toward the high-temperature reaction zone.
                    let inner: &[(i32, i32)] = &[
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (1, -1), (-1, 1), (-1, -1),
                    ];
                    let mut burn_progress = false;
                    if rand::gen_range::<f32>(0.0, 1.0) < 0.80 {
                        for &(dx, dy) in inner {
                            let nx = x + dx;
                            let ny = y + dy;
                            if !Self::in_bounds(nx, ny) { continue; }
                            let ni = Self::idx(nx, ny);
                            if self.cells[ni].is_updated() { continue; }
                            if self.cells[ni].el != Element::CO2 { continue; }
                            let new_cell = if rand::gen_range::<f32>(0.0, 1.0) < 0.05 {
                                let mut carbon = Cell::new(Element::C);
                                carbon.temp = BURN_TEMP;
                                carbon.flag |= Cell::FLAG_UPDATED;
                                carbon
                            } else {
                                Cell::EMPTY
                            };
                            self.cells[ni] = new_cell;
                            burn_progress = true;
                            break;
                        }
                    }
                    if !burn_progress {
                        let mut gap: Option<usize> = None;
                        for &(dx, dy) in inner {
                            let nx = x + dx;
                            let ny = y + dy;
                            if !Self::in_bounds(nx, ny) { continue; }
                            let ni = Self::idx(nx, ny);
                            if self.cells[ni].el == Element::Empty
                                && !self.cells[ni].is_updated()
                            {
                                gap = Some(ni);
                                break;
                            }
                        }
                        if let Some(g) = gap {
                            'pull: for r in 2..=5i32 {
                                for ddy in -r..=r {
                                    for ddx in -r..=r {
                                        if ddx.abs().max(ddy.abs()) != r { continue; }
                                        let nx = x + ddx;
                                        let ny = y + ddy;
                                        if !Self::in_bounds(nx, ny) { continue; }
                                        let src = Self::idx(nx, ny);
                                        if self.cells[src].is_updated() { continue; }
                                        if self.cells[src].el != Element::CO2 { continue; }
                                        let pulled = self.cells[src];
                                        self.cells[g] = pulled;
                                        self.cells[src] = Cell::EMPTY;
                                        burn_progress = true;
                                        break 'pull;
                                    }
                                }
                            }
                        }
                    }
                    if burn_progress {
                        // Either consumed or inhaled — fuel is reaching
                        // the burn, sustain it.
                        self.cells[i].burn = self.cells[i].burn.max(BURN_DURATION / 2);
                    }
                    let c = self.cells[i]; // re-read after possible burn refresh
                    let new_burn = c.burn - 1;
                    if new_burn == 0 {
                        // Transmute to MgO (Mg+O derived compound).
                        let new_cell = match derive_or_lookup(Element::Mg, Element::O) {
                            Some(id) => {
                                let mut s = Cell::new(Element::Derived);
                                s.derived_id = id;
                                s.temp = FINAL_TEMP;
                                s.flag |= Cell::FLAG_UPDATED;
                                s
                            }
                            None => {
                                let mut a = c;
                                a.burn = 0;
                                a.flag |= Cell::FLAG_UPDATED;
                                a
                            }
                        };
                        self.cells[i] = new_cell;
                    } else {
                        let mut burning = c;
                        burning.temp = BURN_TEMP;
                        burning.burn = new_burn;
                        burning.flag |= Cell::FLAG_UPDATED;
                        self.cells[i] = burning;
                    }
                    self.thermite_burn_effects(x, y, i, HEAT_BROADCAST_DELTA);
                    continue;
                }

                // ---- Try to ignite a fresh Mg cell ----
                if c.el != Element::Mg { continue; }
                if c.temp < IGNITION { continue; }
                // Oxidizer access check — Mg can ignite from O₂ or CO₂.
                // Either an explicit O cell neighbor, an Empty cell with
                // ambient O₂, or a CO₂ neighbor (Mg's affinity for O is
                // strong enough to strip C-O bonds even at ignition temps,
                // which is why a CO₂ extinguisher won't put out a Mg fire).
                let has_oxidizer = (0..4).any(|d| {
                    let (dx, dy) = match d {
                        0 => (1i32, 0i32),
                        1 => (-1, 0),
                        2 => (0, 1),
                        _ => (0, -1),
                    };
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { return false; }
                    let n = self.cells[Self::idx(nx, ny)];
                    n.el == Element::O
                        || n.el == Element::CO2
                        || (n.el == Element::Empty && self.ambient_oxygen > 0.05)
                });
                if !has_oxidizer { continue; }
                let mut burning = c;
                burning.burn = BURN_DURATION;
                burning.temp = BURN_TEMP;
                burning.flag |= Cell::FLAG_UPDATED;
                self.cells[i] = burning;
                self.thermite_burn_effects(x, y, i, HEAT_BROADCAST_DELTA);
            }
        }
        // Boudouard reaction — hot soot in a CO₂ atmosphere converts:
        //   C + CO₂ → 2 CO  (above ~700°C, equilibrium shifts strongly
        //   toward CO at temperatures characteristic of Mg burning)
        // We don't model CO as a discrete element, so we approximate
        // the equilibrium by sublimating hot C cells away when they're
        // adjacent to CO₂. Without this, C deposits from Mg+CO₂ build
        // a permanent shell on the Mg pile that eventually starves the
        // burn — this keeps the soot layer in equilibrium with the
        // surrounding gas, mimicking the real reaction's behavior.
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.is_updated() { continue; }
                if c.el != Element::C { continue; }
                if c.temp < 1000 { continue; }
                if rand::gen_range::<f32>(0.0, 1.0) > 0.20 { continue; }
                let has_co2 = (0..4).any(|d| {
                    let (dx, dy) = match d {
                        0 => (1i32, 0i32),
                        1 => (-1, 0),
                        2 => (0, 1),
                        _ => (0, -1),
                    };
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { return false; }
                    self.cells[Self::idx(nx, ny)].el == Element::CO2
                });
                if !has_co2 { continue; }
                self.cells[i] = Cell::EMPTY;
            }
        }
    }

    fn thermite_burn_effects(&mut self, x: i32, y: i32, i: usize, broadcast_delta: i16) {
        // Incremental heat broadcast to 8 non-burning neighbors. Each
        // tick of exposure adds broadcast_delta °C, so cells need
        // sustained adjacency to a burning cell to reach ignition.
        // Skip cells already FLAG_UPDATED (themselves burning or
        // recently transmuted) — only push heat into not-yet-reacting
        // material.
        const BURN_TEMP: i16 = 2500;
        for ddy in -1..=1 {
            for ddx in -1..=1 {
                if ddx == 0 && ddy == 0 { continue; }
                let bx = x + ddx;
                let by = y + ddy;
                if !Self::in_bounds(bx, by) { continue; }
                let bi = Self::idx(bx, by);
                if bi == i { continue; }
                let mut nb = self.cells[bi];
                if nb.el == Element::Empty { continue; }
                if nb.is_frozen() { continue; }
                if nb.is_updated() { continue; }
                let new_temp = (nb.temp as i32 + broadcast_delta as i32)
                    .min(BURN_TEMP as i32) as i16;
                nb.temp = new_temp;
                self.cells[bi] = nb;
            }
        }
        // Smoke at modest rate per burning cell.
        if rand::gen_range::<f32>(0.0, 1.0) < 0.10 {
            for (sx, sy) in [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)] {
                if !Self::in_bounds(sx, sy) { continue; }
                let si = Self::idx(sx, sy);
                if self.cells[si].el == Element::Empty {
                    self.cells[si] = Cell::new(Element::CO2);
                    break;
                }
            }
        }
        // Sparks (Fire cells) eject upward at low per-cell rate.
        if rand::gen_range::<f32>(0.0, 1.0) < 0.04 {
            let sdx = rand::gen_range::<i32>(-2, 3);
            let sy = y - rand::gen_range::<i32>(1, 4);
            let sx = x + sdx;
            if Self::in_bounds(sx, sy) {
                let si = Self::idx(sx, sy);
                if self.cells[si].el == Element::Empty {
                    self.cells[si] = Cell::new(Element::Fire);
                }
            }
        }
        if rand::gen_range::<f32>(0.0, 1.0) < 0.005 {
            self.spawn_shockwave_capped(x, y, 1000.0, 2_500.0);
        }
    }

    // Hg amalgamation — liquid mercury dissolves many metals on contact,
    // forming a liquid amalgam alloy. Real-world: Au/Ag/Na/K/Cs/Zn/Pb/Sn
    // form amalgams readily; Fe and Ni don't (Hg won't wet ferromagnetic
    // metals). The amalgam is liquid even when the parent metal would
    // be solid alone, because the Hg holds it in solution.
    //
    // Differs from `alloy_formation` in two ways: (1) doesn't require
    // both cells in PHASE_LIQUID — the SOLID metal dissolves into Hg's
    // liquid phase. (2) skips ferrous metals (Fe, Ni) which don't amalgamate.
    fn hg_amalgamation(&mut self) {
        if !self.present_elements[Element::Hg as usize] { return; }
        // Slow reaction so the user sees a gradual amalgam spread,
        // not a flash conversion of the entire contact line.
        const RATE: f32 = 0.05;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                if c.el != Element::Hg { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    if !is_atomic_metal(n.el) { continue; }
                    // Hg+Hg — same element, no alloy formation.
                    if n.el == Element::Hg { continue; }
                    // Ferromagnetic metals don't amalgamate — Hg beads
                    // up on Fe/Ni rather than dissolving them.
                    if matches!(n.el, Element::Fe | Element::Ni) { continue; }
                    let mut rate = RATE;
                    if n.is_frozen() { rate *= 0.02; }
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    let Some(alloy_id) = alloy_or_lookup(Element::Hg, n.el)
                        else { continue; };
                    // Volumetric stoichiometry: 1 cell Hg + 1 cell Au →
                    // 2 cells AuHg. Each cell is a fixed unit of volume
                    // and the total atom inventory is preserved (each
                    // AuHg cell contains half the Hg and half the Au of
                    // the parent cells). Consuming Hg is essential —
                    // letting Hg persist while producing AuHg would
                    // duplicate Hg atoms.
                    let mk_amalgam = |src: Cell| -> Cell {
                        let mut a = Cell::new(Element::Derived);
                        a.derived_id = alloy_id;
                        a.set_phase(PHASE_LIQUID);
                        a.temp = src.temp;
                        a.flag |= Cell::FLAG_UPDATED;
                        a
                    };
                    self.cells[i] = mk_amalgam(c);
                    self.cells[ni] = mk_amalgam(n);
                    break;
                }
            }
        }
    }

    // Halogen displacement — a more reactive halogen displaces a less
    // reactive one from its salt. Real reactivity series: F > Cl > Br > I.
    // Currently only F and Cl are fully implemented atomic halogens, so
    // this models F + (metal-Cl salt) → (metal-F salt) + Cl gas. Targets
    // both Element::Salt (bespoke NaCl) and any derived metal-Cl compound.
    fn halogen_displacement(&mut self) {
        if !self.present_elements[Element::F as usize] { return; }
        const RATE: f32 = 0.30;
        const REACTION_HEAT: i16 = 200;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                if c.el != Element::F { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    // Identify metal in the chloride salt.
                    let metal_el: Element = match n.el {
                        Element::Salt => Element::Na,
                        Element::Derived => {
                            let m = DERIVED_COMPOUNDS.with(|r| {
                                let reg = r.borrow();
                                let cd = reg.get(n.derived_id as usize)?;
                                if cd.constituents.len() != 2 { return None; }
                                let (e0, _) = cd.constituents[0];
                                let (e1, _) = cd.constituents[1];
                                if e0 == Element::Cl { Some(e1) }
                                else if e1 == Element::Cl { Some(e0) }
                                else { None }
                            });
                            match m {
                                Some(metal) => metal,
                                None => continue,
                            }
                        }
                        _ => continue,
                    };
                    let mut rate = RATE;
                    if n.is_frozen() { rate *= 0.02; }
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    let Some(fluoride_id) = derive_or_lookup(metal_el, Element::F)
                        else { continue; };
                    // Halide cell → metal fluoride.
                    let mut fluoride = Cell::new(Element::Derived);
                    fluoride.derived_id = fluoride_id;
                    fluoride.temp = (n.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    fluoride.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = fluoride;
                    // F cell → Cl (displaced halogen escapes as gas).
                    let mut cl = Cell::new(Element::Cl);
                    cl.temp = (c.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    cl.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = cl;
                    break;
                }
            }
        }
    }

    // F + Glass etching — fluorine famously eats glass via the reaction
    //   2 F₂ + SiO₂ → SiF₄ + O₂
    // We don't expose Glass to the general chemistry engine (would make it
    // react with O / metals / everything). Instead, this dedicated scan
    // looks for F adjacent to Glass (or MoltenGlass) and runs the reaction
    // as a 1:1 cell pair: the F cell becomes a derived SiF compound (Si
    // mixed with F at 1:4 stoichiometry from derive_or_lookup), the Glass
    // cell becomes O. Stoichiometry isn't exact at the cell scale, but
    // the visual story — glass dissolving while oxygen escapes — is right.
    //
    // Real F+SiO₂ proceeds at room temperature, so no activation gate.
    // Reaction is strongly exothermic; we add 800°C to both products.
    // Frozen (build-mode) Glass etches 50× slower so a glass window
    // doesn't vanish in the instant a single F atom drifts into it.
    fn glass_etching(&mut self) {
        if !self.present_elements[Element::F as usize]
            || (!self.present_elements[Element::Glass as usize]
                && !self.present_elements[Element::MoltenGlass as usize]) {
            return;
        }
        const ETCH_RATE: f32 = 0.20;
        const REACTION_HEAT: i16 = 800;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                if c.el != Element::F { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    if !matches!(n.el, Element::Glass | Element::MoltenGlass) { continue; }
                    let mut rate = ETCH_RATE;
                    if n.is_frozen() { rate *= 0.02; }
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    let Some(sif_id) = derive_or_lookup(Element::Si, Element::F)
                        else { continue; };
                    // Glass cell → SiF (Si stays put, F migrates in to bond).
                    let mut sif = Cell::new(Element::Derived);
                    sif.derived_id = sif_id;
                    sif.temp = (n.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    sif.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = sif;
                    // F cell → O (released oxygen replaces the consumed F).
                    let mut o = Cell::new(Element::O);
                    o.temp = (c.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    o.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = o;
                    break;
                }
            }
        }
    }

    // Selective acid leaching of alloys. Acid adjacent to an alloy cell
    // finds the alloy's most reactive constituent (lowest-E metal) and
    // strips it out: the acid becomes H gas, the alloy cell becomes the
    // LEAST reactive constituent as a pure atom, and the freed reactive
    // metal precipitates as its halide salt into an adjacent empty cell.
    // This is how you purify copper out of a CuFe alloy: drop it in HCl,
    // iron dissolves away as FeCl₂ + H₂, copper is left behind.
    fn alloy_acid_leach(&mut self) {
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                let Some(elems) = alloy_constituents(c) else { continue; };
                if elems.len() < 2 { continue; }
                // Rank constituents by electronegativity. Lowest E is the
                // most reactive (will be stripped); highest E stays.
                let mut ranked: Vec<(Element, f32)> = elems.iter()
                    .filter_map(|&e| atom_profile_for(e)
                        .map(|p| (e, p.electronegativity)))
                    .collect();
                ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let (reactive_el, reactive_e) = ranked[0];
                let (leftover_el, _) = ranked[ranked.len() - 1];
                // Same reactivity-series cutoff as acid_displacement — the
                // constituent must be above H to get stripped out. Cu
                // stays put in the alloy regardless of acid exposure.
                if reactive_e >= 1.88 { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    let Some((acceptor_el, strength)) = acid_signature(n)
                        else { continue; };
                    let metal_reactivity = 2.0 - reactive_e;
                    // Slightly slower than pure-metal displacement because
                    // the reactive atom has to diffuse out of the alloy
                    // lattice — acid on a chunk of alloy eats visibly but
                    // not instantly.
                    let rate = (strength * metal_reactivity * 0.3).min(0.4);
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    let Some(salt_id) = derive_or_lookup(reactive_el, acceptor_el)
                        else { continue; };
                    // Alloy cell → pure leftover metal (Cu staying put).
                    let mut pure = Cell::new(leftover_el);
                    pure.temp = c.temp;
                    pure.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = pure;
                    // Acid cell → H gas.
                    let mut h_cell = Cell::new(Element::H);
                    h_cell.temp = n.temp;
                    h_cell.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = h_cell;
                    // Salt precipitates into an adjacent empty cell if
                    // available. If nowhere to go (totally enclosed), the
                    // salt is lost — imperfect but acceptable.
                    for (dx2, dy2) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let px = x + dx2;
                        let py = y + dy2;
                        if !Self::in_bounds(px, py) { continue; }
                        let pi = Self::idx(px, py);
                        if self.cells[pi].el != Element::Empty { continue; }
                        let mut salt = Cell::new(Element::Derived);
                        salt.derived_id = salt_id;
                        salt.temp = c.temp;
                        salt.flag |= Cell::FLAG_UPDATED;
                        self.cells[pi] = salt;
                        break;
                    }
                    break;
                }
            }
        }
    }

    // Alloy formation. Two different atomic metals, both in liquid phase,
    // fuse into a 1:1 alloy compound. The alloy's properties are averaged
    // from the parents with a mild eutectic drop on melting point, so
    // cooling the fresh alloy solidifies at a slightly lower temperature
    // than either pure metal would. Since alloys are derived compounds,
    // they participate in the normal phase system and other chemistry.
    fn alloy_formation(&mut self) {
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                if c.phase() != PHASE_LIQUID { continue; }
                if !is_atomic_metal(c.el) { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    if n.phase() != PHASE_LIQUID { continue; }
                    if !is_atomic_metal(n.el) { continue; }
                    if c.el == n.el { continue; }
                    let Some(alloy_id) = alloy_or_lookup(c.el, n.el)
                        else { continue; };
                    // Modest per-frame rate so a mixing pool alloys
                    // visibly over a few seconds rather than instantly.
                    if rand::gen_range::<f32>(0.0, 1.0) > 0.15 { continue; }
                    let mk_alloy = |src: Cell| -> Cell {
                        let mut a = Cell::new(Element::Derived);
                        a.derived_id = alloy_id;
                        a.set_phase(PHASE_LIQUID);
                        a.temp = src.temp;
                        a.flag |= Cell::FLAG_UPDATED;
                        a
                    };
                    self.cells[i] = mk_alloy(c);
                    self.cells[ni] = mk_alloy(n);
                    break;
                }
            }
        }
    }

    // Acid-base neutralization. A basic oxide (metal + O) adjacent to an
    // acid (H + halide) rearranges: the metal pairs with the halide to
    // form a salt, and the freed H and O combine into water.
    //
    //     M_xO + 2HX  →  M_xX₂ + H₂O
    //
    // Rate scales with metal basicity × acid strength. The reaction is
    // modestly exothermic — warms the salt and water but not enough to
    // cascade into combustion.
    fn base_neutralization(&mut self) {
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                let Some((metal_el, basicity)) = basic_oxide_signature(c)
                    else { continue; };
                if basicity <= 0.0 { continue; }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    let Some((acceptor_el, strength)) = acid_signature(n)
                        else { continue; };
                    if strength <= 0.0 { continue; }
                    let rate = (basicity * strength * 0.5).min(0.5);
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    // Oxide cell → salt (metal + acid's acceptor).
                    // Acid cell → water.
                    let Some(product) = infer_product(metal_el, acceptor_el, &[])
                        else { continue; };
                    let dt = ((basicity + strength) * 60.0) as i32;
                    let mut salt = match product {
                        InferredProduct::Bespoke(el) => Cell::new(el),
                        InferredProduct::Derived(id) => {
                            let mut sc = Cell::new(Element::Derived);
                            sc.derived_id = id;
                            sc
                        }
                    };
                    salt.temp = (c.temp as i32 + dt).clamp(-273, 5000) as i16;
                    salt.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = salt;
                    let mut water = Cell::new(Element::Water);
                    water.temp = (n.temp as i32 + dt).clamp(-273, 5000) as i16;
                    water.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = water;
                    break;
                }
            }
        }
    }

    // Dissolution: soluble solids adjacent to a liquid get slowly taken
    // into solution. The solid cell vanishes and the liquid's solute_amt
    // climbs by a fixed step per frame (capped at 255 = saturated).
    // Phase 1 scope: Salt in Water. Other soluble solids will be added
    // as the predicate grows.
    fn dissolve(&mut self) {
        // Water with room (solute_amt < ABSORB_THRESHOLD) will fill up to
        // 255 on contact with a salt neighbor, consuming the salt cell.
        // Diffusion (diffuse_solute) lowers saturation over time so fresh
        // neighbors appear at the salt/water interface and dissolution
        // continues until the whole water body equilibrates at 255.
        const ABSORB_THRESHOLD: u8 = 192;
        const TRY_P: f32 = 0.20;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el != Element::Water { continue; }
                if c.solute_amt >= ABSORB_THRESHOLD { continue; } // no room
                if rand::gen_range::<f32>(0.0, 1.0) > TRY_P { continue; }
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    let n = self.cells[ni];
                    if n.is_frozen() { continue; }
                    // Soluble: Salt (bespoke NaCl), or any derived ionic
                    // metal-halide (FeCl, KCl, MgCl₂, …).
                    let soluble = n.el == Element::Salt
                        || (n.el == Element::Derived
                            && derived_is_soluble_salt(n.derived_id));
                    if !soluble { continue; }
                    // Different solute already present — don't mix species.
                    if c.solute_amt > 0
                        && (c.solute_el != n.el
                            || c.solute_derived_id != n.derived_id)
                    {
                        continue;
                    }
                    self.cells[i].solute_el = n.el;
                    self.cells[i].solute_derived_id = n.derived_id;
                    self.cells[i].solute_amt = 255;
                    self.cells[ni] = Cell::EMPTY;
                    break;
                }
            }
        }
    }

    fn diffuse_solute(&mut self) {
        // Solute_amt equalizes between adjacent water cells carrying the
        // same solute (concentration gradient → diffusion). This is what
        // breaks the interface-saturation stall: without it the layer of
        // water touching a salt pile saturates and never mixes away, and
        // no fresh water ever reaches the remaining salt.
        //
        // Picks a random neighbor per cell per frame. Transfers half the
        // gap each event, capped at DIFFUSE_MAX so visual mixing looks
        // gradual rather than snapping to uniform in a frame.
        const DIFFUSE_P: f32 = 0.35;
        const DIFFUSE_MAX: u8 = 24;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el != Element::Water { continue; }
                if c.solute_amt == 0 { continue; }
                if rand::gen_range::<f32>(0.0, 1.0) > DIFFUSE_P { continue; }
                let (dx, dy) = match rand::gen_range::<i32>(0, 4) {
                    0 => (-1i32, 0i32),
                    1 => (1, 0),
                    2 => (0, -1),
                    _ => (0, 1),
                };
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                let n = self.cells[ni];
                if n.el != Element::Water { continue; }
                // Don't mix different solutes — keep a FeCl cell and a
                // NaCl cell distinguishable even in the same water body.
                if n.solute_amt > 0
                    && (n.solute_el != c.solute_el
                        || n.solute_derived_id != c.solute_derived_id)
                {
                    continue;
                }
                if n.solute_amt >= c.solute_amt { continue; }
                let gap = c.solute_amt - n.solute_amt;
                let transfer = (gap / 2).min(DIFFUSE_MAX).max(1);
                self.cells[i].solute_amt -= transfer;
                self.cells[ni].solute_el = c.solute_el;
                self.cells[ni].solute_derived_id = c.solute_derived_id;
                self.cells[ni].solute_amt = self.cells[ni].solute_amt.saturating_add(transfer);
                if self.cells[i].solute_amt == 0 {
                    self.cells[i].solute_el = Element::Empty;
                    self.cells[i].solute_derived_id = 0;
                }
            }
        }
    }

    fn reactions(&mut self) {
        // Only moisture-chemistry reactions live here now. Heat-driven effects
        // (ignition, boiling, lava→obsidian, mud drying) are all in thermal().
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let el = self.get(x, y).el;
                match el {
                    Element::Water => {
                        // Percolate through a mud column to soak sand below.
                        let mut py = y + 1;
                        let mut steps = 0;
                        while Self::in_bounds(x, py) && steps < 30 {
                            let e = self.get(x, py).el;
                            if e == Element::Mud { py += 1; steps += 1; continue; }
                            if e == Element::Sand && rand::gen_range::<u16>(0, 200) < 1 {
                                self.set(x, y, Cell::EMPTY);
                                self.set(x, py, Cell::new(Element::Mud));
                            }
                            break;
                        }
                    }
                    Element::Sand => {
                        for (dx, dy) in [(0, -1), (1, 0), (-1, 0)] {
                            let n = self.get(x + dx, y + dy).el;
                            if n == Element::Water && rand::gen_range::<u16>(0, 60) < 1 {
                                self.set(x, y, Cell::new(Element::Mud));
                                self.set(x + dx, y + dy, Cell::EMPTY);
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn update_cell(&mut self, x: i32, y: i32, wind: Vec2) {
        let c = self.get(x, y);
        if c.is_updated() { return; }
        // Frozen (rigid-body) cells are fully locked: no movement, no phase
        // change, no combustion, no moisture change. Thermal diffusion still
        // flows through them because that uses a separate code path.
        //
        // …unless a nearby explosion or extreme suction creates a pressure
        // differential large enough to rupture the structure. Check the
        // cell's max neighbor pressure gap; above the burst threshold we
        // unfreeze so the usual pressure-shove can fling the piece away.
        if c.is_frozen() {
            // A wall doesn't fail layer-by-layer. It holds as a unit until
            // internal pressure exceeds the structural capacity of the full
            // thickness, then the entire column fractures outward together.
            // Only the INNERMOST face cell — the one actually touching the
            // high-pressure fluid — runs this check; deeper wall cells are
            // surrounded by zero-pressure neighbors (walls don't transmit
            // pressure) and correctly see no gap to react to.
            const BASE_THRESHOLD: i32 = 2500;
            const PER_LAYER: i32 = 350;
            const MAX_PROBE: usize = 30;
            let my_p = c.pressure as i32;
            let mut max_gap: i32 = 0;
            let mut blast_dir: (i32, i32) = (0, 0);
            let mut max_np: i32 = i32::MIN;
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx; let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let n_p = self.cells[Self::idx(nx, ny)].pressure as i32;
                let gap = (my_p - n_p).abs();
                if gap > max_gap { max_gap = gap; }
                if n_p > max_np {
                    max_np = n_p;
                    blast_dir = (dx, dy);
                }
            }
            if max_gap < BASE_THRESHOLD { return; }
            let push = (-blast_dir.0, -blast_dir.1);
            if push.0 == 0 && push.1 == 0 { return; }
            // Probe outward through cells of the SAME ELEMENT. A glass window
            // embedded in an iron wall fails based on the glass's own
            // thickness, not the iron around it — that's how physical
            // windows work. The iron gets its own thickness probe when its
            // inner face is checked separately.
            let wall_el = c.el;
            let mut wall_cells: [(i32, i32); MAX_PROBE + 1] =
                [(0, 0); MAX_PROBE + 1];
            wall_cells[0] = (x, y);
            let mut thickness: usize = 1;
            let mut tx = x + push.0;
            let mut ty = y + push.1;
            while thickness <= MAX_PROBE {
                if !Self::in_bounds(tx, ty) { break; }
                let ti = Self::idx(tx, ty);
                if !self.cells[ti].is_frozen() { break; }
                if self.cells[ti].el != wall_el { break; }
                wall_cells[thickness] = (tx, ty);
                thickness += 1;
                tx += push.0;
                ty += push.1;
            }
            let effective_threshold =
                BASE_THRESHOLD + PER_LAYER * (thickness as i32 - 1);
            if max_gap < effective_threshold { return; }
            // Wall fails as a unit. Fling the entire column outward in this
            // frame — hop count scales with pressure overage so a larger
            // blast throws debris further.
            let overage = max_gap - effective_threshold;
            let hops = (2 + overage / 1000).clamp(2, 10);
            // Unfreeze everyone first; then teleport outermost-first so the
            // inner cells always have vacated space to walk into. If a cell
            // can't move at all (e.g. glass window backed by iron), it
            // SHATTERS to Empty — the material fragments and disperses,
            // giving gas an actual escape path. Otherwise pressure stays
            // trapped and nothing vents.
            for i in 0..thickness {
                let (wx, wy) = wall_cells[i];
                self.cells[Self::idx(wx, wy)].flag &= !Cell::FLAG_FROZEN;
            }
            for i in (0..thickness).rev() {
                let (wx, wy) = wall_cells[i];
                let mut cx_cur = wx;
                let mut cy_cur = wy;
                let mut moved = false;
                for _ in 0..hops {
                    let nx = cx_cur + push.0;
                    let ny = cy_cur + push.1;
                    if !Self::in_bounds(nx, ny) { break; }
                    let ni = Self::idx(nx, ny);
                    let tk = cell_physics(self.cells[ni]).kind;
                    if !matches!(tk, Kind::Empty | Kind::Gas | Kind::Fire) { break; }
                    let cur_idx = Self::idx(cx_cur, cy_cur);
                    self.cells.swap(cur_idx, ni);
                    self.cells[ni].flag |= Cell::FLAG_UPDATED;
                    cx_cur = nx;
                    cy_cur = ny;
                    moved = true;
                }
                if !moved {
                    self.cells[Self::idx(wx, wy)] = Cell::EMPTY;
                }
            }
            return;
        }
        // Falling wood — tree_support_check set life=1 on unsupported cells.
        // We move them every frame so collapse is smooth (60 cells/sec fall).
        if c.el == Element::Wood && c.life > 0 {
            if !self.should_fall() { return; }
            let (gx, gy) = self.gravity_step();
            {
                let bx = x + gx;
                let by = y + gy;
                if Self::in_bounds(bx, by) {
                    let bidx = Self::idx(bx, by);
                    let below = self.cells[bidx];
                    if below.el == Element::Empty && !below.is_updated() {
                        self.cells.swap(Self::idx(x, y), bidx);
                        self.cells[bidx].flag |= Cell::FLAG_UPDATED;
                        return;
                    }
                }
            }
            // Can't fall — landed. Clear the falling flag; the next support
            // check will re-evaluate whether we're truly supported.
            self.cells[Self::idx(x, y)].life = 0;
            return;
        }
        match c.el {
            Element::Seed => { self.update_seed(x, y); return; }
            Element::Leaves  => { self.update_leaves(x, y); return; }
            _ => {}
        }
        // Phase-aware dispatch: a molten-phase atom enters update_liquid
        // even if its element is nominally a Gravel; a boiled-off atom
        // enters update_gas.
        match cell_physics(c).kind {
            Kind::Empty => {}
            Kind::Solid => {
                // Non-frozen solids still respond to extreme pressure
                // gradients — a painted glass wall under a blast should
                // shove outward, not just stand there. Frozen cells are
                // already handled by the earlier burst check.
                self.try_pressure_shove(x, y, c);
            }
            Kind::Gravel => self.update_gravel(x, y),
            Kind::Powder => self.update_powder(x, y),
            Kind::Liquid => self.update_liquid(x, y),
            Kind::Gas | Kind::Fire => self.update_gas(x, y, wind),
        }
    }

    // Moisture at the growth tip (used when seed hasn't rooted yet).
    // Counts both Water/Mud elements *and* the moisture field on other cells —
    // so damp sand / wet wood that's been wicking moisture counts as wet.
    fn moisture_score(&self, x: i32, y: i32) -> f32 {
        let mut score = 0.0;
        for dy in -3..=3i32 {
            for dx in -3..=3i32 {
                let cell = self.get(x + dx, y + dy);
                match cell.el {
                    Element::Water => score += 1.0,
                    Element::Mud   => score += 0.8,
                    _ if cell.moisture > 40 => {
                        score += (cell.moisture as f32 - 40.0) / 220.0;
                    }
                    _ => {}
                }
            }
        }
        score
    }

    // Trace down through the trunk (Wood) to the root zone, then sample
    // a wide horizontal window for water + mud. Roots spread sideways.
    fn root_y(&self, x: i32, y: i32) -> i32 {
        let mut py = y + 1;
        let mut steps = 0;
        while steps < 80 && Self::in_bounds(x, py) && self.get(x, py).el == Element::Wood {
            py += 1;
            steps += 1;
        }
        py
    }

    fn root_moisture(&self, x: i32, y: i32) -> f32 {
        let ry = self.root_y(x, y);
        let mut score = 0.0;
        for dy in -2..=3i32 {
            for dx in -6..=6i32 {
                let cell = self.get(x + dx, ry + dy);
                match cell.el {
                    Element::Water => score += 1.0,
                    Element::Mud   => score += 0.8,
                    _ if cell.moisture > 40 => {
                        score += (cell.moisture as f32 - 40.0) / 220.0;
                    }
                    _ => {}
                }
            }
        }
        score
    }

    // Take one water (→ empty) or mud (→ sand) from near the root or tip.
    fn consume_moisture(&mut self, x: i32, y: i32) {
        let ry = self.root_y(x, y);
        for _ in 0..8 {
            let dx = rand::gen_range::<i32>(-6, 7);
            let dy = rand::gen_range::<i32>(-2, 4);
            let el = self.get(x + dx, ry + dy).el;
            if el == Element::Water {
                self.set(x + dx, ry + dy, Cell::EMPTY);
                return;
            }
            if el == Element::Mud {
                self.set(x + dx, ry + dy, Cell::new(Element::Sand));
                return;
            }
        }
        for _ in 0..4 {
            let dx = rand::gen_range::<i32>(-3, 4);
            let dy = rand::gen_range::<i32>(-3, 4);
            let el = self.get(x + dx, y + dy).el;
            if el == Element::Water {
                self.set(x + dx, y + dy, Cell::EMPTY);
                return;
            }
            if el == Element::Mud {
                self.set(x + dx, y + dy, Cell::new(Element::Sand));
                return;
            }
        }
    }

    // Pick a random wood cell in the trunk and extend the first empty cell
    // encountered stepping outward left or right — trunks widen over time.
    fn thicken_trunk(&mut self, x: i32, y: i32) {
        let mut py = y + 1;
        let mut trunk_len = 0u16;
        let mut steps = 0;
        while steps < 160 && Self::in_bounds(x, py) && self.get(x, py).el == Element::Wood {
            py += 1;
            steps += 1;
            trunk_len += 1;
        }
        if trunk_len == 0 { return; }
        // Bias toward the base of the trunk — 3/4 of thickening attempts
        // land in the lower half, so trees taper naturally (wide root, narrow crown).
        let half = (trunk_len + 1) / 2;
        let pick = if rand::gen_range::<u8>(0, 4) < 3 {
            half + rand::gen_range::<u16>(0, half.max(1))
        } else {
            rand::gen_range::<u16>(0, trunk_len)
        };
        let ty = y + 1 + pick as i32;
        let side: i32 = if rand::gen_range::<u8>(0, 2) == 0 { 1 } else { -1 };
        // Reach also scales by how deep the cell is — base gets fatter faster.
        let depth = pick as i32;
        let max_reach = ((depth / 3).max(3)).min(16);
        for step in 1..=max_reach {
            let cx = x + side * step;
            let el = self.get(cx, ty).el;
            if el == Element::Empty {
                self.set(cx, ty, Cell::new(Element::Wood));
                return;
            }
            if el != Element::Wood { break; }
        }
    }

    fn update_seed(&mut self, x: i32, y: i32) {
        let age = self.get(x, y).life;

        // Unrooted (freshly painted or tumbling): fall into empty cells only
        // — seeds don't plow through sand or water, they land on surfaces.
        if age == 0 {
            if !self.should_fall() { return; }
            let (gx, gy) = self.gravity_step();
            let below = self.get(x + gx, y + gy);
            if below.el == Element::Empty && !below.is_updated() {
                self.swap(x, y, x + gx, y + gy); self.mark(x + gx, y + gy);
                return;
            }
            let (sa, sb) = self.gravity_sides();
            let sides = if rand::gen_range::<u8>(0, 2) == 0 { [sa, sb] } else { [sb, sa] };
            for (sx, sy) in sides {
                let tx = x + gx + sx;
                let ty = y + gy + sy;
                let diag = self.get(tx, ty);
                if diag.el == Element::Empty && !diag.is_updated() {
                    self.swap(x, y, tx, ty); self.mark(tx, ty);
                    return;
                }
            }
        }

        // Germinate/grow — moisture from the root zone via trunk, or local.
        let moisture = self.moisture_score(x, y).max(self.root_moisture(x, y));
        if moisture <= 0.0 { return; }
        let chance = 0.02 * (moisture / 8.0).min(1.0);
        let roll = rand::gen_range::<u16>(0, 10000) as f32 / 10000.0;
        if roll > chance { return; }

        let max_height:   u16 = 80;
        let branch_start: u16 = 8;
        // Once age exceeds this, the seed retires into a leaf. Caps the
        // tree's lifetime so it doesn't endlessly spew leaves.
        let lifetime_cap: u16 = max_height + 80;

        // Canopy reach scales with tree age — older/taller trees spawn leaves
        // and branches over a wider area. Kept modest so young trees look young.
        let canopy_reach = ((age / 8).min(10)) as i32;

        if age < max_height && self.get(x, y - 1).el == Element::Empty {
            self.set(x, y, Cell::new(Element::Wood));
            let mut seed = Cell::new(Element::Seed);
            seed.life = age + 1;
            self.set(x, y - 1, seed);
            self.mark(x, y - 1);

            // Each growth step drinks one water (or dries one mud → sand)
            // and has a chance to widen the trunk.
            self.consume_moisture(x, y - 1);
            if rand::gen_range::<u8>(0, 10) < 7 {
                self.thicken_trunk(x, y - 1);
            }

            // Lateral leaves at every height above branch_start.
            if age >= branch_start {
                for &dx in &[-1i32, 1] {
                    if self.get(x + dx, y).el == Element::Empty
                        && rand::gen_range::<u8>(0, 3) == 0
                    {
                        self.set(x + dx, y, Cell::new(Element::Leaves));
                        self.mark(x + dx, y);
                    }
                }
                // Occasional horizontal branch. Frequency scales with age so
                // older trees grow more limbs.
                let branch_period = if age < 20 { 0 } else { 14 };
                if branch_period > 0 && age % branch_period == 0 {
                    let side: i32 = if rand::gen_range::<u8>(0, 2) == 0 { 1 } else { -1 };
                    let branch_len = canopy_reach.max(2).min(8);
                    for step in 1..=branch_len {
                        let bx = x + side * step;
                        if self.get(bx, y).el == Element::Empty {
                            self.set(bx, y, Cell::new(Element::Wood));
                        }
                    }
                    // Leaves clustered at the branch tip.
                    let tip = x + side * branch_len;
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)] {
                        let lx = tip + dx; let ly = y + dy;
                        if self.get(lx, ly).el == Element::Empty
                            && rand::gen_range::<u8>(0, 2) == 0
                        {
                            self.set(lx, ly, Cell::new(Element::Leaves));
                        }
                    }
                }
            }
        } else if age >= branch_start {
            // Blocked or at max height — bloom phase. Spawn leaves only in a
            // tight pattern around the seed so they stay within support range.
            self.consume_moisture(x, y);
            let spots: [(i32, i32); 7] = [(-1,0),(1,0),(-1,-1),(1,-1),(0,-1),(-2,0),(2,0)];
            for (dx, dy) in spots {
                if self.get(x + dx, y + dy).el == Element::Empty
                    && rand::gen_range::<u8>(0, 5) == 0
                {
                    self.set(x + dx, y + dy, Cell::new(Element::Leaves));
                    self.mark(x + dx, y + dy);
                }
            }
            // Age still ticks while blooming so the seed eventually retires.
            if age < lifetime_cap {
                self.cells[Self::idx(x, y)].life = age + 1;
            } else {
                // Retirement: seed converts to a leaf. The standing trunk
                // remains as a mature tree, no longer consuming resources.
                self.cells[Self::idx(x, y)] = Cell::new(Element::Leaves);
            }
        }
    }

    fn update_leaves(&mut self, x: i32, y: i32) {
        // Supported if any Wood/Seed is within radius 2 (counts other trees' branches too).
        let mut supported = false;
        'outer: for dy in -2..=2i32 {
            for dx in -2..=2i32 {
                if dx == 0 && dy == 0 { continue; }
                let e = self.get(x + dx, y + dy).el;
                if e == Element::Wood || e == Element::Seed {
                    supported = true;
                    break 'outer;
                }
            }
        }
        if supported { return; }

        // Unsupported: slow, light fall (probabilistic to simulate drifting).
        if rand::gen_range::<u8>(0, 4) != 0 { return; }
        if !self.should_fall() { return; }
        let me_cell = self.get(x, y);
        let (gx, gy) = self.gravity_step();
        let gdy = gy.signum();
        if self.can_enter(me_cell, x + gx, y + gy, gdy) {
            self.swap(x, y, x + gx, y + gy); self.mark(x + gx, y + gy); return;
        }
        let (sa, sb) = self.gravity_sides();
        let sides = if rand::gen_range::<u8>(0, 2) == 0 { [sa, sb] } else { [sb, sa] };
        for (sx, sy) in sides {
            let tx = x + gx + sx;
            let ty = y + gy + sy;
            if self.can_enter(me_cell, tx, ty, gdy) {
                self.swap(x, y, tx, ty); self.mark(tx, ty); return;
            }
        }
    }

    fn update_gravel(&mut self, x: i32, y: i32) {
        let me_cell = self.get(x, y);
        let me_p = cell_physics(me_cell);
        // Pressure shove first — strong gradients override gravity.
        if self.try_pressure_shove(x, y, me_cell) { return; }
        if !self.should_fall() { return; }
        let (gx, gy) = self.gravity_step();
        let bx = x + gx;
        let by = y + gy;
        if !Self::in_bounds(bx, by) { return; }
        let below = self.get(bx, by);
        let below_p = cell_physics(below);
        let k = below_p.kind;
        // Rocks fall through empty, and sink into fluids by density — but
        // they sit on top of any rigid matter (other gravel, solids, powders).
        let can_fall = if k == Kind::Empty {
            true
        } else if k.is_fluid() {
            // Same viscosity gate: lava's crust doesn't plunge through the melt.
            if below_p.viscosity > 100 { false }
            else { !below.is_updated() && me_p.density > below_p.density }
        } else {
            false
        };
        if can_fall {
            self.swap(x, y, bx, by);
            self.mark(bx, by);
        }
    }

    fn update_powder(&mut self, x: i32, y: i32) {
        let me_cell = self.get(x, y);
        if self.try_pressure_shove(x, y, me_cell) { return; }
        if !self.should_fall() { return; }
        let (gx, gy) = self.gravity_step();
        let gdy = gy.signum();
        if self.can_enter(me_cell, x + gx, y + gy, gdy) {
            self.swap(x, y, x + gx, y + gy);
            self.mark(x + gx, y + gy);
            return;
        }
        // Two perpendicular slip directions — which we "fall-diagonal" through.
        let (sa, sb) = self.gravity_sides();
        let first_sa = rand::gen_range::<u8>(0, 2) == 0;
        let sides = if first_sa { [sa, sb] } else { [sb, sa] };
        for (sx, sy) in sides {
            let tx = x + gx + sx;
            let ty = y + gy + sy;
            if self.can_enter(me_cell, tx, ty, gdy) {
                self.swap(x, y, tx, ty);
                self.mark(tx, ty);
                return;
            }
        }
    }

    fn update_liquid(&mut self, x: i32, y: i32) {
        let me_cell = self.get(x, y);
        let me_p = cell_physics(me_cell);
        // Pressure shove first — a liquid next to a strong vacuum or
        // explosion gets flung before gravity decides anything.
        if self.try_pressure_shove(x, y, me_cell) { return; }
        let visc = me_p.viscosity;
        if !self.should_fall() { return; }
        let (gx, gy) = self.gravity_step();
        let gdy = gy.signum();
        // Gravity fall is NOT throttled by viscosity. Thick fluids like
        // lava and molten glass still fall at reasonable speed in real
        // life — honey poured from a jar drops cleanly; it's the sideways
        // spreading that molasses and lava do slowly. Throttling descent
        // made lava fall slower than leaves, which looked wrong.
        if self.can_enter(me_cell, x + gx, y + gy, gdy) {
            self.swap(x, y, x + gx, y + gy);
            self.mark(x + gx, y + gy);
            return;
        }
        // Diagonal fall — still gravity-driven, not viscosity-throttled.
        let (sa, sb) = self.gravity_sides();
        let first_sa = rand::gen_range::<u8>(0, 2) == 0;
        let diag_sides = if first_sa { [sa, sb] } else { [sb, sa] };
        for (sx, sy) in diag_sides {
            let tx = x + gx + sx;
            let ty = y + gy + sy;
            if self.can_enter(me_cell, tx, ty, gdy) {
                self.swap(x, y, tx, ty);
                self.mark(tx, ty);
                return;
            }
        }
        // Horizontal (perpendicular to gravity) spread IS throttled by
        // viscosity — this is where thick fluids visibly ooze and pool
        // instead of running flat. Scale: visc 0 always spreads, 400
        // never spreads.
        if visc > 0 && rand::gen_range::<u16>(0, 400) < visc {
            return;
        }
        let dispersion: i32 = if visc > 150 { 1 } else { 5 };
        for (sx, sy) in diag_sides {
            for step in 1..=dispersion {
                let nx = x + sx * step;
                let ny = y + sy * step;
                if !self.can_enter(me_cell, nx, ny, 0) { break; }
                if step == dispersion
                    || !self.can_enter(me_cell, nx + sx, ny + sy, 0)
                {
                    self.swap(x, y, nx, ny);
                    self.mark(nx, ny);
                    return;
                }
            }
        }
    }

    fn update_gas(&mut self, x: i32, y: i32, wind: Vec2) {
        let i = Self::idx(x, y);
        let me_cell = self.cells[i];
        let me = me_cell.el;
        // Edge dissipation: gas at a horizontal edge vents off-world into
        // the implied open atmosphere beyond the play space. Rate is
        // deliberately aggressive AND biased by wind: a gas with strong
        // outward-blowing wind evacuates near-instantly, matching the
        // user-expected "left/right barriers aren't walls" behavior.
        // Ceiling/floor (vertical edges) are kept solid — sky cap and
        // ground.
        if x == 0 || x == W as i32 - 1 {
            let outward = if x == 0 { -1.0 } else { 1.0 };
            let wind_out = (wind.x * outward).max(0.0);
            let rate = (0.30 + wind_out * 0.50).clamp(0.3, 1.0);
            if rand::gen_range::<f32>(0.0, 1.0) < rate {
                self.cells[i] = Cell::EMPTY;
                return;
            }
        }
        // Decay: only gases that carry a finite lifetime (Fire, Steam)
        // tick down and convert. Atomic gases (H, He, N, O, Ne, Cl) and
        // CO₂ are persistent — they start with life = 0 and must NOT
        // be treated as expired on frame one.
        if matches!(me, Element::Fire | Element::Steam) {
            if self.cells[i].life == 0 {
                self.cells[i] = match me {
                    Element::Steam => if rand::gen_range::<u8>(0, 5) == 0 { Cell::new(Element::Water) } else { Cell::EMPTY },
                    Element::Fire  => if rand::gen_range::<u8>(0, 3) == 0 { Cell::new(Element::CO2) } else { Cell::EMPTY },
                    _              => Cell::EMPTY,
                };
                return;
            }
            self.cells[i].life -= 1;
        }

        // Fire lingers near fuel so users can "hold" flame next to wood and
        // let it warm up. Alone in open air, fire rises only sometimes.
        if me == Element::Fire {
            let mut fuel_near = false;
            'f: for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if dx == 0 && dy == 0 { continue; }
                    let e = self.get(x + dx, y + dy).el;
                    if matches!(e, Element::Wood | Element::Leaves | Element::Seed) {
                        fuel_near = true; break 'f;
                    }
                }
            }
            if fuel_near { return; }
            if rand::gen_range::<u8>(0, 4) != 0 { return; }
        }

        // ---- Pressure-gradient bias with acceleration ----
        // Gases flow toward the lowest-pressure reachable neighbor. In steep
        // gradients (strong suction, explosions), gas hops multiple cells
        // per frame — each hop recomputes the gradient from the new
        // position, so gas accelerates as it approaches the low-P center
        // (the gradient gets steeper the closer it gets).
        let my_compl = me.pressure_p().compliance as i32;
        if my_compl > 0 {
            let grad_dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            const MAX_HOPS: u8 = 8;
            let mut cur_x = x;
            let mut cur_y = y;
            let mut moved = false;
            for hop in 0..MAX_HOPS {
                let cur_i = Self::idx(cur_x, cur_y);
                let my_p_now = self.cells[cur_i].pressure as i32;
                let mut best_drop: i32 = 0;
                let mut best_dir: (i32, i32) = (0, 0);
                let grad_start = rand::gen_range::<usize>(0, 4);
                for k in 0..4 {
                    let (dx, dy) = grad_dirs[(grad_start + k) % 4];
                    let nx = cur_x + dx;
                    let ny = cur_y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    if !self.can_enter(me_cell, nx, ny, dy) { continue; }
                    let ni = Self::idx(nx, ny);
                    let drop = my_p_now - self.cells[ni].pressure as i32;
                    if drop > best_drop {
                        best_drop = drop;
                        best_dir = (dx, dy);
                    }
                }
                if best_drop == 0 { break; }
                let take = ((best_drop * my_compl) / 64).clamp(0, 255) as u8;
                if rand::gen_range::<u8>(0, 255) >= take { break; }
                let nx = cur_x + best_dir.0;
                let ny = cur_y + best_dir.1;
                self.swap(cur_x, cur_y, nx, ny);
                self.mark(nx, ny);
                cur_x = nx;
                cur_y = ny;
                moved = true;
                // Continue hopping while gradient is meaningful. Threshold
                // is low so vacuum-scale fields (even with their per-cell
                // drops of 50-200) keep the gas accelerating across the
                // play area, not just at the very center of the well.
                let _ = hop;
                if best_drop < 50 { break; }
            }
            if moved { return; }
        }

        // ---- Empty expansion (runs BEFORE buoyancy) ----
        // A gas next to vacuum rushes into it — *this is the primary drive
        // of gas motion*. We do this before buoyancy because otherwise
        // light gases with a ~90% rise probability never get a chance to
        // spread laterally into side-facing empty pockets; they just
        // stream upward, leaving gaps. With expansion first, gases fill
        // every nearby vacuum in any direction; buoyancy only operates on
        // cells already surrounded by other matter.
        //
        // Wind biases the expansion probability per direction (dot product
        // of wind · dir), but ONLY for cells that are exposed to the
        // atmosphere — reachable from an open edge via non-frozen cells.
        // Gas inside a sealed container sits in an unexposed pocket and
        // doesn't feel external wind; gas in an open-topped beaker IS
        // exposed (through the opening) and does.
        {
            let exposed = self.wind_exposed
                .get(i).copied().unwrap_or(false);
            let exp_dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            let exp_start = rand::gen_range::<usize>(0, 4);
            for k in 0..4 {
                let (dx, dy) = exp_dirs[(exp_start + k) % 4];
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let ni = Self::idx(nx, ny);
                if self.cells[ni].el != Element::Empty { continue; }
                if self.cells[ni].is_updated() { continue; }
                let prob = if exposed {
                    let wind_bias = wind.x * dx as f32 + wind.y * dy as f32;
                    (0.5 + wind_bias * 0.35).clamp(0.05, 0.98)
                } else {
                    0.5 // unexposed: baseline diffusion, no wind push
                };
                if rand::gen_range::<f32>(0.0, 1.0) < prob {
                    self.swap(x, y, nx, ny);
                    self.mark(nx, ny);
                    return;
                }
            }
        }

        // ---- Ambient buoyancy (global force, not neighbor comparison) ----
        // Gases feel a force based on their mass vs the ambient atmosphere
        // (AMBIENT_AIR ≈ 29 g/mol). Lighter → rise against gravity; heavier
        // → sink with gravity. This is a WORLD-LEVEL effect, not a
        // per-neighbor one, so Empty cells don't masquerade as "air" and
        // stratify heavy gases — they're just vacuum. Gas-gas density
        // stratification still emerges because lighter gases have a higher
        // rise probability than heavier ones, so they migrate past each
        // other over time.
        let my_mass = me.physics().molar_mass;
        if my_mass > 0.0 && self.gravity > 0.0 {
            let air_mass = AMBIENT_AIR.molar_mass;
            let bias = (air_mass - my_mass) / air_mass * 255.0
                * self.gravity.clamp(0.0, 1.0);
            let (gx, gy) = self.gravity_step();
            let dir = if bias > 0.0 { (-gx, -gy) } else { (gx, gy) };
            let bias_abs = bias.abs().min(255.0) as u8;
            if bias_abs > 0 && rand::gen_range::<u8>(0, 255) < bias_abs
                && self.can_enter(me_cell, x + dir.0, y + dir.1, dir.1)
            {
                self.swap(x, y, x + dir.0, y + dir.1);
                self.mark(x + dir.0, y + dir.1);
                return;
            }
        }

        // ---- Brownian gas-gas mixing + wind ----
        // Final fallback: random cardinal diffusion into other *gas* cells.
        // Empty was already handled in the expansion pass above.
        const DIFF_P: u8 = 100;
        let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let start = rand::gen_range::<usize>(0, 4);
        for k in 0..4 {
            let (dx, dy) = dirs[(start + k) % 4];
            let nx = x + dx;
            let ny = y + dy;
            if !Self::in_bounds(nx, ny) { continue; }
            if self.cells[Self::idx(nx, ny)].el == Element::Empty { continue; }
            if !self.can_enter(me_cell, nx, ny, dy) { continue; }
            let exposed = self.wind_exposed.get(i).copied().unwrap_or(false);
            let wind_bonus = if exposed {
                wind.x * dx as f32 + wind.y * dy as f32
            } else { 0.0 };
            let p = (DIFF_P as f32 * (1.0 + wind_bonus)).clamp(0.0, 255.0) as u8;
            if rand::gen_range::<u8>(0, 255) < p {
                self.swap(x, y, nx, ny);
                self.mark(nx, ny);
                return;
            }
        }
    }

    // Heat tool: add `delta` degrees to every cell in a circular brush. Works
    // on any cell including frozen matter — heating a frozen ice block still
    // melts it. Clamped to a wide range to tolerate lava-scale sources.
    // Attempt to shove a matter cell based on net pressure force across it.
    // Summed over 4 cardinal neighbors: high-P on one side pushes the cell
    // the opposite direction. This correctly models a piston / wall under
    // blast pressure — even when the cell itself has permeability=0 (so its
    // own pressure doesn't track the blast), the neighbor differential
    // produces a net outward force. Previously we only checked "cell P
    // minus neighbor P" which missed walls entirely because blast pressure
    // couldn't enter them.
    //
    // Threshold prevents ordinary hydrostatic variations from rearranging
    // sand piles; only explosion-scale net force pushes matter.
    fn try_pressure_shove(&mut self, x: i32, y: i32, me_cell: Cell) -> bool {
        let compliance = me_cell.el.pressure_p().compliance as i32;
        if compliance == 0 { return false; }
        // Compute net pressure force vector across the cell.
        let mut net_x: i32 = 0;
        let mut net_y: i32 = 0;
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nx = x + dx;
            let ny = y + dy;
            let n_p = if Self::in_bounds(nx, ny) {
                self.cells[Self::idx(nx, ny)].pressure as i32
            } else {
                0
            };
            // Pressure on side (dx, dy) pushes toward (-dx, -dy).
            net_x -= dx * n_p;
            net_y -= dy * n_p;
        }
        let mag = ((net_x * net_x + net_y * net_y) as f32).sqrt() as i32;
        if mag < 400 { return false; }
        // Discretize direction to the dominant cardinal.
        let (step_x, step_y) = if net_x.abs() >= net_y.abs() {
            (net_x.signum(), 0)
        } else {
            (0, net_y.signum())
        };
        if step_x == 0 && step_y == 0 { return false; }
        let nx = x + step_x;
        let ny = y + step_y;
        if !Self::in_bounds(nx, ny) { return false; }
        if !self.can_enter(me_cell, nx, ny, step_y) { return false; }
        let take = ((mag * compliance) / 512).clamp(0, 255) as u8;
        if rand::gen_range::<u8>(0, 255) >= take { return false; }
        self.swap(x, y, nx, ny);
        self.mark(nx, ny);
        true
    }

    fn apply_heat(&mut self, cx: i32, cy: i32, radius: i32, delta: i16) {
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy > radius * radius { continue; }
                if !Self::in_bounds(x, y) { continue; }
                let idx = Self::idx(x, y);
                let t = self.cells[idx].temp as i32 + delta as i32;
                self.cells[idx].temp = t.clamp(-273, 5000) as i16;
            }
        }
    }

    // Stir tool — random pairwise swaps among non-frozen cells inside
    // the brush disk. Simulates finger-mixing of layered powders for
    // reactions that need homogeneous mixtures (thermite, gunpowder
    // constituents, brine, dissolved salts). One invocation does
    // 5x cell-count swaps, which is enough to fully randomize the
    // disk's contents in a single press — thorough mix, not gradual.
    // Triggered by an X keypress over the sim, no drag required.
    // Frozen cells (built walls) are excluded so structures don't
    // get scrambled. Cell counts and properties are preserved by
    // construction (swap, not regenerate).
    fn stir(&mut self, cx: i32, cy: i32, radius: i32) {
        let r2 = radius * radius;
        let mut indices: Vec<usize> = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if !Self::in_bounds(x, y) { continue; }
                let i = Self::idx(x, y);
                if self.cells[i].is_frozen() { continue; }
                indices.push(i);
            }
        }
        let n = indices.len();
        if n < 2 { return; }
        // 15x cell count gives near-perfect homogenization in a single
        // press, eliminating orphan single-element patches that block
        // thermite cascades. 5x was too patchy — left clusters of pure
        // Rust and pure Al that remained unreacted because they had no
        // cross-element neighbors.
        let swap_count = n * 15;
        for _ in 0..swap_count {
            let a_idx = rand::gen_range::<i32>(0, n as i32) as usize;
            let b_idx = rand::gen_range::<i32>(0, n as i32) as usize;
            if a_idx == b_idx { continue; }
            let ia = indices[a_idx];
            let ib = indices[b_idx];
            self.cells.swap(ia, ib);
        }
    }

    // Vacuum tool: directly pull every gas cell in the play space toward
    // the cursor each frame, then delete gas in the cursor's eat radius.
    // Pulling first and eating second means gas that arrives at the tool
    // this frame is consumed this frame. Hop counts scale with distance
    // so gas accelerates dramatically as it approaches. Walls block the
    // pull so genuinely sealed pockets stay untouched.
    fn apply_vacuum(&mut self, cx: i32, cy: i32, radius: i32) {
        // Pressure field — kept for inspector feedback and for the main
        // pressure-gradient motion system to visualize the suction. Actual
        // motion is driven by the direct pull below.
        let max_dist = (((W * W + H * H) as f32).sqrt()).max(1.0);
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let dx = x - cx;
                let dy = y - cy;
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                let idx = Self::idx(x, y);
                let falloff = (1.0 - d / max_dist).max(0.0);
                let target = -(4000.0 * falloff) as i16;
                if self.cells[idx].pressure > target {
                    self.cells[idx].pressure = target;
                }
            }
        }

        // Direct gas pull — teleport-to-last-empty. For each gas cell, walk
        // a straight-ish path toward the cursor (passing through other gas
        // cells and empty cells, stopped by walls and by already-moved
        // cells). Track the furthest EMPTY cell along that path, and swap
        // the source gas directly into it. Intermediate gas cells are
        // left in place — they'll get the same treatment in their own
        // turn, so the whole stream advances together each frame instead
        // of queueing behind a leader.
        if self.vacuum_moved.len() != W * H {
            self.vacuum_moved.resize(W * H, false);
        }
        for b in self.vacuum_moved.iter_mut() { *b = false; }
        let moved = &mut self.vacuum_moved;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let idx = Self::idx(x, y);
                if moved[idx] { continue; }
                let cell = self.cells[idx];
                if !matches!(cell_physics(cell).kind, Kind::Gas | Kind::Fire) { continue; }
                let d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y);
                let d = (d2 as f32).sqrt();
                if d < 1.0 { continue; }
                // How far we're willing to walk this frame, by distance.
                let max_steps: i32 = if d < 40.0  { 30 }
                                else if d < 100.0 { 20 }
                                else if d < 200.0 { 15 }
                                else              { 10 };
                let mut cur_x = x;
                let mut cur_y = y;
                let mut last_empty: Option<(i32, i32)> = None;
                for _ in 0..max_steps {
                    let ddx = cx - cur_x;
                    let ddy = cy - cur_y;
                    if ddx == 0 && ddy == 0 { break; }
                    let step_x = if ddx.abs() >= ddy.abs() { ddx.signum() } else { 0 };
                    let step_y = if ddy.abs() >  ddx.abs() { ddy.signum() } else { 0 };
                    if step_x == 0 && step_y == 0 { break; }
                    let nx = cur_x + step_x;
                    let ny = cur_y + step_y;
                    if !Self::in_bounds(nx, ny) { break; }
                    let nidx = Self::idx(nx, ny);
                    if moved[nidx] { break; } // another gas already claimed
                    let tgt = self.cells[nidx];
                    let tk = cell_physics(tgt).kind;
                    // Walls stop the walk entirely. Gas is passable but
                    // isn't a valid landing spot — we keep walking past
                    // to find an Empty. Empty is a valid landing spot.
                    if !matches!(tk, Kind::Empty | Kind::Gas | Kind::Fire) { break; }
                    if tk == Kind::Empty {
                        last_empty = Some((nx, ny));
                    }
                    cur_x = nx;
                    cur_y = ny;
                }
                if let Some((fx, fy)) = last_empty {
                    let final_idx = Self::idx(fx, fy);
                    self.cells.swap(idx, final_idx);
                    moved[final_idx] = true;
                }
            }
        }

        // Eat gas/fire in eat radius — AFTER pulling so just-arrived gas
        // is consumed on the same frame it reaches the cursor.
        let eat_r = radius * 2;
        for y in (cy - eat_r)..=(cy + eat_r) {
            for x in (cx - eat_r)..=(cx + eat_r) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy > eat_r * eat_r { continue; }
                if !Self::in_bounds(x, y) { continue; }
                let idx = Self::idx(x, y);
                let k = cell_physics(self.cells[idx]).kind;
                if matches!(k, Kind::Gas | Kind::Fire) {
                    self.cells[idx] = Cell::EMPTY;
                }
            }
        }
    }

    // (grab_cells/drop_cells removed — the pipet tool's unified
    // collect/release replaces positional-snapshot semantics.)

    // Stamp a prefab structure centered at (cx, cy). All wall cells are
    // placed as frozen so they act as rigid structure by default — users
    // can still heat/blast them off afterwards.
    // Stamp a wire along a Bresenham line from (x0, y0) to (x1, y1) with
    // the given material and per-cell disk thickness. Cells are placed
    // frozen (wires are structural). Empty targets only — won't overwrite
    // existing matter (same protection as the paint tool).
    pub fn place_wire_line(
        &mut self, x0: i32, y0: i32, x1: i32, y1: i32,
        el: Element, thickness: i32,
    ) {
        if el == Element::Empty { return; }
        let r = thickness.max(1);
        let r2 = r * r;
        let stamp_disk = |world: &mut Self, cx: i32, cy: i32| {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy > r2 { continue; }
                    let x = cx + dx;
                    let y = cy + dy;
                    if !Self::in_bounds(x, y) { continue; }
                    let idx = Self::idx(x, y);
                    // Wires overwrite whatever's there — including
                    // frozen walls. Makes wiring through containers
                    // sane instead of pixel-hunting the right seam.
                    let mut c = Cell::new(el);
                    c.flag |= Cell::FLAG_FROZEN;
                    world.cells[idx] = c;
                }
            }
        };
        // Bresenham line between endpoints.
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = x0;
        let mut y = y0;
        loop {
            stamp_disk(self, x, y);
            if x == x1 && y == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    }

    pub fn place_prefab(
        &mut self, cx: i32, cy: i32,
        kind: PrefabKind, el: Element,
        thickness: i32, w: i32, h: i32, rotation: u8,
    ) {
        if el == Element::Empty { return; }
        // Rotation swaps width/height for the bounding box so a 40×50
        // rotated 90° occupies 50×40 on screen. Orientation-dependent
        // asymmetries (Beaker's open top, Battery's terminal placement)
        // also shift by reinterpreting which edge is which.
        let rot = rotation & 3;
        let (bw, bh) = if rot == 1 || rot == 3 { (h, w) } else { (w, h) };
        let x0 = cx - bw / 2;
        let y0 = cy - bh / 2;
        let x1 = x0 + bw;
        let y1 = y0 + bh;
        let stamp = |world: &mut Self, x: i32, y: i32, target: Element| {
            if !Self::in_bounds(x, y) { return; }
            let idx = Self::idx(x, y);
            if world.cells[idx].el != Element::Empty { return; }
            let mut c = Cell::new(target);
            c.flag |= Cell::FLAG_FROZEN;
            world.cells[idx] = c;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let d_top    = y - y0;
                let d_bottom = y1 - 1 - y;
                let d_left   = x - x0;
                let d_right  = x1 - 1 - x;
                // Remap which screen direction is the prefab's
                // "canonical" top/bottom/left/right based on rotation.
                // Rotation rotates the prefab clockwise; from the
                // prefab's perspective, its "up" now points in a
                // different screen direction, so we reverse the lookup.
                let (d_up, d_down, d_l, d_r) = match rot {
                    0 => (d_top,    d_bottom, d_left,  d_right),
                    1 => (d_right,  d_left,   d_top,   d_bottom),
                    2 => (d_bottom, d_top,    d_right, d_left),
                    3 => (d_left,   d_right,  d_bottom, d_top),
                    _ => (d_top,    d_bottom, d_left,  d_right),
                };
                let dx_edge = d_l.min(d_r);
                let dy_edge_any = d_up.min(d_down);
                match kind {
                    PrefabKind::Box => {
                        if dx_edge < thickness || dy_edge_any < thickness {
                            stamp(self, x, y, el);
                        }
                    }
                    PrefabKind::Beaker => {
                        // Left + right + bottom walls, open top
                        // (in canonical/unrotated space).
                        if dx_edge < thickness || d_down < thickness {
                            stamp(self, x, y, el);
                        }
                    }
                    PrefabKind::Battery => {
                        // Top band = BattPos, bottom = BattNeg (in
                        // canonical space). Middle is the user-chosen
                        // material — insulator keeps terminals isolated,
                        // conductor deliberately shorts.
                        if d_up < thickness {
                            stamp(self, x, y, Element::BattPos);
                        } else if d_down < thickness {
                            stamp(self, x, y, Element::BattNeg);
                        } else {
                            stamp(self, x, y, el);
                        }
                    }
                }
            }
        }
    }

    // Pipet collect: scan the brush, move matching non-frozen cells into
    // the bucket (preserving their full state — temp, phase, moisture,
    // etc.). If `target` is Some, only cells of that species are taken;
    // if None, any non-frozen non-empty cell qualifies. Frozen cells are
    // never siphoned — the pipet can't drain structural walls.
    fn pipet_collect(
        &mut self, cx: i32, cy: i32, radius: i32,
        target: Option<(Element, u8)>, bucket: &mut Vec<Cell>, limit: usize,
    ) {
        if bucket.len() >= limit { return; }
        let r2 = radius * radius;
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                if bucket.len() >= limit { return; }
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy > r2 { continue; }
                if !Self::in_bounds(x, y) { continue; }
                let idx = Self::idx(x, y);
                let c = self.cells[idx];
                if c.el == Element::Empty { continue; }
                if c.is_frozen() { continue; }
                if let Some((tel, tdid)) = target {
                    if c.el != tel { continue; }
                    if c.el == Element::Derived && c.derived_id != tdid {
                        continue;
                    }
                }
                bucket.push(c);
                self.cells[idx] = Cell::EMPTY;
            }
        }
    }

    // Pipet release: pop cells from the bucket and place them into empty
    // cells in the brush. Cells retain whatever state they had when
    // collected (temperature carries over, etc.). Returns the number
    // placed this call.
    fn pipet_release(
        &mut self, cx: i32, cy: i32, radius: i32,
        bucket: &mut Vec<Cell>,
    ) -> usize {
        if bucket.is_empty() { return 0; }
        let r2 = radius * radius;
        let mut placed = 0;
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                if bucket.is_empty() { return placed; }
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy > r2 { continue; }
                if !Self::in_bounds(x, y) { continue; }
                let idx = Self::idx(x, y);
                if self.cells[idx].el != Element::Empty { continue; }
                if let Some(c) = bucket.pop() {
                    self.cells[idx] = c;
                    placed += 1;
                }
            }
        }
        placed
    }

    pub fn paint(&mut self, cx: i32, cy: i32, radius: i32, el: Element, derived_id: u8, frozen: bool) {
        let painting_solid = matches!(el.physics().kind, Kind::Solid | Kind::Gravel);
        // Liquids paint as scattered drops, not a solid fill. 1-in-50 per
        // frame per brush cell reads as a natural pour/rainfall and avoids
        // the "packed mass falling apart" visual that solid-fill liquids
        // produced (especially Hg/Lava). In build mode we paint solidly
        // regardless.
        let sparsity: u16 = if frozen {
            1
        } else {
            match el.physics().kind {
                Kind::Liquid => 50,
                _ => 1,
            }
        };
        // Over-paint pressure boost. Holding the paint button on existing
        // gas/liquid cells stacks pressure so sealed volumes genuinely
        // pressurize. *Fresh* spawns don't get this boost — they just use
        // the element's formation_pressure (see Cell::new). Without that
        // distinction, heavy gases painted into an open container jet out
        // the opening instead of settling, because the 400 paint pressure
        // overwhelms their weight-driven buoyancy.
        let overpaint_pressure: i16 = match el.physics().kind {
            Kind::Gas | Kind::Fire => 400,
            Kind::Liquid           => 200,
            _                      => 0,
        };
        for y in (cy - radius)..=(cy + radius) {
            for x in (cx - radius)..=(cx + radius) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy > radius * radius { continue; }
                if !Self::in_bounds(x, y) { continue; }
                if sparsity > 1 && rand::gen_range::<u16>(0, sparsity) != 0 { continue; }
                let idx = Self::idx(x, y);
                let existing_cell = self.cells[idx];
                let existing = existing_cell.el;
                let _ = painting_solid; // kept out of the new gate below
                if el == Element::Empty {
                    // Erase — clears frozen state too.
                    self.cells[idx] = Cell::EMPTY;
                } else if existing == el {
                    // Over-paint same element: don't recreate the cell (that
                    // would reset temp/life/etc.), just *stack pressure* on
                    // top. This lets the user pressurize a sealed volume by
                    // holding the paint button — each frame adds to an
                    // already-high pressure instead of clipping at one kick.
                    self.cells[idx].pressure =
                        existing_cell.pressure.saturating_add(overpaint_pressure);
                } else if existing != Element::Empty && !frozen {
                    // Non-build-mode paint is additive: it fills empty
                    // cells and stacks pressure on same-element cells,
                    // but never overwrites different matter. Prevents
                    // accidents like painting gold over your iron stash.
                    // Erase first (or switch to Build mode) to modify
                    // existing structure.
                    continue;
                } else {
                    let mut c = Cell::new(el);
                    // Derived-compound spawn: tag with the caller-provided
                    // registry index so the cell maps to the right
                    // compound entry (color, phase points, reactivity).
                    if el == Element::Derived {
                        c.derived_id = derived_id;
                    }
                    if frozen && el != Element::Empty {
                        c.flag |= Cell::FLAG_FROZEN;
                        // Build-mode solids inherit the replaced cell's
                        // pressure. Otherwise a fresh frozen cell spawns at
                        // pressure 0 and immediately registers a pressure gap
                        // against its hydrostatically-settled neighbors — or
                        // worse, against a nearby pressurized region — and
                        // crumbles mid-stroke as the user draws.
                        c.pressure = existing_cell.pressure;
                    }
                    // Non-frozen spawns use the element's natural formation
                    // pressure from Cell::new. No extra kick — heavy gases
                    // then settle instead of jetting through any opening.
                    self.cells[idx] = c;
                }
            }
        }
    }
}

// Bundled TTF font — embedded so the binary is self-contained and we get
// crisp text instead of macroquad's default bitmap atlas, which renders
// poorly at non-default sizes and stretched aspect ratios.
const UI_FONT_BYTES: &[u8] = include_bytes!("../assets/Roboto-Regular.ttf");

thread_local! {
    static UI_FONT: std::cell::OnceCell<Font> = const { std::cell::OnceCell::new() };
}

fn init_ui_font() {
    let font = load_ttf_font_from_bytes(UI_FONT_BYTES)
        .expect("bundled Roboto font should load");
    UI_FONT.with(|cell| { let _ = cell.set(font); });
}

// Central UI-text helper — routes all text through the bundled TTF so
// rendering stays consistent regardless of window scale.
fn draw_ui_text(text: &str, x: f32, y: f32, size: f32, color: Color) {
    UI_FONT.with(|cell| {
        let font = cell.get();
        draw_text_ex(text, x, y, TextParams {
            font,
            font_size: size as u16,
            font_scale: 1.0,
            font_scale_aspect: 1.0,
            rotation: 0.0,
            color,
        });
    });
}

// Measure using the bundled TTF so tooltip backgrounds line up with the
// text. macroquad's `measure_text` with font=None uses the default bitmap
// font which produces a different glyph width.
fn measure_ui_text(text: &str, size: u16) -> TextDimensions {
    UI_FONT.with(|cell| {
        measure_text(text, cell.get(), size, 1.0)
    })
}

// RGB-returning version used in the hot render loop — avoids the
// float-to-u8 detour through macroquad's Color. Called once per cell per
// frame so it has to stay tight.
#[inline]
fn color_rgb(c: Cell) -> [u8; 3] {
    // Derived compounds pull their color from the runtime registry.
    let (r, g, b) = if c.el == Element::Derived {
        derived_color_of(c.derived_id)
    } else {
        c.el.base_color()
    };
    let clamp_u8 = |v: i16| -> u8 { v.clamp(0, 255) as u8 };
    let v = (c.seed as i16 - 128) / 16;
    let (mut r, mut g, mut b) = match c.el {
        Element::Fire => {
            let t = (c.life as f32 / 70.0).min(1.0);
            let mut rr = (240.0 - 40.0 * (1.0 - t)) as u8;
            let mut gg = (50.0 + 100.0 * t) as u8;
            let mut bb = 20u8;
            // Metal-salt flame coloring — real flame-test colors. When
            // the Fire cell carries a known metal in its solute_el slot
            // (set by the color_fires() pass when adjacent to that
            // metal/salt), tint the flame toward the metal's
            // characteristic emission color. 70% metal / 30% base
            // gives a saturated colored flame that still reads as fire.
            if let Some((fr, fg, fb)) = flame_color(c.solute_el) {
                let mix = 0.7f32;
                rr = ((rr as f32) * (1.0 - mix) + fr as f32 * mix) as u8;
                gg = ((gg as f32) * (1.0 - mix) + fg as f32 * mix) as u8;
                bb = ((bb as f32) * (1.0 - mix) + fb as f32 * mix) as u8;
            }
            (rr, gg, bb)
        }
        Element::Lava => {
            let flick = (c.seed as i16 - 128) / 6;
            (clamp_u8(r as i16 + flick), clamp_u8(g as i16 + flick / 2), b)
        }
        _ => (clamp_u8(r as i16 + v), clamp_u8(g as i16 + v), clamp_u8(b as i16 + v)),
    };
    if c.moisture > 20 && c.el != Element::Water && c.el != Element::Empty {
        let wet = ((c.moisture as f32 - 20.0) / 235.0).clamp(0.0, 1.0) * 0.55;
        r = ((r as f32) * (1.0 - wet)) as u8;
        g = ((g as f32) * (1.0 - wet * 0.9)) as u8;
        b = ((b as f32) * (1.0 - wet * 0.6) + 70.0 * wet) as u8;
    }
    // Dissolved solute tints the water toward the solute's own color. Mixes
    // up to ~0.55 at saturation — enough to read (saltwater pales, FeCl
    // water yellows, CuCl turns cyan) without losing the water identity.
    if c.el == Element::Water && c.solute_amt > 0 {
        let (sr, sg, sb) = if c.solute_el == Element::Derived {
            derived_color_of(c.solute_derived_id)
        } else {
            c.solute_el.base_color()
        };
        let t = (c.solute_amt as f32 / 255.0) * 0.55;
        r = ((r as f32) * (1.0 - t) + (sr as f32) * t) as u8;
        g = ((g as f32) * (1.0 - t) + (sg as f32) * t) as u8;
        b = ((b as f32) * (1.0 - t) + (sb as f32) * t) as u8;
    }
    if c.temp > 250 && c.el != Element::Fire {
        // Stage 1: cool → red → orange → yellow (250-1750°C). Models
        // iron glowing red at ~700°C, orange at ~1100°C, yellow at
        // ~1500°C. Saturates by 1750°C at RGB(255, 200, 80).
        let warm_heat = ((c.temp - 250) as f32 / 1500.0).clamp(0.0, 1.0);
        let warm_mix = warm_heat * 0.8;
        r = ((r as f32) * (1.0 - warm_mix) + 255.0 * warm_mix) as u8;
        g = ((g as f32) * (1.0 - warm_mix) + 200.0 * warm_mix) as u8;
        b = ((b as f32) * (1.0 - warm_mix) + 80.0  * warm_mix) as u8;
        // Stage 2: yellow → white-hot (1750-3000°C). Real incandescence
        // shifts toward pure white at very high temps — Mg combustion
        // (~3000°C), thermite peak (~2500°C), and other extreme
        // exotherms should read as brilliant white, not just bright
        // orange. Blue channel ramps up faster than red/green, which
        // shifts the perceived color from yellow into white.
        if c.temp > 1750 {
            let white_t = ((c.temp - 1750) as f32 / 1250.0).clamp(0.0, 1.0);
            let white_mix = white_t * 0.9;
            r = ((r as f32) * (1.0 - white_mix) + 255.0 * white_mix) as u8;
            g = ((g as f32) * (1.0 - white_mix) + 255.0 * white_mix) as u8;
            b = ((b as f32) * (1.0 - white_mix) + 255.0 * white_mix) as u8;
        }
    }
    // Subtle brighten for frozen (rigid-body) cells so you can see what's locked.
    if c.is_frozen() && c.el != Element::Empty {
        r = r.saturating_add(20);
        g = g.saturating_add(20);
        b = b.saturating_add(30);
    }
    // Radioactive glow — bright pulsing cyan-green tint whose amplitude
    // tracks the atom's half-life (shorter → brighter). Seed-driven phase
    // + temp-driven secondary phase so a pile shimmers instead of flashing
    // in sync. Baseline is always visibly tinted so the atom reads as
    // radioactive even at rest; pulse rides on top.
    if let Some(a) = atom_profile_for(c.el) {
        if a.half_life_frames > 0 {
            let activity = (30000.0 / a.half_life_frames as f32).clamp(0.35, 1.0);
            let phase_byte = c.seed.wrapping_add((c.temp & 0xFF) as u8);
            let pulse_norm = ((phase_byte as f32) / 255.0) * 2.0 - 1.0;
            let pulse = 0.6 + 0.4 * pulse_norm.abs();
            let mix = (activity * pulse * 0.75).min(0.85);
            r = ((r as f32) * (1.0 - mix) + 120.0 * mix) as u8;
            g = ((g as f32) * (1.0 - mix) + 255.0 * mix) as u8;
            b = ((b as f32) * (1.0 - mix) + 160.0 * mix) as u8;
        }
    }
    // Phase tint for forced (non-native) phases. Molten atoms render darker
    // and warmer; boiled atoms wash toward the background; frozen-out
    // gases/liquids read as a cold bluish-grey. PHASE_NATIVE uses the
    // element's hand-tuned color with no tint.
    match c.phase() {
        PHASE_SOLID => {
            // shift toward cold bluish grey
            r = clamp_u8((r as i16) * 7 / 10);
            g = clamp_u8((g as i16) * 7 / 10);
            b = clamp_u8((b as i16) * 9 / 10 + 20);
        }
        PHASE_LIQUID => {
            // shift toward red/orange, darken slightly
            r = clamp_u8((r as i16) * 9 / 10 + 30);
            g = clamp_u8((g as i16) * 7 / 10 + 15);
            b = clamp_u8((b as i16) * 5 / 10);
        }
        PHASE_GAS => {
            // wash toward the dark background so gas atoms are faint
            r = clamp_u8((r as i16) / 2 + 8);
            g = clamp_u8((g as i16) / 2 + 8);
            b = clamp_u8((b as i16) / 2 + 12);
        }
        _ => {}
    }
    [r, g, b]
}


// Retained as a zero-height constant so existing code that offsets by
// TOP_BAR still compiles while the top bar itself is gone. The panel
// uses a small internal padding (PANEL_TOP_PAD) instead.
const TOP_BAR: f32 = 0.0;
const PANEL_TOP_PAD: f32 = 10.0;

// Paintable compound materials. Wood is intentionally omitted — trees make
// wood. MoltenGlass is intentionally omitted — you get it by melting sand.
// Fire is intentionally omitted — it emerges from combustion, driven by the
// Heat tool raising a fuel's temperature past its ignition threshold.
// These live in the periodic-table overlay under the atom grid.
const COMPOUND_PALETTE: [Element; 18] = [
    Element::Sand, Element::Water, Element::Stone,
    Element::CO2, Element::Steam, Element::Lava,
    Element::Obsidian, Element::Seed, Element::Mud, Element::Leaves,
    Element::Oil, Element::Ice, Element::Glass,
    Element::Gunpowder,
    Element::Quartz, Element::Firebrick,
    Element::Rust,
    Element::Empty,
];

// Compute how the sim viewport is laid out inside the current window.
// The sim texture scales to fit the available space (window minus the top
// bar), preserving aspect, centered horizontally.
// Width of the side control panel, in pixels. Reserved from the right edge
// of the window; the sim fits into whatever remains. Collapsing the panel
// (U key) returns this space to the simulation.
const PANEL_WIDTH: f32 = 240.0;

// Shared panel color — also used as the window clear color so any area of
// the window not covered by the sim (bottom dead space from aspect-ratio
// mismatch, etc.) visually merges with the panel instead of reading as
// ugly blank.
fn panel_bg() -> Color { Color::from_rgba(18, 18, 24, 255) }

// Button rect layout inside the panel. Returned in a fixed order so both
// the input phase (hit-testing) and the render phase (drawing) can agree
// without duplicating the y-accumulator logic. Order is:
// Indices into panel_button_rects(prefab_open, wire_open):
//   0 Paint  1 Heat  2 Vacuum  3 Pipet  4 Prefab  5 Wire  6 Build-toggle
const PANEL_BUTTON_COUNT: usize = 7;
const PANEL_BUTTON_BUILD: usize = 6;
const PANEL_BUTTON_WIRE: usize = 5;
const PANEL_BUTTON_PREFAB: usize = 4;

// Heights of the inline dropdown sub-panels that expand below their
// parent button when that tool is active. Push the buttons below them
// (and everything after) down by this amount so the content fits.
fn prefab_dropdown_height(open: bool) -> f32 {
    if open { 232.0 } else { 0.0 }
}
fn wire_dropdown_height(open: bool) -> f32 {
    if open { 110.0 } else { 0.0 }
}

fn panel_button_rects(prefab_open: bool, wire_open: bool) -> [(f32, f32, f32, f32); PANEL_BUTTON_COUNT] {
    let px = screen_width() - PANEL_WIDTH;
    let bx = px + 12.0;
    let bw = PANEL_WIDTH - 24.0;
    let bh = 30.0;
    let gap = 6.0;
    let mut y = PANEL_TOP_PAD + 22.0;  // room for the "TOOLS" section header
    let mut out = [(0.0, 0.0, 0.0, 0.0); PANEL_BUTTON_COUNT];
    // Tool buttons: Paint, Heat, Vacuum, Pipet — above the Prefab dropdown.
    for i in 0..4 {
        out[i] = (bx, y, bw, bh);
        y += bh + gap;
    }
    // Prefab button + its dropdown.
    out[PANEL_BUTTON_PREFAB] = (bx, y, bw, bh);
    y += bh + gap + prefab_dropdown_height(prefab_open);
    // Wire button + its dropdown.
    out[PANEL_BUTTON_WIRE] = (bx, y, bw, bh);
    y += bh + gap + wire_dropdown_height(wire_open);
    // Extra gap before Build — it's a modifier, not a mode.
    y += 14.0;
    out[PANEL_BUTTON_BUILD] = (bx, y, bw, bh);
    out
}

// Current-element readout — a single text line between the Build toggle
// and the SIMULATION section. Shown outside the button rects so hit-
// testing doesn't get confused with a button.
fn panel_element_rect(prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let build = panel_button_rects(prefab_open, wire_open)[PANEL_BUTTON_BUILD];
    let x = build.0;
    let y = build.1 + build.3 + 14.0;
    let w = build.2;
    let h = 18.0;
    (x, y, w, h)
}

// Ambient control rows — Temp, O₂, Gravity. Positioned below the element
// readout with a SIMULATION section header above them. Each row is a hit-
// target for scroll-on-hover adjustment.
const PANEL_AMBIENT_COUNT: usize = 3;
fn panel_ambient_rects(prefab_open: bool, wire_open: bool) -> [(f32, f32, f32, f32); PANEL_AMBIENT_COUNT] {
    let el = panel_element_rect(prefab_open, wire_open);
    let x = el.0;
    let w = el.2;
    let h = 28.0;
    let gap = 5.0;
    // Space for the SIMULATION header between the element row and the
    // first ambient row.
    let mut y = el.1 + el.3 + 28.0;
    let mut out = [(0.0, 0.0, 0.0, 0.0); PANEL_AMBIENT_COUNT];
    for i in 0..PANEL_AMBIENT_COUNT {
        out[i] = (x, y, w, h);
        y += h + gap;
    }
    out
}

// Wind widget — an 84×84 click/drag pad positioned below the three
// ambient rows. Click inside: wind direction = vector from center to
// click point; magnitude = distance scaled into [0, WIND_MAX].
const WIND_MAX: f32 = 2.0;
fn wind_pad_rect(prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let amb = panel_ambient_rects(prefab_open, wire_open);
    let last = amb[PANEL_AMBIENT_COUNT - 1];
    let size = 84.0;
    let x = last.0;
    let y = last.1 + last.3 + 28.0;
    (x, y, size, size)
}

// Reset-wind button — directly below the wind pad, full row width so the
// user has an obvious zero button.
fn wind_reset_rect(prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let pad = wind_pad_rect(prefab_open, wire_open);
    let row_w = panel_button_rects(prefab_open, wire_open)[PANEL_BUTTON_BUILD].2;
    let x = pad.0;
    let y = pad.1 + pad.3 + 10.0;
    (x, y, row_w, 22.0)
}

// Prefab tool dropdown — appears inline between the Prefab button and
// the Build button when Prefab is the active tool. Two kind-select
// buttons and three hover-scroll rows for thickness, width, and height.
const PREFAB_ROW_COUNT: usize = 4;
fn prefab_kind_rects() -> [(f32, f32, f32, f32); 3] {
    // Anchored to the Prefab button's unshifted position (dropdown lives
    // right below it). Use `false` so we get the Prefab button position
    // regardless of dropdown state.
    let prefab_btn = panel_button_rects(false, false)[4];
    let y = prefab_btn.1 + prefab_btn.3 + 10.0;
    let row_w = prefab_btn.2;
    let gap = 4.0;
    let each = (row_w - gap * 2.0) / 3.0;
    let h = 26.0;
    [
        (prefab_btn.0, y, each, h),
        (prefab_btn.0 + (each + gap), y, each, h),
        (prefab_btn.0 + (each + gap) * 2.0, y, each, h),
    ]
}
fn prefab_slider_rects() -> [(f32, f32, f32, f32); PREFAB_ROW_COUNT] {
    let kr = prefab_kind_rects();
    let x = kr[0].0;
    let w = panel_button_rects(false, false)[4].2;
    let h = 26.0;
    let gap = 5.0;
    let mut y = kr[0].1 + kr[0].3 + 10.0;
    let mut out = [(0.0, 0.0, 0.0, 0.0); PREFAB_ROW_COUNT];
    for i in 0..PREFAB_ROW_COUNT {
        out[i] = (x, y, w, h);
        y += h + gap;
    }
    out
}

fn prefab_material_rect() -> (f32, f32, f32, f32) {
    let sr = prefab_slider_rects();
    let last = sr[PREFAB_ROW_COUNT - 1];
    let x = last.0;
    let y = last.1 + last.3 + 6.0;
    (x, y, last.2, 24.0)
}

// Wire-tool dropdown layout. Sits directly below the Wire tool button.
// Reads the button position with wire_open=false so the sub-panel rects
// anchor to the button's un-shifted location (dropdown lives right below
// its own button regardless of its expanded state).
fn wire_material_rect() -> (f32, f32, f32, f32) {
    let wb = panel_button_rects(false, false)[PANEL_BUTTON_WIRE];
    (wb.0, wb.1 + wb.3 + 10.0, wb.2, 26.0)
}
fn wire_thickness_rect() -> (f32, f32, f32, f32) {
    let mr = wire_material_rect();
    (mr.0, mr.1 + mr.3 + 8.0, mr.2, 26.0)
}

// Pipet status panel. Height is chosen large enough to comfortably hold
// a short species breakdown list (grab-all mode with mixed bucket); in
// species-filter mode the extra vertical space is left empty.
fn pipet_status_rect(prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let rst = wind_reset_rect(prefab_open, wire_open);
    let x = rst.0;
    let y = rst.1 + rst.3 + 18.0;
    let w = rst.2;
    let h = 170.0;
    (x, y, w, h)
}

fn pipet_clear_rect(prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let status = pipet_status_rect(prefab_open, wire_open);
    let x = status.0;
    let y = status.1 + status.3 + 8.0;
    let w = status.2;
    let h = 26.0;
    (x, y, w, h)
}

// "Species present in scene" list — shown beneath the Clear button
// whenever the pipet is active. Each row is a clickable target picker.
// Rect for row index i (0-based).
fn species_list_row_rect(i: usize, prefab_open: bool, wire_open: bool) -> (f32, f32, f32, f32) {
    let clear = pipet_clear_rect(prefab_open, wire_open);
    let x = clear.0;
    let w = clear.2;
    let h = 20.0;
    // Gap below Clear button, then header (14 px) + small padding.
    let start_y = clear.1 + clear.3 + 30.0;
    (x, start_y + (i as f32) * (h + 2.0), w, h)
}

fn draw_panel_button(
    rect: (f32, f32, f32, f32),
    label: &str,
    selected: bool,
    hovered: bool,
) {
    let (x, y, w, h) = rect;
    let bg = if selected {
        Color::from_rgba(62, 90, 140, 255)
    } else if hovered {
        Color::from_rgba(42, 42, 54, 255)
    } else {
        Color::from_rgba(30, 30, 38, 255)
    };
    draw_rectangle(x, y, w, h, bg);
    draw_rectangle_lines(x, y, w, h, 1.0, Color::from_rgba(60, 60, 72, 255));
    let dim = measure_ui_text(label, 14);
    let tx = x + (w - dim.width) * 0.5;
    let ty = y + (h + dim.height) * 0.5 - 2.0;
    draw_ui_text(label, tx, ty, 14.0, Color::from_rgba(220, 220, 230, 255));
}

fn sim_layout(panel_w: f32, zoom: f32, pan_x: f32, pan_y: f32) -> (f32, f32, f32, f32, f32) {
    let avail_w = (screen_width() - panel_w).max(1.0);
    let avail_h = (screen_height() - TOP_BAR).max(1.0);
    let scale_fit = (avail_w / W as f32).min(avail_h / H as f32).max(0.5);
    let scale = scale_fit * zoom;
    let sim_w = W as f32 * scale;
    let sim_h = H as f32 * scale;
    let sim_x = (avail_w - sim_w) * 0.5 + pan_x;
    let sim_y = TOP_BAR + pan_y;
    (scale, sim_x, sim_y, sim_w, sim_h)
}

// ============================================================================
// PERIODIC TABLE OVERLAY
// ============================================================================

const PT_TILE: f32 = 44.0;
const PT_GAP: f32 = 3.0;
const PT_COLS: usize = 18;

// Vertical space reserved below the atom grid: compound row (tile + label)
// plus a gap plus the detail panel.
const PT_COMPOUND_ROW_H: f32 = PT_TILE + 40.0;
const PT_DETAIL_PANEL_H: f32 = 180.0;
const PT_BELOW_TABLE: f32 = PT_COMPOUND_ROW_H + 30.0 + PT_DETAIL_PANEL_H;

// Main table: 7 rows (periods 1-7). Then a 1-row gap, then 2 rows for
// lanthanides (period 8) and actinides (period 9).
fn pt_layout() -> (f32, f32, f32, f32) {
    let table_w = PT_TILE * PT_COLS as f32 + PT_GAP * (PT_COLS - 1) as f32;
    let rows: f32 = 7.0 + 1.0 + 2.0; // main + spacer + f-block
    let table_h = PT_TILE * rows + PT_GAP * (rows - 1.0);
    let tx = (screen_width() - table_w) * 0.5;
    let ty = (screen_height() - table_h - PT_BELOW_TABLE).max(60.0) * 0.5;
    (tx, ty, table_w, table_h)
}

// Top-left of the compound-submenu strip (directly below the atom grid).
fn pt_compound_origin() -> (f32, f32) {
    let (tx, ty, _, table_h) = pt_layout();
    (tx, ty + table_h + 28.0)
}

fn pt_compound_slot(i: usize) -> (f32, f32) {
    let (cx, cy) = pt_compound_origin();
    (cx + i as f32 * (PT_TILE + PT_GAP), cy)
}

fn pt_compound_hit(mx: f32, my: f32) -> Option<usize> {
    for i in 0..COMPOUND_PALETTE.len() {
        let (sx, sy) = pt_compound_slot(i);
        if mx >= sx && mx < sx + PT_TILE && my >= sy && my < sy + PT_TILE {
            return Some(i);
        }
    }
    None
}

// Hit-test for derived-compound tiles appended after the main compound
// palette. Returns an index into the caller-supplied derived-palette
// slice, which is rendered at COMPOUND_PALETTE.len() + i.
fn pt_derived_hit(mx: f32, my: f32, count: usize) -> Option<usize> {
    let base = COMPOUND_PALETTE.len();
    for i in 0..count {
        let (sx, sy) = pt_compound_slot(base + i);
        if mx >= sx && mx < sx + PT_TILE && my >= sy && my < sy + PT_TILE {
            return Some(i);
        }
    }
    None
}

// Map an atom's (period, group) to its on-screen tile position.
fn pt_tile_xy(atom: &AtomProfile, tx: f32, ty: f32) -> (f32, f32) {
    // Visual row: period 1-7 map to rows 0-6. The f-block gap takes row 7.
    // Period 8 (lanthanides) → row 8; period 9 (actinides) → row 9.
    let row = if atom.period <= 7 {
        atom.period as i32 - 1
    } else {
        (atom.period as i32) - 8 + 8  // period 8 → row 8, period 9 → row 9
    };
    let col = atom.group as i32 - 1;
    let x = tx + col as f32 * (PT_TILE + PT_GAP);
    let y = ty + row as f32 * (PT_TILE + PT_GAP);
    (x, y)
}

fn pt_hit(mx: f32, my: f32) -> Option<usize> {
    let (tx, ty, _, _) = pt_layout();
    for (i, a) in ATOMS.iter().enumerate() {
        let (x, y) = pt_tile_xy(a, tx, ty);
        if mx >= x && mx < x + PT_TILE && my >= y && my < y + PT_TILE {
            return Some(i);
        }
    }
    None
}

// Renders the PT detail panel's compound section — all the physical
// properties a user might want to glance at before painting.
fn draw_compound_detail(el: Element, tx: f32, panel_y: f32) {
    let px = tx + 14.0;
    let mut py = panel_y + 28.0;
    let phys = el.physics();
    let therm = el.thermal();
    let moist = el.moisture();
    let press = el.pressure_p();
    let elec = el.electrical();
    let kind_name = match phys.kind {
        Kind::Empty  => "empty",
        Kind::Solid  => "solid",
        Kind::Gravel => "gravel",
        Kind::Powder => "powder",
        Kind::Liquid => "liquid",
        Kind::Gas    => "gas",
        Kind::Fire   => "fire",
    };
    let dim = Color::from_rgba(175, 175, 190, 255);
    let section = Color::from_rgba(130, 140, 170, 255);

    draw_ui_text(&format!("{}    (compound)", el.name()), px, py, 24.0, WHITE);
    py += 28.0;

    // Physical:
    let mass_frag = if phys.molar_mass > 0.0 {
        format!("   |   mass {:.1} g/mol", phys.molar_mass)
    } else { String::new() };
    let visc_frag = if phys.viscosity > 0 {
        format!("   |   viscosity {}", phys.viscosity)
    } else { String::new() };
    draw_ui_text(
        &format!("{}   |   density {}{}{}",
            kind_name, phys.density, visc_frag, mass_frag),
        px, py, 14.0, dim,
    );
    py += 20.0;

    // Thermal phase transitions:
    let mut phase_bits: Vec<String> = Vec::new();
    if let Some(p) = therm.freeze_below {
        phase_bits.push(format!("freeze → {} below {}°C", p.target.name(), p.threshold));
    }
    if let Some(p) = therm.melt_above {
        phase_bits.push(format!("melt → {} at {}°C", p.target.name(), p.threshold));
    }
    if let Some(p) = therm.boil_above {
        phase_bits.push(format!("boil → {} at {}°C", p.target.name(), p.threshold));
    }
    if let Some(p) = therm.condense_below {
        phase_bits.push(format!("condense → {} below {}°C", p.target.name(), p.threshold));
    }
    if let Some(ig) = therm.ignite_above {
        let bt = therm.burn_temp.unwrap_or(0);
        phase_bits.push(format!("ignite at {}°C (burns {}°C)", ig, bt));
    }
    draw_ui_text("Thermal:", px, py, 13.0, section);
    py += 16.0;
    if phase_bits.is_empty() {
        draw_ui_text("   no phase transitions (stable)", px, py, 13.0, dim);
    } else {
        draw_ui_text(&format!("   {}", phase_bits.join("   |   ")), px, py, 13.0, dim);
    }
    py += 18.0;

    // Electrical:
    let elec_desc = if elec.conductivity < 0.01 {
        "insulator".to_string()
    } else if elec.conductivity >= 0.7 {
        format!("strong conductor (σ={:.2})", elec.conductivity)
    } else {
        format!("partial conductor (σ={:.2})", elec.conductivity)
    };
    let glow_frag = if let Some((r, g, b)) = elec.glow_color {
        format!("   |   plasma glow rgb({},{},{})", r, g, b)
    } else { String::new() };
    draw_ui_text(
        &format!("Electrical:   {}{}", elec_desc, glow_frag),
        px, py, 13.0, dim,
    );
    py += 18.0;

    // Moisture:
    let mut moist_bits: Vec<String> = Vec::new();
    if moist.is_source { moist_bits.push("wet source".to_string()); }
    if moist.is_sink   { moist_bits.push("absorbs water".to_string()); }
    if let Some((m, t)) = moist.wet_above {
        moist_bits.push(format!("→ {} at moisture {}", t.name(), m));
    }
    if moist_bits.is_empty() && moist.default_moisture == 0 {
        // skip line entirely for dry/inert materials
    } else {
        if moist_bits.is_empty() {
            moist_bits.push("dry".to_string());
        }
        draw_ui_text(
            &format!("Moisture:   {}", moist_bits.join("   |   ")),
            px, py, 13.0, dim,
        );
        py += 18.0;
    }

    // Pressure:
    draw_ui_text(
        &format!(
            "Pressure:   permeability {}   |   compliance {}{}",
            press.permeability, press.compliance,
            if press.formation_pressure != 0 {
                format!("   |   injected on spawn: {}", press.formation_pressure)
            } else { String::new() },
        ),
        px, py, 13.0, dim,
    );
    py += 20.0;

    draw_ui_text("click to paint", px, py, 14.0, GREEN);
}

fn draw_periodic_table(
    hovered: Option<usize>, hovered_compound: Option<usize>,
    hovered_derived: Option<usize>,
    selected: Element, selected_did: u8,
    derived_palette: &[u8],
) {
    let sw = screen_width();
    let sh = screen_height();
    // Dim sim behind the overlay.
    draw_rectangle(0.0, 0.0, sw, sh, Color::from_rgba(8, 8, 14, 220));

    let (tx, ty, table_w, _table_h) = pt_layout();

    // Title bar
    draw_ui_text("Periodic Table of Elements",
        tx, ty - 44.0, 22.0, WHITE);
    draw_ui_text("Tab or Esc to close",
        tx + table_w - 150.0, ty - 44.0, 14.0, LIGHTGRAY);

    draw_ui_text("click an atom or compound to paint   (dimmed atoms are placeholders)",
        tx, ty - 24.0, 12.0, GRAY);

    // Draw each atom tile.
    for (i, a) in ATOMS.iter().enumerate() {
        let (x, y) = pt_tile_xy(a, tx, ty);
        let (r, g, b) = atom_category_color(a.category);
        let alpha = if a.implemented { 255 } else { 90 };
        draw_rectangle(x, y, PT_TILE, PT_TILE, Color::from_rgba(r, g, b, alpha));

        // Subtle border
        draw_rectangle_lines(x, y, PT_TILE, PT_TILE, 1.0,
            Color::from_rgba(40, 40, 50, 200));

        // Atomic number (tiny, top-left)
        let num = a.number.to_string();
        draw_ui_text(&num, x + 3.0, y + 11.0, 11.0, BLACK);

        // Symbol (big, centered-ish)
        let sym_font = 20.0;
        let sym_dim = measure_ui_text(a.symbol, sym_font as u16);
        draw_ui_text(a.symbol,
            x + (PT_TILE - sym_dim.width) * 0.5,
            y + PT_TILE - 12.0,
            sym_font, BLACK);

        // Selected outline — persistent green — for the currently-painting atom.
        if atom_to_element(i) == Some(selected) {
            draw_rectangle_lines(x - 2.0, y - 2.0, PT_TILE + 4.0, PT_TILE + 4.0,
                3.0, GREEN);
        }

        // Hover highlight
        if hovered == Some(i) {
            draw_rectangle_lines(x - 1.0, y - 1.0, PT_TILE + 2.0, PT_TILE + 2.0,
                3.0, YELLOW);
        }
    }

    // Compound submenu — a labeled row of tiles directly below the grid.
    let (cx0, cy0) = pt_compound_origin();
    draw_ui_text("Compounds", cx0, cy0 - 6.0, 14.0, LIGHTGRAY);
    for (i, el) in COMPOUND_PALETTE.iter().enumerate() {
        let (sx, sy) = pt_compound_slot(i);
        if *el == Element::Empty {
            draw_rectangle(sx, sy, PT_TILE, PT_TILE, Color::from_rgba(40, 40, 46, 255));
            let red = Color::from_rgba(200, 70, 70, 255);
            draw_line(sx + 8.0, sy + 8.0, sx + PT_TILE - 8.0, sy + PT_TILE - 8.0, 2.0, red);
            draw_line(sx + PT_TILE - 8.0, sy + 8.0, sx + 8.0, sy + PT_TILE - 8.0, 2.0, red);
        } else {
            let (r, g, b) = el.base_color();
            draw_rectangle(sx, sy, PT_TILE, PT_TILE, Color::from_rgba(r, g, b, 255));
        }
        draw_rectangle_lines(sx, sy, PT_TILE, PT_TILE, 1.0,
            Color::from_rgba(40, 40, 50, 200));
        if selected == *el {
            draw_rectangle_lines(sx - 2.0, sy - 2.0, PT_TILE + 4.0, PT_TILE + 4.0,
                3.0, GREEN);
        }
        if hovered_compound == Some(i) {
            draw_rectangle_lines(sx - 1.0, sy - 1.0, PT_TILE + 2.0, PT_TILE + 2.0,
                3.0, YELLOW);
        }
    }

    // Derived-compound tiles appended after the main palette. Each entry
    // in `derived_palette` is a registry index; we look up color/formula
    // on the fly so the tile stays in sync with the compound registry.
    let base_idx = COMPOUND_PALETTE.len();
    for (i, &did) in derived_palette.iter().enumerate() {
        let (sx, sy) = pt_compound_slot(base_idx + i);
        let (r, g, b) = derived_color_of(did);
        draw_rectangle(sx, sy, PT_TILE, PT_TILE, Color::from_rgba(r, g, b, 255));
        draw_rectangle_lines(sx, sy, PT_TILE, PT_TILE, 1.0,
            Color::from_rgba(40, 40, 50, 200));
        let formula = derived_formula_of(did);
        let fdim = measure_ui_text(&formula, 11);
        draw_ui_text(
            &formula,
            sx + (PT_TILE - fdim.width) * 0.5,
            sy + PT_TILE - 4.0, 11.0, WHITE,
        );
        if selected == Element::Derived && selected_did == did {
            draw_rectangle_lines(sx - 2.0, sy - 2.0, PT_TILE + 4.0, PT_TILE + 4.0,
                3.0, GREEN);
        }
        if hovered_derived == Some(i) {
            draw_rectangle_lines(sx - 1.0, sy - 1.0, PT_TILE + 2.0, PT_TILE + 2.0,
                3.0, YELLOW);
        }
    }

    // Detail panel below the compound row.
    let panel_y = cy0 + PT_COMPOUND_ROW_H;
    let panel_h = PT_DETAIL_PANEL_H;
    draw_rectangle(tx, panel_y, table_w, panel_h,
        Color::from_rgba(18, 18, 26, 230));
    draw_rectangle_lines(tx, panel_y, table_w, panel_h, 1.0,
        Color::from_rgba(60, 60, 76, 255));

    if let Some(i) = hovered {
        let a = &ATOMS[i];
        let px = tx + 14.0;
        let mut py = panel_y + 28.0;

        draw_ui_text(&format!("{} ({})    #{}", a.name, a.symbol, a.number),
            px, py, 24.0, WHITE);
        py += 28.0;

        let cat_name = match a.category {
            Hydrogen => "Hydrogen (unique)",
            AlkaliMetal => "Alkali Metal",
            AlkalineEarth => "Alkaline Earth Metal",
            TransitionMetal => "Transition Metal",
            PostTransition => "Post-Transition Metal",
            Metalloid => "Metalloid",
            Nonmetal => "Nonmetal",
            Halogen => "Halogen",
            NobleGas => "Noble Gas",
            Lanthanide => "Lanthanide",
            Actinide => "Actinide",
        };
        draw_ui_text(&format!("period {}    group {}    |    {}    |    {:?} at STP",
            a.period, a.group, cat_name, a.stp_state),
            px, py, 14.0, LIGHTGRAY);
        py += 20.0;

        if a.implemented {
            draw_ui_text(&format!(
                "mass {:.3} amu    |    melt {}°C    |    boil {}°C    |    density {:.3} g/cm3",
                a.atomic_mass, a.melting_point, a.boiling_point, a.density_stp),
                px, py, 14.0, LIGHTGRAY);
            py += 18.0;
            draw_ui_text(&format!(
                "electronegativity {:.2}    |    {} valence e⁻",
                a.electronegativity, a.valence_electrons),
                px, py, 14.0, LIGHTGRAY);
        } else {
            draw_ui_text(&format!(
                "mass {:.3} amu    |    {} valence e⁻    |    (other properties not yet entered)",
                a.atomic_mass, a.valence_electrons),
                px, py, 14.0, LIGHTGRAY);
        }
        py += 22.0;

        let (status_text, status_color) = if a.implemented {
            ("IMPLEMENTED", GREEN)
        } else {
            ("PLACEHOLDER", YELLOW)
        };
        draw_ui_text(status_text, px, py, 14.0, status_color);
        py += 20.0;

        draw_ui_text(a.notes, px, py, 14.0, WHITE);
    } else if let Some(i) = hovered_compound {
        let el = COMPOUND_PALETTE[i];
        draw_compound_detail(el, tx, panel_y);
    } else {
        draw_ui_text("hover an atom or compound to inspect",
            tx + 14.0, panel_y + panel_h * 0.5, 16.0, GRAY);
    }
}

// ============================================================================
// GPU RENDERING — shader sources and helpers.
//
// Bloom blur runs as a separable two-pass fragment shader instead of the
// CPU per-pixel kernel sweep. Same 19-tap triangular kernel as the CPU
// reference (weights 1..10..1, sum 100). Pass Direction = (1/W, 0) for
// the horizontal sweep and (0, 1/H) for the vertical sweep.
//
// The vertex shader is a standard textured quad pass-through; macroquad
// supplies Model and Projection matrices.
// ============================================================================
const VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
varying lowp vec2 uv;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1);
    uv = texcoord;
}"#;

const BLOOM_BLUR_FRAGMENT: &str = r#"#version 100
precision mediump float;
varying lowp vec2 uv;
uniform sampler2D Texture;
uniform vec2 Direction;
void main() {
    vec3 sum = vec3(0.0);
    sum += texture2D(Texture, uv + Direction * -9.0).rgb * 1.0;
    sum += texture2D(Texture, uv + Direction * -8.0).rgb * 2.0;
    sum += texture2D(Texture, uv + Direction * -7.0).rgb * 3.0;
    sum += texture2D(Texture, uv + Direction * -6.0).rgb * 4.0;
    sum += texture2D(Texture, uv + Direction * -5.0).rgb * 5.0;
    sum += texture2D(Texture, uv + Direction * -4.0).rgb * 6.0;
    sum += texture2D(Texture, uv + Direction * -3.0).rgb * 7.0;
    sum += texture2D(Texture, uv + Direction * -2.0).rgb * 8.0;
    sum += texture2D(Texture, uv + Direction * -1.0).rgb * 9.0;
    sum += texture2D(Texture, uv).rgb * 10.0;
    sum += texture2D(Texture, uv + Direction *  1.0).rgb * 9.0;
    sum += texture2D(Texture, uv + Direction *  2.0).rgb * 8.0;
    sum += texture2D(Texture, uv + Direction *  3.0).rgb * 7.0;
    sum += texture2D(Texture, uv + Direction *  4.0).rgb * 6.0;
    sum += texture2D(Texture, uv + Direction *  5.0).rgb * 5.0;
    sum += texture2D(Texture, uv + Direction *  6.0).rgb * 4.0;
    sum += texture2D(Texture, uv + Direction *  7.0).rgb * 3.0;
    sum += texture2D(Texture, uv + Direction *  8.0).rgb * 2.0;
    sum += texture2D(Texture, uv + Direction *  9.0).rgb * 1.0;
    gl_FragColor = vec4(sum / 100.0, 1.0);
}"#;

// Passthrough material with additive blending. Used to composite the
// blurred bloom layer onto the base sim image without a custom fragment
// shader — macroquad's default textured quad is fine; we only need the
// blend state to be Add(One, One).
const PASSTHROUGH_FRAGMENT: &str = r#"#version 100
precision mediump float;
varying lowp vec2 uv;
uniform sampler2D Texture;
void main() {
    gl_FragColor = texture2D(Texture, uv);
}"#;

// Gas cloud blur — radius 5, 11-tap triangular kernel (weights 1..6..1,
// sum 36). Blurs RGBA together; alpha carries per-cell density (220
// if gas else 0). Used for both horizontal and vertical passes — pass
// Direction = (1/W, 0) or (0, 1/H).
const GAS_BLUR_FRAGMENT: &str = r#"#version 100
precision mediump float;
varying lowp vec2 uv;
uniform sampler2D Texture;
uniform vec2 Direction;
void main() {
    vec4 sum = vec4(0.0);
    sum += texture2D(Texture, uv + Direction * -5.0) * 1.0;
    sum += texture2D(Texture, uv + Direction * -4.0) * 2.0;
    sum += texture2D(Texture, uv + Direction * -3.0) * 3.0;
    sum += texture2D(Texture, uv + Direction * -2.0) * 4.0;
    sum += texture2D(Texture, uv + Direction * -1.0) * 5.0;
    sum += texture2D(Texture, uv)                    * 6.0;
    sum += texture2D(Texture, uv + Direction *  1.0) * 5.0;
    sum += texture2D(Texture, uv + Direction *  2.0) * 4.0;
    sum += texture2D(Texture, uv + Direction *  3.0) * 3.0;
    sum += texture2D(Texture, uv + Direction *  4.0) * 2.0;
    sum += texture2D(Texture, uv + Direction *  5.0) * 1.0;
    gl_FragColor = sum / 36.0;
}"#;

// Gas composite — read fully-blurred gas RGBA, scale RGB by amplified
// density (alpha × 6, clamped) so even isolated atoms produce a visible
// halo, output for additive blending onto the base sim layer.
const GAS_COMPOSITE_FRAGMENT: &str = r#"#version 100
precision mediump float;
varying lowp vec2 uv;
uniform sampler2D Texture;
void main() {
    vec4 v = texture2D(Texture, uv);
    float amped = clamp(v.a * 6.0, 0.0, 1.0);
    gl_FragColor = vec4(v.rgb * amped, 1.0);
}"#;

// Pressure diffusion — runs one explicit-Euler iteration of the same
// 4-neighbor flux model the CPU `pressure()` uses. Pressure values
// are signed i16 packed into RG bytes (R = low, G = high; signedness
// preserved by treating the unpacked u16 as two's-complement 16-bit).
// Permeability is sampled as a single u8 in the R channel of PermTex.
//
// Boundary conditions match the CPU reference:
//   * Horizontal out-of-bounds (uv.x < 0 or > 1) acts as open atmosphere
//     (P=0, perm=255). Pressure leaks out the sides.
//   * Vertical out-of-bounds (uv.y < 0 or > 1) is sealed — that
//     neighbor contributes nothing.
//   * Wall cells (perm == 0) keep their current pressure unchanged.
//
// `precision highp float` is required to faithfully encode 16-bit
// integer values via floats — mediump may quantize and lose precision.
const PRESSURE_DIFFUSION_FRAGMENT: &str = r#"#version 100
precision highp float;
varying lowp vec2 uv;
uniform sampler2D PressureTex;
uniform sampler2D PermTex;
uniform vec2 TexelSize;

float decode_p(vec2 uv_p) {
    vec4 t = texture2D(PressureTex, uv_p);
    float val = floor(t.r * 255.0 + 0.5) + floor(t.g * 255.0 + 0.5) * 256.0;
    if (val >= 32768.0) val -= 65536.0;
    return val;
}

float decode_perm(vec2 uv_p) {
    return floor(texture2D(PermTex, uv_p).r * 255.0 + 0.5);
}

vec4 encode_p(float val) {
    float v = val;
    if (v < 0.0) v += 65536.0;
    v = clamp(v, 0.0, 65535.0);
    float hi = floor(v / 256.0);
    float lo = floor(v - hi * 256.0);
    return vec4(lo / 255.0, hi / 255.0, 0.0, 1.0);
}

void main() {
    float me_perm = decode_perm(uv);
    float me_p = decode_p(uv);
    if (me_perm < 0.5) {
        gl_FragColor = encode_p(me_p);
        return;
    }
    float new_p = me_p;
    // LEFT
    {
        vec2 nuv = uv + vec2(-TexelSize.x, 0.0);
        float n_p, n_perm;
        if (nuv.x < 0.0) { n_p = 0.0; n_perm = 255.0; }
        else { n_p = decode_p(nuv); n_perm = decode_perm(nuv); }
        float min_perm = min(me_perm, n_perm);
        if (min_perm > 0.0) new_p += (n_p - me_p) * min_perm / 2048.0;
    }
    // RIGHT
    {
        vec2 nuv = uv + vec2(TexelSize.x, 0.0);
        float n_p, n_perm;
        if (nuv.x > 1.0) { n_p = 0.0; n_perm = 255.0; }
        else { n_p = decode_p(nuv); n_perm = decode_perm(nuv); }
        float min_perm = min(me_perm, n_perm);
        if (min_perm > 0.0) new_p += (n_p - me_p) * min_perm / 2048.0;
    }
    // UP — sealed at top boundary
    if (uv.y - TexelSize.y >= 0.0) {
        vec2 nuv = uv + vec2(0.0, -TexelSize.y);
        float n_p = decode_p(nuv);
        float n_perm = decode_perm(nuv);
        float min_perm = min(me_perm, n_perm);
        if (min_perm > 0.0) new_p += (n_p - me_p) * min_perm / 2048.0;
    }
    // DOWN — sealed at bottom boundary
    if (uv.y + TexelSize.y <= 1.0) {
        vec2 nuv = uv + vec2(0.0, TexelSize.y);
        float n_p = decode_p(nuv);
        float n_perm = decode_perm(nuv);
        float min_perm = min(me_perm, n_perm);
        if (min_perm > 0.0) new_p += (n_p - me_p) * min_perm / 2048.0;
    }
    gl_FragColor = encode_p(new_p);
}"#;

pub async fn run_game() {
    init_ui_font();
    let mut world = World::new();
    let mut image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let texture = Texture2D::from_image(&image);
    texture.set_filter(FilterMode::Nearest);

    // ---- GPU bloom pipeline ----
    // Replaces the CPU bloom blur (19-tap separable, ~3M ops/frame) with
    // a fragment-shader pass on offscreen render targets. CPU still
    // computes the per-cell bloom seed (base color × emission) since
    // emission depends on sim state (cell.temp + Fire/Lava + glow), but
    // the spatial blur and additive composite move to the GPU.
    let bloom_seed_image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let bloom_seed_tex = Texture2D::from_image(&bloom_seed_image);
    bloom_seed_tex.set_filter(FilterMode::Linear);
    // Full-resolution blur targets. Half-res experiments didn't deliver
    // a measurable speedup in this stack (macroquad RT overhead dominates
    // the per-pixel work at this scale), so we keep full size.
    const BLUR_W: u32 = W as u32;
    const BLUR_H: u32 = H as u32;
    let bloom_h_rt = render_target(BLUR_W, BLUR_H);
    bloom_h_rt.texture.set_filter(FilterMode::Linear);
    let bloom_v_rt = render_target(BLUR_W, BLUR_H);
    bloom_v_rt.texture.set_filter(FilterMode::Linear);
    let bloom_blur_material = match load_material(
        ShaderSource::Glsl {
            vertex: VERTEX_SHADER,
            fragment: BLOOM_BLUR_FRAGMENT,
        },
        MaterialParams {
            uniforms: vec![UniformDesc::new("Direction", UniformType::Float2)],
            ..Default::default()
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FATAL: bloom blur shader compile failed: {:?}", e);
            std::process::exit(1);
        }
    };
    // Additive-blend passthrough material — composites the blurred bloom
    // layer onto the screen without overwriting (Add(One, One)).
    let additive_material = match load_material(
        ShaderSource::Glsl {
            vertex: VERTEX_SHADER,
            fragment: PASSTHROUGH_FRAGMENT,
        },
        MaterialParams {
            pipeline_params: PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::One,
                )),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FATAL: additive shader compile failed: {:?}", e);
            std::process::exit(1);
        }
    };
    // Per-frame scratch buffer for the bloom seed (base × emission).
    // Allocated once and reused; the same Image gets uploaded to
    // bloom_seed_tex each frame after CPU populates it.
    let mut bloom_seed_image = bloom_seed_image;

    // ---- GPU gas cloud pipeline ----
    // Mirror of bloom infra but with radius-5 blur and a density-amp
    // composite shader. Seed image: RGB = the gas atom's color,
    // A = 220 (density flag) for gas cells; zero everywhere else.
    let gas_seed_image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let gas_seed_tex = Texture2D::from_image(&gas_seed_image);
    gas_seed_tex.set_filter(FilterMode::Linear);
    let gas_h_rt = render_target(BLUR_W, BLUR_H);
    gas_h_rt.texture.set_filter(FilterMode::Linear);
    let gas_v_rt = render_target(BLUR_W, BLUR_H);
    gas_v_rt.texture.set_filter(FilterMode::Linear);
    let gas_blur_material = match load_material(
        ShaderSource::Glsl {
            vertex: VERTEX_SHADER,
            fragment: GAS_BLUR_FRAGMENT,
        },
        MaterialParams {
            uniforms: vec![UniformDesc::new("Direction", UniformType::Float2)],
            ..Default::default()
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FATAL: gas blur shader compile failed: {:?}", e);
            std::process::exit(1);
        }
    };
    // Gas composite: density-amp scale + additive blend onto screen.
    let gas_composite_material = match load_material(
        ShaderSource::Glsl {
            vertex: VERTEX_SHADER,
            fragment: GAS_COMPOSITE_FRAGMENT,
        },
        MaterialParams {
            pipeline_params: PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::One,
                )),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FATAL: gas composite shader compile failed: {:?}", e);
            std::process::exit(1);
        }
    };
    let mut gas_seed_image = gas_seed_image;

    // GPU pressure scaffolding (currently unused — kept as reference).
    // The fragment-shader-based GPU pressure proved slower than the
    // CPU SoA implementation in this stack because of sync glReadPixels
    // overhead and per-iteration framebuffer-bind state changes. The
    // proper fix is moving to wgpu with real compute shaders + storage
    // buffers + async readback (see v0.3 migration branch).
    let _gpu_pressure_ctx = GpuPressureCtx::new();

    let mut selected: Element = Element::Sand;
    // derived_id sidecar: non-zero only when `selected == Element::Derived`,
    // pointing into the runtime compound registry. Used by paint() so
    // derived compounds can be spawned directly from the PT palette.
    let mut selected_did: u8 = 0;
    // Pre-register derived compounds that get dedicated palette entries
    // (skipping the "mix the constituents and wait" workflow for testing).
    let hcl_id: u8 = derive_or_lookup(Element::H, Element::Cl).unwrap_or(0);
    // AuCl is a testing shortcut — Au + Cl won't react at STP (noble metal,
    // EN gap below the bonding threshold), so without this you'd have to
    // melt the Au first every time you want to set up an electroplating
    // bath. Pre-register it here so it can be painted directly.
    let aucl_id: u8 = derive_or_lookup(Element::Au, Element::Cl).unwrap_or(0);
    let mut brush_radius: i32 = 4;
    let mut wind = vec2(0.0, 0.0);
    let mut last_seed_cell: Option<(i32, i32)> = None;
    let mut screenshot_notice: Option<String> = None;
    let mut screenshot_timer: u32 = 0;
    let mut paused: bool = false;
    // Lightweight per-frame timing summary printed once per second.
    let mut prof_sim_us: u64 = 0;
    let mut prof_render_cpu_us: u64 = 0;
    let mut prof_upload_us: u64 = 0;
    let mut prof_gpu_submit_us: u64 = 0;
    let mut prof_ui_us: u64 = 0;
    let mut prof_total_us: u64 = 0;
    let mut prof_frames: u32 = 0;
    let mut prof_last_print = std::time::Instant::now();
    let mut build_mode: bool = false;
    let mut tool_mode: ToolMode = ToolMode::Paint;
    // Camera zoom + pan. zoom multiplies the auto-fit base scale;
    // pan_x/pan_y shift the rendered sim relative to centered fit.
    // Defaults (1.0, 0, 0) reproduce the original full-window fit.
    let mut zoom: f32 = 1.0;
    let mut pan_x: f32 = 0.0;
    let mut pan_y: f32 = 0.0;
    // Last mouse position — used to compute middle-button-drag pan
    // deltas. Set to None at startup so the first frame can't produce
    // a bogus delta.
    let mut last_mouse: Option<(f32, f32)> = None;
    // Side control panel — always-open strip on the right with tool
    // buttons and ambient sliders. Toggling hides it and returns the space
    // to the sim. Currently empty (visible background only) — tool buttons
    // and sliders land in subsequent phases.
    let mut panel_visible: bool = true;
    // Pipet tool state. A single bucket of collected cells, optionally
    // filtered by target species. Capacity is generous so a full beaker
    // can be siphoned and moved.
    const PIPET_CAPACITY: usize = 4000;
    let mut pipet_target: Option<(Element, u8)> = None;
    let mut pipet_bucket: Vec<Cell> = Vec::new();
    // Frames remaining on the "empty pipet first" warning flash. Shown
    // when the user tries to change target while bucket is non-empty.
    let mut pipet_warning_frames: u32 = 0;
    // "Species present in scene" cache. Rescanning the full grid each
    // frame would be wasteful; we refresh at a few Hz while pipet is the
    // active tool so sparse/mingled species are still pickable without
    // having to click a single pixel. Tuples are (element, derived_id,
    // count), sorted descending by count.
    let mut species_cache: Vec<(Element, u8, usize)> = Vec::new();
    let mut species_cache_frame: u32 = 0;
    // Prefab tool configuration. Material is its own selection (decoupled
    // from the paint element) — switching between "build iron walls" and
    // "paint gunpowder" shouldn't require reselecting. All dimensions in
    // cells.
    let mut prefab_kind: PrefabKind = PrefabKind::Box;
    let mut prefab_thickness: i32 = 10;
    let mut prefab_width: i32 = 145;
    let mut prefab_height: i32 = 200;
    let mut prefab_material: Element = Element::Glass;
    // Battery output voltage — applied by all batteries in the scene
    // (MVP global). Drives Joule heating in the energized circuit.
    let mut prefab_voltage: i32 = 100;
    // Rotation quarter-turns (0..=3) applied to the prefab before
    // placement. Press R while in Prefab mode to cycle. Rotates the
    // bounding box too, so a 40×50 Battery laid sideways becomes 50×40.
    let mut prefab_rotation: u8 = 0;
    // Wire tool state. Two-click drawing: first click sets wire_start,
    // second click draws a line from start to cursor with the selected
    // material and thickness. wire_material defaults to Cu (best real-
    // world conductor we have implemented).
    let mut wire_material: Element = Element::Cu;
    let mut wire_thickness: i32 = 2;
    let mut wire_start: Option<(i32, i32)> = None;
    // Which "slot" a subsequent periodic-table selection will fill. Set
    // by clicking the Paint-element tile vs the Prefab Material tile; the
    // PT overlay's click handler branches on this.
    #[derive(Clone, Copy, PartialEq)]
    enum PtTarget { Paint, PrefabMaterial, WireMaterial }
    let mut pt_target: PtTarget = PtTarget::Paint;
    let mut periodic_open: bool = false;
    // When the user clicks an atom/compound tile to select it, the overlay
    // closes on the same mouse-down. Without this latch, the still-held
    // button on the next frame would immediately paint a pile into the sim.
    // Cleared once the mouse is released.
    let mut consume_stroke: bool = false;

    loop {
        let prof_frame_start = std::time::Instant::now();
        // --- keyboard ---
        let keymap = [
            (KeyCode::Key1, Element::Sand),
            (KeyCode::Key2, Element::Water),
            (KeyCode::Key3, Element::Stone),
            (KeyCode::Key5, Element::CO2),
            (KeyCode::Key6, Element::Steam),
            (KeyCode::Key7, Element::Lava),
            (KeyCode::Key8, Element::Obsidian),
            (KeyCode::Key9, Element::Mud),
            (KeyCode::Key0, Element::Empty),
            (KeyCode::Q,    Element::Seed),
            (KeyCode::L,    Element::Leaves),
            (KeyCode::O,    Element::Oil),
            (KeyCode::I,    Element::Ice),
        ];
        for (k, el) in keymap {
            if is_key_pressed(k) { selected = el; }
        }
        if (is_key_pressed(KeyCode::LeftBracket) || is_key_pressed(KeyCode::Minus))
            && brush_radius > 1 { brush_radius -= 1; }
        if (is_key_pressed(KeyCode::RightBracket) || is_key_pressed(KeyCode::Equal))
            && brush_radius < 30 { brush_radius += 1; }
        let (_, wheel_y_raw) = mouse_wheel();
        let mut wheel_y = wheel_y_raw;
        let shift_scroll = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
        let g_scroll = is_key_down(KeyCode::G);
        let a_scroll = is_key_down(KeyCode::A);
        let t_scroll = is_key_down(KeyCode::T);
        let ctrl_held = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
        // Mouse position — captured here so hover-over-ambient scrolling
        // can test panel row bounds. Used again later in the frame without
        // re-querying.
        let (mx, my) = mouse_position();
        // ---- Camera zoom + pan ----
        // Approximate "mouse in sim" check (anything left of the panel and
        // below the top bar). Used to gate zoom/pan so they don't fire while
        // hovering panel rows or chrome.
        let panel_w_pre = if panel_visible { PANEL_WIDTH } else { 0.0 };
        let mouse_in_sim_approx = mx >= 0.0
            && mx < (screen_width() - panel_w_pre)
            && my >= TOP_BAR;
        // Ctrl+wheel zooms the view, anchored on the cell under the cursor
        // (that cell stays under the cursor as the zoom changes). Consumes
        // the wheel event so brush-radius scroll doesn't also fire.
        if ctrl_held && wheel_y != 0.0 && mouse_in_sim_approx {
            let (old_scale, old_sim_x, old_sim_y, _, _) =
                sim_layout(panel_w_pre, zoom, pan_x, pan_y);
            let gx_under_mouse = (mx - old_sim_x) / old_scale;
            let gy_under_mouse = (my - old_sim_y) / old_scale;
            let factor = if wheel_y > 0.0 { 1.15 } else { 1.0 / 1.15 };
            let new_zoom = (zoom * factor).clamp(0.5, 12.0);
            // Recompute the layout at the new zoom (ignoring pan), then
            // adjust pan so the cell under the cursor lands on (mx, my).
            let avail_w = (screen_width() - panel_w_pre).max(1.0);
            let avail_h = (screen_height() - TOP_BAR).max(1.0);
            let scale_fit = (avail_w / W as f32).min(avail_h / H as f32).max(0.5);
            let new_scale = scale_fit * new_zoom;
            let new_sim_w = W as f32 * new_scale;
            let new_centered_x = (avail_w - new_sim_w) * 0.5;
            let new_centered_y = TOP_BAR;
            zoom = new_zoom;
            pan_x = mx - new_centered_x - gx_under_mouse * new_scale;
            pan_y = my - new_centered_y - gy_under_mouse * new_scale;
            wheel_y = 0.0;
        }
        // Middle-mouse drag pans the view by 1:1 pixel deltas. Cheap
        // and predictable — drag continues as long as the button is
        // held. Tracked via last_mouse since macroquad has no built-in
        // delta accessor.
        if is_mouse_button_down(MouseButton::Middle) {
            if let Some((lx, ly)) = last_mouse {
                pan_x += mx - lx;
                pan_y += my - ly;
            }
        }
        // Reset zoom + pan to the auto-fit default with Backspace.
        if is_key_pressed(KeyCode::Backspace) {
            zoom = 1.0;
            pan_x = 0.0;
            pan_y = 0.0;
        }
        last_mouse = Some((mx, my));
        // Prefab/Wire dropdown state — threaded into the layout helpers
        // so the Build button and everything below it shift down when
        // either inline sub-panel is visible.
        let prefab_open = tool_mode == ToolMode::Prefab;
        let wire_open = tool_mode == ToolMode::Wire;
        // Hover-over-ambient-row scrolling. If the cursor is parked on one
        // of the three ambient-control rows in the side panel, the wheel
        // adjusts that property instead of the brush / scrub / etc. Gives
        // a discoverable alternative to the G/A/T+wheel keybinds.
        let hovered_ambient: Option<usize> = if panel_visible && wheel_y != 0.0 {
            let ar = panel_ambient_rects(prefab_open, wire_open);
            (0..PANEL_AMBIENT_COUNT).find(|&i| {
                let r = ar[i];
                mx >= r.0 && mx < r.0 + r.2 && my >= r.1 && my < r.1 + r.3
            })
        } else { None };
        // Hover-over-prefab-slider scrolling — mirrors the ambient rows.
        let hovered_prefab_row: Option<usize> =
            if panel_visible && tool_mode == ToolMode::Prefab && wheel_y != 0.0 {
                let sr = prefab_slider_rects();
                (0..PREFAB_ROW_COUNT).find(|&i| {
                    let r = sr[i];
                    mx >= r.0 && mx < r.0 + r.2 && my >= r.1 && my < r.1 + r.3
                })
            } else { None };
        if let Some(i) = hovered_prefab_row {
            let dir = if wheel_y > 0.0 { 1 } else if wheel_y < 0.0 { -1 } else { 0 };
            // Voltage uses a bigger step because its range (1..=1000) is
            // much wider than dimension rows.
            let step: i32 = match i {
                3 => if shift_scroll { 50 } else { 10 },
                _ => if shift_scroll { 5 } else { 1 },
            };
            match i {
                0 => prefab_thickness = (prefab_thickness + dir * step).clamp(1, 20),
                1 => prefab_width  = (prefab_width  + dir * step).clamp(6, 200),
                2 => prefab_height = (prefab_height + dir * step).clamp(6, 200),
                3 => prefab_voltage = (prefab_voltage + dir * step).clamp(1, 1000),
                _ => {}
            }
        } else if let Some(i) = hovered_ambient {
            match i {
                0 => {
                    // Temp — same step curve as the T+wheel shortcut.
                    let step: i16 = if shift_scroll { 250 } else { 25 };
                    if wheel_y > 0.0 {
                        world.ambient_offset = (world.ambient_offset + step).min(4980);
                    } else if wheel_y < 0.0 {
                        world.ambient_offset = (world.ambient_offset - step).max(-293);
                    }
                }
                1 => {
                    if wheel_y > 0.0 {
                        world.ambient_oxygen = (world.ambient_oxygen + 0.05).min(2.0);
                    } else if wheel_y < 0.0 {
                        world.ambient_oxygen = (world.ambient_oxygen - 0.05).max(0.0);
                    }
                }
                2 => {
                    if wheel_y > 0.0 {
                        world.gravity = (world.gravity + 0.1).min(2.0);
                    } else if wheel_y < 0.0 {
                        world.gravity = (world.gravity - 0.1).max(0.0);
                    }
                }
                _ => {}
            }
        } else if g_scroll {
            // Hold G + scroll: tweak gravity strength. Step is 0.1 per notch,
            // clamped to [0, 2]. Above 1.0 doesn't currently double up motion
            // but signals "extra strong" intent for future heavy-G work.
            if wheel_y > 0.0 { world.gravity = (world.gravity + 0.1).min(2.0); }
            if wheel_y < 0.0 { world.gravity = (world.gravity - 0.1).max(0.0); }
        } else if a_scroll {
            // Hold A + scroll: tweak ambient-oxygen fraction. 0 = vacuum
            // (no combustion/oxidation without explicit O painted),
            // 0.21 = Earth air, >1 = enriched pure-O world where everything
            // burns on contact.
            if wheel_y > 0.0 { world.ambient_oxygen = (world.ambient_oxygen + 0.05).min(2.0); }
            if wheel_y < 0.0 { world.ambient_oxygen = (world.ambient_oxygen - 0.05).max(0.0); }
        } else if t_scroll {
            // Hold T + scroll: tweak ambient temperature offset. Displayed
            // ambient = 20°C + offset, so we clamp the offset to cover
            // -273..=5000°C actual. Shift speeds steps by 10× for fast sweeps.
            let step: i16 = if shift_scroll { 250 } else { 25 };
            if wheel_y > 0.0 {
                world.ambient_offset = (world.ambient_offset + step).min(4980);
            }
            if wheel_y < 0.0 {
                world.ambient_offset = (world.ambient_offset - step).max(-293);
            }
        } else if shift_scroll && paused {
            // Shift+scroll while paused scrubs through time:
            // wheel down = backward (into history),
            // wheel up   = forward — replays recorded frames if we've scrubbed
            //              back, otherwise advances the sim by one new frame.
            if wheel_y > 0.0 {
                if world.rewind_offset > 0 {
                    world.seek(-1);
                } else {
                    world.step(wind);
                }
            }
            if wheel_y < 0.0 { world.seek(1); }
        } else {
            // Prefab sizing via sim-area scroll (takes priority over the
            // brush-radius default): wheel = height, shift+wheel = width.
            let panel_w_here = if panel_visible { PANEL_WIDTH } else { 0.0 };
            let (_, sxh, syh, swh, shh) = sim_layout(panel_w_here, zoom, pan_x, pan_y);
            let in_sim_here = mx >= sxh && mx < sxh + swh
                && my >= syh && my < syh + shh;
            let prefab_in_sim = tool_mode == ToolMode::Prefab && in_sim_here;
            if prefab_in_sim && wheel_y != 0.0 {
                let dir: i32 = if wheel_y > 0.0 { 1 } else { -1 };
                // Larger steps than the panel-row scroll — sizing in the
                // sim is coarse/visual, so cover the 6..200 range in a
                // few dozen notches instead of a few hundred.
                let step = 5;
                if shift_scroll {
                    prefab_width = (prefab_width + dir * step).clamp(6, 200);
                } else {
                    prefab_height = (prefab_height + dir * step).clamp(6, 200);
                }
            } else if !shift_scroll {
                if wheel_y > 0.0 && brush_radius < 30 { brush_radius += 1; }
                else if wheel_y < 0.0 && brush_radius > 1 { brush_radius -= 1; }
            }
        }
        // Pause toggle — rewind is preview-only, unpausing always returns
        // to the pause-point and resumes the real timeline.
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
            world.seek(0);
        }
        // Build mode — painted cells become rigid bodies (anchored wood,
        // static sand, immovable stone). Useful for chambers and fixtures.
        if is_key_pressed(KeyCode::B) {
            build_mode = !build_mode;
        }
        // Tool mode keys — each toggles that tool on/off (pressing again
        // returns to Paint mode). H for Heat, V for Vacuum, M for Move/grab.
        if is_key_pressed(KeyCode::H) {
            tool_mode = if tool_mode == ToolMode::Heat
                { ToolMode::Paint } else { ToolMode::Heat };
        }
        if is_key_pressed(KeyCode::V) {
            tool_mode = if tool_mode == ToolMode::Vacuum
                { ToolMode::Paint } else { ToolMode::Vacuum };
        }
        // Pipet: a unified collect/move tool. With no target set it grabs
        // everything in the brush as a positional snapshot (old "grab"
        // behavior); with a target set (via Shift+L-click eyedropper) it
        // siphons only matching cells as a count. Toggling in and out of
        // pipet mode preserves contents and snapshot so you don't lose
        // what you were holding.
        if is_key_pressed(KeyCode::F) {
            tool_mode = if tool_mode == ToolMode::Prefab
                { ToolMode::Paint } else { ToolMode::Prefab };
        }
        if is_key_pressed(KeyCode::W) {
            tool_mode = if tool_mode == ToolMode::Wire
                { ToolMode::Paint } else { ToolMode::Wire };
            wire_start = None;
        }
        // R cycles the prefab rotation 0 → 1 → 2 → 3 → 0 (90° each).
        if tool_mode == ToolMode::Prefab && is_key_pressed(KeyCode::R) {
            prefab_rotation = (prefab_rotation + 1) & 3;
        }
        if is_key_pressed(KeyCode::P) || is_key_pressed(KeyCode::M) {
            tool_mode = if tool_mode == ToolMode::Pipet
                { ToolMode::Paint } else { ToolMode::Pipet };
        }
        // Periodic table overlay (Tab to open/close, Esc to close).
        if is_key_pressed(KeyCode::Tab) {
            periodic_open = !periodic_open;
        }
        if periodic_open && is_key_pressed(KeyCode::Escape) {
            periodic_open = false;
        }
        if is_key_pressed(KeyCode::C) {
            // C alone clears non-frozen matter; Shift+C nukes everything
            // including built structures. Lets you wipe a mess without
            // losing the beaker you built to hold it in.
            let shift_c = is_key_down(KeyCode::LeftShift)
                || is_key_down(KeyCode::RightShift);
            if shift_c {
                for c in world.cells.iter_mut() { *c = Cell::EMPTY; }
            } else {
                for c in world.cells.iter_mut() {
                    if !c.is_frozen() { *c = Cell::EMPTY; }
                }
            }
        }
        // Ambient temperature is now adjusted via hold-T + mouse wheel,
        // matching the G / A scroll pattern. The old comma/period/slash
        // keys were removed — inconsistent with the rest of the scheme.
        // F2 = full-window screenshot. Deferred to AFTER we draw this
        // frame (so the saved image includes sim + UI + any open overlay).
        let take_screenshot = is_key_pressed(KeyCode::F2);
        if is_key_pressed(KeyCode::U) { panel_visible = !panel_visible; }

        // --- mouse ---
        // (mx, my) was captured earlier to feed the hover-scroll check.
        let panel_w = if panel_visible { PANEL_WIDTH } else { 0.0 };
        let (scale_fit, sim_x, sim_y, sim_w, sim_h) = sim_layout(panel_w, zoom, pan_x, pan_y);

        // --- side panel input ---
        // Hit-test tool/build buttons before any sim tool handlers fire, so
        // a click on a button updates tool_mode *this frame* and also sets
        // consume_stroke, preventing the held press from leaking into the
        // sim if the user drags off the button.
        if panel_visible {
            let rects = panel_button_rects(prefab_open, wire_open);
            let hit = |r: (f32, f32, f32, f32)|
                mx >= r.0 && mx < r.0 + r.2 && my >= r.1 && my < r.1 + r.3;
            if is_mouse_button_pressed(MouseButton::Left) {
                if hit(rects[0]) { tool_mode = ToolMode::Paint;  consume_stroke = true; }
                if hit(rects[1]) { tool_mode = ToolMode::Heat;   consume_stroke = true; }
                if hit(rects[2]) { tool_mode = ToolMode::Vacuum; consume_stroke = true; }
                if hit(rects[3]) { tool_mode = ToolMode::Pipet;  consume_stroke = true; }
                if hit(rects[PANEL_BUTTON_PREFAB]) {
                    tool_mode = ToolMode::Prefab;
                    consume_stroke = true;
                }
                if hit(rects[PANEL_BUTTON_WIRE]) {
                    tool_mode = ToolMode::Wire;
                    wire_start = None;
                    consume_stroke = true;
                }
                if hit(rects[PANEL_BUTTON_BUILD]) {
                    build_mode = !build_mode;
                    consume_stroke = true;
                }
            }
            // Pipet "Clear" button — only hit-tested while the pipet is
            // active. Empties both the species target and whatever cells
            // were collected, returning the pipet to its initial state.
            if tool_mode == ToolMode::Pipet
                && is_mouse_button_pressed(MouseButton::Left)
            {
                let cr = pipet_clear_rect(prefab_open, wire_open);
                if mx >= cr.0 && mx < cr.0 + cr.2
                    && my >= cr.1 && my < cr.1 + cr.3
                {
                    pipet_target = None;
                    pipet_bucket.clear();
                    consume_stroke = true;
                }
            }
            // Species-list row hit-test — clicking a row auto-switches to
            // the Pipet tool with the clicked species preloaded. Works
            // from any tool mode since the list is always visible.
            if is_mouse_button_pressed(MouseButton::Left) {
                for (i, &(el, did, _)) in species_cache.iter().enumerate() {
                    let r = species_list_row_rect(i, prefab_open, wire_open);
                    if r.1 + r.3 > screen_height() { break; }
                    if mx >= r.0 && mx < r.0 + r.2
                        && my >= r.1 && my < r.1 + r.3
                    {
                        let new_target = Some((el, did));
                        if new_target != pipet_target && !pipet_bucket.is_empty() {
                            // Holding cells for a different species — warn
                            // rather than silently clobber the pipet target.
                            pipet_warning_frames = 120;
                        } else {
                            pipet_target = new_target;
                            tool_mode = ToolMode::Pipet;
                        }
                        consume_stroke = true;
                        break;
                    }
                }
            }
            // Wire tool — material picker click + thickness scroll.
            if tool_mode == ToolMode::Wire {
                let mr = wire_material_rect();
                if is_mouse_button_pressed(MouseButton::Left)
                    && mx >= mr.0 && mx < mr.0 + mr.2
                    && my >= mr.1 && my < mr.1 + mr.3
                {
                    pt_target = PtTarget::WireMaterial;
                    periodic_open = true;
                    consume_stroke = true;
                }
                let tr = wire_thickness_rect();
                if wheel_y != 0.0
                    && mx >= tr.0 && mx < tr.0 + tr.2
                    && my >= tr.1 && my < tr.1 + tr.3
                {
                    let step = if shift_scroll { 2 } else { 1 };
                    let dir = if wheel_y > 0.0 { 1 } else { -1 };
                    wire_thickness = (wire_thickness + dir * step).clamp(1, 20);
                }
            }
            // Prefab kind selectors — click to choose Beaker or Box. Also
            // the Material button opens the periodic table in "pick prefab
            // material" mode so the selection feeds prefab_material rather
            // than the paint element.
            if tool_mode == ToolMode::Prefab
                && is_mouse_button_pressed(MouseButton::Left)
            {
                let kr = prefab_kind_rects();
                let hit_k = |r: (f32, f32, f32, f32)|
                    mx >= r.0 && mx < r.0 + r.2
                    && my >= r.1 && my < r.1 + r.3;
                if hit_k(kr[0]) {
                    prefab_kind = PrefabKind::Beaker;
                    consume_stroke = true;
                }
                if hit_k(kr[1]) {
                    prefab_kind = PrefabKind::Box;
                    consume_stroke = true;
                }
                if hit_k(kr[2]) {
                    // When first switching to Battery, nudge the
                    // dimensions + material to sensible defaults for a
                    // small working battery. User can still change these
                    // afterwards (Cu casing for shorted-battery demos,
                    // bigger dimensions for higher-capacity, etc.).
                    if prefab_kind != PrefabKind::Battery {
                        prefab_material = Element::Quartz;
                        prefab_thickness = 10;
                        prefab_width = 30;
                        prefab_height = 40;
                    }
                    prefab_kind = PrefabKind::Battery;
                    consume_stroke = true;
                }
                let mr = prefab_material_rect();
                if hit_k(mr) {
                    pt_target = PtTarget::PrefabMaterial;
                    periodic_open = true;
                    consume_stroke = true;
                }
            }
            // Wind reset button — one click zeros the wind vector.
            {
                let rst = wind_reset_rect(prefab_open, wire_open);
                if is_mouse_button_pressed(MouseButton::Left)
                    && mx >= rst.0 && mx < rst.0 + rst.2
                    && my >= rst.1 && my < rst.1 + rst.3
                {
                    wind = vec2(0.0, 0.0);
                    consume_stroke = true;
                }
            }
            // Wind pad — click or drag inside to set wind direction and
            // magnitude. Center point = (0,0); vector from center to
            // cursor gives the direction; distance (clamped to pad radius)
            // mapped into [0, WIND_MAX] sets the magnitude.
            let wr = wind_pad_rect(prefab_open, wire_open);
            if (is_mouse_button_pressed(MouseButton::Left)
                || is_mouse_button_down(MouseButton::Left))
                && mx >= wr.0 && mx < wr.0 + wr.2
                && my >= wr.1 && my < wr.1 + wr.3
            {
                let cx = wr.0 + wr.2 * 0.5;
                let cy = wr.1 + wr.3 * 0.5;
                let dx = mx - cx;
                let dy = my - cy;
                let r = wr.2 * 0.5;
                let dist = (dx * dx + dy * dy).sqrt().min(r);
                let mag = if r > 0.0 { (dist / r) * WIND_MAX } else { 0.0 };
                if dist > 1.0 {
                    wind = vec2(dx / dist, dy / dist) * mag;
                } else {
                    wind = vec2(0.0, 0.0);
                }
                if is_mouse_button_pressed(MouseButton::Left) {
                    consume_stroke = true;
                }
            }
        }
        let in_sim = my >= sim_y && my < sim_y + sim_h
                  && mx >= sim_x && mx < sim_x + sim_w;
        let gx = ((mx - sim_x) / scale_fit) as i32;
        let gy = ((my - sim_y) / scale_fit) as i32;

        // X = stir at cursor. Discrete burst (one keypress = one full
        // mix); doesn't depend on tool mode or mouse drag. Useful for
        // homogenizing pre-mixed powders (thermite, gunpowder feedstock)
        // or stirring solutes into water.
        if in_sim && is_key_pressed(KeyCode::X) {
            world.stir(gx, gy, brush_radius);
        }

        if periodic_open {
            // Overlay open — clicks pick a new selection and close. Also
            // drop the active tool back to Paint: the user picking an
            // element almost always wants to paint it somewhere, not keep
            // whatever specialty tool was active before.
            if is_mouse_button_pressed(MouseButton::Left) {
                // Dispatch: the PT picker feeds either the paint slot or
                // the prefab material slot depending on what opened it.
                // Paint target also auto-switches to Paint mode so the
                // user can immediately paint; prefab-material target stays
                // in Prefab mode.
                // Resolve the pick as (Element, derived_id).
                let picked: Option<(Element, u8)> = if let Some(i) = pt_hit(mx, my) {
                    atom_to_element(i).map(|el| (el, 0))
                } else if let Some(i) = pt_compound_hit(mx, my) {
                    Some((COMPOUND_PALETTE[i], 0))
                } else if let Some(i) = pt_derived_hit(mx, my, 2) {
                    let did = [hcl_id, aucl_id][i];
                    Some((Element::Derived, did))
                } else { None };
                if let Some((el, did)) = picked {
                    match pt_target {
                        PtTarget::Paint => {
                            // Tab-driven picks always go to the paint
                            // slot AND kick back to Paint mode. Material
                            // selection for prefabs/wires happens only
                            // via their dedicated material buttons (which
                            // set pt_target to PrefabMaterial/WireMaterial).
                            selected = el;
                            selected_did = did;
                            tool_mode = ToolMode::Paint;
                        }
                        PtTarget::PrefabMaterial => {
                            // Prefabs still only accept atomic/bespoke
                            // materials — derived selections fall back to
                            // the element itself, which for Derived isn't
                            // meaningful. Skip derived picks here.
                            if el != Element::Derived {
                                prefab_material = el;
                            }
                        }
                        PtTarget::WireMaterial => {
                            if el != Element::Derived {
                                wire_material = el;
                            }
                        }
                    }
                    pt_target = PtTarget::Paint;
                    periodic_open = false;
                    consume_stroke = true;
                }
            }
        } else if in_sim {
            match tool_mode {
                ToolMode::Paint => {
                    if selected == Element::Seed {
                        // One seed per click, plus one per new cell crossed while dragging.
                        let held = is_mouse_button_down(MouseButton::Left);
                        let pressed = is_mouse_button_pressed(MouseButton::Left);
                        if !consume_stroke
                            && (pressed || (held && last_seed_cell != Some((gx, gy))))
                        {
                            world.paint(gx, gy, 0, Element::Seed, 0, build_mode);
                            last_seed_cell = Some((gx, gy));
                        }
                        if !held { last_seed_cell = None; }
                    } else if !consume_stroke && is_mouse_button_down(MouseButton::Left) {
                        world.paint(gx, gy, brush_radius, selected, selected_did, build_mode);
                    }
                    if is_mouse_button_down(MouseButton::Right) {
                        // Right-click always erases, never freezes.
                        world.paint(gx, gy, brush_radius, Element::Empty, 0, false);
                    }
                }
                ToolMode::Heat => {
                    // Shift boosts the per-frame delta 5× so you can rapidly
                    // preheat something or snap-cool it; default is gentle
                    // enough to watch thresholds cross in real time.
                    let shift = is_key_down(KeyCode::LeftShift)
                        || is_key_down(KeyCode::RightShift);
                    let base: i16 = if shift { 25 } else { 5 };
                    if !consume_stroke && is_mouse_button_down(MouseButton::Left) {
                        world.apply_heat(gx, gy, brush_radius, base);
                    }
                    if is_mouse_button_down(MouseButton::Right) {
                        world.apply_heat(gx, gy, brush_radius, -base);
                    }
                    last_seed_cell = None;
                }
                ToolMode::Vacuum => {
                    // L-click held: suck gas into the brush. Gas cells in
                    // the brush are deleted; strong negative pressure is
                    // injected to pull neighboring gas in via gradient.
                    if !consume_stroke && is_mouse_button_down(MouseButton::Left) {
                        world.apply_vacuum(gx, gy, brush_radius);
                    }
                    last_seed_cell = None;
                }
                ToolMode::Prefab => {
                    // L-click places the configured prefab centered on the
                    // cursor. Material, kind, thickness, and size all
                    // come from the prefab's own state — decoupled from
                    // the paint element.
                    if !consume_stroke
                        && is_mouse_button_pressed(MouseButton::Left)
                    {
                        world.place_prefab(
                            gx, gy, prefab_kind, prefab_material,
                            prefab_thickness, prefab_width, prefab_height,
                            prefab_rotation,
                        );
                        consume_stroke = true;
                    }
                    last_seed_cell = None;
                }
                ToolMode::Pipet => {
                    // Unified collection tool. L-hold collects, R-hold
                    // releases, regardless of whether a target species is
                    // set. Shift+L-click picks a species filter (or clears
                    // it); while a filter is set only matching cells are
                    // siphoned, otherwise any non-frozen cell qualifies.
                    // Frozen walls are never siphoned.
                    let shift_held = is_key_down(KeyCode::LeftShift)
                        || is_key_down(KeyCode::RightShift);
                    if !consume_stroke && shift_held
                        && is_mouse_button_pressed(MouseButton::Left)
                    {
                        if gx >= 0 && gx < W as i32 && gy >= 0 && gy < H as i32 {
                            let c = world.cells[gy as usize * W + gx as usize];
                            let picked = if c.el == Element::Empty {
                                None
                            } else {
                                Some((c.el, c.derived_id))
                            };
                            let new_target = if pipet_target == picked {
                                None
                            } else {
                                picked
                            };
                            // Bucket contents have preserved cell state.
                            // Letting the user switch filters mid-hold
                            // would cause visual confusion — warn and
                            // block instead.
                            if new_target != pipet_target {
                                if !pipet_bucket.is_empty() {
                                    pipet_warning_frames = 120;
                                } else {
                                    pipet_target = new_target;
                                }
                            }
                        }
                        consume_stroke = true;
                    } else {
                        if is_mouse_button_down(MouseButton::Left)
                            && !consume_stroke
                        {
                            world.pipet_collect(
                                gx, gy, brush_radius, pipet_target,
                                &mut pipet_bucket, PIPET_CAPACITY,
                            );
                        }
                        if is_mouse_button_down(MouseButton::Right) {
                            world.pipet_release(
                                gx, gy, brush_radius, &mut pipet_bucket,
                            );
                        }
                    }
                    last_seed_cell = None;
                }
                ToolMode::Wire => {
                    // Two-click line drawing: first click sets the start,
                    // second click draws the wire to the cursor. Hold
                    // Shift on the second click to keep the endpoint as
                    // the next segment's start — chain segments into a
                    // polyline without re-clicking the pickup every
                    // time. R-click cancels.
                    if !consume_stroke
                        && is_mouse_button_pressed(MouseButton::Left)
                    {
                        if let Some((sx, sy)) = wire_start {
                            world.place_wire_line(
                                sx, sy, gx, gy,
                                wire_material, wire_thickness,
                            );
                            let chaining = is_key_down(KeyCode::LeftShift)
                                || is_key_down(KeyCode::RightShift);
                            wire_start = if chaining {
                                Some((gx, gy))
                            } else {
                                None
                            };
                        } else {
                            wire_start = Some((gx, gy));
                        }
                        consume_stroke = true;
                    }
                    if is_mouse_button_pressed(MouseButton::Right) {
                        wire_start = None;
                    }
                    last_seed_cell = None;
                }
            }
        } else {
            last_seed_cell = None;
        }

        // Release re-arms painting after a PT-overlay selection stroke.
        if !is_mouse_button_down(MouseButton::Left) { consume_stroke = false; }

        // Per-frame UI timers — tick regardless of pause state so flashes
        // and fades aren't frozen while the sim is paused.
        if pipet_warning_frames > 0 { pipet_warning_frames -= 1; }

        // Refresh the "species present" cache roughly 4× per second. The
        // list is always visible at the bottom of the panel (clicking a
        // row jumps to Pipet tool with that species preloaded), so the
        // cache needs to stay populated regardless of the active tool.
        // Scans the whole grid once per refresh — ~100k simple cell
        // reads + a linear tally by species.
        species_cache_frame = species_cache_frame.wrapping_add(1);
        if species_cache_frame % 15 == 0 {
            species_cache.clear();
            for c in &world.cells {
                if c.el == Element::Empty { continue; }
                // Hide prefab/wire-only species the user can't paint: battery
                // terminals live on prefabs, and frozen rigid-body cells
                // would pollute the list with every wall they've placed.
                if c.is_frozen() { continue; }
                if matches!(c.el, Element::BattPos | Element::BattNeg) { continue; }
                let key = (c.el, c.derived_id);
                if let Some(entry) = species_cache.iter_mut()
                    .find(|(el, did, _)| (*el, *did) == key)
                {
                    entry.2 += 1;
                } else {
                    species_cache.push((c.el, c.derived_id, 1));
                }
            }
            species_cache.sort_by(|a, b| b.2.cmp(&a.2));
        }

        // --- simulate ---
        // Propagate the battery voltage setting to the sim state so
        // Joule heating uses the currently-configured value.
        world.battery_voltage = prefab_voltage as f32;
        let t_sim_start = std::time::Instant::now();
        if !paused { world.step(wind); }
        let t_sim = t_sim_start.elapsed();

        // --- render sim ---
        // Direct byte writes — much faster than set_pixel which does a
        // function call + bounds checks + float->u8 conversion per pixel.
        {
            let bytes = &mut image.bytes;
            for i in 0..(W * H) {
                let c = world.cells[i];
                // Empty fast path: skip color_rgb and downstream checks.
                // Empty cells have a fixed render color (seed always 0,
                // no temp glow, no phase, no liquid styling). For an
                // empty world this short-circuits 100K function calls
                // per frame.
                if c.el == Element::Empty {
                    let base = i * 4;
                    bytes[base]     = 2;
                    bytes[base + 1] = 2;
                    bytes[base + 2] = 6;
                    bytes[base + 3] = 255;
                    continue;
                }
                let [mut r, mut g, mut b] = color_rgb(c);
                // Energized cells get their electrical glow color (noble
                // gases light up; conducting metals stay their normal
                // color since glow_color is None for them). This is the
                // neon-sign rendering path — no per-cell state needed.
                if world.energized[i] {
                    if let Some((gr, gg, gb)) = c.el.electrical().glow_color {
                        r = gr; g = gg; b = gb;
                    }
                }
                // Liquid styling — surface highlight + depth shading +
                // gentle animated texture. All effects scale the cell's
                // existing color, so dark liquids stay dark and bright
                // liquids stay bright. Phase-aware via cell_physics so
                // molten metals get the same treatment.
                if cell_physics(c).kind == Kind::Liquid {
                    let cx = (i % W) as i32;
                    let cy = (i / W) as i32;
                    // Surface highlight: only flag a cell as "true
                    // surface" when its left/right neighbors at the
                    // same row are also surface cells. Filters out
                    // jagged single-cell spikes that produced
                    // vertical streaks on Hg/Water/Lava.
                    let is_top = |x: i32, y: i32| -> bool {
                        if y <= 0 || x < 0 || x >= W as i32 { return false; }
                        let here = (y as usize) * W + x as usize;
                        if cell_physics(world.cells[here]).kind != Kind::Liquid { return false; }
                        let above = ((y - 1) as usize) * W + x as usize;
                        cell_physics(world.cells[above]).kind != Kind::Liquid
                    };
                    let self_top = is_top(cx, cy);
                    let neigh_top = is_top(cx - 1, cy) as i32 + is_top(cx + 1, cy) as i32;
                    let on_surface = self_top && neigh_top >= 1;
                    if on_surface {
                        r = ((r as u32 * 122 / 100).min(255)) as u8;
                        g = ((g as u32 * 122 / 100).min(255)) as u8;
                        b = ((b as u32 * 122 / 100).min(255)) as u8;
                    }
                    // Depth shading: count liquid cells stacked above
                    // for cx-1, cx, cx+1 and take the MIN. Smooths
                    // out column-by-column variation that read as
                    // vertical streaks. ~3% per cell, capped 24%.
                    let col_depth = |x: i32| -> i32 {
                        if x < 0 || x >= W as i32 { return 0; }
                        let mut d = 0i32;
                        for dy in 1..=8 {
                            let py = cy - dy;
                            if py < 0 { break; }
                            let pi = py as usize * W + x as usize;
                            if cell_physics(world.cells[pi]).kind == Kind::Liquid { d += 1; }
                            else { break; }
                        }
                        d
                    };
                    let depth = col_depth(cx).min(col_depth(cx - 1)).min(col_depth(cx + 1));
                    if depth > 0 && !on_surface {
                        let darken = 100 - (depth * 3).min(24);
                        r = (r as u32 * darken as u32 / 100) as u8;
                        g = (g as u32 * darken as u32 / 100) as u8;
                        b = (b as u32 * darken as u32 / 100) as u8;
                    }
                }
                let base = i * 4;
                bytes[base]     = r;
                bytes[base + 1] = g;
                bytes[base + 2] = b;
                bytes[base + 3] = 255;
            }
            // ---- Gas cloud seed (CPU prep for GPU blur) ----
            // Build a per-pixel RGBA seed: RGB = the gas atom's color,
            // A = 220 (density flag) for cells whose CURRENT phase is
            // Gas (so boiled metals — Pb vapor from U fission, etc. —
            // get detected even when their static Kind is Gravel/Powder).
            // Also hide the discrete atom in the main image so the gas
            // cloud halo is the ONLY visible representation of the gas.
            // GPU then runs the 11-tap separable blur and applies the
            // density amp on composite.
            const GAS_PER_ATOM: u8 = 220;
            let n = W * H;
            let gas_seed_bytes = &mut gas_seed_image.bytes;
            for i in 0..n {
                let c = world.cells[i];
                let base_i = i * 4;
                if !matches!(cell_physics(c).kind, Kind::Gas) {
                    gas_seed_bytes[base_i]     = 0;
                    gas_seed_bytes[base_i + 1] = 0;
                    gas_seed_bytes[base_i + 2] = 0;
                    gas_seed_bytes[base_i + 3] = 0;
                    continue;
                }
                gas_seed_bytes[base_i]     = bytes[base_i];
                gas_seed_bytes[base_i + 1] = bytes[base_i + 1];
                gas_seed_bytes[base_i + 2] = bytes[base_i + 2];
                gas_seed_bytes[base_i + 3] = GAS_PER_ATOM;
                bytes[base_i]     = 0;
                bytes[base_i + 1] = 0;
                bytes[base_i + 2] = 0;
            }
            // ---- Bloom seed (CPU prep for GPU blur) ----
            // Bloom contribution is driven by EMISSION (cell temperature
            // and "always glows" elements like Fire/Lava), NOT by pixel
            // brightness. This stops bright-but-cool elements (silvery
            // Mg, white MgO powder, glass) from triggering bloom while
            // letting actually-hot cells glow even if their base color
            // is dark. The pixel's RENDERED color is then tinted by
            // emission intensity — so the bloom inherits the source
            // pixel's hue (yellow lava → yellow halo, white-hot Mg →
            // white halo, orange fire → orange halo) automatically.
            //
            // CPU produces ONLY the seed buffer (base × emission/255).
            // The spatial blur and additive composite run on the GPU
            // via a separable two-pass fragment shader.
            let n = W * H;
            let seed_bytes = &mut bloom_seed_image.bytes;
            for i in 0..n {
                let c = world.cells[i];
                let mut emission: u32 = if c.temp > 500 {
                    (((c.temp - 500) as i32 * 255 / 2000).clamp(0, 255)) as u32
                } else {
                    0
                };
                if matches!(c.el, Element::Fire | Element::Lava) {
                    emission = emission.max(200);
                }
                if world.energized[i] && c.el.electrical().glow_color.is_some() {
                    emission = emission.max(160);
                }
                let base = i * 4;
                if emission == 0 {
                    seed_bytes[base]     = 0;
                    seed_bytes[base + 1] = 0;
                    seed_bytes[base + 2] = 0;
                    seed_bytes[base + 3] = 255;
                    continue;
                }
                let r = bytes[base] as u32;
                let g = bytes[base + 1] as u32;
                let b = bytes[base + 2] as u32;
                seed_bytes[base]     = ((r * emission) / 255).min(255) as u8;
                seed_bytes[base + 1] = ((g * emission) / 255).min(255) as u8;
                seed_bytes[base + 2] = ((b * emission) / 255).min(255) as u8;
                seed_bytes[base + 3] = 255;
            }
        }
        let t_render_cpu = t_sim_start.elapsed() - t_sim;
        let t_upload_start = std::time::Instant::now();
        texture.update(&image);
        bloom_seed_tex.update(&bloom_seed_image);
        gas_seed_tex.update(&gas_seed_image);
        let t_upload = t_upload_start.elapsed();
        let t_gpu_submit_start = std::time::Instant::now();

        // ---- GPU gas cloud blur (separable 11-tap, radius 5) ----
        let blur_wf = BLUR_W as f32;
        let blur_hf = BLUR_H as f32;
        set_camera(&Camera2D {
            zoom: vec2(2.0 / blur_wf, 2.0 / blur_hf),
            target: vec2(blur_wf / 2.0, blur_hf / 2.0),
            render_target: Some(gas_h_rt.clone()),
            ..Default::default()
        });
        clear_background(BLACK);
        gl_use_material(&gas_blur_material);
        gas_blur_material.set_uniform("Direction", vec2(1.0 / blur_wf, 0.0));
        draw_texture_ex(
            &gas_seed_tex, 0.0, 0.0, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(blur_wf, blur_hf)),
                ..Default::default()
            },
        );
        gl_use_default_material();

        set_camera(&Camera2D {
            zoom: vec2(2.0 / blur_wf, 2.0 / blur_hf),
            target: vec2(blur_wf / 2.0, blur_hf / 2.0),
            render_target: Some(gas_v_rt.clone()),
            ..Default::default()
        });
        clear_background(BLACK);
        gl_use_material(&gas_blur_material);
        gas_blur_material.set_uniform("Direction", vec2(0.0, 1.0 / blur_hf));
        draw_texture_ex(
            &gas_h_rt.texture, 0.0, 0.0, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(blur_wf, blur_hf)),
                ..Default::default()
            },
        );
        gl_use_default_material();

        // ---- GPU bloom: separable two-pass blur on render targets ----
        set_camera(&Camera2D {
            zoom: vec2(2.0 / blur_wf, 2.0 / blur_hf),
            target: vec2(blur_wf / 2.0, blur_hf / 2.0),
            render_target: Some(bloom_h_rt.clone()),
            ..Default::default()
        });
        clear_background(BLACK);
        gl_use_material(&bloom_blur_material);
        bloom_blur_material.set_uniform("Direction", vec2(1.0 / blur_wf, 0.0));
        draw_texture_ex(
            &bloom_seed_tex, 0.0, 0.0, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(blur_wf, blur_hf)),
                ..Default::default()
            },
        );
        gl_use_default_material();

        set_camera(&Camera2D {
            zoom: vec2(2.0 / blur_wf, 2.0 / blur_hf),
            target: vec2(blur_wf / 2.0, blur_hf / 2.0),
            render_target: Some(bloom_v_rt.clone()),
            ..Default::default()
        });
        clear_background(BLACK);
        gl_use_material(&bloom_blur_material);
        bloom_blur_material.set_uniform("Direction", vec2(0.0, 1.0 / blur_hf));
        draw_texture_ex(
            &bloom_h_rt.texture, 0.0, 0.0, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(blur_wf, blur_hf)),
                ..Default::default()
            },
        );
        gl_use_default_material();

        // Back to default screen camera for the rest of the frame.
        set_default_camera();

        // Clear with the panel color — any area the sim doesn't cover
        // (bottom strip from aspect mismatch, space beside a centered sim)
        // reads as panel instead of blank dead space.
        clear_background(panel_bg());
        draw_texture_ex(
            &texture, sim_x, sim_y, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(sim_w, sim_h)),
                ..Default::default()
            },
        );
        // Gas cloud composite — density-amp scale + additive blend.
        gl_use_material(&gas_composite_material);
        draw_texture_ex(
            &gas_v_rt.texture, sim_x, sim_y, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(sim_w, sim_h)),
                ..Default::default()
            },
        );
        gl_use_default_material();
        // Bloom composite — additive blend over base + gas cloud.
        gl_use_material(&additive_material);
        draw_texture_ex(
            &bloom_v_rt.texture, sim_x, sim_y, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(sim_w, sim_h)),
                ..Default::default()
            },
        );
        gl_use_default_material();
        let t_gpu_submit = t_gpu_submit_start.elapsed();
        let t_ui_start = std::time::Instant::now();

        // Shockwave leading edges — a bright ring at each active wave's
        // current radius so blasts read visually as expanding fronts, not
        // just invisible forces knocking things around. Alpha scales with
        // remaining magnitude so the wave visibly dims as it spreads out.
        for s in &world.shockwaves {
            let decay = 1.0 + s.radius / 6.0;
            let mag = s.yield_p / (decay * decay);
            if mag < 200.0 { continue; }
            let alpha = (mag / 40.0).clamp(20.0, 220.0) as u8;
            let cx = sim_x + s.cx * scale_fit + scale_fit * 0.5;
            let cy = sim_y + s.cy * scale_fit + scale_fit * 0.5;
            let r = s.radius * scale_fit;
            draw_circle_lines(
                cx, cy, r, (scale_fit * 1.8).max(1.5),
                Color::from_rgba(255, 230, 180, alpha),
            );
        }

        // --- side panel (phase 2: tool buttons) ---
        if panel_visible {
            let px = screen_width() - PANEL_WIDTH;
            draw_line(
                px, TOP_BAR, px, screen_height(), 1.0,
                Color::from_rgba(60, 60, 72, 255),
            );
            // Section header
            draw_ui_text(
                "TOOLS", px + 14.0, PANEL_TOP_PAD + 14.0, 11.0,
                Color::from_rgba(130, 130, 150, 255),
            );
            let rects = panel_button_rects(prefab_open, wire_open);
            let hit = |r: (f32, f32, f32, f32)|
                mx >= r.0 && mx < r.0 + r.2 && my >= r.1 && my < r.1 + r.3;
            draw_panel_button(
                rects[0], "Paint",
                tool_mode == ToolMode::Paint, hit(rects[0]),
            );
            draw_panel_button(
                rects[1], "Heat",
                tool_mode == ToolMode::Heat, hit(rects[1]),
            );
            draw_panel_button(
                rects[2], "Vacuum",
                tool_mode == ToolMode::Vacuum, hit(rects[2]),
            );
            draw_panel_button(
                rects[3], "Pipet",
                tool_mode == ToolMode::Pipet, hit(rects[3]),
            );
            draw_panel_button(
                rects[PANEL_BUTTON_PREFAB], "Prefab",
                tool_mode == ToolMode::Prefab, hit(rects[PANEL_BUTTON_PREFAB]),
            );
            draw_panel_button(
                rects[PANEL_BUTTON_WIRE], "Wire",
                tool_mode == ToolMode::Wire, hit(rects[PANEL_BUTTON_WIRE]),
            );
            draw_panel_button(
                rects[PANEL_BUTTON_BUILD],
                if build_mode { "Build: ON" } else { "Build: OFF" },
                build_mode, hit(rects[PANEL_BUTTON_BUILD]),
            );

            // SIMULATION section — ambient controls. Scroll on hover to
            // adjust; works in addition to the G/A/T+wheel keybinds. Each
            // row shows label + current value, with a subtle hover glow.
            let ar = panel_ambient_rects(prefab_open, wire_open);
            let amb_header_y = ar[0].1 - 14.0;
            draw_ui_text(
                "SIMULATION", px + 14.0, amb_header_y, 11.0,
                Color::from_rgba(130, 130, 150, 255),
            );
            let dim_label = Color::from_rgba(150, 150, 165, 255);
            let value_color = Color::from_rgba(230, 230, 240, 255);
            let ambient_actual = 20 + world.ambient_offset;
            let rows = [
                ("Temp", format!("{:+}°C", ambient_actual)),
                ("O₂",   format!("{:.0}%", world.ambient_oxygen * 100.0)),
                ("Grav", format!("{:.1}×", world.gravity)),
            ];
            for (i, (label, value)) in rows.iter().enumerate() {
                let r = ar[i];
                let row_hovered = hit(r);
                let bg = if row_hovered {
                    Color::from_rgba(38, 38, 48, 255)
                } else {
                    Color::from_rgba(24, 24, 32, 255)
                };
                draw_rectangle(r.0, r.1, r.2, r.3, bg);
                draw_rectangle_lines(
                    r.0, r.1, r.2, r.3, 1.0,
                    Color::from_rgba(50, 50, 62, 255),
                );
                draw_ui_text(label, r.0 + 10.0, r.1 + r.3 * 0.5 + 5.0, 13.0, dim_label);
                let vd = measure_ui_text(value, 14);
                draw_ui_text(
                    value,
                    r.0 + r.2 - vd.width - 10.0,
                    r.1 + r.3 * 0.5 + 5.0,
                    14.0, value_color,
                );
            }

            // Wind widget — a circular pad showing the current wind vector
            // as an arrow from center. Click or drag inside to set wind.
            let wr = wind_pad_rect(prefab_open, wire_open);
            let wr_hov = mx >= wr.0 && mx < wr.0 + wr.2
                && my >= wr.1 && my < wr.1 + wr.3;
            draw_ui_text(
                "Wind", wr.0, wr.1 - 6.0, 11.0, dim_label,
            );
            draw_rectangle(
                wr.0, wr.1, wr.2, wr.3,
                if wr_hov { Color::from_rgba(34, 34, 46, 255) }
                else { Color::from_rgba(24, 24, 32, 255) },
            );
            draw_rectangle_lines(
                wr.0, wr.1, wr.2, wr.3, 1.0,
                Color::from_rgba(50, 50, 62, 255),
            );
            let wcx = wr.0 + wr.2 * 0.5;
            let wcy = wr.1 + wr.3 * 0.5;
            let wrad = wr.2 * 0.5 - 2.0;
            draw_circle_lines(
                wcx, wcy, wrad, 1.0,
                Color::from_rgba(50, 50, 62, 255),
            );
            // Cardinal tick marks for visual reference.
            for (dx, dy) in [(wrad, 0.0), (-wrad, 0.0), (0.0, wrad), (0.0, -wrad)] {
                draw_line(
                    wcx + dx * 0.85, wcy + dy * 0.85,
                    wcx + dx, wcy + dy, 1.0,
                    Color::from_rgba(70, 70, 84, 255),
                );
            }
            // Arrow from center → current wind vector (scaled to pad).
            let mag = wind.length();
            if mag > 0.001 {
                let scale_ = (wrad / WIND_MAX) * mag.min(WIND_MAX);
                let dir = wind.normalize();
                let tx = wcx + dir.x * scale_;
                let ty = wcy + dir.y * scale_;
                draw_line(
                    wcx, wcy, tx, ty, 2.0,
                    Color::from_rgba(230, 200, 120, 255),
                );
                draw_circle(tx, ty, 3.0,
                    Color::from_rgba(240, 210, 130, 255),
                );
            }
            draw_circle(wcx, wcy, 2.0, Color::from_rgba(140, 140, 160, 255));
            // Numeric readout to the right of the pad — wind vector and
            // magnitude in the same lane as the ambient rows above.
            let info_x = wr.0 + wr.2 + 10.0;
            let info_w = ar[0].0 + ar[0].2 - info_x;
            let _ = info_w;
            draw_ui_text(
                &format!("x {:+.2}", wind.x),
                info_x, wr.1 + 16.0, 12.0, value_color,
            );
            draw_ui_text(
                &format!("y {:+.2}", wind.y),
                info_x, wr.1 + 34.0, 12.0, value_color,
            );
            draw_ui_text(
                &format!("|v| {:.2}", mag),
                info_x, wr.1 + 52.0, 12.0, dim_label,
            );
            // Wind reset button — sets wind to zero.
            let rst = wind_reset_rect(prefab_open, wire_open);
            let rst_hov = mx >= rst.0 && mx < rst.0 + rst.2
                && my >= rst.1 && my < rst.1 + rst.3;
            draw_panel_button(rst, "Reset Wind", false, rst_hov);

            // Prefab sub-panel — kind selectors + thickness/width/height.
            if tool_mode == ToolMode::Prefab {
                let kr = prefab_kind_rects();
                draw_ui_text(
                    "PREFAB",
                    kr[0].0 + 2.0, kr[0].1 - 10.0, 11.0,
                    Color::from_rgba(130, 130, 150, 255),
                );
                let hit_k = |r: (f32, f32, f32, f32)|
                    mx >= r.0 && mx < r.0 + r.2
                    && my >= r.1 && my < r.1 + r.3;
                draw_panel_button(
                    kr[0], "Beaker",
                    prefab_kind == PrefabKind::Beaker, hit_k(kr[0]),
                );
                draw_panel_button(
                    kr[1], "Box",
                    prefab_kind == PrefabKind::Box, hit_k(kr[1]),
                );
                draw_panel_button(
                    kr[2], "Batt",
                    prefab_kind == PrefabKind::Battery, hit_k(kr[2]),
                );
                let sr = prefab_slider_rects();
                let labels = [
                    ("Thickness", prefab_thickness.to_string()),
                    ("Width",     prefab_width.to_string()),
                    ("Height",    prefab_height.to_string()),
                    ("Voltage",   format!("{} V", prefab_voltage)),
                ];
                for i in 0..PREFAB_ROW_COUNT {
                    let r = sr[i];
                    let hov = mx >= r.0 && mx < r.0 + r.2
                        && my >= r.1 && my < r.1 + r.3;
                    let bg = if hov {
                        Color::from_rgba(38, 38, 48, 255)
                    } else {
                        Color::from_rgba(24, 24, 32, 255)
                    };
                    draw_rectangle(r.0, r.1, r.2, r.3, bg);
                    draw_rectangle_lines(
                        r.0, r.1, r.2, r.3, 1.0,
                        Color::from_rgba(50, 50, 62, 255),
                    );
                    draw_ui_text(
                        labels[i].0, r.0 + 10.0, r.1 + r.3 * 0.5 + 5.0, 13.0,
                        Color::from_rgba(150, 150, 165, 255),
                    );
                    let vd = measure_ui_text(&labels[i].1, 13);
                    draw_ui_text(
                        &labels[i].1,
                        r.0 + r.2 - vd.width - 10.0,
                        r.1 + r.3 * 0.5 + 5.0, 13.0,
                        Color::from_rgba(220, 220, 230, 255),
                    );
                }
                // Material picker button. Displays current prefab material
                // and opens the periodic table (in material-picker mode)
                // when clicked.
                let mr = prefab_material_rect();
                let mr_hov = mx >= mr.0 && mx < mr.0 + mr.2
                    && my >= mr.1 && my < mr.1 + mr.3;
                draw_panel_button(
                    mr,
                    &format!("Material: {}", prefab_material.name()),
                    false, mr_hov,
                );
                // Hint below — just the "click to place" reminder.
                draw_ui_text(
                    "click in sim to place",
                    mr.0 + 2.0, mr.1 + mr.3 + 14.0, 11.0,
                    Color::from_rgba(130, 130, 150, 255),
                );
            }

            // Wire sub-panel — material picker + thickness slider. Shown
            // only when Wire is the active tool.
            if tool_mode == ToolMode::Wire {
                let mr = wire_material_rect();
                let mr_hov = mx >= mr.0 && mx < mr.0 + mr.2
                    && my >= mr.1 && my < mr.1 + mr.3;
                draw_ui_text(
                    "WIRE",
                    mr.0 + 2.0, mr.1 - 10.0, 11.0,
                    Color::from_rgba(130, 130, 150, 255),
                );
                draw_panel_button(
                    mr,
                    &format!("Material: {}", wire_material.name()),
                    false, mr_hov,
                );
                let tr = wire_thickness_rect();
                let tr_hov = mx >= tr.0 && mx < tr.0 + tr.2
                    && my >= tr.1 && my < tr.1 + tr.3;
                let bg = if tr_hov {
                    Color::from_rgba(38, 38, 48, 255)
                } else {
                    Color::from_rgba(24, 24, 32, 255)
                };
                draw_rectangle(tr.0, tr.1, tr.2, tr.3, bg);
                draw_rectangle_lines(
                    tr.0, tr.1, tr.2, tr.3, 1.0,
                    Color::from_rgba(50, 50, 62, 255),
                );
                draw_ui_text(
                    "Thickness", tr.0 + 10.0, tr.1 + tr.3 * 0.5 + 5.0, 13.0,
                    Color::from_rgba(150, 150, 165, 255),
                );
                let vt = wire_thickness.to_string();
                let vd = measure_ui_text(&vt, 13);
                draw_ui_text(
                    &vt,
                    tr.0 + tr.2 - vd.width - 10.0,
                    tr.1 + tr.3 * 0.5 + 5.0, 13.0,
                    Color::from_rgba(220, 220, 230, 255),
                );
                draw_ui_text(
                    if wire_start.is_some() {
                        "L-click endpoint  •  R cancel"
                    } else {
                        "L-click start point"
                    },
                    tr.0 + 2.0, tr.1 + tr.3 + 14.0, 11.0,
                    Color::from_rgba(130, 130, 150, 255),
                );
            }

            // Pipet status — target species + contents readout, plus a
            // Clear button. Only visible while pipet is the active tool.
            if tool_mode == ToolMode::Pipet {
                let sr = pipet_status_rect(prefab_open, wire_open);
                draw_rectangle(
                    sr.0, sr.1, sr.2, sr.3,
                    Color::from_rgba(24, 28, 36, 255),
                );
                draw_rectangle_lines(
                    sr.0, sr.1, sr.2, sr.3, 1.0,
                    Color::from_rgba(60, 60, 72, 255),
                );
                let dim_label = Color::from_rgba(130, 130, 150, 255);
                let value_color = Color::from_rgba(220, 220, 230, 255);
                draw_ui_text("TARGET", sr.0 + 10.0, sr.1 + 16.0, 11.0, dim_label);
                let target_text = match pipet_target {
                    None => "any (unfiltered)".to_string(),
                    Some((el, did)) => if el == Element::Derived {
                        derived_formula_of(did)
                    } else { el.name().to_string() },
                };
                draw_ui_text(&target_text, sr.0 + 10.0, sr.1 + 34.0, 15.0, value_color);
                let txt = format!("{} / {}", pipet_bucket.len(), PIPET_CAPACITY);
                draw_ui_text(&txt, sr.0 + 10.0, sr.1 + 58.0, 13.0, value_color);

                // Breakdown list — only interesting in grab-all mode where
                // the bucket can hold mixed species. Counts duplicates by
                // (element, derived_id) so atomic and derived cells group
                // separately, sorted descending.
                if pipet_target.is_none() && !pipet_bucket.is_empty() {
                    let mut tally: Vec<((Element, u8), usize)> =
                        Vec::with_capacity(16);
                    for c in &pipet_bucket {
                        let key = (c.el, c.derived_id);
                        if let Some(entry) =
                            tally.iter_mut().find(|(k, _)| *k == key)
                        {
                            entry.1 += 1;
                        } else {
                            tally.push((key, 1));
                        }
                    }
                    tally.sort_by(|a, b| b.1.cmp(&a.1));
                    let max_rows = 6usize;
                    let row_h = 14.0;
                    let row_x = sr.0 + 10.0;
                    let mut row_y = sr.1 + 80.0;
                    let shown = tally.len().min(max_rows);
                    for &(key, count) in &tally[..shown] {
                        let name = if key.0 == Element::Derived {
                            derived_formula_of(key.1)
                        } else { key.0.name().to_string() };
                        let line = format!("{:>4} {}", count, name);
                        draw_ui_text(
                            &line, row_x, row_y, 12.0,
                            Color::from_rgba(200, 200, 215, 255),
                        );
                        row_y += row_h;
                    }
                    if tally.len() > max_rows {
                        let rest = tally.len() - max_rows;
                        let line = format!("+{} more species", rest);
                        draw_ui_text(
                            &line, row_x, row_y, 11.0, dim_label,
                        );
                    }
                }
                // Warning flash — overlays the status box with a red
                // border + message while the countdown is active.
                if pipet_warning_frames > 0 {
                    let fade = (pipet_warning_frames as f32 / 120.0).min(1.0);
                    let alpha = (fade * 220.0) as u8;
                    draw_rectangle_lines(
                        sr.0, sr.1, sr.2, sr.3, 2.0,
                        Color::from_rgba(220, 80, 80, alpha),
                    );
                    let msg = "Empty pipet first!";
                    let mdim = measure_ui_text(msg, 12);
                    let mx_ = sr.0 + (sr.2 - mdim.width) * 0.5;
                    let my_ = sr.1 - 4.0;
                    draw_rectangle(
                        mx_ - 6.0, my_ - 12.0,
                        mdim.width + 12.0, 16.0,
                        Color::from_rgba(40, 20, 20, alpha),
                    );
                    draw_ui_text(
                        msg, mx_, my_, 12.0,
                        Color::from_rgba(240, 130, 130, alpha),
                    );
                }
                // Clear button — empties both target and contents.
                let cr = pipet_clear_rect(prefab_open, wire_open);
                let cr_hovered = mx >= cr.0 && mx < cr.0 + cr.2
                    && my >= cr.1 && my < cr.1 + cr.3;
                draw_panel_button(cr, "Clear", false, cr_hovered);
            }

            // Species-present list — always visible at the bottom of the
            // panel. Clicking a row jumps to the Pipet tool with that
            // species pre-selected as the target filter.
            {
                if !species_cache.is_empty() {
                    let first = species_list_row_rect(0, prefab_open, wire_open);
                    draw_ui_text(
                        "PICK SPECIES",
                        first.0 + 2.0, first.1 - 10.0, 11.0,
                        Color::from_rgba(130, 130, 150, 255),
                    );
                    for (i, &(el, did, count)) in species_cache.iter().enumerate() {
                        let r = species_list_row_rect(i, prefab_open, wire_open);
                        if r.1 + r.3 > screen_height() - 8.0 { break; }
                        let row_hovered = mx >= r.0 && mx < r.0 + r.2
                            && my >= r.1 && my < r.1 + r.3;
                        let is_sel = pipet_target == Some((el, did));
                        let bg = if is_sel {
                            Color::from_rgba(62, 90, 140, 255)
                        } else if row_hovered {
                            Color::from_rgba(38, 38, 50, 255)
                        } else {
                            Color::from_rgba(22, 22, 30, 255)
                        };
                        draw_rectangle(r.0, r.1, r.2, r.3, bg);
                        draw_rectangle_lines(
                            r.0, r.1, r.2, r.3, 1.0,
                            Color::from_rgba(48, 48, 60, 255),
                        );
                        let name = if el == Element::Derived {
                            derived_formula_of(did)
                        } else { el.name().to_string() };
                        draw_ui_text(
                            &name, r.0 + 8.0, r.1 + r.3 * 0.5 + 4.0, 12.0,
                            Color::from_rgba(220, 220, 230, 255),
                        );
                        let ctext = format!("{}", count);
                        let cdim = measure_ui_text(&ctext, 12);
                        draw_ui_text(
                            &ctext,
                            r.0 + r.2 - cdim.width - 8.0,
                            r.1 + r.3 * 0.5 + 4.0, 12.0,
                            Color::from_rgba(170, 170, 185, 255),
                        );
                    }
                }
            }

            // Paused indicator — top of panel, next to the TOOLS header,
            // so the state is visible even without the old top bar.
            if paused {
                let tag = if world.rewind_offset > 0 {
                    format!("PAUSED −{}", world.rewind_offset)
                } else { "PAUSED".to_string() };
                let td = measure_ui_text(&tag, 11);
                draw_ui_text(
                    &tag,
                    px + PANEL_WIDTH - td.width - 12.0,
                    PANEL_TOP_PAD + 10.0, 11.0, YELLOW,
                );
            }

            // Current element + brush radius — its own row between tools
            // and the SIMULATION section so both headers breathe.
            let el_r = panel_element_rect(prefab_open, wire_open);
            draw_ui_text(
                &format!(
                    "{}   B{}",
                    if selected == Element::Derived {
                        derived_formula_of(selected_did)
                    } else { selected.name().to_string() },
                    brush_radius,
                ),
                el_r.0 + 4.0, el_r.1 + el_r.3 - 4.0, 13.0,
                Color::from_rgba(200, 200, 220, 255),
            );

            // FPS at the bottom of the panel — right-aligned with a small
            // margin so digit width changes don't reflow anything.
            let fps_text = format!("{} fps", get_fps());
            let fps_dim = measure_ui_text(&fps_text, 12);
            draw_ui_text(
                &fps_text,
                px + PANEL_WIDTH - fps_dim.width - 12.0,
                screen_height() - 10.0, 12.0,
                Color::from_rgba(140, 140, 160, 255),
            );
        }

        // Screenshot notice — shown briefly over the sim after F2. Used to
        // live in the top bar; now it floats near the top-left of the sim.
        if screenshot_timer > 0 {
            screenshot_timer -= 1;
            if let Some(msg) = &screenshot_notice {
                draw_ui_text(msg, sim_x + 8.0, sim_y + 16.0, 16.0, GREEN);
            }
        }

        if in_sim {
            if tool_mode == ToolMode::Wire {
                // Wire ghost preview. When a start is set, show the line
                // that will be drawn. When no start is set, show a small
                // crosshair at the pending-start position.
                let cx = sim_x + gx as f32 * scale_fit + scale_fit * 0.5;
                let cy = sim_y + gy as f32 * scale_fit + scale_fit * 0.5;
                let (r, g, b) = wire_material.base_color();
                let accent = Color::from_rgba(r, g, b, 220);
                if let Some((sx, sy)) = wire_start {
                    let sx_s = sim_x + sx as f32 * scale_fit + scale_fit * 0.5;
                    let sy_s = sim_y + sy as f32 * scale_fit + scale_fit * 0.5;
                    let thick = (wire_thickness as f32 * 2.0 * scale_fit)
                        .max(2.0);
                    draw_line(sx_s, sy_s, cx, cy, thick, accent);
                    draw_circle(sx_s, sy_s, thick * 0.5 + 1.0, accent);
                }
                draw_circle(cx, cy, (wire_thickness as f32 * scale_fit).max(2.0), accent);
            } else if tool_mode == ToolMode::Prefab {
                // Ghost preview of the prefab footprint at the cursor.
                // Rotation swaps w/h when laying sideways, matching the
                // placement code.
                let rot = prefab_rotation & 3;
                let sideways = rot == 1 || rot == 3;
                let (pw, ph) = if sideways {
                    (prefab_height as f32, prefab_width as f32)
                } else {
                    (prefab_width as f32, prefab_height as f32)
                };
                let px0 = sim_x + (gx as f32 - pw * 0.5) * scale_fit;
                let py0 = sim_y + (gy as f32 - ph * 0.5) * scale_fit;
                draw_rectangle_lines(
                    px0, py0, pw * scale_fit, ph * scale_fit, 1.5,
                    Color::from_rgba(230, 200, 120, 220),
                );
                let tk = prefab_thickness as f32;
                // Battery preview: color the two end bands red/blue so
                // you can see which way + and − are pointing, rotating
                // with the prefab.
                if prefab_kind == PrefabKind::Battery && tk > 0.0 {
                    let pos_color = Color::from_rgba(170, 50, 50, 100);
                    let neg_color = Color::from_rgba(40, 70, 130, 100);
                    let band = tk * scale_fit;
                    // (pos_band_rect, neg_band_rect) in screen coords.
                    let pw_s = pw * scale_fit;
                    let ph_s = ph * scale_fit;
                    let (pos_r, neg_r) = match rot {
                        0 => (
                            (px0, py0, pw_s, band),
                            (px0, py0 + ph_s - band, pw_s, band),
                        ),
                        1 => (
                            (px0 + pw_s - band, py0, band, ph_s),
                            (px0, py0, band, ph_s),
                        ),
                        2 => (
                            (px0, py0 + ph_s - band, pw_s, band),
                            (px0, py0, pw_s, band),
                        ),
                        _ => (
                            (px0, py0, band, ph_s),
                            (px0 + pw_s - band, py0, band, ph_s),
                        ),
                    };
                    draw_rectangle(pos_r.0, pos_r.1, pos_r.2, pos_r.3, pos_color);
                    draw_rectangle(neg_r.0, neg_r.1, neg_r.2, neg_r.3, neg_color);
                }
            } else {
                let cx = sim_x + gx as f32 * scale_fit + scale_fit * 0.5;
                let cy = sim_y + gy as f32 * scale_fit + scale_fit * 0.5;
                let r = if selected == Element::Seed { scale_fit }
                        else { brush_radius as f32 * scale_fit };
                draw_circle_lines(cx, cy, r, 1.0, WHITE);
            }
        }

        // Cell inspector: hover shows element + physical state of the cell
        // directly under the cursor. Floats near the mouse, flips to the other
        // side when near the screen edge.
        if in_sim && gx >= 0 && gx < W as i32 && gy >= 0 && gy < H as i32 {
            let cell = world.cells[gy as usize * W + gx as usize];
            let phase_suffix = match cell.phase() {
                PHASE_SOLID  => " (solid)",
                PHASE_LIQUID => " (liquid)",
                PHASE_GAS    => " (gas)",
                _            => "",
            };
            let display_name: String = if cell.el == Element::Derived {
                derived_formula_of(cell.derived_id)
            } else {
                cell.el.name().to_string()
            };
            let idx_here = gy as usize * W + gx as usize;
            let mut info = format!(
                "{}{}  T{:+}°  P{:+}",
                display_name, phase_suffix, cell.temp, cell.pressure,
            );
            // Conductivity — always shown so users can see at a glance
            // whether a material is inert (Quartz 0.00) or conductive
            // (Cu 0.95) without guessing.
            let cond = cell.conductivity();
            info.push_str(&format!("  σ{:.2}", cond));
            // Voltage only when the cell is part of an active circuit.
            if world.energized[idx_here] && world.active_emf > 0.0 {
                info.push_str(&format!("  ⚡{:.0}V", world.active_emf));
            }
            // Moisture suppressed when zero (most cells).
            if cell.moisture > 0 {
                info.push_str(&format!("  m{}", cell.moisture));
            }
            // Dissolved solute — only shown when a liquid is carrying one.
            if cell.solute_amt > 0 {
                let label = if cell.solute_el == Element::Derived {
                    derived_formula_of(cell.solute_derived_id)
                } else {
                    cell.solute_el.name().to_string()
                };
                info.push_str(&format!("  {} {}", label, cell.solute_amt));
            }
            let dims = measure_ui_text(&info, 14);
            // Offset tooltip outside the brush-highlight circle so the
            // outline doesn't cut through the text. Add a small margin past
            // the circle's radius in screen pixels.
            let brush_px = brush_radius as f32 * scale_fit;
            let offset = (brush_px + 10.0).max(12.0);
            let mut tx = mx + offset;
            let mut ty = my - 10.0;
            if tx + dims.width + 6.0 > screen_width() {
                tx = mx - offset - dims.width;
            }
            if ty - 14.0 < TOP_BAR { ty = my + offset + 14.0; }
            draw_rectangle(tx - 3.0, ty - 12.0, dims.width + 6.0, 18.0,
                Color::from_rgba(0, 0, 0, 220));
            draw_ui_text(&info, tx, ty, 14.0, WHITE);
        }

        // Periodic table overlay draws on top of everything when open.
        // It's the only element/compound picker now — no bottom bar.
        if periodic_open {
            let hover = pt_hit(mx, my);
            let hover_c = pt_compound_hit(mx, my);
            let derived_palette: [u8; 2] = [hcl_id, aucl_id];
            let hover_d = pt_derived_hit(mx, my, derived_palette.len());
            draw_periodic_table(
                hover, hover_c, hover_d,
                selected, selected_did,
                &derived_palette,
            );
        }

        // Screenshot: captured AFTER all drawing so the saved image includes
        // UI and the periodic-table overlay if it's open. Uses the OpenGL
        // framebuffer via get_screen_data.
        if take_screenshot {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let path = format!("screenshot_{}.png", ts);
            let img = get_screen_data();
            img.export_png(&path);
            screenshot_notice = Some(format!("saved {}", path));
            screenshot_timer = 120;
        }

        let t_ui = t_ui_start.elapsed();
        prof_sim_us         += t_sim.as_micros() as u64;
        prof_render_cpu_us  += t_render_cpu.as_micros() as u64;
        prof_upload_us      += t_upload.as_micros() as u64;
        prof_gpu_submit_us  += t_gpu_submit.as_micros() as u64;
        prof_ui_us          += t_ui.as_micros() as u64;
        prof_total_us       += prof_frame_start.elapsed().as_micros() as u64;
        prof_frames         += 1;
        if prof_last_print.elapsed().as_secs_f32() >= 1.0 {
            let f = prof_frames as f32;
            eprintln!(
                "[prof] {:>3} fps | sim {:>4.1} render {:>4.1} upload {:>4.1} gpu_sub {:>4.1} ui {:>4.1} total {:>4.1}ms",
                prof_frames,
                (prof_sim_us as f32 / f) / 1000.0,
                (prof_render_cpu_us as f32 / f) / 1000.0,
                (prof_upload_us as f32 / f) / 1000.0,
                (prof_gpu_submit_us as f32 / f) / 1000.0,
                (prof_ui_us as f32 / f) / 1000.0,
                (prof_total_us as f32 / f) / 1000.0,
            );
            prof_sim_us = 0;
            prof_render_cpu_us = 0;
            prof_upload_us = 0;
            prof_gpu_submit_us = 0;
            prof_ui_us = 0;
            prof_total_us = 0;
            prof_frames = 0;
            prof_last_print = std::time::Instant::now();
        }
        next_frame().await
    }
}
