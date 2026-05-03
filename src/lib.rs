use macroquad::prelude::*;

// Sim grid dimensions. Sized so the play area fills the 1280×1024 window
// with zero dead space after reserving the right control panel (240 px):
//   avail = (1280 − 240) × 1024 = 1040 × 1024
//   cell scale = 1040 / 320 = 3.25 → H = 1024 / 3.25 ≈ 315
// Window is non-resizable so these proportions hold exactly. The old
// top status bar is gone — all state now lives in the side panel.
pub const W: usize = 320;
pub const H: usize = 315;

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
    // ---- Periodic-table fill, batch 1: rest of the main table ----
    // Period 2 missing. Lithium and beryllium.
    Li = 55,  Be = 56,
    // Period 4 transition + post-transition + p-block remainders.
    Sc = 57,  Ti = 58,  V  = 59,  Cr = 60,  Mn = 61,  Co = 62,
    Ga = 63,  Ge = 64,  As = 65,  Se = 66,  Br = 67,  Kr = 68,
    // Period 5.
    Rb = 69,  Sr = 70,  Y  = 71,  Zr = 72,  Nb = 73,  Mo = 74,
    Tc = 75,  Ru = 76,  Rh = 77,  Pd = 78,  Cd = 79,  In = 80,
    Sn = 81,  Sb = 82,  Te = 83,  I  = 84,  Xe = 85,
    // Period 6 main + lanthanides (period 8 in our layout).
    Ba = 86,
    La = 87,  Ce = 88,  Pr = 89,  Nd = 90,  Pm = 91,  Sm = 92,
    Eu = 93,  Gd = 94,  Tb = 95,  Dy = 96,  Ho = 97,  Er = 98,
    Tm = 99,  Yb = 100, Lu = 101,
    Hf = 102, Ta = 103, W  = 104, Re = 105, Os = 106, Ir = 107, Pt = 108,
    Tl = 109, Bi = 110, Po = 111, At = 112, Rn = 113,
    // Period 7 + actinides (period 9 in our layout).
    Fr = 114, Ac = 115, Th = 116, Pa = 117,
}
const ELEMENT_COUNT: usize = 118;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind { Empty, Solid, Gravel, Powder, Liquid, Gas, Fire }

// Mouse-tool mode. Paint places matter; Heat raises/lowers temperature;
// Vacuum sucks gas into a cursor-centered low-pressure zone and deletes it;
// Grab lifts existing cells from the sim and lets you reposition them —
// useful for dropping reactive elements into containers you've built.
#[derive(Clone, Copy, PartialEq)]
enum ToolMode { Paint, Heat, Vacuum, Pipet, Prefab, Wire }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PrefabKind { Beaker, Box, Battery, Line, Circle, Bowl }
const PREFAB_KIND_COUNT: usize = 6;
const PREFAB_KINDS: [PrefabKind; PREFAB_KIND_COUNT] = [
    PrefabKind::Beaker,
    PrefabKind::Box,
    PrefabKind::Battery,
    PrefabKind::Line,
    PrefabKind::Circle,
    PrefabKind::Bowl,
];
impl PrefabKind {
    fn label(self) -> &'static str {
        match self {
            PrefabKind::Beaker  => "Beaker",
            PrefabKind::Box     => "Box",
            PrefabKind::Battery => "Battery",
            PrefabKind::Line    => "Line",
            PrefabKind::Circle  => "Circle",
            PrefabKind::Bowl    => "Bowl",
        }
    }
}

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
    // Porosity for combustion: fraction of ambient O₂ that reaches a
    // fully-buried (no-air-cardinal) cell of this material via gaps
    // between same-material cells in a pile. 0.0 = airtight (stone,
    // dense metals), 1.0 = fully porous. Loose granular materials
    // (leaves, seed, gunpowder) get high values; dense waxy/crystalline
    // solids (P, sulfur) low values. Used by oxygen_available so a
    // leaves pile burns through its interior naturally while a packed
    // carbon pile chars layer-by-layer.
    air_permeability: f32,
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
    // Quartz / Firebrick: Kind::Gravel (was Solid). When frozen as a
    // wall they're inert anyway, but once an explosion ruptures
    // them and flips off the FROZEN flag, the loose chunks need to
    // fall under gravity — Kind::Solid only dispatches to
    // try_pressure_shove (no gravity path), so chunks were floating
    // mid-air after the wall broke.
    a[Element::Quartz    as usize] = PhysicsProfile { density: 33, kind: Kind::Gravel, viscosity: 0, molar_mass: 60.1 };
    a[Element::Firebrick as usize] = PhysicsProfile { density: 42, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    // Argon — monatomic noble gas, denser than air (Ar₂O doesn't exist,
    // it's chemically inert). Forms 1% of real atmosphere.
    a[Element::Ar        as usize] = PhysicsProfile { density: -1, kind: Kind::Gas,   viscosity: 0, molar_mass: 40.0 };
    // Battery — a solid structural cell representing a voltage source.
    // Behaves like a wall that injects energy into connected conductors.
    a[Element::BattPos as usize] = PhysicsProfile { density: 60, kind: Kind::Solid, viscosity: 0, molar_mass: 0.0 };
    a[Element::BattNeg as usize] = PhysicsProfile { density: 60, kind: Kind::Solid, viscosity: 0, molar_mass: 0.0 };
    // ---- Periodic-table fill ----
    // Density values are real_density × 10 to fit our integer
    // scale (Stone=100, Iron=79, Mg=17). Negative for gases so the
    // gas-phase logic correctly buoys/sinks vs ambient air.
    // Period 2.
    a[Element::Li as usize] = PhysicsProfile { density:   5, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Be as usize] = PhysicsProfile { density:  19, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    // Period 4 fill.
    a[Element::Sc as usize] = PhysicsProfile { density:  30, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ti as usize] = PhysicsProfile { density:  45, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::V  as usize] = PhysicsProfile { density:  60, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Cr as usize] = PhysicsProfile { density:  72, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Mn as usize] = PhysicsProfile { density:  72, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Co as usize] = PhysicsProfile { density:  89, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ga as usize] = PhysicsProfile { density:  59, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ge as usize] = PhysicsProfile { density:  53, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::As as usize] = PhysicsProfile { density:  57, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Se as usize] = PhysicsProfile { density:  48, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    // Bromine — only nonmetal liquid at STP besides Hg.
    a[Element::Br as usize] = PhysicsProfile { density:  31, kind: Kind::Liquid, viscosity: 80, molar_mass: 159.8 };
    a[Element::Kr as usize] = PhysicsProfile { density:  -2, kind: Kind::Gas,    viscosity: 0, molar_mass: 83.8 };
    // Period 5.
    a[Element::Rb as usize] = PhysicsProfile { density:  15, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Sr as usize] = PhysicsProfile { density:  26, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Y  as usize] = PhysicsProfile { density:  45, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Zr as usize] = PhysicsProfile { density:  65, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Nb as usize] = PhysicsProfile { density:  86, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Mo as usize] = PhysicsProfile { density: 103, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Tc as usize] = PhysicsProfile { density: 110, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ru as usize] = PhysicsProfile { density: 124, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Rh as usize] = PhysicsProfile { density: 124, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Pd as usize] = PhysicsProfile { density: 120, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Cd as usize] = PhysicsProfile { density:  87, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::In as usize] = PhysicsProfile { density:  73, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Sn as usize] = PhysicsProfile { density:  73, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Sb as usize] = PhysicsProfile { density:  67, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Te as usize] = PhysicsProfile { density:  62, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::I  as usize] = PhysicsProfile { density:  49, kind: Kind::Gravel, viscosity: 0, molar_mass: 253.8 };
    a[Element::Xe as usize] = PhysicsProfile { density:  -1, kind: Kind::Gas,    viscosity: 0, molar_mass: 131.3 };
    // Period 6 main + lanthanides.
    a[Element::Ba as usize] = PhysicsProfile { density:  35, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::La as usize] = PhysicsProfile { density:  61, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ce as usize] = PhysicsProfile { density:  67, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Pr as usize] = PhysicsProfile { density:  67, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Nd as usize] = PhysicsProfile { density:  70, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Pm as usize] = PhysicsProfile { density:  72, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Sm as usize] = PhysicsProfile { density:  75, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Eu as usize] = PhysicsProfile { density:  52, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Gd as usize] = PhysicsProfile { density:  79, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Tb as usize] = PhysicsProfile { density:  82, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Dy as usize] = PhysicsProfile { density:  86, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ho as usize] = PhysicsProfile { density:  88, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Er as usize] = PhysicsProfile { density:  91, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Tm as usize] = PhysicsProfile { density:  93, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Yb as usize] = PhysicsProfile { density:  70, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Lu as usize] = PhysicsProfile { density:  98, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Hf as usize] = PhysicsProfile { density: 133, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ta as usize] = PhysicsProfile { density: 167, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::W  as usize] = PhysicsProfile { density: 193, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Re as usize] = PhysicsProfile { density: 210, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Os as usize] = PhysicsProfile { density: 226, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ir as usize] = PhysicsProfile { density: 226, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Pt as usize] = PhysicsProfile { density: 215, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Tl as usize] = PhysicsProfile { density: 119, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Bi as usize] = PhysicsProfile { density:  98, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Po as usize] = PhysicsProfile { density:  92, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::At as usize] = PhysicsProfile { density:  64, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Rn as usize] = PhysicsProfile { density:  -2, kind: Kind::Gas,    viscosity: 0, molar_mass: 222.0 };
    // Period 7 + actinides.
    a[Element::Fr as usize] = PhysicsProfile { density:  19, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Ac as usize] = PhysicsProfile { density: 100, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Th as usize] = PhysicsProfile { density: 117, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a[Element::Pa as usize] = PhysicsProfile { density: 154, kind: Kind::Gravel, viscosity: 0, molar_mass: 0.0 };
    a
};

static THERMAL: [ThermalProfile; ELEMENT_COUNT] = {
    const NONE_PH: Option<Phase> = None;
    const fn base() -> ThermalProfile {
        ThermalProfile {
            initial_temp: 20, ambient_temp: 20, ambient_rate: 0.001,
            conductivity: 0.02, heat_capacity: 1.0,
            freeze_below: NONE_PH, melt_above: NONE_PH,
            boil_above: NONE_PH, condense_below: NONE_PH,
            ignite_above: None, burn_duration: None, burn_temp: None,
            air_permeability: 0.0,
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
        burn_temp: Some(700),
        // Logs char from the outside in — solid grain, low interior O₂
        // access. Surface burns fast, interior chars slowly.
        air_permeability: 0.15,
        ..base()
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
        ignite_above: Some(320), burn_duration: Some(80),
        burn_temp: Some(650),
        // Loose granular pile, lots of inter-grain air gaps.
        air_permeability: 0.55,
        ..base()
    };
    a[Element::Mud as usize]      = ThermalProfile {
        ambient_rate: 0.003, conductivity: 0.030, heat_capacity: 1.2, ..base()
    };
    a[Element::Leaves as usize]   = ThermalProfile {
        ambient_rate: 0.005, conductivity: 0.020, heat_capacity: 1.0,
        ignite_above: Some(230), burn_duration: Some(20),
        burn_temp: Some(600),
        // Crumpled non-conformal leaves pack with huge air gaps —
        // a leaf pile lights up almost uniformly fast. Short burn
        // duration so leaves visibly flash through in a few frames.
        air_permeability: 0.95,
        ..base()
    };
    a[Element::Oil as usize]      = ThermalProfile {
        ambient_rate: 0.006, conductivity: 0.035, heat_capacity: 2.2,
        ignite_above: Some(180), burn_duration: Some(180),
        // Realistic hydrocarbon flame: hotter than wood. Water adjacent still
        // wins in sufficient volume, thanks to the increased latent heat of
        // vaporization — each boil drains 1200 energy from this cell.
        burn_temp: Some(800),
        // Liquid oil burns at the surface; submerged interior gets
        // very little O₂. Low porosity matches "pool fire" behavior.
        air_permeability: 0.05,
        ..base()
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
        ignite_above: Some(500), burn_duration: Some(255), burn_temp: Some(900),
        // Carbon piles (lump charcoal, coal) burn from the surface
        // inward — moderate porosity. Matches grill behavior where
        // outer lumps glow while inner ones char. Long burn duration
        // so a coal pile sustains for minutes of game time.
        air_permeability: 0.20,
        ..base()
    };
    a[Element::Na as usize] = ThermalProfile {
        ignite_above: Some(100), burn_duration: Some(80), burn_temp: Some(700), ..base()
    };
    a[Element::Mg as usize] = ThermalProfile {
        // Magnesium burns with an intense white flame at 2000°C+.
        ignite_above: Some(470), burn_duration: Some(90), burn_temp: Some(2200), ..base()
    };
    a[Element::Ca as usize] = ThermalProfile {
        // Calcium auto-ignites in air around 480°C as fine powder; bulk
        // Ca needs higher. Burns with a brick-red flame at ~1500°C.
        ignite_above: Some(500), burn_duration: Some(100), burn_temp: Some(1500), ..base()
    };
    a[Element::Sc as usize] = ThermalProfile {
        // Bulk Sc oxidizes via chemistry (Sc + O → Sc₂O₃ at any temp via
        // emergent reaction), it doesn't sustain a fire cascade. Leaving
        // ignite_above unset matches the Rust→Fe pattern: oxide
        // decomposition produces clean bulk metal that doesn't re-burn
        // back into Sc₂O₃. Real Sc combustion is a fine-powder behavior,
        // not bulk; we don't model that distinction. Sc still tarnishes
        // ambiently and forms a passivating Sc₂O₃ Gravel layer.
        ..base()
    };
    a[Element::P  as usize] = ThermalProfile {
        // White phosphorus auto-ignites just above room temperature.
        ignite_above: Some(30), burn_duration: Some(60), burn_temp: Some(900),
        // Waxy solid, low interior porosity — surface-only ignition
        // when piled, and effectively zero when submerged in water.
        air_permeability: 0.05,
        ..base()
    };
    a[Element::S  as usize] = ThermalProfile {
        // Match Na/K's "normal colored burner" profile so sulfur
        // doesn't visually outshout the other flame-color metals.
        // The blue tint still applies to nearby Fire via flame_color
        // + the existing color_fires pass; we just don't want S
        // double-emitting flame-test Fire on top of its burn cascade.
        ignite_above: Some(232), burn_duration: Some(80), burn_temp: Some(700),
        air_permeability: 0.0,
        ..base()
    };
    a[Element::Se as usize] = ThermalProfile {
        // Real Se ignites in air around 340°C, burns with a vivid
        // azure-blue flame producing SeO₂ vapor. Use a slightly cooler
        // burn_temp than S (Se's combustion is less energetic) so the
        // surrounding cells don't all flash to combustion.
        ignite_above: Some(340), burn_duration: Some(100), burn_temp: Some(620),
        air_permeability: 0.0,
        ..base()
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
        // Coarse grains with lots of air gaps; gunpowder is mostly
        // self-oxidizing anyway (KNO₃ + S + C) so this is mostly
        // belt-and-suspenders for chain ignition through a pile.
        air_permeability: 0.60,
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
    // ---- Periodic-table fill: bespoke thermal behaviour ----
    // Reactive alkali metals that ignite in air at low temps. Heavier
    // alkalis (Rb, Fr) react more violently than Na — modeled with a
    // lower ignition threshold and shorter burn.
    a[Element::Li as usize] = ThermalProfile {
        ignite_above: Some(180), burn_duration: Some(60), burn_temp: Some(1200), ..base()
    };
    a[Element::Rb as usize] = ThermalProfile {
        ignite_above: Some(40), burn_duration: Some(70), burn_temp: Some(750), ..base()
    };
    a[Element::Fr as usize] = ThermalProfile {
        // All Fr isotopes are radioactive; in real life nothing has
        // ever been observed in bulk. We treat it as the most reactive
        // alkali — ignites essentially on contact.
        ignite_above: Some(28), burn_duration: Some(8), burn_temp: Some(1400), ..base()
    };
    // Strontium and Barium burn with the characteristic flame-test
    // colours (Sr crimson, Ba green). Magnesium-style ignition.
    a[Element::Sr as usize] = ThermalProfile {
        ignite_above: Some(720), burn_duration: Some(80), burn_temp: Some(2000), ..base()
    };
    a[Element::Ba as usize] = ThermalProfile {
        ignite_above: Some(700), burn_duration: Some(80), burn_temp: Some(2000), ..base()
    };
    // Be and Ti are bulk-non-combustible in this model — same as Fe.
    // Real Be/Ti combustion is a fine-powder phenomenon (sparkler /
    // pyrotechnic flake), not a bulk-metal behavior; we don't model
    // particle size. Oxidation runs through chemistry (Be+O → BeO,
    // Ti+O → TiO₂) and the oxide layer passivates as a Gravel
    // coating. Treating them as flammable gave the Rust→Fe-style
    // oxide decomposition no clean exit: the freed metal would
    // re-ignite in any sustained heat zone and reform the oxide,
    // making decomp visually identical to a phase cycle. Default
    // base() leaves ignite_above/burn_temp/burn_duration unset.
    a[Element::Be as usize] = ThermalProfile { ..base() };
    a[Element::Ti as usize] = ThermalProfile { ..base() };
    // Yttrium — same Sc/Ti/Be tier: bulk Y is non-combustible in this
    // model. Real Y combustion is a fine-powder phenomenon (sparkler /
    // pyrotechnic flake), not a bulk-metal behavior; we don't model
    // particle size. Oxidation runs through chemistry (Y+O → Y₂O₃)
    // and the oxide layer passivates as a Gravel coating. Treating Y
    // as combustible gave the same Rust→Fe oxide-decomposition no
    // clean exit: freed metal would re-ignite in any sustained heat
    // zone and reform the oxide, making decomp visually identical to
    // a phase cycle. Default base() leaves ignite_above unset.
    a[Element::Y as usize] = ThermalProfile { ..base() };
    // Boron — finely divided B powder burns with a vivid green flame
    // (boric acid + boron itself give the same flame test). Real
    // amorphous boron auto-ignites in air around 700°C; we set the
    // threshold a touch lower so it's reachable without a torch on
    // every sustained heat source. Slow burn (long duration) compared
    // to Mg's flash, since B combustion in air is more deliberate.
    a[Element::B as usize] = ThermalProfile {
        ignite_above: Some(600), burn_duration: Some(80), burn_temp: Some(1800), ..base()
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
    // Salt is hygroscopic — absorbs water (and other liquids like Br
    // for halogen displacement) so a NaCl pile soaks instead of just
    // being a surface contact line.
    a[Element::Salt   as usize] = MoistureProfile {
        conductivity: 0.07, is_sink: true, ..base()
    };
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
    // ---- Periodic-table fill ----
    // Soft alkali/alkaline-earth metals (Li/Rb/Sr/Ba/Fr) get higher
    // compliance — they yield easily under blast pressure. Hard
    // transition metals (Ti/Cr/W/Re/Os/Ir/Pt) get low compliance — like
    // Fe/Pt. Post-transition (Sn/Tl/Bi) and lanthanides land in between.
    a[Element::Li as usize] = PressureProfile { permeability: 0, compliance: 32, formation_pressure: 0 };
    a[Element::Be as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Sc as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Ti as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::V  as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Cr as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Mn as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::Co as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::Ga as usize] = PressureProfile { permeability: 0, compliance: 25, formation_pressure: 0 };
    a[Element::Ge as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::As as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::Se as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    // Bromine — liquid, transmits pressure like other liquids.
    a[Element::Br as usize] = PressureProfile { permeability: 100, compliance: 50, formation_pressure: 0 };
    // Krypton/Xenon/Radon — gases.
    a[Element::Kr as usize] = PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 };
    a[Element::Xe as usize] = PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 };
    a[Element::Rn as usize] = PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 };
    a[Element::Rb as usize] = PressureProfile { permeability: 0, compliance: 32, formation_pressure: 0 };
    a[Element::Sr as usize] = PressureProfile { permeability: 0, compliance: 22, formation_pressure: 0 };
    a[Element::Y  as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Zr as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Nb as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Mo as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Tc as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Ru as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Rh as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Pd as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::Cd as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::In as usize] = PressureProfile { permeability: 0, compliance: 22, formation_pressure: 0 };
    a[Element::Sn as usize] = PressureProfile { permeability: 0, compliance: 22, formation_pressure: 0 };
    a[Element::Sb as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::Te as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
    a[Element::I  as usize] = PressureProfile { permeability: 20, compliance: 18, formation_pressure: 0 };
    a[Element::Ba as usize] = PressureProfile { permeability: 0, compliance: 20, formation_pressure: 0 };
    // Lanthanides — modest compliance (similar across the series).
    a[Element::La as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Ce as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Pr as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Nd as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Pm as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Sm as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Eu as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Gd as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Tb as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Dy as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Ho as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Er as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Tm as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Yb as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Lu as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Hf as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Ta as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::W  as usize] = PressureProfile { permeability: 0, compliance:  8, formation_pressure: 0 };
    a[Element::Re as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Os as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Ir as usize] = PressureProfile { permeability: 0, compliance: 10, formation_pressure: 0 };
    a[Element::Pt as usize] = PressureProfile { permeability: 0, compliance: 12, formation_pressure: 0 };
    a[Element::Tl as usize] = PressureProfile { permeability: 0, compliance: 22, formation_pressure: 0 };
    a[Element::Bi as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Po as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::At as usize] = PressureProfile { permeability: 0, compliance: 18, formation_pressure: 0 };
    a[Element::Fr as usize] = PressureProfile { permeability: 0, compliance: 32, formation_pressure: 0 };
    a[Element::Ac as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Th as usize] = PressureProfile { permeability: 0, compliance: 16, formation_pressure: 0 };
    a[Element::Pa as usize] = PressureProfile { permeability: 0, compliance: 14, formation_pressure: 0 };
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
    // ---- Periodic-table fill ----
    // Conductivity values keyed to the real-world ordering Ag > Cu > Au
    // > Al > Ca > Be > W > Mo > Mg > Zn > Co > Ni > Fe > Pt > Sn > Pb,
    // mapped onto our 0–1 sandbox scale.
    a[Element::Li as usize] = ElectricalProfile { conductivity: 0.32, glow_color: None };
    a[Element::Be as usize] = ElectricalProfile { conductivity: 0.55, glow_color: None };
    a[Element::Sc as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Ti as usize] = ElectricalProfile { conductivity: 0.12, glow_color: None };
    a[Element::V  as usize] = ElectricalProfile { conductivity: 0.12, glow_color: None };
    a[Element::Cr as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Mn as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Co as usize] = ElectricalProfile { conductivity: 0.32, glow_color: None };
    a[Element::Ga as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Ge as usize] = ElectricalProfile { conductivity: 0.04, glow_color: None };
    // As/Se/Te are metalloids/nonmetals; near-insulating.
    a[Element::Rb as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::Sr as usize] = ElectricalProfile { conductivity: 0.18, glow_color: None };
    a[Element::Y  as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Zr as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Nb as usize] = ElectricalProfile { conductivity: 0.30, glow_color: None };
    a[Element::Mo as usize] = ElectricalProfile { conductivity: 0.55, glow_color: None };
    a[Element::Tc as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::Ru as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Rh as usize] = ElectricalProfile { conductivity: 0.55, glow_color: None };
    a[Element::Pd as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Cd as usize] = ElectricalProfile { conductivity: 0.25, glow_color: None };
    a[Element::In as usize] = ElectricalProfile { conductivity: 0.30, glow_color: None };
    a[Element::Sn as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::Sb as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Ba as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    // Lanthanides — generally ~0.06–0.16 sim units (modest conductors).
    a[Element::La as usize] = ElectricalProfile { conductivity: 0.07, glow_color: None };
    a[Element::Ce as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Pr as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Nd as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Pm as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Sm as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Eu as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Gd as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Tb as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Dy as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Ho as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Er as usize] = ElectricalProfile { conductivity: 0.05, glow_color: None };
    a[Element::Tm as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Yb as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Lu as usize] = ElectricalProfile { conductivity: 0.06, glow_color: None };
    a[Element::Hf as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Ta as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::W  as usize] = ElectricalProfile { conductivity: 0.50, glow_color: None };
    a[Element::Re as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::Os as usize] = ElectricalProfile { conductivity: 0.30, glow_color: None };
    a[Element::Ir as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Pt as usize] = ElectricalProfile { conductivity: 0.40, glow_color: None };
    a[Element::Tl as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Bi as usize] = ElectricalProfile { conductivity: 0.02, glow_color: None };
    // Po/At — modest semiconductor-ish.
    a[Element::Fr as usize] = ElectricalProfile { conductivity: 0.18, glow_color: None };
    a[Element::Ac as usize] = ElectricalProfile { conductivity: 0.10, glow_color: None };
    a[Element::Th as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    a[Element::Pa as usize] = ElectricalProfile { conductivity: 0.20, glow_color: None };
    // Noble gases get glow colors (neon-tube behaviour) when energized.
    // Krypton: pale white-blue. Xenon: deep blue. Radon: same family —
    // but radon is so rare in the sim we skip glow.
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
    // Nitrogen — discharge tubes glow pink-orange / red-violet from
    // the broad N₂ band emission. Real "neon" novelty signs that look
    // pink are actually nitrogen, not neon.
    a[Element::N as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((255, 130, 160)),
    };
    // Oxygen — discharge tubes glow pale violet/lavender. The atomic
    // O emission is mostly in a few violet lines; molecular O₂ adds
    // some red-orange, blending toward a pinkish-violet overall.
    a[Element::O as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((200, 150, 240)),
    };
    // Fluorine — discharge tubes glow pale pink-violet / red-violet.
    // Strongest visible F atomic lines are in the red (624/685/703 nm)
    // and the molecular band emission adds violet — combined to a dim
    // pink-purple, distinctively different from the yellow-green gas
    // color it would have when un-energized.
    a[Element::F as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((220, 140, 200)),
    };
    // Chlorine — discharge tubes glow pale yellow-green to apple-green
    // depending on pressure. Cl₂ atomic emission is dominated by green
    // bands (around 540 nm) with a touch of red. The glow is dimmer
    // and more uniform than the noble-gas tubes.
    a[Element::Cl as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((180, 240, 130)),
    };
    // Krypton — pale white with a faint blue. Real Kr discharge tubes
    // are off-white from the broad multi-line emission spectrum.
    a[Element::Kr as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((220, 230, 255)),
    };
    // Xenon — deep blue-violet, classic Xe-arc lamp colour.
    a[Element::Xe as usize] = ElectricalProfile {
        conductivity: 0.0, glow_color: Some((90, 120, 255)),
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
    AtomProfile { number: 3, symbol: "Li", name: "Lithium",
        period: 2, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 6.94, melting_point: 180, boiling_point: 1342, density_stp: 0.534,
        electronegativity: 0.98, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "lightest metal; reactive alkali; ignites in air; lithium battery anode" },
    AtomProfile { number: 4, symbol: "Be", name: "Beryllium",
        period: 2, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 9.012, melting_point: 1287, boiling_point: 2469, density_stp: 1.85,
        electronegativity: 1.57, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "toxic; very high mp for a light metal; copper-beryllium alloys" },
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
    AtomProfile { number: 21, symbol: "Sc", name: "Scandium",
        period: 4, group: 3, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 44.956, melting_point: 1541, boiling_point: 2836, density_stp: 2.985,
        electronegativity: 1.36, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "rare-earth-adjacent transition metal; aerospace alloy additive" },
    AtomProfile { number: 22, symbol: "Ti", name: "Titanium",
        period: 4, group: 4, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 47.867, melting_point: 1668, boiling_point: 3287, density_stp: 4.506,
        electronegativity: 1.54, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "strong, light, corrosion-resistant; aerospace and implants" },
    AtomProfile { number: 23, symbol: "V", name: "Vanadium",
        period: 4, group: 5, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 50.942, melting_point: 1910, boiling_point: 3407, density_stp: 6.0,
        electronegativity: 1.63, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "hard transition metal; steel additive for spring/tool steels" },
    AtomProfile { number: 24, symbol: "Cr", name: "Chromium",
        period: 4, group: 6, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 51.996, melting_point: 1907, boiling_point: 2671, density_stp: 7.19,
        electronegativity: 1.66, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "lustrous corrosion-resistant; chrome plating, stainless-steel additive" },
    AtomProfile { number: 25, symbol: "Mn", name: "Manganese",
        period: 4, group: 7, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 54.938, melting_point: 1246, boiling_point: 2061, density_stp: 7.21,
        electronegativity: 1.55, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "hard brittle transition metal; steel alloying, batteries" },
    AtomProfile { number: 26, symbol: "Fe", name: "Iron",
        period: 4, group: 8, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 55.845, melting_point: 1538, boiling_point: 2862, density_stp: 7.874,
        electronegativity: 1.83, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "workhorse metal; rusts in moist air (Fe + O + H2O); magnetic" },
    AtomProfile { number: 27, symbol: "Co", name: "Cobalt",
        period: 4, group: 9, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 58.933, melting_point: 1495, boiling_point: 2927, density_stp: 8.90,
        electronegativity: 1.88, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "ferromagnetic; classic blue-pigment metal; superalloy and battery cathodes" },
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
    AtomProfile { number: 31, symbol: "Ga", name: "Gallium",
        period: 4, group: 13, stp_state: SSolid, category: PostTransition,
        atomic_mass: 69.723, melting_point: 30, boiling_point: 2204, density_stp: 5.91,
        electronegativity: 1.81, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "famously melts in your hand (mp ~30°C); semiconductor; embrittles aluminum" },
    AtomProfile { number: 32, symbol: "Ge", name: "Germanium",
        period: 4, group: 14, stp_state: SSolid, category: Metalloid,
        atomic_mass: 72.630, melting_point: 938, boiling_point: 2833, density_stp: 5.32,
        electronegativity: 2.01, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "metalloid semiconductor; precursor to silicon in early transistors" },
    AtomProfile { number: 33, symbol: "As", name: "Arsenic",
        period: 4, group: 15, stp_state: SSolid, category: Metalloid,
        atomic_mass: 74.922, melting_point: 817, boiling_point: 614, density_stp: 5.73,
        electronegativity: 2.18, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "toxic metalloid; sublimes (bp < mp at 1 atm); historical poison" },
    AtomProfile { number: 34, symbol: "Se", name: "Selenium",
        period: 4, group: 16, stp_state: SSolid, category: Nonmetal,
        atomic_mass: 78.971, melting_point: 221, boiling_point: 685, density_stp: 4.81,
        electronegativity: 2.55, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "nonmetal; semiconductor used in xerography and red glass tints" },
    AtomProfile { number: 35, symbol: "Br", name: "Bromine",
        period: 4, group: 17, stp_state: SLiquid, category: Halogen,
        atomic_mass: 79.904, melting_point: -7, boiling_point: 59, density_stp: 3.10,
        electronegativity: 2.96, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "only nonmetal liquid at STP besides Hg; toxic red-brown; reacts with metals" },
    AtomProfile { number: 36, symbol: "Kr", name: "Krypton",
        period: 4, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 83.798, melting_point: -157, boiling_point: -153, density_stp: 0.003733,
        electronegativity: 0.0, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "noble gas; whitish glow when energized; inert to everything except F at extremes" },

    // ---- Period 5 ----
    AtomProfile { number: 37, symbol: "Rb", name: "Rubidium",
        period: 5, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 85.468, melting_point: 39, boiling_point: 688, density_stp: 1.532,
        electronegativity: 0.82, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "very low-mp alkali; ignites on contact with air; atomic clocks" },
    AtomProfile { number: 38, symbol: "Sr", name: "Strontium",
        period: 5, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 87.62, melting_point: 777, boiling_point: 1382, density_stp: 2.64,
        electronegativity: 0.95, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "burns crimson-red in flame tests; signal flares and fireworks" },
    AtomProfile { number: 39, symbol: "Y", name: "Yttrium",
        period: 5, group: 3, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 88.906, melting_point: 1526, boiling_point: 2930, density_stp: 4.472,
        electronegativity: 1.22, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "rare-earth-adjacent; YBCO superconductor compound" },
    AtomProfile { number: 40, symbol: "Zr", name: "Zirconium",
        period: 5, group: 4, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 91.224, melting_point: 1855, boiling_point: 4377, density_stp: 6.52,
        electronegativity: 1.33, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "low neutron absorption; nuclear fuel rod cladding (Zircaloy)" },
    AtomProfile { number: 41, symbol: "Nb", name: "Niobium",
        period: 5, group: 5, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 92.906, melting_point: 2477, boiling_point: 4744, density_stp: 8.57,
        electronegativity: 1.6, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "superconducting alloys; jet-engine superalloys" },
    AtomProfile { number: 42, symbol: "Mo", name: "Molybdenum",
        period: 5, group: 6, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 95.95, melting_point: 2623, boiling_point: 4639, density_stp: 10.28,
        electronegativity: 2.16, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "very high mp; high-strength steel additive (Mo steel)" },
    AtomProfile { number: 43, symbol: "Tc", name: "Technetium",
        period: 5, group: 7, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 98.0, melting_point: 2157, boiling_point: 4265, density_stp: 11.0,
        electronegativity: 1.9, valence_electrons: 7,
        // Tc-99 real half-life is ~211k years — permanent in any
        // realistic timescale, but for sandbox visibility we set a
        // U-tier game-compressed value (~14 minutes) so multi-cell
        // Tc piles produce visible Ru transmutation activity in real
        // time. β⁻ decay → Ru-99; decay_heat 0 since β decay heat
        // per atom is negligible (no thermal accumulation).
        half_life_frames: 3_000_000,
        decay_product: Element::Ru,
        decay_heat: 0,
        implemented: true,
        notes: "first synthetically-produced element; all isotopes radioactive (β⁻ → Ru)" },
    AtomProfile { number: 44, symbol: "Ru", name: "Ruthenium",
        period: 5, group: 8, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 101.07, melting_point: 2334, boiling_point: 4150, density_stp: 12.45,
        electronegativity: 2.2, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "platinum-group metal; corrosion-resistant; hardens Pt/Pd alloys" },
    AtomProfile { number: 45, symbol: "Rh", name: "Rhodium",
        period: 5, group: 9, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 102.91, melting_point: 1964, boiling_point: 3695, density_stp: 12.41,
        electronegativity: 2.28, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "platinum-group; catalytic converters; one of the rarest stable metals" },
    AtomProfile { number: 46, symbol: "Pd", name: "Palladium",
        period: 5, group: 10, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 106.42, melting_point: 1555, boiling_point: 2963, density_stp: 12.023,
        electronegativity: 2.20, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "absorbs hydrogen up to 900× its volume; catalysis, fuel cells" },
    AtomProfile { number: 47, symbol: "Ag", name: "Silver",
        period: 5, group: 11, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 107.87, melting_point: 962, boiling_point: 2162, density_stp: 10.49,
        electronegativity: 1.93, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "highest electrical/thermal conductivity of any metal" },
    AtomProfile { number: 48, symbol: "Cd", name: "Cadmium",
        period: 5, group: 12, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 112.41, melting_point: 321, boiling_point: 767, density_stp: 8.65,
        electronegativity: 1.69, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "toxic; nickel-cadmium batteries; electroplating" },
    AtomProfile { number: 49, symbol: "In", name: "Indium",
        period: 5, group: 13, stp_state: SSolid, category: PostTransition,
        atomic_mass: 114.82, melting_point: 156, boiling_point: 2072, density_stp: 7.31,
        electronegativity: 1.78, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "soft, low-mp metal; ITO touchscreens; low-temp solder" },
    AtomProfile { number: 50, symbol: "Sn", name: "Tin",
        period: 5, group: 14, stp_state: SSolid, category: PostTransition,
        atomic_mass: 118.71, melting_point: 232, boiling_point: 2602, density_stp: 7.265,
        electronegativity: 1.96, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "low-mp post-transition; bronze with Cu, pewter, classic solder" },
    AtomProfile { number: 51, symbol: "Sb", name: "Antimony",
        period: 5, group: 15, stp_state: SSolid, category: Metalloid,
        atomic_mass: 121.76, melting_point: 631, boiling_point: 1587, density_stp: 6.685,
        electronegativity: 2.05, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "metalloid; flame retardants; lead-acid battery additive" },
    AtomProfile { number: 52, symbol: "Te", name: "Tellurium",
        period: 5, group: 16, stp_state: SSolid, category: Metalloid,
        atomic_mass: 127.60, melting_point: 449, boiling_point: 988, density_stp: 6.232,
        electronegativity: 2.10, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "metalloid; thermoelectric and rewritable optical media (CdTe, GeSbTe)" },
    AtomProfile { number: 53, symbol: "I", name: "Iodine",
        period: 5, group: 17, stp_state: SSolid, category: Halogen,
        atomic_mass: 126.90, melting_point: 113, boiling_point: 184, density_stp: 4.93,
        electronegativity: 2.66, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "halogen; sublimes to a violet vapor; antiseptic, thyroid biochemistry" },
    AtomProfile { number: 54, symbol: "Xe", name: "Xenon",
        period: 5, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 131.29, melting_point: -111, boiling_point: -108, density_stp: 0.005887,
        electronegativity: 2.6, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "noble gas; bright blue-white discharge lamps; first noble to form compounds" },

    // ---- Period 6 (main row skips lanthanides 57-71) ----
    AtomProfile { number: 55, symbol: "Cs", name: "Caesium",
        period: 6, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 132.91, melting_point: 28, boiling_point: 671, density_stp: 1.93,
        electronegativity: 0.79, valence_electrons: 1,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "most reactive stable alkali; melts at body temp; detonates in water" },
    AtomProfile { number: 56, symbol: "Ba", name: "Barium",
        period: 6, group: 2, stp_state: SSolid, category: AlkalineEarth,
        atomic_mass: 137.33, melting_point: 727, boiling_point: 1845, density_stp: 3.51,
        electronegativity: 0.89, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "burns vivid green in flame tests; barium sulfate radio contrast" },

    // ---- Period 8 row: Lanthanides (57-71) ----
    AtomProfile { number: 57, symbol: "La", name: "Lanthanum",
        period: 8, group: 3, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 138.91, melting_point: 920, boiling_point: 3464, density_stp: 6.145,
        electronegativity: 1.10, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare-earth; namesake of the lanthanide series; nickel-metal-hydride batteries" },
    AtomProfile { number: 58, symbol: "Ce", name: "Cerium",
        period: 8, group: 4, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 140.12, melting_point: 798, boiling_point: 3443, density_stp: 6.770,
        electronegativity: 1.12, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "most abundant rare earth; flint sparks (mischmetal)" },
    AtomProfile { number: 59, symbol: "Pr", name: "Praseodymium",
        period: 8, group: 5, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 140.91, melting_point: 931, boiling_point: 3520, density_stp: 6.773,
        electronegativity: 1.13, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; yellow-green pigments and welding goggles" },
    AtomProfile { number: 60, symbol: "Nd", name: "Neodymium",
        period: 8, group: 6, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 144.24, melting_point: 1024, boiling_point: 3074, density_stp: 7.007,
        electronegativity: 1.14, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; world's strongest permanent magnets (NdFeB)" },
    AtomProfile { number: 61, symbol: "Pm", name: "Promethium",
        period: 8, group: 7, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 145.0, melting_point: 1042, boiling_point: 3000, density_stp: 7.26,
        electronegativity: 1.13, valence_electrons: 3,
        // Pm-145 real half-life ~17.7 yr; β⁻ decay → Sm. Sandbox-
        // compressed, no decay heat (β decay).
        half_life_frames: 4_000_000, decay_product: Element::Sm, decay_heat: 0,
        implemented: true, notes: "only radioactive lanthanide; longest isotope ~17.7 yr (β⁻ → Sm)" },
    AtomProfile { number: 62, symbol: "Sm", name: "Samarium",
        period: 8, group: 8, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 150.36, melting_point: 1072, boiling_point: 1794, density_stp: 7.520,
        electronegativity: 1.17, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; SmCo high-temperature magnets" },
    AtomProfile { number: 63, symbol: "Eu", name: "Europium",
        period: 8, group: 9, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 151.96, melting_point: 822, boiling_point: 1529, density_stp: 5.243,
        electronegativity: 1.20, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "softest lanthanide; red phosphor in CRT displays and Euro banknotes" },
    AtomProfile { number: 64, symbol: "Gd", name: "Gadolinium",
        period: 8, group: 10, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 157.25, melting_point: 1313, boiling_point: 3273, density_stp: 7.895,
        electronegativity: 1.20, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "ferromagnetic near room temp; MRI contrast agents; high neutron capture" },
    AtomProfile { number: 65, symbol: "Tb", name: "Terbium",
        period: 8, group: 11, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 158.93, melting_point: 1356, boiling_point: 3230, density_stp: 8.229,
        electronegativity: 1.20, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; green phosphor and magnetostrictive alloys" },
    AtomProfile { number: 66, symbol: "Dy", name: "Dysprosium",
        period: 8, group: 12, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 162.50, melting_point: 1412, boiling_point: 2567, density_stp: 8.55,
        electronegativity: 1.22, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; high-temperature stability for neodymium magnets" },
    AtomProfile { number: 67, symbol: "Ho", name: "Holmium",
        period: 8, group: 13, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 164.93, melting_point: 1474, boiling_point: 2700, density_stp: 8.795,
        electronegativity: 1.23, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "highest magnetic permeability of any element; medical lasers" },
    AtomProfile { number: 68, symbol: "Er", name: "Erbium",
        period: 8, group: 14, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 167.26, melting_point: 1529, boiling_point: 2868, density_stp: 9.066,
        electronegativity: 1.24, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "fiber-optic amplifiers (EDFA); pink-tinted ceramics" },
    AtomProfile { number: 69, symbol: "Tm", name: "Thulium",
        period: 8, group: 15, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 168.93, melting_point: 1545, boiling_point: 1950, density_stp: 9.321,
        electronegativity: 1.25, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "least abundant naturally-occurring lanthanide; portable X-ray sources" },
    AtomProfile { number: 70, symbol: "Yb", name: "Ytterbium",
        period: 8, group: 16, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 173.05, melting_point: 819, boiling_point: 1196, density_stp: 6.965,
        electronegativity: 1.10, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "rare earth; atomic clocks (Yb-171); stainless-steel additive" },
    AtomProfile { number: 71, symbol: "Lu", name: "Lutetium",
        period: 8, group: 17, stp_state: SSolid, category: Lanthanide,
        atomic_mass: 174.97, melting_point: 1663, boiling_point: 3402, density_stp: 9.841,
        electronegativity: 1.27, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "densest lanthanide; petroleum cracking catalyst" },

    // ---- Period 6 resumes (72-86) ----
    AtomProfile { number: 72, symbol: "Hf", name: "Hafnium",
        period: 6, group: 4, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 178.49, melting_point: 2233, boiling_point: 4603, density_stp: 13.31,
        electronegativity: 1.3, valence_electrons: 4,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "very high mp; high neutron-capture, opposite Zr — nuclear control rods" },
    AtomProfile { number: 73, symbol: "Ta", name: "Tantalum",
        period: 6, group: 5, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 180.95, melting_point: 3017, boiling_point: 5458, density_stp: 16.65,
        electronegativity: 1.5, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "extremely corrosion-resistant; capacitors, surgical implants" },
    AtomProfile { number: 74, symbol: "W", name: "Tungsten",
        period: 6, group: 6, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 183.84, melting_point: 3422, boiling_point: 5555, density_stp: 19.25,
        electronegativity: 2.36, valence_electrons: 6,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "highest melting point of any pure element; lamp filaments, armor-piercing rounds" },
    AtomProfile { number: 75, symbol: "Re", name: "Rhenium",
        period: 6, group: 7, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 186.21, melting_point: 3186, boiling_point: 5596, density_stp: 21.02,
        electronegativity: 1.9, valence_electrons: 7,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "third-highest mp; jet-engine superalloys" },
    AtomProfile { number: 76, symbol: "Os", name: "Osmium",
        period: 6, group: 8, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 190.23, melting_point: 3033, boiling_point: 5012, density_stp: 22.59,
        electronegativity: 2.2, valence_electrons: 8,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "densest natural element; fountain pen tips, electrical contacts" },
    AtomProfile { number: 77, symbol: "Ir", name: "Iridium",
        period: 6, group: 9, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 192.22, melting_point: 2466, boiling_point: 4428, density_stp: 22.56,
        electronegativity: 2.2, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "most corrosion-resistant metal; spark-plug tips; Cretaceous boundary marker" },
    AtomProfile { number: 78, symbol: "Pt", name: "Platinum",
        period: 6, group: 10, stp_state: SSolid, category: TransitionMetal,
        atomic_mass: 195.08, melting_point: 1768, boiling_point: 3825, density_stp: 21.45,
        electronegativity: 2.28, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "noble metal catalyst; jewelry; resists oxidation completely" },
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
    AtomProfile { number: 81, symbol: "Tl", name: "Thallium",
        period: 6, group: 13, stp_state: SSolid, category: PostTransition,
        atomic_mass: 204.38, melting_point: 304, boiling_point: 1473, density_stp: 11.85,
        electronegativity: 1.62, valence_electrons: 3,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "infamously toxic; rat poison; low-mp post-transition" },
    AtomProfile { number: 82, symbol: "Pb", name: "Lead",
        period: 6, group: 14, stp_state: SSolid, category: PostTransition,
        atomic_mass: 207.2, melting_point: 327, boiling_point: 1749, density_stp: 11.34,
        electronegativity: 1.87, valence_electrons: 2,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true,
        notes: "heavy, soft, unreactive; future radiation shielding material" },
    AtomProfile { number: 83, symbol: "Bi", name: "Bismuth",
        period: 6, group: 15, stp_state: SSolid, category: PostTransition,
        atomic_mass: 208.98, melting_point: 271, boiling_point: 1564, density_stp: 9.78,
        electronegativity: 2.02, valence_electrons: 5,
        half_life_frames: 0, decay_product: Element::Empty, decay_heat: 0,
        implemented: true, notes: "low-toxicity heavy metal; pink iridescent crystals; stomach medicine, fishing weights" },
    AtomProfile { number: 84, symbol: "Po", name: "Polonium",
        period: 6, group: 16, stp_state: SSolid, category: Metalloid,
        atomic_mass: 209.0, melting_point: 254, boiling_point: 962, density_stp: 9.20,
        electronegativity: 2.0, valence_electrons: 6,
        // Po-210 real half-life ~138 days; α → Pb-206. Game-compressed
        // to ~5 min so a small Po pile produces visible transmutation.
        // High decay heat — Po-210 produces ~140 W/g in real life.
        half_life_frames: 800_000, decay_product: Element::Pb, decay_heat: 70,
        implemented: true, notes: "intensely radioactive α-emitter; α → Pb; discovered by the Curies" },
    AtomProfile { number: 85, symbol: "At", name: "Astatine",
        period: 6, group: 17, stp_state: SSolid, category: Halogen,
        atomic_mass: 210.0, melting_point: 302, boiling_point: 337, density_stp: 6.35,
        electronegativity: 2.2, valence_electrons: 7,
        // At-211 real half-life ~7 hr; α → Bi-207. Game-compressed to
        // ~1 min — At is the rarest natural element exactly because
        // every isotope decays fast.
        half_life_frames: 300_000, decay_product: Element::Bi, decay_heat: 40,
        implemented: true, notes: "rarest natural element; all isotopes α/β; α → Bi" },
    AtomProfile { number: 86, symbol: "Rn", name: "Radon",
        period: 6, group: 18, stp_state: SGas, category: NobleGas,
        atomic_mass: 222.0, melting_point: -71, boiling_point: -62, density_stp: 0.00973,
        electronegativity: 0.0, valence_electrons: 8,
        // Rn-222 real half-life ~3.8 days; α → Po-218. Game-
        // compressed to U-tier (~7 min). Radon as a gas means decay
        // products spawn into open air, which is a cool emergent
        // tell — a Rn cloud slowly seeds Po dust.
        half_life_frames: 1_500_000, decay_product: Element::Po, decay_heat: 30,
        implemented: true, notes: "radioactive noble gas; α → Po; basement-air carcinogen" },

    // ---- Period 7 (main row stops at Ra; actinides go to strip) ----
    AtomProfile { number: 87, symbol: "Fr", name: "Francium",
        period: 7, group: 1, stp_state: SSolid, category: AlkaliMetal,
        atomic_mass: 223.0, melting_point: 27, boiling_point: 677, density_stp: 1.87,
        electronegativity: 0.7, valence_electrons: 1,
        // Fr-223 real half-life ~22 min; β⁻ → Ra-227. Game-compressed
        // to ~30 sec — fastest natural decay we model. β decay so no
        // significant heat. Pile vanishes quickly.
        half_life_frames: 100_000, decay_product: Element::Ra, decay_heat: 0,
        implemented: true, notes: "most reactive alkali metal; β⁻ → Ra; shortest natural half-life" },
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
    AtomProfile { number: 89, symbol: "Ac", name: "Actinium",
        period: 9, group: 3, stp_state: SSolid, category: Actinide,
        atomic_mass: 227.0, melting_point: 1050, boiling_point: 3198, density_stp: 10.07,
        electronegativity: 1.1, valence_electrons: 3,
        // Ac-227 real half-life ~21.7 yr; β⁻ → Th-227 (real chain
        // continues through Ra/Rn/etc but for sandbox we collapse to
        // the daughter atom). Game-compressed to ~20 min.
        half_life_frames: 2_000_000, decay_product: Element::Th, decay_heat: 30,
        implemented: true, notes: "namesake of the actinide series; β⁻ → Th; self-glow from air ionization" },
    AtomProfile { number: 90, symbol: "Th", name: "Thorium",
        period: 9, group: 4, stp_state: SSolid, category: Actinide,
        atomic_mass: 232.04, melting_point: 1750, boiling_point: 4788, density_stp: 11.7,
        electronegativity: 1.3, valence_electrons: 4,
        // Th-232 real half-life ~14 billion years (longer than U).
        // Game-compressed to ~30 min — slowest visible decay we
        // model. α → Ra-228. Modest heat (slow alpha).
        half_life_frames: 5_000_000, decay_product: Element::Ra, decay_heat: 20,
        implemented: true, notes: "weakly radioactive; α → Ra; thorium fuel cycle" },
    AtomProfile { number: 91, symbol: "Pa", name: "Protactinium",
        period: 9, group: 5, stp_state: SSolid, category: Actinide,
        atomic_mass: 231.04, melting_point: 1568, boiling_point: 4027, density_stp: 15.37,
        electronegativity: 1.5, valence_electrons: 5,
        // Pa-231 real half-life ~32k yr; α → Ac-227. Game-compressed
        // to ~25 min between Ac and Th.
        half_life_frames: 3_000_000, decay_product: Element::Ac, decay_heat: 30,
        implemented: true, notes: "rare radioactive actinide; intermediate in the U-235 to Ac-227 decay chain" },
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
fn is_waterish(el: Element) -> bool {
    matches!(el, Element::Water | Element::Ice | Element::Steam)
}

fn is_water_reactive_metal(el: Element) -> bool {
    matches!(el, Element::Li | Element::Na | Element::K
        | Element::Rb | Element::Cs | Element::Fr)
}

// Explicit list of metal+halogen pairs that detonate violently rather
// than just forming a hot ionic salt. Real K/Rb/Cs/Fr halogenation
// (with F or Cl₂) is effectively explosive — adiabatic flame temps
// >3000°C, the bulk of the reactant flash-vaporizes; classic chem-
// class demos drop K into Cl₂ atmosphere for the bright flash + boom.
// Li/Na fluorination is exothermic but doesn't produce a macroscopic
// blast in the same tier; Br and I bonds are weaker so K-Cs + Br/I
// are vigorous-but-not-detonating. Keeping LiF/NaF/NaCl out of this
// set lets the user actually inspect the formed salt without it
// being shockwaved out of existence.
fn is_violent_halide_pair(donor: Element, acceptor: Element) -> bool {
    matches!((donor, acceptor),
        (Element::K,  Element::F) | (Element::K,  Element::Cl)
        | (Element::Rb, Element::F) | (Element::Rb, Element::Cl)
        | (Element::Cs, Element::F) | (Element::Cs, Element::Cl)
        | (Element::Fr, Element::F) | (Element::Fr, Element::Cl))
}

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
        // Iron rusts in water too — real chemistry, the dissolved O₂
        // in liquid water is the actual oxidizer (more so in cold
        // water, since dissolved O concentration is higher). Without
        // this clause Fe in water does nothing because the derived
        // registry can't synthesize from a non-atom acceptor.
        (Element::Fe, Element::Water)
        | (Element::Fe, Element::Ice)
        | (Element::Fe, Element::Steam) => Some(Element::Rust),
        (Element::C, Element::O) => Some(Element::CO2),
        // Si + O is SiO₂ — the same compound as Element::Sand. Without
        // this mapping the derived registry creates a parallel "SiO₂"
        // entry that's distinct from Sand even though they're the same
        // chemistry, so the same atom pair can produce two different-
        // looking products depending on which code path triggered. Map
        // explicitly to Sand to keep compound identity unified.
        (Element::Si, Element::O) => Some(Element::Sand),
        (m, Element::Water) | (m, Element::Ice) | (m, Element::Steam)
            if atom_profile_for(m).map_or(false, |a|
                a.implemented
                && a.electronegativity > 0.0
                && a.electronegativity < 1.4
                && a.valence_electrons <= 2)
            => Some(Element::H),
        // Sc + water — slow but real (Sc(OH)₃ + H₂). Sc's valence 3
        // misses the EN-and-valence-≤-2 clause above, but the chemistry
        // is similar to the Mg/Ca tier — needs hot water to fizz.
        (Element::Sc, Element::Water)
        | (Element::Sc, Element::Ice)
        | (Element::Sc, Element::Steam) => Some(Element::H),
        // Mn + water — inert at room temp, releases H₂ in hot water /
        // steam (Mn(OH)₂ + H₂). Same Mg/Ca/Sc tier, valence 7 misses
        // the EN-and-valence clause above.
        (Element::Mn, Element::Water)
        | (Element::Mn, Element::Ice)
        | (Element::Mn, Element::Steam) => Some(Element::H),
        // Y + water — slow but real (Y(OH)₃ + H₂). Y's valence 3 also
        // misses the EN-and-valence clause; explicit override puts it
        // on the slow-hydrolysis tier alongside Sc/Mn.
        (Element::Y, Element::Water)
        | (Element::Y, Element::Ice)
        | (Element::Y, Element::Steam) => Some(Element::H),
        // Zr + steam — real high-temp reaction (>300°C):
        // Zr + 2H₂O → ZrO₂ + 2H₂. Famously the runaway path in
        // overheated nuclear cladding (Fukushima). Cold water no-op,
        // hot water/steam fires. Activation floor below pins this to
        // hot temperatures only.
        (Element::Zr, Element::Water)
        | (Element::Zr, Element::Ice)
        | (Element::Zr, Element::Steam) => Some(Element::H),
        // Nb + steam — same pattern, ~350°C onset for Nb + H₂O →
        // Nb₂O₅ + H₂. Cold-water inert, hot-water/steam reactive.
        (Element::Nb, Element::Water)
        | (Element::Nb, Element::Ice)
        | (Element::Nb, Element::Steam) => Some(Element::H),
        // Mo + steam — same Zr/Nb tier. Real Mo + steam at high temp
        // → MoO₂ + H₂. Cold-water passivated by MoO₃ coating; only
        // hot steam breaks through.
        (Element::Mo, Element::Water)
        | (Element::Mo, Element::Ice)
        | (Element::Mo, Element::Steam) => Some(Element::H),
        // Tc + steam — same passivating-refractory tier. Cold water
        // inert (Tc₂O₇ coating); hot steam reacts.
        (Element::Tc, Element::Water)
        | (Element::Tc, Element::Ice)
        | (Element::Tc, Element::Steam) => Some(Element::H),
        // Ru + steam — platinum-group metal, cold water inert
        // (corrosion-resistant). Hot steam (>500°C) oxidizes slowly,
        // higher floor than Zr/Nb/Mo/Tc since real Ru is more
        // resistant than Mo.
        (Element::Ru, Element::Water)
        | (Element::Ru, Element::Ice)
        | (Element::Ru, Element::Steam) => Some(Element::H),
        _ => None,
    };
    if let Some(el) = bespoke {
        return Some(InferredProduct::Bespoke(el));
    }
    // Water-first hydrolysis guard: a water-reactive alkali touching
    // water/ice/steam should ALWAYS resolve to the H+Steam bespoke
    // path, never fall through to the derived registry to produce an
    // oxide. The bespoke match above already handles this for atoms
    // with EN < 1.4 and valence ≤ 2 — this is defense in depth for
    // any future low-EN metal that doesn't match those exact bounds.
    if is_waterish(acceptor) && is_water_reactive_metal(donor) {
        return Some(InferredProduct::Bespoke(Element::H));
    }
    // Water-passivation block: Be does NOT react with water at any
    // temperature in real life — the BeO surface layer protects it.
    // Without this gate, the derived registry happily produces BeO
    // when Be touches water, since Water exposes O's chemistry face.
    // Block at the product level so the Be cell stays Be.
    if is_waterish(acceptor) && donor == Element::Be {
        return None;
    }
    // As is inert to water — no As + H₂O reaction at any temperature.
    // Without this gate, As cells in water happily form As₂O₅ since
    // water's chemistry face exposes O. Same pattern as the Be gate.
    if is_waterish(acceptor) && donor == Element::As {
        return None;
    }
    // Se is inert to water (and to non-oxidizing acids in general).
    // Real Se requires concentrated HNO₃ or hot conditions for any
    // water-side reaction. Block bulk Se + H₂O at the product level
    // so the derived registry doesn't auto-form SeO₃ on contact.
    if is_waterish(acceptor) && donor == Element::Se {
        return None;
    }
    // Xe — real xenon is noble for almost everything; only the
    // xenon fluorides (XeF₂/XeF₄/XeF₆) are stable, and only form
    // at high temp or under electric discharge. Block Xe + any
    // non-F partner so the bucket math doesn't auto-fire reactions
    // with Cl/Br/O/metals (Xe's profile EN 2.6 dodges the standard
    // noble-gas EN==0 early-out, which is why these were happening
    // unintentionally). Xe+F still goes through normal chemistry
    // path but gets a 400°C activation floor (see activation block
    // in try_emergent_reaction).
    if (donor == Element::Xe && acceptor != Element::F)
        || (acceptor == Element::Xe && donor != Element::F)
    {
        return None;
    }
    // Nitride block: only Li reacts with N₂ at room temperature to form
    // Li₃N. Real Na/K/Rb/Cs/Fr do NOT form nitrides under normal
    // conditions — Li⁺ has uniquely high charge density (small ion)
    // that stabilizes the nitride lattice; the heavier alkalis don't.
    // Without this gate the emergent engine produces Na₃N, K₃N, etc.
    if acceptor == Element::N && matches!(donor,
        Element::Na | Element::K | Element::Rb | Element::Cs | Element::Fr)
    {
        return None;
    }
    // Interhalogen block. ClF/ClF₃/BrF₃ etc. exist but require extreme
    // lab conditions (fluorine streams at high temp/pressure). The
    // sandbox produced ClF₇ from displaced Cl gas immediately re-
    // reacting with adjacent F, eating the freed halogen instead of
    // letting it escape as gas. Block at the product level so freed
    // halogens stay free.
    let is_halogen = |e: Element| matches!(e,
        Element::F | Element::Cl | Element::Br | Element::I);
    if is_halogen(donor) && is_halogen(acceptor) {
        return None;
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
    // Optional byproduct gas to spawn into a third (empty) neighbor
    // when the reaction fires. Used by slow-hydrolysis (M + water →
    // MO residue + H₂ released as bubbles, water cell preserved) so
    // the reaction can cycle indefinitely on a small puddle without
    // consuming the water away.
    byproduct: Option<Element>,
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
    // 0.34 threshold (was 0.35, originally 0.4). Lowered to 0.34 to
    // cover f32 precision: H + Se delta_e is exactly 0.35 in math
    // (2.55 - 2.20) but evaluates to ~0.349999 in f32, falling just
    // below a 0.35 strict-less-than cutoff. Other sub-0.4 pairs in
    // our table are same-period nonmetal/metalloid mismatches that
    // don't form interesting compounds. The 0.34 floor still
    // excludes those (Si+P = 0.29, C+I = 0.11, etc.).
    // Te+H bypasses the delta_e gate. Te EN 2.10 and H EN 2.20 give
    // delta_e 0.10, far below the 0.34 floor. Real H₂Te does form
    // (chalcogen hydride series H₂O / H₂S / H₂Se / H₂Te) but needs
    // significant heat — explicit activation override below sets
    // the threshold to 500°C.
    let te_h_special = (a_el == Element::Te && b_el == Element::H)
        || (a_el == Element::H && b_el == Element::Te);
    if delta_e < 0.34 && !te_h_special { return None; }

    // Donor is lower-electronegativity (gives electrons); acceptor pulls.
    let (donor_el, donor_v, acceptor_el, acceptor_v) = if ea < eb {
        (a_el, va, b_el, vb)
    } else {
        (b_el, vb, a_el, va)
    };
    // Valence compatibility — donor should want to lose, acceptor to gain.
    // Donor ≤4 permits carbon (v=4) to participate in covalent bonds; the
    // acceptor still needs a half-full-or-more outer shell to attract.
    // Hydrogen is special: real metal hydrides (LiH, NaH, CaH₂, …) have
    // H acting as the H⁻ anion — effectively "needs 1 to fill" like a
    // halogen. Treat H-as-acceptor with effective valence 7 so the
    // metal-hydride pair passes this filter.
    let effective_acceptor_v = if acceptor_el == Element::H { 7 } else { acceptor_v };
    // High-valence donors don't actually give up all their valence
    // electrons — they oxidize to a much lower state (S→SO₂ is +4,
    // Cr→Cr₂O₃ is +3, etc.). The raw valence_electrons field counts
    // their full outer shell which fails the donor_v ≤ 4 gate, so
    // without an override S+O / Cr+S / Mn+anything etc. all silently
    // return None even though they're real reactions.
    //
    // Pnictogens (N/P/As, valence 5) → 2-3 electrons donated.
    // Chalcogens (O/S/Se, valence 6) → 2 electrons donated.
    // Halogens (Cl/Br/I, valence 7) → 1 electron donated (interhalogens).
    // Transition metals with valence > 4 (V, Cr, Mn, Fe, etc.) → 2.
    let effective_donor_v = match donor_el {
        Element::N | Element::P | Element::As => 2,
        Element::O | Element::S | Element::Se => 2,
        Element::Cl | Element::Br | Element::I => 1,
        _ if donor_v > 4 => 2,
        _ => donor_v,
    };
    if effective_donor_v > 4 || effective_acceptor_v < 5 { return None; }

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
    // [0.35, 0.9) bucket dropped from 800 → 200 so weak-polar pairs
    // can actually fire before their constituents start burning. H+S
    // (Δe 0.38) needs to react before S ignites at 232°C, otherwise
    // S just burns to CO₂ and never sees the H. Same for H+Br /
    // H+I — they sit in this bucket and need to fire before H itself
    // hits 500°C ignition.
    let mut activation: i16 = if delta_e >= 2.5 { -200 }
        else if delta_e >= 1.6 { 100 }
        else if delta_e >= 0.9 { 400 }
        else { 200 };
    activation -= acceptor_bonus;
    activation -= donor_metal_bonus;
    if has_electrolyte { activation -= 200; }

    // Product lookup via donor/acceptor stoichiometry.
    let inferred = infer_product(donor_el, acceptor_el, catalysts)?;

    // Carbon combustion needs a real ignition kick. Δe 0.89 + O's
    // acceptor bonus would land C+O activation at -82°C, so a C pile
    // would smoulder in ambient air at room temperature. Real charcoal
    // auto-ignition is ~250-400°C — without a torch or hot surface,
    // C should not burn. Force a 400°C floor on bespoke CO₂ formation.
    if matches!(inferred, InferredProduct::Bespoke(Element::CO2)) {
        activation = activation.max(400);
    }
    // Nitrogen activation floor — N₂'s triple bond is the strongest in
    // common chemistry. Real N₂ reactions need very high activation:
    // Haber NH₃ ~400°C, Mg₃N₂ ~700°C, NOx >1000°C. Without this floor
    // the engine would form NH₃ at 38°C and Mg₃N₂ at room temp from
    // the generic activation math. Li is the lone exception — Li⁺'s
    // small ionic radius lets it crack N₂ at room temperature, which
    // is the textbook anomaly we want to preserve.
    if (donor_el == Element::N || acceptor_el == Element::N)
        && donor_el != Element::Li
    {
        activation = activation.max(500);
    }
    // Al + heavier halogens (Br, I) is famously vigorous at room
    // temp in real chemistry — Al strip in liquid Br₂ ignites in
    // seconds. The generic bucket math lands Al+Br at 62°C activation
    // (just above ambient) which makes the reaction silently fail to
    // start without a heat source. Force ambient-fireable.
    if donor_el == Element::Al
        && matches!(acceptor_el, Element::Br | Element::I)
    {
        activation = activation.min(0);
    }
    // Si + Br / Si + I — proceeds in real life but slowly at room
    // temp; vigorous with mild heat. Bucket math lands Si+Br at 62°C
    // (just above ambient, never fires). Force room-temp fireable so
    // submerging Si in liquid Br₂ produces a visible reaction even
    // without an external heat source.
    if donor_el == Element::Si
        && matches!(acceptor_el, Element::Br | Element::I)
    {
        activation = activation.min(0);
    }
    // P + halogens — white phosphorus reacts vigorously with all
    // halogens at room temperature (PF₅, PCl₃/PCl₅, PBr₃/PBr₅, PI₃).
    // Bucket math lands P+Cl at 202°C and P+Br at 62°C, neither of
    // which fires at ambient. Force room-temp fireable so a P-on-Cl
    // pile reacts on contact like real life. P+F already passes
    // (activation -344) so isn't strictly necessary in this list,
    // but included for symmetry.
    if donor_el == Element::P
        && matches!(acceptor_el,
            Element::F | Element::Cl | Element::Br | Element::I)
    {
        activation = activation.min(0);
    }
    // Transition + post-transition metal + heavy-halogen (Br / I)
    // general rule. The bucket math systematically under-prioritizes
    // these pairs (Δe lands them in the [0.9, 1.6) bucket at 400°C
    // activation, then small acceptor and donor-metal bonuses don't
    // drop it below ambient). In real chemistry M + Br₂/I₂ → MBr_x /
    // MI_x fires on contact at room temp for every transition metal
    // and post-transition metal we audit (Sc, Ti, V, Cr, Mn, Ga all
    // needed/needs this; Al + Br₂ is a textbook ignition demo).
    // Forcing activation ≤ 0 lets the salt form on contact and the
    // halide can then proceed through its normal decomp/melt/boil
    // cycle. F and Cl are already covered by their stronger acceptor
    // bonuses, so this rule narrowly targets the gap.
    let donor_is_metal_block = atom_profile_for(donor_el)
        .map_or(false, |a| matches!(a.category,
            AtomCategory::TransitionMetal | AtomCategory::PostTransition));
    if donor_is_metal_block && matches!(acceptor_el, Element::Br | Element::I) {
        activation = activation.min(0);
    }
    // Nonmetal & metalloid donors + heavy halogens (Cl/Br/I). Bucket
    // math systematically under-activates these because the strong
    // halogen acceptor_bonus (~200 for Cl, ~280 for F) dominates a
    // moderate Δe bucket (200-400°C), netting activation values in
    // the single digits — pairs silently fire on contact at room
    // temperature even though real chemistry requires sustained heat
    // (industrial chlorination of C, Si, P, S etc. runs 300-500°C).
    // 400°C floor matches real ignition tier. F stays uncapped — F
    // is uniquely aggressive (real C + F₂, Si + F₂ etc. ARE contact-
    // reactive at room temp due to F's outsized electron affinity).
    if matches!(donor_el,
        Element::C | Element::Si | Element::P | Element::S | Element::B)
        && matches!(acceptor_el, Element::Cl | Element::Br | Element::I)
    {
        activation = activation.max(400);
    }
    // Ge halogenation activation override (CAP, not floor). Bucket
    // math lands Ge+F at -344, Ge+Cl at 202, Ge+Br at 262, Ge+I
    // at 152 — F auto-fires at STP (don't want), and Cl/Br need
    // furnace temps the user can't realistically reach in a
    // sandbox. Real Ge + Cl₂ proceeds at ~250°C in industrial
    // chlorination, but Br bp is 59°C and I sublimes at 184°C
    // so any activation above their phase points means the
    // halogen disperses before reacting. Cap activation at
    // 150 (F/Cl) and 50 (Br/I) so heated Ge in halogen vapor
    // actually fires.
    if donor_el == Element::Ge
        && matches!(acceptor_el, Element::F | Element::Cl)
    {
        activation = 150;
    }
    if donor_el == Element::Ge
        && matches!(acceptor_el, Element::Br | Element::I)
    {
        activation = 50;
    }
    // Y + I — real YI₃ formation needs ~150°C heat. Bucket math gives
    // 152°C native, but the broad transition-metal + Br/I override
    // upstream forces ≤0 (textbook Al+Br ignition demo) and Y rides
    // along incorrectly. Restore the bucket-math threshold here so a
    // Y+I pile in cool conditions doesn't auto-fire — needs warming.
    // Y + Br is left at the broad ≤0 (real Y+Br₂ does fire on contact).
    if donor_el == Element::Y && acceptor_el == Element::I {
        activation = 150;
    }
    // Zr + halogens — real Zr halides need heat to form (Zr+Cl₂ ~250°C
    // industrially). Override the broad transition-metal+Br/I forced-
    // low and the bucket math's near-zero for Cl. Br/I caps below
    // their bp/sublimation point so warmed halogen vapor reacts before
    // dispersing.
    if donor_el == Element::Zr && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Zr && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Zr && acceptor_el == Element::I {
        activation = 150;
    }
    // Nb halogenation — same Zr-tier pattern: real NbCl₅ industrial
    // synthesis runs at ~250°C, NbBr₅/NbI₅ also need heat. Override
    // the broad transition-metal forced-low so user has to warm the
    // pile to see the reaction.
    if donor_el == Element::Nb && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Nb && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Nb && acceptor_el == Element::I {
        activation = 150;
    }
    // Mo halogenation — Mo + Br/I needs heat, real MoBr₄/MoI₄ form at
    // 250-350°C. Overrides the broad transition-metal+Br/I forced-low.
    // Mo+Cl already gated at 202°C by bucket math. Mo+F stays
    // contact-reactive (real MoF₆ is gas at STP, real Mo+F₂ on
    // contact).
    if donor_el == Element::Mo && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Mo && acceptor_el == Element::I {
        activation = 150;
    }
    // Tc halogenation — real Tc halides need heat (TcCl₄ ~300°C, etc).
    // Override the broad transition-metal forced-low.
    if donor_el == Element::Tc && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Tc && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Tc && acceptor_el == Element::I {
        activation = 150;
    }
    // Ru — platinum-group metal, corrosion-resistant. Real Ru is
    // air-stable at room temp; oxidation only fires at >500°C.
    // Halogenation needs heat (real Ru+Cl₂ at ~700°C, Ru+F₂ with
    // strong fluorinating agents). The activation gate uses both
    // cells, but global ambient_offset propagates to the virtual-O
    // n_temp via line 7823 (20 + ambient_offset), so when the user
    // raises world ambient ≥500°C both Ru cells AND atmospheric O
    // cross the threshold together.
    if donor_el == Element::Ru && acceptor_el == Element::O {
        activation = 500;
    }
    if donor_el == Element::Ru && acceptor_el == Element::F {
        activation = 100;
    }
    if donor_el == Element::Ru && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Ru && acceptor_el == Element::Br {
        activation = 100;
    }
    if donor_el == Element::Ru && acceptor_el == Element::I {
        activation = 200;
    }
    // Rh — platinum-group, even more inert than Ru. Real Rh starts
    // oxidizing in O₂ above ~600°C; halides need significant heat.
    // Halide thresholds slightly higher than Ru to reflect Rh's
    // greater corrosion resistance.
    if donor_el == Element::Rh && acceptor_el == Element::O {
        activation = 600;
    }
    if donor_el == Element::Rh && acceptor_el == Element::F {
        activation = 100;
    }
    if donor_el == Element::Rh && acceptor_el == Element::Cl {
        activation = 300;
    }
    if donor_el == Element::Rh && acceptor_el == Element::Br {
        activation = 100;
    }
    if donor_el == Element::Rh && acceptor_el == Element::I {
        activation = 200;
    }
    // Pd — platinum-group, air-stable until ~350°C (PdO formation
    // temp). Halides need heat; PdCl₂ at ~250°C is sandbox-realistic.
    if donor_el == Element::Pd && acceptor_el == Element::O {
        activation = 350;
    }
    if donor_el == Element::Pd && acceptor_el == Element::F {
        activation = 100;
    }
    if donor_el == Element::Pd && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Pd && acceptor_el == Element::Br {
        activation = 100;
    }
    if donor_el == Element::Pd && acceptor_el == Element::I {
        activation = 200;
    }
    // Cd — Zn-family transition metal. Real Cd halides need
    // moderate heat (CdCl₂ at ~250°C). Same Mo/Tc/Pd-style
    // halide gating.
    if donor_el == Element::Cd && acceptor_el == Element::Cl {
        activation = 250;
    }
    if donor_el == Element::Cd && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Cd && acceptor_el == Element::I {
        activation = 150;
    }
    // In — group 13 post-transition metal. Real In halides need
    // moderate heat (InCl₃ at ~200°C industrial).
    if donor_el == Element::In && acceptor_el == Element::Cl {
        activation = 200;
    }
    if donor_el == Element::In && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::In && acceptor_el == Element::I {
        activation = 150;
    }
    // Sn — group 14 post-transition metal. Real Sn halides need
    // moderate heat (SnCl₄ at ~200°C industrial).
    if donor_el == Element::Sn && acceptor_el == Element::Cl {
        activation = 200;
    }
    if donor_el == Element::Sn && acceptor_el == Element::Br {
        activation = 50;
    }
    if donor_el == Element::Sn && acceptor_el == Element::I {
        activation = 150;
    }
    // Ge + O activation override: real Ge does tarnish slowly at
    // room temp; bucket math gates it at 118°C and prevents any
    // ambient oxidation. Drop activation so chemistry fires at STP;
    // rate is capped below to keep tarnish gradual.
    if donor_el == Element::Ge && acceptor_el == Element::O
        && matches!(inferred, InferredProduct::Derived(_))
    {
        activation = activation.min(0);
    }
    // Sb + O — same as Ge. Real Sb tarnishes slowly in air at
    // room temp; bucket math gives 118°C floor (Sb EN 2.05 ≥ 2.0
    // so no donor_metal_bonus, leaving the threshold high).
    // Force STP-firing; donor_passivates 0.002 rate cap keeps it
    // visibly slow.
    if donor_el == Element::Sb && acceptor_el == Element::O
        && matches!(inferred, InferredProduct::Derived(_))
    {
        activation = activation.min(0);
    }
    // Te + O — same as Sb. Real Te tarnishes slowly at room temp;
    // bucket math gives 118°C floor. Force STP firing with 0.002
    // passivation rate cap.
    if donor_el == Element::Te && acceptor_el == Element::O
        && matches!(inferred, InferredProduct::Derived(_))
    {
        activation = activation.min(0);
    }
    // Te + H → H₂Te (sim formula TeH₆ via valence math). Real
    // synthesis at ~650°C; sandbox 500°C floor lets the user trigger
    // it with a moderately hot Te+H atmosphere.
    if (donor_el == Element::Te && acceptor_el == Element::H)
        || (donor_el == Element::H && acceptor_el == Element::Te)
    {
        activation = 500;
    }
    // Xe + F — real XeF₂/F₄/F₆ form at ~400°C or under electric
    // discharge. Sandbox 400°C floor matches the heat-required
    // synthesis path; STP Xe+F is blocked.
    if (donor_el == Element::Xe && acceptor_el == Element::F)
        || (donor_el == Element::F && acceptor_el == Element::Xe)
    {
        activation = 400;
    }
    // Metal hydride formation needs Haber-tier conditions in real life
    // (high temp + pressurized H₂). Without a high activation floor,
    // fresh H₂ gas produced by M+water reactions is at ~1300°C, which
    // easily clears the bucket math's 200°C activation for K+H,
    // dampening the dramatic alkali-water reaction. 500°C floor keeps
    // metal hydrides accessible in a hot enriched atmosphere but
    // doesn't let them auto-form from drifting H gas.
    if acceptor_el == Element::H && donor_e < 2.0 {
        activation = activation.max(500);
    }
    // Slow-hydrolysis activation floors, tiered by real reactivity with
    // water:
    //   * Mg / Sc / Mn — essentially inert in cold water, fizz weakly
    //     in hot water (~80°C), vigorous with steam. 80°C floor matches
    //     the "warm water needed" textbook threshold.
    //   * Ca — fizzes in cold water but slow. 30°C floor lets cool-room
    //     ambient (above 30) trigger it without wholly blocking the
    //     reaction.
    //   * Sr — more reactive than Ca, less than Na. 10°C floor — only
    //     refrigerated water won't trigger it.
    //   * Ba/Ra — fully cold-water reactive (no floor; bucket math
    //     gives them very low activation already, leave alone).
    // The bespoke (metal, water) → H clause is what gates this; without
    // these tiers, all alkaline-earth metals would fire at full alkali-
    // tier rate at any temperature.
    if matches!(donor_el, Element::Mg | Element::Sc | Element::Mn | Element::Y)
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H))
    {
        activation = activation.max(80);
    }
    // Zr/Nb + steam — passivating refractory metals only react with
    // water at high temp (Zr ~300°C, Nb ~350°C). Cold water no-op;
    // hot steam fires. 300°C floor blocks ambient-temp reaction
    // entirely so the passivating coating story stays intact.
    if matches!(donor_el, Element::Zr | Element::Nb | Element::Mo | Element::Tc)
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H))
    {
        activation = activation.max(300);
    }
    // Ru steam — higher floor than Mo/Tc tier. Real Ru is markedly
    // more corrosion-resistant; needs hotter steam to break the
    // RuO₂ surface layer.
    if donor_el == Element::Ru
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H))
    {
        activation = activation.max(500);
    }
    if donor_el == Element::Ca
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H))
    {
        activation = activation.max(30);
    }
    if donor_el == Element::Sr
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H))
    {
        activation = activation.max(10);
    }
    // Surface-mediated oxidation: for passivating-metal + O pairs,
    // the reaction is catalyzed at the hot solid surface — bulk gas
    // temp barely matters. Use max-cell gate so a hot metal pile in
    // cool spawned O₂ still oxidizes (matches the virtual_o n_temp
    // override for ambient air, and matches real surface chemistry
    // where O₂ molecules pick up activation energy from contact with
    // the hot surface, not from bulk-gas thermal equilibration).
    let surface_oxidation = acceptor_el == Element::O
        && matches!(donor_el,
            Element::Al | Element::Cr | Element::Ti
            | Element::V | Element::Sc | Element::Be
            | Element::Ni | Element::Cu | Element::Zn
            | Element::Ga | Element::Ge | Element::As
            | Element::Se | Element::Sr | Element::Y
            | Element::Zr | Element::Nb | Element::Mo
            | Element::Tc | Element::Ru | Element::Rh);
    let gate_temp = if surface_oxidation {
        a_temp.max(b_temp)
    } else {
        a_temp.min(b_temp)
    };
    if gate_temp < activation { return None; }

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
    // Rust formation is direct cell-vs-cell, not virtual_o-damped, so
    // submerged Fe gets dense, continuous reaction opportunities. Set
    // explicit rates instead of trying to derive them through the
    // catalyst-multiplier cap chain. Equivalences:
    //   Fe + plain water == Fe + ambient O₂   (slow surface tarnish)
    //   Fe + salt water  == Fe + spawned O₂   (visible accelerated)
    if matches!(inferred, InferredProduct::Bespoke(Element::Rust)) {
        const RUST_AMBIENT_RATE: f32 = 0.00005;
        const RUST_EXPLICIT_O_RATE: f32 = 0.0005;
        let water_acceptor = matches!(acceptor_el,
            Element::Water | Element::Ice | Element::Steam);
        let has_salt = catalysts.iter().any(|&c| c == Element::Salt);
        rate = if water_acceptor && has_salt {
            RUST_EXPLICIT_O_RATE
        } else if water_acceptor {
            RUST_AMBIENT_RATE
        } else {
            RUST_EXPLICIT_O_RATE
        };
    }
    // Si passivation — Si + O → Sand should NOT consume a Si pile in
    // seconds. Real Si oxidizes very slowly at room temperature; a
    // thin SiO₂ surface layer (a few nanometers) protects the bulk.
    // With our cell-based engine we can't truly "stop after one
    // layer", but we can cap the rate so it forms slowly enough that
    // the visible behavior reads as gradual surface tarnish rather
    // than instant conversion.
    if matches!(inferred, InferredProduct::Bespoke(Element::Sand)) {
        rate = 0.0005;
    }
    // Slow-hydrolysis rate caps, tiered. Even when the activation gate
    // clears, alkaline-earth + water reactions need to be visibly slow
    // versus the alkali "flash" tier — bubble-stream look, not whole-
    // pile conversion in one frame. Tiers:
    //   * Mg/Sc/Mn — slowest (Mg in boiling water still takes minutes)
    //   * Ca/Sr — visibly fizzy. Same numeric cap; the "more reactive"
    //     feel of Sr vs Ca comes from Sr's lower activation floor
    //     (fires at 10°C ambient where Ca needs 30°C+), not from a
    //     faster per-event rate.
    if matches!(inferred, InferredProduct::Bespoke(Element::H))
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
    {
        if matches!(donor_el, Element::Mg | Element::Sc | Element::Mn) {
            rate = (rate * 0.05).min(0.03);
        } else if donor_el == Element::Y {
            // Y forms a passivating Y(OH)₃ layer that retards further
            // reaction; tighter cap than Mg-tier so warmed water shows
            // visible-but-slow fizz instead of rapid surface conversion.
            rate = (rate * 0.025).min(0.015);
        } else if matches!(donor_el, Element::Zr | Element::Nb | Element::Mo | Element::Tc | Element::Ru) {
            // Zr/Nb/Mo/Tc/Ru + steam — slow visible fizz once
            // activation cleared.
            rate = (rate * 0.05).min(0.03);
        } else if matches!(donor_el, Element::Ca | Element::Sr) {
            rate = (rate * 0.10).min(0.05);
        }
    }
    // Derived compounds (non-bespoke products) reflect slow surface
    // corrosion/tarnish processes, not energetic combustion. Very low rate
    // so a metal in air oxidizes visibly over seconds rather than snapping
    // to its compound in one frame.
    if matches!(inferred, InferredProduct::Derived(_)) {
        rate = (rate * 0.01).min(0.2);
    }
    // Passivating donors (Al, Cr, Ti, V, Sc, Be, Ni, Cu, Zn, Ga, Ge,
    // As, Se) form a tight protective oxide layer that slows further
    // oxidation. Real Ni is the textbook example — used as corrosion-
    // resistant plating exactly because NiO passivates the surface.
    // Without a rate floor *here* (after the generic Derived ×0.01
    // slowdown above), the cap collapses to ~2e-5 — visually nothing
    // forms even in spawned O₂ atmospheres. 0.002 reads as gradual
    // surface tarnish rather than rapid consumption, while staying
    // above the floor needed to actually be visible. List mirrors
    // `coating_oxide` in derive_or_lookup so these oxides land as
    // Gravel coatings rather than flaky Powder.
    let donor_passivates = acceptor_el == Element::O
        && matches!(donor_el,
            Element::Al | Element::Cr | Element::Ti
            | Element::V | Element::Sc | Element::Be
            | Element::Ni | Element::Cu | Element::Zn
            | Element::Ga | Element::Ge | Element::As
            | Element::Se | Element::Sr | Element::Y
            | Element::Zr | Element::Nb | Element::Mo
            | Element::Tc | Element::Ru | Element::Rh
            | Element::Pd | Element::Ag | Element::Cd
            | Element::In | Element::Sn | Element::Sb
            | Element::Te);
    if donor_passivates && matches!(inferred, InferredProduct::Derived(_)) {
        rate = 0.002;
    }
    // Halide passivation for noble metals — Au/Cu form tight halide
    // surface layers in real chemistry: AuBr₃, CuBr/CuI form slowly
    // and coat the parent metal before flaking. Ag is intentionally
    // excluded — silver halides (AgF, AgCl, AgBr, AgI) are the
    // famous photographic "flash salts" that form rapidly on
    // contact, not slow tarnish. Without a cap on Au/Cu, the
    // metal+halogen contact-reactive rule consumes the metal at
    // ~0.002 per pair-frame and reads as rapid uniform conversion.
    // 0.0005 reads as gradual surface tarnish — slower than oxide
    // passivation because halide bond formation is less thermo-
    // dynamically favorable than oxide formation on these metals.
    // Mirror in `coating_halide` in derive_or_lookup so the product
    // lands as Gravel rather than Powder.
    let halide_acceptor = matches!(acceptor_el,
        Element::F | Element::Cl | Element::Br | Element::I);
    let donor_halide_passivates = halide_acceptor
        && matches!(donor_el, Element::Au | Element::Cu);
    if donor_halide_passivates && matches!(inferred, InferredProduct::Derived(_)) {
        rate = 0.0005;
    }
    // Ge halogenation rate floor — once the activation barrier is
    // crossed, the reaction needs to proceed visibly before the
    // volatile halogen drifts away. The generic Derived ×0.01
    // throttle above lands Ge+Br at ~0.0019 per frame, so even
    // hot Ge in Br vapor would barely form anything before the
    // vapor diffused out of contact range. 0.35 is fast enough
    // that a heated Ge cell in adjacent Br/I vapor converts in
    // a handful of frames, matching real-world hot-halogenation
    // kinetics. Narrowed to Ge specifically — broader donors
    // (Si/B/P/S) stay on the slow path until their audit pass
    // revisits the same issue.
    if donor_el == Element::Ge
        && matches!(acceptor_el, Element::F | Element::Cl | Element::Br | Element::I)
        && matches!(inferred, InferredProduct::Derived(_))
    {
        rate = rate.max(0.35);
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
    // Violent halide pairs — explicit list of K/Rb/Cs/Fr + F. Bypass
    // the derived-product slowdown that would otherwise cap these at
    // surface-corrosion timescales. Li/Na fluorination is hot but not
    // detonation-class, so LiF/NaF form via the slow path and stay
    // inspectable instead of getting shockwaved out of existence.
    if is_violent_halide_pair(donor_el, acceptor_el)
        && matches!(inferred, InferredProduct::Derived(_))
    {
        rate = rate.max(0.85);
    }

    // Heat released. Bespoke reactions get hand-tuned values matched to
    // real-world enthalpy of formation; emergent reactions default to a
    // *small* Δe-based release. The derived fallback is intentionally mild —
    // surface-level oxidation, tarnish, slow corrosion — not combustion.
    // Anything that should cascade into a fireball (hydrogen, carbon) needs
    // to be listed as bespoke with a tuned heat release.
    let mut delta_temp: i16 = match inferred {
        // Hydrogen combustion — full thermodynamic release. The
        // reaction is gated at ~118°C activation, so it never
        // fires at ambient; the user has to heat the H + O mix to
        // trigger it. Once it does fire, the temp-scaling cubic
        // ramps from a tiny trickle near activation to the full
        // 1800°C release as the reaction zone heats further,
        // crossing the 1200°C shockwave threshold for the proper
        // detonation.
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
    // Hyper-reactive alkali (Cs/Fr) reactions aren't tarnish — they're
    // full combustion. Real Cs + O₂ releases enough heat to self-ignite
    // neighbors; the derived-oxide 80°C cap would make the whole pile read
    // as quiet evaporation instead of the visibly-burning spray of oxide
    // and fire you see in lab footage. Restricted to Cs/Fr by explicit
    // match — using donor_e < 0.85 also swept in K (0.82) and Rb (0.82),
    // and the 450°C heat dump pushed those past their ignite_above
    // threshold on first contact with O, making them auto-ignite at
    // ambient temperature when real K/Rb just tarnish.
    if matches!(donor_el, Element::Cs | Element::Fr)
        && matches!(inferred, InferredProduct::Derived(_))
    {
        // 450°C: below the 500°C mp floor for derived oxides, so Cs₂O
        // stays solid instead of phase-liquefying into a weird puddle
        // while it waits to cool. The actual drama comes from the burn
        // cascade (ignite_above 50, burn_temp 1400) and the Fire-spawn
        // rule below; the oxide product doesn't need to be combustion-hot
        // by itself.
        delta_temp = 450;
    }
    // Tier metal-water exotherm by donor reactivity to match the real
    // reactivity series. Mg/Ca barely fizz; Li gentle; Na vigorous;
    // K/Rb/Cs/Fr violently detonate (clear the 1200°C chem_blast
    // threshold so the accumulator emits a shockwave). Without this
    // tiering K-in-water looked just like Na-in-water — a moderate
    // fizz instead of the iconic ignite-the-H₂ pop K is famous for.
    if matches!(inferred, InferredProduct::Bespoke(Element::H))
        && matches!(acceptor_el, Element::Water | Element::Ice | Element::Steam)
    {
        delta_temp = match donor_el {
            // Slow tier — gently exothermic. Real Mg/Ca/Mn + water
            // releases ~50 kJ/mol H₂; with our cell granularity and
            // the cumulative heat that piles up across hundreds of
            // reactions, 80°C/event boils the entire pool before
            // saturation can build at the interface. 15°C is a more
            // honest per-event warming that lets the simmer be
            // visible without overheating the pool. The pool will
            // still slowly approach boil with sustained reaction —
            // that's correct chemistry — just not racing past it
            // every reaction.
            Element::Mg | Element::Ca | Element::Sc | Element::Mn => 15,
            Element::Li => 400,
            Element::Na => 700,
            Element::K  => 1300,
            Element::Rb => 1400,
            Element::Cs | Element::Fr => 1500,
            _ => delta_temp,
        };
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
    if is_violent_halide_pair(donor_el, acceptor_el)
        && matches!(inferred, InferredProduct::Derived(_))
    {
        delta_temp = (delta_e * 1000.0) as i16;
    }

    // For reactive-metal + water, acceptor cell becomes Steam (water ripped
    // apart), donor cell becomes H gas. For rust formation, only the
    // metal cell becomes Rust — water/O is the oxidizer, not bulk
    // material being converted into rust (without this, every Fe+Water
    // reaction creates TWO rust cells, doubling rust volume out of
    // nowhere). Explicit O is consumed (Empty), water is preserved.
    // For all other pairs, both cells become the product.
    let metal_in_water = matches!(acceptor_el,
            Element::Water | Element::Ice | Element::Steam)
        && matches!(inferred, InferredProduct::Bespoke(Element::H));
    let rust_reaction = matches!(inferred, InferredProduct::Bespoke(Element::Rust));
    // Slow-tier hydrolysis donors leave a hydroxide residue in real
    // life (Mg(OH)₂, Ca(OH)₂, Sc(OH)₃, Mn(OH)₂) — visible white/brown
    // powder where the metal sat. We approximate with the metal's
    // derived oxide (closest sim analogue, since hydroxides aren't
    // modeled). Alkali metals genuinely dissolve into solution so
    // their cells vanish; the slow tier doesn't.
    let slow_hydrolysis = matches!(donor_el,
        Element::Mg | Element::Ca | Element::Sr | Element::Ba | Element::Ra
        | Element::Sc | Element::Mn | Element::Y
        | Element::Zr | Element::Nb | Element::Mo | Element::Tc
        | Element::Ru);
    let mut byproduct: Option<Element> = None;
    let products = if metal_in_water && slow_hydrolysis {
        // Donor → oxide residue. Acceptor (water) PRESERVED — the
        // reaction does not consume the solvent. This is a sim
        // simplification of real chemistry (Mn + 2H₂O → Mn(OH)₂ +
        // H₂ does consume water), but our cell-granular model
        // makes 1:1 water consumption gameplay-hostile: every
        // reaction event destroys a solvent cell, the residue
        // can never accumulate enough dissolved solute for the
        // pool to saturate. By preserving water, the residue can
        // dissolve up to its low-solubility cap (~64) and the user
        // sees saturation climb. H₂ is still emitted as byproduct;
        // a probabilistic Steam visual fires at the chemistry-pass
        // emission point (chemistry-pass-side, see byproduct loop).
        let residue = match derive_or_lookup(donor_el, Element::O) {
            Some(id) => ProductSpec::derived(id),
            None     => ProductSpec::bespoke(donor_el),
        };
        byproduct = Some(Element::H);
        [residue, ProductSpec::bespoke(acceptor_el)]
    } else if metal_in_water {
        [ProductSpec::bespoke(Element::H), ProductSpec::bespoke(Element::Steam)]
    } else if rust_reaction {
        let acceptor_product = match acceptor_el {
            Element::O => Element::Empty,
            _ => acceptor_el,  // water/ice/steam preserved
        };
        [ProductSpec::bespoke(Element::Rust), ProductSpec::bespoke(acceptor_product)]
    } else {
        let s = ProductSpec::from_inferred(inferred);
        [s, s]
    };

    Some(ReactionOutcome { products, delta_temp, rate, byproduct })
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
    // Optional "wet" color for compounds with hydration-dependent
    // appearance. Real CoCl₂ is the textbook example: anhydrous deep
    // blue, hydrated pink/magenta. The render path interpolates
    // between `color` (dry) and `hydration_color` (wet) based on the
    // cell's moisture. None means no hydration shift — the cell stays
    // the static `color` regardless of moisture (just gets the
    // standard wet-darken tint applied uniformly later).
    hydration_color: Option<(u8, u8, u8)>,
}

// Process-global derived-compound registry. Two storage tiers:
//
// * `DERIVED_COMPOUNDS` — the authoritative store, holds the full
//   compound data (formula String, constituents Vec, etc.). Behind
//   parking_lot::RwLock because writes mutate the Vec on registration.
//   Touched on the cold paths only (UI, chemistry classification).
//
// * `DERIVED_HOT` — a lock-free per-id cache of the fields that hot
//   paths read constantly: physics profile, color, mp/bp, decomp
//   threshold. Each slot is `OnceLock`, so reads are a single atomic
//   load. Writes are one-shot at registration time. Without this,
//   the parallel byte-build, pressure pass, motion pass, and can_enter
//   all hammer the same RwLock cache line millions of times per frame
//   when derived cells are on the grid, tanking the framerate.
//
// Must not be thread_local — rayon worker threads need to see the
// same data the main thread registered.
static DERIVED_COMPOUNDS: parking_lot::RwLock<Vec<DerivedCompound>> =
    parking_lot::RwLock::new(Vec::new());

#[derive(Clone, Copy)]
struct DerivedHot {
    physics: PhysicsProfile,
    color: (u8, u8, u8),
    melting_point: i16,
    boiling_point: i16,
    decomposes_above: Option<i16>,
    hydration_color: Option<(u8, u8, u8)>,
    // True if any constituent atom has half_life_frames > 0. Real
    // chemistry: a Tc-bearing oxide / halide is just as radioactive
    // as elemental Tc — the decay happens at the atomic core,
    // independent of bonding. Used by the rendering halo so a
    // TcO₃ pile glows the same as a Tc pile.
    is_radioactive: bool,
}

static DERIVED_HOT: [std::sync::OnceLock<DerivedHot>; 256] =
    [const { std::sync::OnceLock::new() }; 256];

#[inline]
fn derived_hot(idx: u8) -> Option<&'static DerivedHot> {
    DERIVED_HOT[idx as usize].get()
}

// Builds a formula string from constituent atoms with subscripts. Uses real
// subscript glyphs where possible.
fn format_formula(atoms: &[(Element, u8)]) -> String {
    let mut out = String::new();
    for &(el, count) in atoms {
        let symbol = el.formula().unwrap_or("?");
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

// Cell-level identity / label helpers. These dispatch over both
// bespoke Element variants AND derived compounds (Element::Derived
// with a registry id), so callers don't need to know which case
// they're in.

// Canonical chemical formula for a cell. Returns the static formula
// for atoms and known-formula compounds, the registry-stored formula
// for derived compounds. Atoms always have a formula (the symbol);
// some bespoke macro-materials (Wood, Lava, etc.) may return None.
fn cell_formula(c: Cell) -> Option<String> {
    if c.el == Element::Derived {
        let reg = DERIVED_COMPOUNDS.read();
        return reg.get(c.derived_id as usize).map(|d| d.formula.clone());
    }
    c.el.formula().map(|s| s.to_string())
}

// User-facing label for a cell. Audit-phase format mirrors
// Element::display_label: "Formula (Common Name)" so the player can
// distinguish materials of the same compound (Sand vs Glass vs
// Quartz, all SiO₂). Derived compounds show their formula only (no
// common name yet — that's what the post-audit discovery system
// will assign). Compounds without a clean formula fall back to the
// common name. See display_label for the post-audit refactor TODO.
fn cell_label(c: Cell) -> String {
    if c.el == Element::Derived {
        let reg = DERIVED_COMPOUNDS.read();
        return reg.get(c.derived_id as usize)
            .map(|d| d.formula.clone())
            .unwrap_or_else(|| "Unknown".to_string());
    }
    c.el.display_label()
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
    {
        let mut reg = DERIVED_COMPOUNDS.write();
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
            hydration_color: None,
        });
        let new_idx = (reg.len() - 1) as u8;
        // Mirror the hot-read fields into the lock-free cache so
        // every per-cell read in pressure / motion / render bypasses
        // the RwLock entirely.
        let is_radioactive = atom_profile_for(a).map_or(false, |p| p.half_life_frames > 0)
            || atom_profile_for(b).map_or(false, |p| p.half_life_frames > 0);
        let _ = DERIVED_HOT[new_idx as usize].set(DerivedHot {
            physics, color, melting_point, boiling_point,
            decomposes_above: None,
            hydration_color: None,
            is_radioactive,
        });
        Some(new_idx)
    }
}

// Per-formula color overrides for derived compounds whose averaged
// constituent colors don't match real-world appearance. Used by
// derive_or_lookup. Add entries during element audits where a
// compound's averaged color reads wrong (Sc₂O₃ is real-world pinkish-
// white, but the average of Sc silver + O cyan comes out plain gray).
fn bespoke_color_for_formula(formula: &str) -> Option<(u8, u8, u8)> {
    match formula {
        "Sc₂O₃" => Some((235, 215, 220)),  // pinkish white
        "V₂O₅"  => Some((250, 175,  60)),  // yellow-orange (vanadium pentoxide)
        // Silver halides — the photographic flash salts. Real-life
        // colors are pale and very specific (these are why we have
        // film). Generic constituent-color averaging gives wildly
        // wrong results (Ag silver + Br red-brown = muddy red, Ag +
        // I purple = blue-violet) so override.
        "AgBr"  => Some((235, 220, 170)),  // pale yellow (real AgBr)
        "AgI"   => Some((240, 220, 110)),  // yellow (real AgI)
        "AgCl"  => Some((240, 240, 235)),  // white (real AgCl)
        // Silver sulfide — the iconic black silver tarnish. Real
        // Ag₂S is deep black (sometimes brown-black with patina).
        // Constituent average of Ag silver + S yellow gives muddy
        // pale yellow — completely wrong for the recognizable
        // tarnish color.
        "Ag₂S"  => Some(( 35,  30,  35)),  // near-black (silver tarnish)
        // Silver oxide — real Ag₂O is dark brown. Override skips
        // the silver-and-O-averaged pale grey.
        "Ag₂O"  => Some(( 80,  60,  55)),  // dark brown
        // Cadmium sulfide — the famous "cadmium yellow" pigment used
        // in art for centuries. Bright yellow-orange. Real CdS is
        // a vivid yellow that constituent averaging (Cd silver + S
        // pale yellow) approximates poorly.
        "CdS"   => Some((250, 210,  60)),  // cadmium yellow
        // Cadmium oxide — real CdO is brown to red-brown.
        "CdO"   => Some((150,  90,  60)),  // brown
        // SnI₄ — real tin tetraiodide is a vivid orange-red solid.
        // Constituent average of Sn silver + I purple gives a muddy
        // grey-purple — wrong for the iconic orange-red crystal.
        "SnI₄"  => Some((220, 110,  55)),  // orange-red
        // SbI₅ rendered as SbI₃ (real). Real SbI₃ is dark red-brown.
        "SbI₅"  => Some((150,  60,  55)),  // dark red-brown
        // Anhydrous cobalt(II) chloride — the iconic deep cobalt
        // blue (humidity indicator: blue dry → pink hydrated).
        // The hydrated/wet color comes from the moisture-aware
        // override below; this is the dry baseline.
        "CoCl₂" => Some(( 40,  85, 200)),
        // Copper(I) oxide — sim approximation of the iconic green/
        // blue copper patina. Real Cu₂O is actually deep red/brown
        // (cuprite), and the actual patina is a basic copper
        // carbonate (CuCO₃·Cu(OH)₂) we don't model — but visually
        // "green oxide on a copper roof or Statue of Liberty" is
        // what users associate with copper oxide. The bespoke
        // override skips real-Cu₂O-red and renders the recognizable
        // patina green directly.
        "Cu₂O"  => Some(( 90, 165, 130)),
        // Copper(I) chloride — pale white-yellow when anhydrous;
        // hydrates and dissolves to the iconic teal/cyan copper
        // chloride solution color via the moisture-aware override
        // below. Solution shade pushes more cyan-green than
        // straight blue so it reads distinct from water's own
        // blue tint at saturation.
        "CuCl"  => Some((230, 220, 170)),
        _ => None,
    }
}

// Wet-state color for compounds whose visible appearance shifts with
// hydration. Real CoCl₂ is the textbook indicator: anhydrous deep
// blue, hexahydrate (CoCl₂·6H₂O) bright pink/magenta — the basis of
// every humidity-indicating drying agent. The render path
// interpolates between the dry `bespoke_color_for_formula` value
// and this wet color based on the cell's moisture, so dropping a
// CoCl₂ pile in water visibly shifts the residue and the resulting
// CoCl₂-saturated water solution toward pink. Returns None for
// compounds without hydration-driven color change.
// Per-formula kind override for compounds where the heuristic
// (Solid+Gas → Powder, etc.) gets the state wrong. Real-world
// volatile metalloid fluorides (GeF₄, SiF₄, BF₃) are gases at
// STP, but our averaging gives them a wildly-off bp that fails
// the < 100°C molecular-gas detection. Returns Some(Kind) to
// force the kind; None falls through to the heuristic.
fn bespoke_kind_for_formula(formula: &str) -> Option<Kind> {
    match formula {
        // GeF₄: real bp -36°C, gas at STP. Averaging Ge bp 2833 with
        // F bp -188 gives 1322°C, so "molecular_gas" check fails and
        // it lands as Powder. Force Gas.
        "GeF₄" => Some(Kind::Gas),
        // AsF₅: real bp -53°C, gas at STP. Same averaging issue as
        // GeF₄. Sim produces +5 oxidation state from valence math
        // (real-world stable form is AsF₃ but engine isn't variable-
        // valence) — both AsF₅ and AsF₃ are gases at STP, so the
        // kind override is correct regardless.
        "AsF₅" => Some(Kind::Gas),
        // AsCl₅ (sim formula from As valence 5): real AsCl₅ is unstable
        // above -50°C, basically doesn't exist. Render as the
        // textbook +3 form's behavior: AsCl₃ is liquid at STP
        // (mp -16, bp 130). Treat the sim's AsCl₅ cell as that
        // liquid for plausibility.
        "AsCl₅" => Some(Kind::Liquid),
        // SnCl₄ — real fuming-tin-tetrachloride is a liquid at STP
        // (mp -33, bp 114). Famous demonstration: smokes vigorously
        // in moist air. Phase points handle the temp behavior; this
        // forces Liquid kind so its base color/density/viscosity
        // match a real liquid puddle.
        "SnCl₄" => Some(Kind::Liquid),
        // SbF₅ and SbCl₅ — both real STP liquids (mp 8/4, bp 141/79).
        // Force Liquid kind so the puddle reads correctly at room
        // temp instead of being rendered as a metallic powder.
        "SbF₅"  => Some(Kind::Liquid),
        "SbCl₅" => Some(Kind::Liquid),
        // TeF₆ — real STP gas (bp -39°C), like SeF₆ and UF₆. Force
        // Gas kind so it puffs out as vapor instead of being rendered
        // as a solid powder.
        "TeF₆"  => Some(Kind::Gas),
        // SeF₆: real bp -34°C, gas at STP. Same averaging issue as
        // GeF₄ / AsF₅ — averaging Se bp 685 with F bp -188 lands well
        // above ambient and the molecular_gas check fails.
        "SeF₆" => Some(Kind::Gas),
        // H₂Se: real bp -41°C, gas at STP. Real toxic chalcogen
        // hydride, similar to H₂S but heavier. Sim doesn't model
        // the toxicity — just gets the kind right.
        "H₂Se" => Some(Kind::Gas),
        // TeH₆ (sim formula from Te valence 6 + H acceptor) ≈ real
        // H₂Te, bp -2°C, gas at STP. Toxic chalcogen hydride.
        "TeH₆" => Some(Kind::Gas),
        // MoF₆: real bp 35°C, gas at STP. Volatile metal hexafluoride
        // similar to UF₆ behavior — used in real chemistry as a
        // gas-phase fluorination agent.
        "MoF₆" => Some(Kind::Gas),
        _ => None,
    }
}

// Per-formula phase points override. Must be paired with
// bespoke_kind_for_formula whenever a kind is forced — otherwise the
// generic phase pass compares cell temp against averaged mp/bp,
// concludes "this is currently solid", sets PHASE_SOLID, and
// cell_physics() turns the cell into Gravel regardless of the base
// Kind. Returns Some((mp, bp)) to override the averaged values.
fn bespoke_phase_points_for_formula(formula: &str) -> Option<(i16, i16)> {
    match formula {
        // GeF₄ real bp -36°C. Both points kept below ambient so the
        // phase pass agrees with the forced Kind::Gas at room temp.
        "GeF₄" => Some((-50, -36)),
        // AsF₅ real bp -53°C, mp -80°C.
        "AsF₅" => Some((-80, -53)),
        // AsCl₅ rendered as AsCl₃-equivalent: real AsCl₃ mp -16,
        // bp 130 — liquid at STP, boils with mild heating.
        "AsCl₅" => Some((-16, 130)),
        // As₂O₅: render with sublimation behavior. Real As₂O₃
        // (white arsenic) sublimes at 193°C; As₂O₅ in sandbox
        // gameplay reads better as a sublimating volatile oxide
        // than a refractory powder. bp < mp = same sublimation
        // pattern As itself uses (bp 614, mp 817).
        "As₂O₅" => Some((400, 200)),
        // SeF₆ real bp -34°C, mp -50°C.
        "SeF₆" => Some((-50, -34)),
        // H₂Se real bp -41°C, mp -65°C.
        "H₂Se" => Some((-65, -41)),
        // TeH₆ rendered as H₂Te (real bp -2°C, mp -49°C).
        "TeH₆" => Some((-49, -2)),
        // SeO₃ — sublimation pattern (bp < mp) for volatile chalcogen
        // oxide. Real SeO₃ sublimes at 119°C, SeO₂ at 315°C; pick
        // ~190°C as a plausible midpoint that gives a heat-to-vaporize
        // gameplay window.
        "SeO₃" => Some((400, 190)),
        // SeCl₆ (sim formula): real SeCl₄ sublimes at 191°C, decomposes
        // at higher temps. Render the sim's +6 product with the +4
        // form's sublimation behavior — Powder/Gravel coating that
        // vaporizes when heated past ~190°C.
        "SeCl₆" => Some((400, 190)),
        // Zr halides — all sublime as the whole compound at moderate
        // heat. mp set high so phase pass classifies cold solid
        // correctly; bp set to real sublimation point so heating
        // converts cell to gas without going through liquid.
        "ZrCl₄" => Some((500, 331)),
        "ZrBr₄" => Some((500, 357)),
        "ZrI₄"  => Some((500, 431)),
        // Nb halides — all volatile, mostly sublime. NbF₅ has a normal
        // mp/bp range (mp 79, bp 234), the others sublime as the whole
        // compound at moderate heat.
        "NbF₅"  => Some((79, 234)),
        "NbCl₅" => Some((500, 254)),
        "NbBr₅" => Some((500, 270)),
        "NbI₅"  => Some((500, 330)),
        // Mo halides — sim's +6 products rendered with +4/+5 behavior.
        // MoF₆ real bp 35°C → gas at room temp. The others sublime.
        "MoF₆"  => Some((17, 35)),
        "MoCl₆" => Some((500, 268)),
        "MoBr₆" => Some((500, 350)),
        "MoI₆"  => Some((500, 250)),
        // Tc compounds — Tc₂O₇ real mp 119, bp 311 (volatile).
        // Halides rendered with +4 form sublimation behavior.
        "Tc₂O₇" => Some((119, 311)),
        "TcF₇"  => Some((500, 100)),
        "TcCl₇" => Some((500, 300)),
        "TcBr₇" => Some((500, 350)),
        "TcI₇"  => Some((500, 300)),
        // Ru compounds — sandbox treats Ru+O as the stable refractory
        // RuO₂-like coating (real RuO₂ mp ~1200, very stable) rather
        // than the famous volatile RuO₄. Don't override mp/bp; the
        // refractory_oxide floor (2000/3500) applies. Halides
        // rendered with +3 form sublimation behavior.
        "RuF₈"  => Some((500, 250)),
        "RuCl₈" => Some((500, 500)),
        "RuBr₈" => Some((500, 400)),
        "RuI₈"  => Some((500, 350)),
        // Rh compounds — RhO uses refractory floor. Halides rendered
        // with +3 form sublimation/decomp temps.
        "RhF₂"  => Some((500, 300)),
        "RhCl₂" => Some((500, 718)),
        "RhBr₂" => Some((500, 400)),
        "RhI₂"  => Some((500, 350)),
        // Pd compounds — PdO uses refractory floor. Halides at real
        // decomp temps.
        "PdF₂"  => Some((500, 750)),
        "PdCl₂" => Some((500, 600)),
        "PdBr₂" => Some((500, 250)),
        "PdI₂"  => Some((500, 350)),
        // Cd compounds — CdO uses refractory floor. Halides at real
        // sublimation temps.
        "CdF₂"  => Some((1110, 1748)),
        "CdCl₂" => Some((568, 960)),
        "CdBr₂" => Some((568, 860)),
        "CdI₂"  => Some((388, 742)),
        // In compounds — In₂O₃ uses refractory floor. Halides at
        // real sublimation/melt temps.
        "InF₃"  => Some((1170, 1200)),
        "InCl₃" => Some((586, 800)),
        "InBr₃" => Some((436, 600)),
        "InI₃"  => Some((210, 500)),
        // Sn compounds — SnO₂ uses refractory floor. SnCl₄ is the
        // famous fuming-tin-tetrachloride STP liquid (mp -33, bp 114).
        // SnBr₄ liquid above mp 31. SnI₄ orange solid mp 144.
        // SnF₄ sublimates 705.
        "SnF₄"  => Some((1000, 705)),
        "SnCl₄" => Some((-33, 114)),
        "SnBr₄" => Some((31, 205)),
        "SnI₄"  => Some((144, 348)),
        // Sb compounds (sim +5 / real +3 mix). Sb₂O₅ rendered with
        // Sb₂O₃ behavior (sublimes ~1425). SbF₅/SbCl₅ are real STP
        // liquids — phase points tuned to land in the liquid window
        // at room temp. SbBr₅/SbI₅ use sublimation patterns (bp < mp)
        // with bp set above their formation temps so they appear as
        // solids at the reaction threshold + reaction-heat bump,
        // then sublimate when heated further.
        "Sb₂O₅" => Some((800, 1425)),
        "SbF₅"  => Some((8, 141)),
        "SbCl₅" => Some((4, 79)),
        "SbBr₅" => Some((500, 400)),
        "SbI₅"  => Some((500, 350)),
        // Te compounds (sim +6 / real +4). TeO₃ rendered as TeO₂
        // (subl 1245). TeF₆ real STP gas. TeCl₆/TeBr₆/TeI₆
        // rendered as TeCl₄/Br₄/I₄ with sublimation pattern.
        "TeO₃"  => Some((733, 1245)),
        "TeF₆"  => Some((-37, -39)),
        "TeCl₆" => Some((500, 380)),
        "TeBr₆" => Some((500, 420)),
        "TeI₆"  => Some((500, 380)),
        _ => None,
    }
}

fn bespoke_hydration_color_for_formula(formula: &str) -> Option<(u8, u8, u8)> {
    match formula {
        "CoCl₂" => Some((230,  85, 145)),  // hydrated cobalt-chloride pink
        // Hydrated/dissolved copper chloride — saturated cyan-teal,
        // pushed toward green-blue-cyan (vs straight blue) so it
        // reads distinct from water's own blue tint at solute
        // saturation. Real CuCl₂·2H₂O crystals and CuCl₂ aqueous
        // solutions are this same blue-green range.
        "CuCl"  => Some(( 30, 195, 200)),
        _ => None,
    }
}

// Map a stoichiometric formula string back to the bespoke Element
// variant that represents it, if any. Used by derive_or_lookup to
// avoid creating parallel derived entries that would visually and
// behaviorally diverge from their hand-tuned bespoke twins.
fn bespoke_for_formula(formula: &str) -> Option<Element> {
    match formula {
        "H₂O"   => Some(Element::Water),
        "NaCl"  => Some(Element::Salt),
        "Fe₂O₃" => Some(Element::Rust),
        "CO₂"   => Some(Element::CO2),
        "SiO₂"  => Some(Element::Sand),
        _ => None,
    }
}

// Per-(donor, acceptor) compound decomposition threshold. Returns None
// for compounds that are stable across the sim's temp range, Some(°C)
// for ones that break apart on heating into donor + acceptor atom
// (Rust-style). Covers oxides AND halides — halide bond strength
// drops F > Cl > Br > I, so iodides decompose lowest, fluorides
// essentially never. Alkali halides (NaCl, KBr) and alkaline-earth
// halides are stable; transition-metal halides decompose into metal
// + halogen on heating, matching real-world behavior (CrI₃ → CrI₂ +
// I₂ at 500-600°C; van Arkel-de Boer Ti from TiI₄ at 1400°C; etc.).
// Stability vs. decomp is chosen for chemistry plausibility AND
// loop-prevention.
fn compound_decomposition_threshold(donor: Element, acceptor: Element, melting_point: i16) -> Option<i16> {
    let donor_is_alkali = matches!(donor,
        Element::Li | Element::Na | Element::K | Element::Rb
        | Element::Cs | Element::Fr);
    let donor_is_alkaline_earth = matches!(donor,
        Element::Be | Element::Mg | Element::Ca
        | Element::Sr | Element::Ba | Element::Ra);

    match acceptor {
        // ---- Oxides ----
        Element::O => match donor {
            // Alkali / alkaline-earth oxides — Cs₂O, Na₂O, CaO, MgO, BeO
            // etc. Don't reduce back to bulk metal at sim temps. Letting
            // them decompose creates an oxidize-decompose loop with the
            // alkali metals' fast O reactivity.
            _ if donor_is_alkali || donor_is_alkaline_earth => None,
            // Al₂O₃ — most thermodynamically stable oxide; the reason
            // thermite works. Stable at sim temps.
            Element::Al => None,
            // Nonmetal oxides — SO₃, P₂O₅, NO₂, B₂O₃. Stable in sim
            // temp range. Prevents burning-S → SO₃ → decompose loops.
            Element::S | Element::P | Element::N | Element::B => None,
            // As₂O₅: don't decompose back to As. Real arsenic oxides
            // (As₂O₃ in particular) sublime as the volatile compound
            // — the cell vaporizes whole, NOT splits into As + O.
            // Skipping decomp puts the compound on the phase-pass
            // track instead, where its bespoke phase points
            // (mp 400, bp 200) drive sublimation at 200°C.
            Element::As => None,
            // SeO₃: same sublimation behavior as As₂O₅. Real SeO₃
            // sublimes at 119°C, real SeO₂ at 315°C — both volatile,
            // neither decomposes back to elemental Se under normal
            // conditions. Phase points handle the sublimation.
            Element::Se => None,
            // Sb₂O₅ rendered as Sb₂O₃ (sim's +5 product, real is +3).
            // Sb₂O₃ sublimes ~1425°C, doesn't decompose. Phase points
            // handle sublimation.
            Element::Sb => None,
            // TeO₃ rendered as TeO₂ (real, sublimes 1245°C). Yellow
            // crystalline solid that vaporizes whole, no decomp.
            Element::Te => None,
            // Tc₂O₇: real bp 311°C, mp 119°C — volatile yellow oxide
            // that sublimes/melts rather than decomposing. Same
            // volatile-non-refractory-oxide pattern as As/Se.
            Element::Tc => None,
            // Rust-like decomposing transition-metal oxides. Real-
            // world decomp temps used as thresholds.
            Element::Sc => Some(2400),  // Sc₂O₃ ~2485°C
            Element::Ti => Some(3200),  // TiO₂ decomp >3000°C
            Element::V  => Some(1800),  // V₂O₅  ~1750°C
            Element::Cr => Some(2400),  // Cr₂O₃ ~2435°C
            // Generic metal-oxide fallback. Catches Cu₂O, MnO₂, etc.
            // without per-donor entries until audit reaches them.
            _ => Some(melting_point.max(600)),
        },
        // ---- Halides ----
        // Alkali / alkaline-earth halides (NaCl, KBr, MgF₂, CaCl₂)
        // are very stable in real chemistry — most don't decompose,
        // they just melt or boil. Skip the decomp path entirely.
        _ if donor_is_alkali || donor_is_alkaline_earth => None,
        // M-F bonds are too strong for sim-reachable decomp.
        Element::F => None,
        // Volatile non-metal halides (SeCl₆ rendered as SeCl₄-like)
        // sublimate as the whole compound, like the chalcogen oxides.
        // Don't decompose back to the constituents — the bespoke
        // phase points drive sublimation at ~190°C instead.
        _ if donor == Element::Se => None,
        // Zr halides (ZrCl₄ 331°C, ZrBr₄ 357°C, ZrI₄ 431°C all sublime)
        // — same volatile-as-whole-compound behavior; let phase points
        // drive sublimation rather than the cell breaking back to Zr+X.
        _ if donor == Element::Zr => None,
        // Nb halides (NbCl₅ 254°C, NbBr₅ 270°C, NbI₅ ~330°C, NbF₅ 234°C)
        // — same volatile-as-whole-compound behavior.
        _ if donor == Element::Nb => None,
        // Mo halides — sim's +6 products rendered with the +5/+4
        // form's behavior. MoF₆ is real gas (bp 35°C), MoCl₅ subl
        // 268°C, MoBr₄ ~350°C, MoI₄ ~250°C.
        _ if donor == Element::Mo => None,
        // Tc halides — sim's +7 products rendered with +4 behavior
        // (TcCl₄, TcBr₄, TcI₄ all sublime/decompose around 300°C).
        _ if donor == Element::Tc => None,
        // Ru halides — sim's +8 products rendered with +3 form
        // behavior (RuCl₃ subl/decomp ~500°C, RuBr₃/RuI₃ similar).
        _ if donor == Element::Ru => None,
        // Rh halides — sim's +2 products rendered with +3 form
        // behavior (RhCl₃ decomp 718°C, RhBr₃ decomp 350°C, etc.).
        _ if donor == Element::Rh => None,
        // Pd halides — PdCl₂ decomp ~600°C, PdBr₂ ~250°C, PdI₂
        // decomp ~350°C, PdF₂ ~750°C. Sublime as whole compound.
        _ if donor == Element::Pd => None,
        // Cd halides — CdCl₂ subl 960°C, CdBr₂ subl ~860°C, CdI₂
        // subl 388°C, CdF₂ very stable.
        _ if donor == Element::Cd => None,
        // In halides — InCl₃ subl 600°C, InBr₃ mp 436, InI₃ mp 210.
        _ if donor == Element::In => None,
        // Sn halides — SnCl₄ liquid (bp 114), SnBr₄ mp 31/bp 205,
        // SnI₄ mp 144, SnF₄ subl 705. Sublimate as whole compound.
        _ if donor == Element::Sn => None,
        // Sb halides — SbF₅/SbCl₅ liquids at STP, SbBr₃/SbI₃ solids.
        // Render +5 sim products with +3-form decomp temps.
        _ if donor == Element::Sb => None,
        // Te halides — TeF₆ gas (real bp -39°C), TeCl₄/TeBr₄/TeI₄
        // solids that sublimate at moderate temps.
        _ if donor == Element::Te => None,
        // Transition / post-transition metal Cl / Br / I — decompose
        // at halide-specific thresholds. Heavier halogens go lower
        // because the M-X bond is weaker. These thresholds are coarse
        // (one number per halogen, not per metal) — the sim doesn't
        // model stepwise oxidation states (CrI₃ vs CrI₂), just the
        // end-state metal + halogen split.
        Element::Cl => Some(1500),
        Element::Br => Some(800),
        Element::I  => Some(500),
        _ => None,
    }
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
    // H acceptor takes a single electron (H⁻ anion), like a halogen
    // would — that's why metal hydrides are 1:1 (LiH, NaH, KH, RbH)
    // or 1:2 (MgH₂, CaH₂). The 8-valence math would otherwise give
    // M₇H (treating H as if it needs 7 electrons to reach an octet,
    // which doesn't apply to H — its full shell is just 2 electrons).
    let needs = if acceptor == Element::H {
        1
    } else {
        8u8.saturating_sub(aa.valence_electrons).max(1)
    };
    let g = gcd_u8(gives, needs);
    let donor_count = (needs / g).max(1);
    let acceptor_count = (gives / g).max(1);
    let constituents: Vec<(Element, u8)> = vec![
        (donor, donor_count),
        (acceptor, acceptor_count),
    ];
    let formula = format_formula(&constituents);

    // Bespoke compounds win — if this formula already has a hand-tuned
    // Element variant (Water = H₂O, Sand = SiO₂, Salt = NaCl, Rust =
    // Fe₂O₃, CO2 = CO₂), refuse to create a duplicate derived entry.
    // Forces all callers to either map through infer_product's bespoke
    // table or get None back. Keeps compound identity unified.
    if bespoke_for_formula(&formula).is_some() {
        return None;
    }

    {
        let mut reg = DERIVED_COMPOUNDS.write();
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
        // Color mix. For hydrogen-as-donor compounds (HCl, HBr, HI,
        // H₂S, …) H's near-white color drowns out the partner atom's
        // identity in a flat 50/50 average. Weight the acceptor 4×
        // when donor is H so the halogen/chalcogen colour dominates
        // — HBr reads as bromine red-brown, HI reads as iodine
        // purple. Other compounds keep the standard stoichiometric
        // mix.
        let (r_mix, g_mix, b_mix) = if donor == Element::H {
            let blend = |d: u8, a: u8| -> u8 {
                ((d as u32 + a as u32 * 4) / 5) as u8
            };
            (
                blend(donor_color.0, acceptor_color.0),
                blend(donor_color.1, acceptor_color.1),
                blend(donor_color.2, acceptor_color.2),
            )
        } else {
            let r_mix = mix(donor_color.0, donor_count, acceptor_color.0, acceptor_count);
            let g_mix = mix(donor_color.1, donor_count, acceptor_color.1, acceptor_count);
            let b_mix = mix(donor_color.2, donor_count, acceptor_color.2, acceptor_count);
            (r_mix, g_mix, b_mix)
        };
        // Per-formula color overrides for compounds whose averaged
        // constituent colors don't match real-world appearance. Adds
        // up as needed during the audit pass.
        let color = bespoke_color_for_formula(&formula)
            .unwrap_or((r_mix, g_mix, b_mix));
        // Kind heuristic. Earlier version used STP-state of the
        // constituents only — H (gas) + Br (liquid) hit (Gas, _)
        // → Powder, even though HBr is a textbook gas. Augment with
        // a "molecular gas" detection: small light-element compounds
        // whose averaged boiling point is near-or-below room temp
        // should render as a gas regardless of constituent phases.
        let avg_mp_pre = (da.melting_point as i32 + aa.melting_point as i32) / 2;
        let avg_bp_pre = (da.boiling_point as i32 + aa.boiling_point as i32) / 2;
        let molecular_gas = avg_bp_pre < 100;
        // Per-formula kind override takes priority over the heuristic.
        let kind = if let Some(forced) = bespoke_kind_for_formula(&formula) {
            forced
        } else if molecular_gas {
            Kind::Gas
        } else {
            // Ionic salts and oxides — anything pairing a solid metal
            // donor with a typical anion-forming nonmetal acceptor
            // (halogen, O, S, N, P) — render as Powder regardless of
            // the acceptor's STP phase. Real metal halides and oxides
            // are granular crystalline products that flake/slide like
            // rust does, not rigid Gravel stacks. Without this rule,
            // NaBr from Na+liquid-Br comes out as Gravel and stays
            // glued to the Na pile, blocking the Br liquid from
            // reaching fresh metal. (The earlier Solid+Gas → Powder
            // rule didn't catch liquid acceptors like Br₂.)
            let anion_acceptor = matches!(acceptor,
                Element::F | Element::Cl | Element::Br | Element::I
                | Element::O | Element::S | Element::N | Element::P);
            // Tight passivating oxides — Al₂O₃, Cr₂O₃, TiO₂, V₂O₅,
            // Sc₂O₃ — form coherent cohesive coatings that don't flake.
            // Real-world textbook examples of self-protecting layers
            // (V₂O₅ is why V is corrosion-resistant despite EN
            // suggesting it should oxidize easily; Sc₂O₃ is similar
            // for rare-earth-adjacent metals); modeled by keeping
            // them as Gravel (rigid) instead of the default Powder
            // for ionic salts.
            let coating_oxide = acceptor == Element::O
                && matches!(donor,
                    Element::Al | Element::Cr | Element::Ti
                    | Element::V | Element::Sc | Element::Be
                    | Element::Ni | Element::Cu | Element::Zn
                    | Element::Ga | Element::Ge | Element::As
                    | Element::Se | Element::Sr | Element::Y
                    | Element::Zr | Element::Nb | Element::Mo
                    | Element::Tc | Element::Ru | Element::Rh
                    | Element::Pd | Element::Ag | Element::Cd
                    | Element::In | Element::Sn | Element::Sb
                    | Element::Te);
            // Noble metal halide coating — Au, Cu form tight
            // halide surface layers (CuI/CuBr, AuBr₃) that read
            // as cohesive tarnish, not flaky powder. Ag is
            // excluded — silver halides are the photographic
            // flash salts and should remain Powder (rapid
            // formation, granular). Mirrors the rate cap in
            // try_emergent_reaction's donor_halide_passivates list.
            let coating_halide = matches!(acceptor,
                    Element::F | Element::Cl | Element::Br | Element::I)
                && matches!(donor,
                    Element::Au | Element::Cu);
            match (da.stp_state, aa.stp_state) {
                (AtomState::Gas, AtomState::Gas) => Kind::Gas,
                (AtomState::Solid, _) if coating_oxide => Kind::Gravel,
                (AtomState::Solid, _) if coating_halide => Kind::Gravel,
                (AtomState::Solid, _) if anion_acceptor => Kind::Powder,
                (AtomState::Solid, _)            => Kind::Gravel,
                (AtomState::Liquid, _)           => Kind::Liquid,
                (AtomState::Gas, _)              => Kind::Powder,
            }
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
            // Oxide-specific O density: O's gas density (-1) clamped to 0
            // makes every solid oxide read lighter than its parent metal,
            // which is wrong — real metal oxides are typically denser than
            // the metal (BeO 3.01 vs Be 1.85, MgO 3.58 vs Mg 1.74). Using
            // a representative bound-oxide value (~20, ≈2 g/cm³) for O
            // pulls oxide densities up to where they belong, so e.g. BeO
            // sinks in water instead of floating.
            // Bound-state density for nonmetal-gas acceptors. Their gas
            // densities (negative or near-zero) drag the average down,
            // so a metal+halogen salt comes out lighter than the parent
            // metal — wrong, real ionic salts are heavier (LiF 2.64 vs
            // Li 0.53; NaCl 2.16 vs Na 0.97). Without this override, a
            // LiF rind floats on Li and stops the F attack at the
            // surface; with it, LiF sinks through and exposes fresh Li.
            let aa_d = match acceptor {
                Element::O                                 => 20,
                Element::F  | Element::Cl                  => 22,
                // Br bound density bumped to 60 so NaBr/LiBr/KBr come
                // out denser than liquid Br₂ (sim density 31). Without
                // this, the salt formed at a Na-on-Br interface stays
                // sandwiched between metal and liquid and the reaction
                // dies; with it, the salt sinks INTO the bromine,
                // exposing fresh metal and letting the reaction
                // consume the Na pile (matching real chemistry — Na
                // in Br₂ reacts vigorously to completion).
                Element::Br                                => 60,
                Element::I                                 => 45,
                Element::N  | Element::S  | Element::P     => 22,
                _ => acceptor.physics().density.max(0) as i32,
            };
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
        let avg_mp = avg_mp_pre;
        let avg_bp = avg_bp_pre;
        let gas_gas = da.stp_state == AtomState::Gas
            && aa.stp_state == AtomState::Gas;
        // Refractory metal oxides — strong ionic bonding gives them very
        // high mp AND bp (Sc₂O₃ mp 2485 / bp 3900, Cr₂O₃ mp 2435 / bp
        // 4000, Al₂O₃ mp 2072 / bp 3000, MgO mp 2852 / bp 3600, V₂O₅
        // mp 690 / decomposes ~1750). The averaging heuristic
        // underestimates both severely. The bp floor matters as much
        // as the mp floor: if bp lands below the decomposition
        // threshold set further down, heating the oxide boils it into
        // a gas before it can decompose back to its constituents
        // (Sc₂O₃ would gas at 1500°C instead of decomposing at
        // 2400°C, killing the oxide-coating illusion).
        let refractory_oxide = acceptor == Element::O
            && matches!(donor,
                Element::Al | Element::Mg | Element::Be
                | Element::Sc | Element::Ti | Element::V | Element::Cr
                | Element::Zr | Element::Nb | Element::Mo
                | Element::Ru | Element::Rh | Element::Pd
                | Element::Cd | Element::In | Element::Sn);
        let oxide_mp_floor: i32 = if refractory_oxide { 2000 } else { 500 };
        let oxide_bp_floor: i32 = if refractory_oxide { 3500 } else { 1500 };
        // Molecular gases (HCl/HBr/HI/H₂S/NH₃/etc.) keep their real
        // sub-zero boiling points — flooring them to 1500 °C would
        // wrongly hold them solid up through any reasonable temp.
        let (melting_point, boiling_point) = if gas_gas || molecular_gas {
            (avg_mp as i16, avg_bp as i16)
        } else {
            (avg_mp.max(oxide_mp_floor) as i16, avg_bp.max(oxide_bp_floor) as i16)
        };
        // Per-formula phase-point override — paired with the
        // bespoke kind override above. Volatile metalloid fluorides
        // (GeF₄ etc.) need explicit sub-ambient mp/bp so the phase
        // pass doesn't classify them as currently-solid and force
        // PHASE_SOLID, which would override the forced Kind::Gas
        // back to Gravel.
        let (melting_point, boiling_point) =
            if let Some((mp, bp)) = bespoke_phase_points_for_formula(&formula) {
                (mp, bp)
            } else {
                (melting_point, boiling_point)
            };
        // Compound decomposition policy lives in
        // compound_decomposition_threshold — covers oxides and halides
        // uniformly. Gas-gas products (NO, SO₂ from gas reactants)
        // skip decomp because their averaged "melting point" is below
        // ambient and would trigger spurious decomposition.
        let decomposes_above = if !gas_gas {
            compound_decomposition_threshold(donor, acceptor, melting_point)
        } else {
            None
        };
        let hydration_color = bespoke_hydration_color_for_formula(&formula);
        let is_radioactive = constituents.iter().any(|(el, _)|
            atom_profile_for(*el).map_or(false, |p| p.half_life_frames > 0));
        reg.push(DerivedCompound {
            formula,
            constituents,
            physics,
            color,
            melting_point,
            boiling_point,
            decomposes_above,
            hydration_color,
        });
        let new_idx = (reg.len() - 1) as u8;
        let _ = DERIVED_HOT[new_idx as usize].set(DerivedHot {
            physics, color, melting_point, boiling_point, decomposes_above,
            hydration_color, is_radioactive,
        });
        Some(new_idx)
    }
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
// Hot path: lock-free atomic load via DERIVED_HOT.
fn derived_physics_of(idx: u8) -> PhysicsProfile {
    derived_hot(idx).map(|h| h.physics).unwrap_or(PhysicsProfile {
        density: 20, kind: Kind::Powder, viscosity: 0, molar_mass: 0.0,
    })
}

fn derived_color_of(idx: u8) -> (u8, u8, u8) {
    derived_hot(idx).map(|h| h.color).unwrap_or((160, 130, 140))
}

fn derived_formula_of(idx: u8) -> String {
    let reg = DERIVED_COMPOUNDS.read();
    reg.get(idx as usize)
        .map(|c| c.formula.clone())
        .unwrap_or_else(|| "?".to_string())
}

// First metal constituent of a derived compound. Used by electrolysis to
// figure out which metal ion plates out of a dissolved salt (CuCl → Cu,
// FeCl → Fe, NaCl → Na). Returns None for compounds that have no metal
// in their composition.
fn compound_metal_component(idx: u8) -> Option<Element> {
    let reg = DERIVED_COMPOUNDS.read();
    let c = reg.get(idx as usize)?;
    for &(el, _) in &c.constituents {
        if is_atomic_metal(el) { return Some(el); }
    }
    None
}

// Ionic metal-halide / metal-halogen compounds dissolve in water. Predicate
// used by dissolve() to decide whether a derived cell is a candidate
// solute. Atom categories are what make a compound "a salt" — if the
// constituents include a metal and a halogen, it's one.
fn derived_is_soluble_salt(idx: u8) -> bool {
    let reg = DERIVED_COMPOUNDS.read();
    let Some(c) = reg.get(idx as usize) else { return false; };
    let mut has_metal = false;
    let mut has_halogen = false;
    for &(el, _) in &c.constituents {
        if is_atomic_metal(el) { has_metal = true; }
        if let Some(a) = atom_profile_for(el) {
            if matches!(a.category, AtomCategory::Halogen) { has_halogen = true; }
        }
    }
    has_metal && has_halogen
}

// Slow-hydrolysis residues — oxides of Mg/Ca/Sr/Ba/Ra/Sc/Mn formed when
// these metals react with water (donor cell → oxide, water cell → H₂).
// Their real-world hydroxides Mg(OH)₂/Ca(OH)₂/etc. have slight water
// solubility, so the residue cells slowly dissolve into adjacent water
// as solute. Once water saturates (solute_amt ≥ ABSORB_THRESHOLD in
// dissolve()), fresh residue stops dissolving and piles up — matches
// real saturation kinetics. Without this, the residue layer at the
// metal-water interface permanently shells over fresh metal and the
// reaction stops; with it, the layer cycles through form → dissolve →
// expose-fresh-metal → repeat.
fn derived_is_hydrolysis_residue(idx: u8) -> bool {
    let reg = DERIVED_COMPOUNDS.read();
    let Some(c) = reg.get(idx as usize) else { return false; };
    let has_o = c.constituents.iter().any(|&(el, _)| el == Element::O);
    let slow_donor = c.constituents.iter().any(|&(el, _)| matches!(el,
        Element::Mg | Element::Ca | Element::Sr | Element::Ba | Element::Ra
        | Element::Sc | Element::Mn));
    has_o && slow_donor
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
        // Periodic-table fill.
         3 => Element::Li,
         4 => Element::Be,
        21 => Element::Sc,
        22 => Element::Ti,
        23 => Element::V,
        24 => Element::Cr,
        25 => Element::Mn,
        27 => Element::Co,
        31 => Element::Ga,
        32 => Element::Ge,
        33 => Element::As,
        34 => Element::Se,
        35 => Element::Br,
        36 => Element::Kr,
        37 => Element::Rb,
        38 => Element::Sr,
        39 => Element::Y,
        40 => Element::Zr,
        41 => Element::Nb,
        42 => Element::Mo,
        43 => Element::Tc,
        44 => Element::Ru,
        45 => Element::Rh,
        46 => Element::Pd,
        48 => Element::Cd,
        49 => Element::In,
        50 => Element::Sn,
        51 => Element::Sb,
        52 => Element::Te,
        53 => Element::I,
        54 => Element::Xe,
        56 => Element::Ba,
        57 => Element::La,
        58 => Element::Ce,
        59 => Element::Pr,
        60 => Element::Nd,
        61 => Element::Pm,
        62 => Element::Sm,
        63 => Element::Eu,
        64 => Element::Gd,
        65 => Element::Tb,
        66 => Element::Dy,
        67 => Element::Ho,
        68 => Element::Er,
        69 => Element::Tm,
        70 => Element::Yb,
        71 => Element::Lu,
        72 => Element::Hf,
        73 => Element::Ta,
        74 => Element::W,
        75 => Element::Re,
        76 => Element::Os,
        77 => Element::Ir,
        78 => Element::Pt,
        81 => Element::Tl,
        83 => Element::Bi,
        84 => Element::Po,
        85 => Element::At,
        86 => Element::Rn,
        87 => Element::Fr,
        89 => Element::Ac,
        90 => Element::Th,
        91 => Element::Pa,
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
        Element::Cl => 17,  Element::Ar => 18,
        Element::K  => 19,  Element::Ca => 20,
        Element::Fe => 26,  Element::Cu => 29,  Element::Au => 79,
        Element::Hg => 80,  Element::U  => 92,
        Element::B  => 5,
        Element::Ni => 28,  Element::Zn => 30,  Element::Ag => 47,
        Element::Cs => 55,  Element::Pb => 82,  Element::Ra => 88,
        // Periodic-table fill.
        Element::Li => 3,   Element::Be => 4,
        Element::Sc => 21,  Element::Ti => 22,  Element::V  => 23,
        Element::Cr => 24,  Element::Mn => 25,  Element::Co => 27,
        Element::Ga => 31,  Element::Ge => 32,  Element::As => 33,
        Element::Se => 34,  Element::Br => 35,  Element::Kr => 36,
        Element::Rb => 37,  Element::Sr => 38,  Element::Y  => 39,
        Element::Zr => 40,  Element::Nb => 41,  Element::Mo => 42,
        Element::Tc => 43,  Element::Ru => 44,  Element::Rh => 45,
        Element::Pd => 46,  Element::Cd => 48,  Element::In => 49,
        Element::Sn => 50,  Element::Sb => 51,  Element::Te => 52,
        Element::I  => 53,  Element::Xe => 54,  Element::Ba => 56,
        Element::La => 57,  Element::Ce => 58,  Element::Pr => 59,
        Element::Nd => 60,  Element::Pm => 61,  Element::Sm => 62,
        Element::Eu => 63,  Element::Gd => 64,  Element::Tb => 65,
        Element::Dy => 66,  Element::Ho => 67,  Element::Er => 68,
        Element::Tm => 69,  Element::Yb => 70,  Element::Lu => 71,
        Element::Hf => 72,  Element::Ta => 73,  Element::W  => 74,
        Element::Re => 75,  Element::Os => 76,  Element::Ir => 77,
        Element::Pt => 78,  Element::Tl => 81,  Element::Bi => 83,
        Element::Po => 84,  Element::At => 85,  Element::Rn => 86,
        Element::Fr => 87,  Element::Ac => 89,  Element::Th => 90,
        Element::Pa => 91,
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
    let reg = DERIVED_COMPOUNDS.read();
    let cd = reg.get(cell.derived_id as usize)?;
    let elements: Vec<Element> =
        cd.constituents.iter().map(|(e, _)| *e).collect();
    for e in &elements {
        if !is_atomic_metal(*e) { return None; }
    }
    Some(elements)
}

// Basic-oxide classification. A metal-oxide compound (M + O) acts as a
// base in Brønsted terms: when it meets an acid, the metal captures the
// acid's acceptor and the released O pairs with H to form water.
// Returns (metal element, basicity) where basicity = 2.0 − metal_e
// (lower-E metals are more strongly basic — same cutoff as acid
// displacement's metal reactivity).
fn basic_oxide_signature(cell: Cell) -> Option<(Element, f32)> {
    if cell.el != Element::Derived { return None; }
    let reg = DERIVED_COMPOUNDS.read();
    let cd = reg.get(cell.derived_id as usize)?;
    if cd.constituents.len() < 2 { return None; }
    let (d_el, _) = cd.constituents[0];
    let (a_el, _) = cd.constituents[1];
    if a_el != Element::O { return None; }
    let d_e = atom_profile_for(d_el)?.electronegativity;
    if d_e <= 0.0 || d_e >= 2.0 { return None; }
    Some((d_el, 2.0 - d_e))
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
    let reg = DERIVED_COMPOUNDS.read();
    let cd = reg.get(cell.derived_id as usize)?;
    if cd.constituents.len() < 2 { return None; }
    let (d_el, _) = cd.constituents[0];
    if d_el != Element::H { return None; }
    let (a_el, _) = cd.constituents[1];
    if !matches!(a_el, Element::F | Element::Cl) { return None; }
    let a_e = atom_profile_for(a_el)?.electronegativity;
    let strength = (a_e - 2.20).max(0.0);
    Some((a_el, strength))
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
        Element::Derived => {
            // decomposition_of needs both the (cold) constituents AND the
            // (hot) decompose threshold — but most callers hit the early
            // None when there's no decomposes_above, so test that first
            // via the hot cache to avoid taking the lock.
            let thr = derived_hot(cell.derived_id)?.decomposes_above?;
            let reg = DERIVED_COMPOUNDS.read();
            let cd = reg.get(cell.derived_id as usize)?;
            if cd.constituents.len() < 2 { return None; }
            Some((thr, cd.constituents[0].0, cd.constituents[1].0))
        }
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
            // Hot-path lookup: physics + mp/bp all live in the
            // lock-free hot cache.
            let h = derived_hot(cell.derived_id)?;
            // Decomposable derived oxides (Sc₂O₃, V₂O₅, Cr₂O₃, Cu₂O, …)
            // behave like bespoke Rust: they don't melt or vaporize into
            // a liquid/gas of themselves. They sit there until the
            // decomposition pass breaks them into donor metal + O at
            // their threshold. Returning None here pulls them out of
            // the generic phase-transition system entirely, which
            // otherwise would convert hot oxide → gas → condense-back-
            // to-oxide and read as a phase cycle instead of a real
            // decomp. Bespoke Rust avoids this implicitly because it's
            // not in the phase-points table; derived oxides need the
            // same exclusion explicitly.
            if h.decomposes_above.is_some() { return None; }
            let stp_state = match h.physics.kind {
                Kind::Gas | Kind::Fire => AtomState::Gas,
                Kind::Liquid => AtomState::Liquid,
                _ => AtomState::Solid,
            };
            Some((h.melting_point, h.boiling_point, stp_state))
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
        Element::Ca   => Some((235,  80,  70)),  // brick-red
        Element::Sr   => Some((230,  35,  55)),  // crimson (signal-flare red)
        Element::Mg   => Some((255, 255, 255)), // brilliant white
        Element::B    => Some((130, 220, 100)), // bright green
        Element::Salt => Some((255, 220, 80)),  // NaCl → Na yellow
        Element::Li   => Some((220,  60,  80)), // crimson
        Element::Rb   => Some((200,  80, 140)), // red-violet
        Element::Cs   => Some((130,  90, 220)), // blue-violet
        Element::Fr   => Some((150, 110, 240)), // extrapolated, bluer
        Element::S    => Some(( 80, 160, 255)), // sky-blue (iconic SO₂ flame)
        Element::Se   => Some(( 80, 140, 255)), // azure (Se burns vivid blue)
        Element::Ti   => Some((250, 245, 230)), // white-hot incandescence
        Element::Be   => Some((245, 250, 255)), // dazzling white-blue (Be flame)
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

// Phase-aware pressure profile lookup. Atoms/compounds keep their hand-
// tuned static profiles, but Element::Derived has no static entry —
// without this helper, every derived compound (HCl, MgO, CuCl, …) would
// inherit the default `permeability: 0` which the pressure-diffusion
// pass treats as a wall. Result: derived gases sit rigid in their spawn
// shape because flux can't propagate through them. This routes derived
// cells through a kind-appropriate template (Gas → high perm, Liquid →
// medium, Solid → wall) so HCl wafts like Steam, MgO behaves like
// Stone, and so on.
fn cell_pressure_p(c: Cell) -> PressureProfile {
    if c.el != Element::Derived {
        return *c.el.pressure_p();
    }
    match cell_physics(c).kind {
        // Bumped to 70 (between atomic gas's 20-30 and Steam's 80)
        // so heavier-than-air derived gases like HCl get a visible
        // initial puff before gravity wins out and pools them on
        // the floor. Trade: slightly louder initial venting in
        // closed containers — matches the v0.1 "Steam vents the
        // kettle" pattern.
        Kind::Gas    => PressureProfile { permeability: 230, compliance: 200, formation_pressure: 70 },
        Kind::Fire   => PressureProfile { permeability: 230, compliance: 200, formation_pressure: 0 },
        Kind::Liquid => PressureProfile { permeability: 100, compliance:  50, formation_pressure: 0 },
        Kind::Powder => PressureProfile { permeability:  25, compliance:  15, formation_pressure: 0 },
        Kind::Gravel => PressureProfile { permeability:   0, compliance:  15, formation_pressure: 0 },
        Kind::Solid  => PressureProfile { permeability:   0, compliance:  25, formation_pressure: 0 },
        Kind::Empty  => PressureProfile { permeability: 255, compliance:   0, formation_pressure: 0 },
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

    // Canonical chemical formula. This is the IDENTITY of a compound.
    // Atoms return their atomic symbol; known-formula compounds return
    // their formula string (e.g., Water → "H₂O", Sand → "SiO₂").
    // Compounds without a clean formula (Wood, Lava, mixtures, etc.)
    // return None — they're treated as named macro-materials rather
    // than discrete chemical species.
    fn formula(self) -> Option<&'static str> {
        if let Some(a) = atom_profile_for(self) {
            return Some(a.symbol);
        }
        match self {
            Element::Water | Element::Steam | Element::Ice => Some("H₂O"),
            Element::Sand | Element::Glass | Element::MoltenGlass
                | Element::Quartz | Element::Obsidian => Some("SiO₂"),
            Element::Salt   => Some("NaCl"),
            Element::Rust   => Some("Fe₂O₃"),
            Element::CO2    => Some("CO₂"),
            Element::Oil    => Some("CₙHₙ₊₂"),  // generic alkane mixture
            Element::Firebrick => Some("Al₂O₃·SiO₂"),  // refractory mix
            _ => None,
        }
    }

    // English common name. Same string the periodic-table palette uses
    // for atoms, and the sandbox material name for compounds. Distinct
    // from formula() so the future discovery system can swap which one
    // gets shown to the player without renaming any internal data.
    #[inline] fn common_name(self) -> &'static str { self.name() }

    // User-facing display label.
    //
    // Audit-phase format: "Formula (Common Name)" so the player can
    // see at a glance which material is which while we work through
    // periodic-table tuning. Atoms render as "H (Hydrogen)", bespoke
    // compounds as "H₂O (Water)", "SiO₂ (Sand)". Compounds without a
    // clean formula fall back to common name only ("Wood", "Lava").
    //
    // POST-AUDIT TODO: split Element into separate Atom and Material
    // enums + a first-class Compound registry keyed by formula. Then
    // wire the discovery system: encountering a compound (any of its
    // materials) reveals the compound entry with common name + fun
    // facts via a modal. After discovery, this label collapses to
    // just the common name (with user-overrideable). See also
    // bespoke_for_formula and cell_label below — those are the seams
    // where compound-level identity already operates.
    fn display_label(self) -> String {
        match self.formula() {
            Some(f) => {
                let cn = self.common_name();
                if cn == f {
                    f.to_string()
                } else {
                    format!("{} ({})", f, cn)
                }
            }
            None => self.common_name().to_string(),
        }
    }

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
            // Quartz: smoky cool grey — clearly darker than Glass
            // (200,230,235) and distinct from the silver BattNeg
            // (~200,210,225) so battery casings don't blend into
            // their negative terminal.
            Element::Quartz    => (138, 150, 170),
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
            // ---- Periodic-table fill ----
            // Pushed harder than literal "silvery gray" so each tile
            // reads distinctly through the edge shading + AO. Still
            // anchored to real-world appearance: cool tints for left-
            // side metals (alkali / group-3 / 4 / 5), warm for the
            // right (Cu/Au/Bi), green-blue for Co/Os/Ir/Ta. Lanthanides
            // get a hue walk through colours their salts/oxides
            // actually take (Pr green, Nd lilac, Sm pale yellow, Eu
            // peach, Er pink, Tm green-blue) so the f-block isn't 15
            // identical silver tiles.
            Element::Li => (190, 205, 230),  // cool silver-blue
            Element::Be => (225, 215, 180),  // warm silver
            Element::Sc => (200, 215, 220),  // pale cool
            Element::Ti => (175, 185, 215),  // cool blue-grey
            Element::V  => (155, 160, 220),  // distinctly blue
            Element::Cr => (170, 215, 230),  // cyan-silver
            Element::Mn => (165, 130, 110),  // brownish (real Mn)
            Element::Co => (110, 145, 230),  // strong cobalt blue
            Element::Ga => (210, 220, 245),  // pale cool
            Element::Ge => (170, 195, 215),  // cool grey-blue
            Element::As => (130, 115, 110),  // dark warm grey
            Element::Se => (175,  60,  55),  // selenium red
            Element::Br => (160,  50,  35),  // red-brown halogen
            Element::Kr => (210, 225, 235),  // pale noble
            Element::Rb => (210, 200, 230),  // pale lavender
            Element::Sr => (230, 230, 175),  // warm gold-silver
            Element::Y  => (180, 210, 215),  // cool silver
            Element::Zr => (160, 200, 220),  // cool blue
            Element::Nb => (175, 170, 235),  // distinct lavender
            Element::Mo => (180, 195, 230),  // cool blue-silver
            Element::Tc => (215, 200, 165),  // warm yellow tint
            Element::Ru => (170, 175, 210),  // cool blue
            Element::Rh => (235, 240, 245),  // bright silver-white
            Element::Pd => (210, 220, 220),  // pale silver
            Element::Cd => (165, 200, 220),  // cool blue-silver
            Element::In => (200, 220, 245),  // pale blue
            Element::Sn => (220, 210, 215),  // pinkish silver
            Element::Sb => (140, 165, 215),  // cool blue
            Element::Te => (130, 110,  85),  // dark brown
            Element::I  => ( 90,  45, 130),  // purple halogen
            Element::Xe => (170, 195, 240),  // cool noble blue
            Element::Ba => (210, 235, 195),  // green-tinted (Ba flame)
            Element::La => (215, 220, 230),  // pale silver
            Element::Ce => (225, 220, 195),  // warm tan
            Element::Pr => (180, 220, 175),  // green (real Pr glass)
            Element::Nd => (210, 175, 220),  // lilac (real Nd glass)
            Element::Pm => (220, 180, 195),  // pale pink
            Element::Sm => (225, 220, 165),  // pale yellow
            Element::Eu => (235, 215, 195),  // peach
            Element::Gd => (200, 215, 220),  // cool silver
            Element::Tb => (190, 220, 195),  // pale green
            Element::Dy => (220, 215, 180),  // warm silver
            Element::Ho => (220, 195, 200),  // pinkish
            Element::Er => (225, 175, 195),  // pink (real Er salts)
            Element::Tm => (175, 220, 215),  // green-blue
            Element::Yb => (215, 215, 185),  // warm silver
            Element::Lu => (215, 220, 220),  // pale silver
            Element::Hf => (175, 195, 220),  // cool blue-grey
            Element::Ta => (160, 195, 230),  // distinct cool blue
            Element::W  => (175, 190, 220),  // cool grey
            Element::Re => (180, 185, 205),  // cool grey
            Element::Os => (115, 130, 195),  // strong osmium blue
            Element::Ir => (215, 235, 245),  // bright cool silver
            Element::Pt => (230, 225, 200),  // slight warm silver
            Element::Tl => (220, 220, 210),  // pale silver
            Element::Bi => (230, 165, 180),  // pink iridescent (real)
            Element::Po => (180, 145, 120),  // warm brown
            Element::At => (110,  80, 105),  // dark muted purple
            Element::Rn => (190, 175, 230),  // pale lavender
            Element::Fr => (220, 195, 220),  // pale pink-silver
            Element::Ac => (215, 220, 190),  // warm pale
            Element::Th => (220, 215, 195),  // warm silver
            Element::Pa => (165, 175, 230),  // cool blue
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
            Element::Li => "Lithium",
            Element::Be => "Beryllium",
            Element::Sc => "Scandium",
            Element::Ti => "Titanium",
            Element::V  => "Vanadium",
            Element::Cr => "Chromium",
            Element::Mn => "Manganese",
            Element::Co => "Cobalt",
            Element::Ga => "Gallium",
            Element::Ge => "Germanium",
            Element::As => "Arsenic",
            Element::Se => "Selenium",
            Element::Br => "Bromine",
            Element::Kr => "Krypton",
            Element::Rb => "Rubidium",
            Element::Sr => "Strontium",
            Element::Y  => "Yttrium",
            Element::Zr => "Zirconium",
            Element::Nb => "Niobium",
            Element::Mo => "Molybdenum",
            Element::Tc => "Technetium",
            Element::Ru => "Ruthenium",
            Element::Rh => "Rhodium",
            Element::Pd => "Palladium",
            Element::Cd => "Cadmium",
            Element::In => "Indium",
            Element::Sn => "Tin",
            Element::Sb => "Antimony",
            Element::Te => "Tellurium",
            Element::I  => "Iodine",
            Element::Xe => "Xenon",
            Element::Ba => "Barium",
            Element::La => "Lanthanum",
            Element::Ce => "Cerium",
            Element::Pr => "Praseodymium",
            Element::Nd => "Neodymium",
            Element::Pm => "Promethium",
            Element::Sm => "Samarium",
            Element::Eu => "Europium",
            Element::Gd => "Gadolinium",
            Element::Tb => "Terbium",
            Element::Dy => "Dysprosium",
            Element::Ho => "Holmium",
            Element::Er => "Erbium",
            Element::Tm => "Thulium",
            Element::Yb => "Ytterbium",
            Element::Lu => "Lutetium",
            Element::Hf => "Hafnium",
            Element::Ta => "Tantalum",
            Element::W  => "Tungsten",
            Element::Re => "Rhenium",
            Element::Os => "Osmium",
            Element::Ir => "Iridium",
            Element::Pt => "Platinum",
            Element::Tl => "Thallium",
            Element::Bi => "Bismuth",
            Element::Po => "Polonium",
            Element::At => "Astatine",
            Element::Rn => "Radon",
            Element::Fr => "Francium",
            Element::Ac => "Actinium",
            Element::Th => "Thorium",
            Element::Pa => "Protactinium",
        }
    }

}

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
    // Identity of the absorbed liquid (Water, Br, future acids/etc.).
    // Default Empty (= dry). Set when a porous solid first absorbs from
    // a liquid neighbor; reset to Empty when moisture drains to 0.
    // Reactions that care about WHICH liquid is in a cell (halogen
    // displacement through a salt pile, future acid-on-metal) gate on
    // this field. Wicking won't mix incompatible liquids — flow stops
    // at the boundary between water-wet and Br-wet cells.
    pub moisture_el: Element,
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
        moisture_el: Element::Empty,
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
            moisture_el: Element::Empty,
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
    pub pressure_scratch: Vec<i16>,
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
    // Per-frame chemistry-blast accumulator. Every detonation-class
    // reaction (delta_temp ≥ 1200) pushes its energy + position into
    // this bucket; flush_chem_blast at the end of chemical_reactions
    // emits ONE shockwave at the energy-weighted centroid with
    // sublinear yield scaling. Replaces the old per-reaction shockwave
    // emit, which produced a ring stack as the cloud chain-reacted
    // across many frames.
    chem_blast_x: f32,
    chem_blast_y: f32,
    chem_blast_energy: f32,
    chem_blast_count: u32,
    // Per-element count in `cells`, refreshed at the start of each
    // step(). Lets systems early-out in O(1) when their required
    // element isn't on the grid — sand-only scenes skip ~10 systems
    // entirely instead of full-grid scanning each one for nothing.
    pub present_count: [u32; ELEMENT_COUNT],
    // True if any cell's temp exceeds the bloom threshold (500°C).
    // Refreshed alongside `present_count`. Used by the render path
    // to gate the bloom GPU passes — sand-only scenes have no hot
    // cells, so uploading the bright sidecar + running 2 blur
    // passes for an all-zero image is pure waste.
    pub any_hot: bool,
    // True if any cell is currently in the Gas phase (via
    // cell_physics, which is phase-aware) — covers static gases
    // like Steam/CO2 and derived gas compounds like HCl plus boiled
    // metal vapors. The render path uses this to gate the volumetric
    // gas-cloud blur.
    pub any_gas: bool,
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
            pressure_scratch: vec![0; W * H],
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
            chem_blast_x: 0.0,
            chem_blast_y: 0.0,
            chem_blast_energy: 0.0,
            chem_blast_count: 0,
            present_count: [0; ELEMENT_COUNT],
            any_hot: false,
            any_gas: false,
        }
    }

    // Cheap O(N) presence rebuild: zeros the count array, then walks
    // every cell once. Run at the start of step() so each system can
    // gate on `self.has(Element::X)` before doing any work. The count
    // is "as of start of this frame" — systems that create new
    // elements mid-step (thermal igniting Wood into Fire) will be
    // visible to next frame's gates, which is fine for visual systems
    // and acceptable for chemistry chains (one-frame lag).
    #[inline]
    fn has(&self, el: Element) -> bool {
        self.present_count[el as usize] > 0
    }
    // True if any element with a chemistry profile (atom 18..ELEMENT_COUNT,
    // or Water/Ice/Steam/Oil) is on the grid. Gates `chemical_reactions`.
    #[inline]
    fn has_reactive_chem(&self) -> bool {
        for el_id in 18..ELEMENT_COUNT {
            if self.present_count[el_id] > 0 { return true; }
        }
        self.present_count[Element::Water as usize] > 0
            || self.present_count[Element::Ice as usize] > 0
            || self.present_count[Element::Steam as usize] > 0
            || self.present_count[Element::Oil as usize] > 0
    }
    // Total atomic-metal cells on the grid. Gates alloy / acid /
    // base systems that all require at least one metal to do
    // anything. Hardcoded list — these are the 16 paintable atomic
    // metals; updating it costs nothing per frame and beats a
    // dynamic is_atomic_metal probe inside the loop.
    #[inline]
    fn atomic_metal_count(&self) -> u32 {
        // Includes every element categorized AlkaliMetal, AlkalineEarth,
        // TransitionMetal, PostTransition, Lanthanide, Actinide. Updated
        // to cover the periodic-table fill — galvanic detection now sees
        // any pair of the new metals as candidates.
        const METALS: [Element; 67] = [
            // Originals
            Element::Na, Element::Mg, Element::Al, Element::K, Element::Ca,
            Element::Fe, Element::Cu, Element::Au, Element::Hg, Element::U,
            Element::Zn, Element::Ag, Element::Ni, Element::Pb, Element::Ra,
            Element::Cs,
            // Period 2/4 fill
            Element::Li, Element::Be,
            Element::Sc, Element::Ti, Element::V, Element::Cr, Element::Mn,
            Element::Co, Element::Ga,
            // Period 5
            Element::Rb, Element::Sr, Element::Y, Element::Zr, Element::Nb,
            Element::Mo, Element::Tc, Element::Ru, Element::Rh, Element::Pd,
            Element::Cd, Element::In, Element::Sn,
            // Period 6 main + lanthanides
            Element::Ba,
            Element::La, Element::Ce, Element::Pr, Element::Nd, Element::Pm,
            Element::Sm, Element::Eu, Element::Gd, Element::Tb, Element::Dy,
            Element::Ho, Element::Er, Element::Tm, Element::Yb, Element::Lu,
            Element::Hf, Element::Ta, Element::W, Element::Re, Element::Os,
            Element::Ir, Element::Pt, Element::Tl, Element::Bi,
            // Period 7 + actinides
            Element::Fr, Element::Ac, Element::Th, Element::Pa,
        ];
        let mut total = 0u32;
        for &m in METALS.iter() {
            total += self.present_count[m as usize];
        }
        total
    }
    fn refresh_presence(&mut self) {
        for v in self.present_count.iter_mut() { *v = 0; }
        let mut any_hot = false;
        let mut any_gas = false;
        for c in &self.cells {
            self.present_count[c.el as usize] += 1;
            if c.temp > 500 { any_hot = true; }
            if cell_physics(*c).kind == Kind::Gas { any_gas = true; }
        }
        self.any_hot = any_hot;
        self.any_gas = any_gas;
    }
    // True if anything on the grid currently glows: hot cells
    // (temp > 500), Fire, Lava, or energized noble gases (active
    // EMF > 0 means a circuit is closed and noble-gas glow_color
    // pixels light up). Gates the bloom GPU passes.
    pub fn has_emission(&self) -> bool {
        self.has(Element::Fire)
            || self.has(Element::Lava)
            || self.active_emf > 0.0
            || self.any_hot
    }
    // True if any cell is currently in the Gas phase — covers
    // static gases (Steam/CO2/H/etc.), derived gas compounds (HCl,
    // NH3 …) and boiled-off vapors. Gates the volumetric blur
    // passes; without this the gas pipeline would freeze for
    // anything that isn't hard-coded as a static gas element.
    pub fn has_any_gas(&self) -> bool { self.any_gas }

    pub fn spawn_shockwave(&mut self, cx: i32, cy: i32, yield_p: f32) {
        self.spawn_shockwave_capped(cx, cy, yield_p, 50000.0);
    }

    // Dump the current cell grid + ambient offset to disk. Format is a
    // tiny custom binary: 4-byte magic, u32 cell count, i16 ambient,
    // then the raw Cell bytes. No serde/bincode — the file is tied to
    // this exact build's Cell layout, which is fine for "F11 save my
    // testing scene; F12 reload it" usage.
    pub fn save_state(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"ALEM")?;
        f.write_all(&(self.cells.len() as u32).to_le_bytes())?;
        f.write_all(&self.ambient_offset.to_le_bytes())?;
        let cell_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.cells.as_ptr() as *const u8,
                std::mem::size_of::<Cell>() * self.cells.len(),
            )
        };
        f.write_all(cell_bytes)?;
        Ok(())
    }

    // Restore a saved scene. Replaces cells and ambient_offset, clears
    // transient runtime state (shockwaves, history, chem-blast bucket)
    // so the loaded scene starts fresh rather than inheriting whatever
    // was mid-flight. Refuses to load if magic/length doesn't match.
    pub fn load_state(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::{Read, ErrorKind};
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"ALEM" {
            return Err(std::io::Error::new(ErrorKind::InvalidData, "bad magic"));
        }
        let mut cnt_buf = [0u8; 4];
        f.read_exact(&mut cnt_buf)?;
        let cnt = u32::from_le_bytes(cnt_buf) as usize;
        if cnt != self.cells.len() {
            return Err(std::io::Error::new(ErrorKind::InvalidData, "cell count mismatch"));
        }
        let mut amb_buf = [0u8; 2];
        f.read_exact(&mut amb_buf)?;
        let ambient = i16::from_le_bytes(amb_buf);
        let bytes_needed = std::mem::size_of::<Cell>() * cnt;
        let cell_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                self.cells.as_mut_ptr() as *mut u8,
                bytes_needed,
            )
        };
        f.read_exact(cell_bytes)?;
        self.ambient_offset = ambient;
        self.shockwaves.clear();
        self.chem_blast_x = 0.0;
        self.chem_blast_y = 0.0;
        self.chem_blast_energy = 0.0;
        self.chem_blast_count = 0;
        self.history_count = 0;
        self.history_write = 0;
        self.rewind_offset = 0;
        Ok(())
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

    // Push a detonation event into this frame's chemistry-blast bucket.
    // Position is energy-weighted so the eventual centroid sits at the
    // mass of the reaction, not at whichever cell happened to fire first.
    fn add_chem_blast(&mut self, x: i32, y: i32, energy: f32) {
        if energy <= 0.0 { return; }
        self.chem_blast_x += x as f32 * energy;
        self.chem_blast_y += y as f32 * energy;
        self.chem_blast_energy += energy;
        self.chem_blast_count += 1;
    }

    // Emit ONE shockwave for everything that detonated this frame.
    // Sublinear yield scaling (sqrt) so a 50-cell H cloud produces a
    // bigger blast than a 5-cell pocket without going hyperbolic.
    fn flush_chem_blast(&mut self) {
        if self.chem_blast_count == 0 || self.chem_blast_energy <= 0.0 {
            self.chem_blast_x = 0.0;
            self.chem_blast_y = 0.0;
            self.chem_blast_energy = 0.0;
            self.chem_blast_count = 0;
            return;
        }
        let cx = (self.chem_blast_x / self.chem_blast_energy).round() as i32;
        let cy = (self.chem_blast_y / self.chem_blast_energy).round() as i32;
        // Punchier yield curve. powf(0.58) gives bigger clouds more
        // authority than sqrt without going fully linear; the 2500
        // floor makes even small ignitions hit harder than the old
        // 1200 + sqrt*140 baseline. 180k cap holds back the absurd
        // upper end while still letting a full H/O room dwarf a small
        // pocket.
        let e = self.chem_blast_energy;
        let yield_p = (2500.0 + e.powf(0.58) * 105.0).min(180_000.0);
        self.spawn_shockwave(cx, cy, yield_p);
        self.chem_blast_x = 0.0;
        self.chem_blast_y = 0.0;
        self.chem_blast_energy = 0.0;
        self.chem_blast_count = 0;
    }

    // Push neighbor H/O cells over their activation threshold so an
    // ignition front actually consumes the cloud instead of crawling
    // cell-by-cell over many frames. Only targets H and O (so stone
    // walls and unrelated cells aren't heated), and plants Fire in a
    // fraction of nearby Empty cells for visible flame. The temp bump
    // is loud enough (+450) that even a 220°C ambient cloud crosses
    // the H+O activation in one tick once the flame front arrives.
    fn flash_ignite_h_o_neighbors(&mut self, cx: i32, cy: i32, radius: i32) {
        let r2 = radius * radius;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if !Self::in_bounds(x, y) { continue; }
                let i = Self::idx(x, y);
                let el = self.cells[i].el;
                if matches!(el, Element::H | Element::O) {
                    let t = self.cells[i].temp as i32 + 450;
                    self.cells[i].temp = t.clamp(-273, 5000) as i16;
                } else if el == Element::Empty
                    && rand::gen_range::<f32>(0.0, 1.0) < 0.18
                {
                    let mut f = Cell::new(Element::Fire);
                    f.temp = 1600;
                    f.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = f;
                }
            }
        }
    }

    // Local pressure shove right at the reaction site. Complements the
    // expanding shockwave ring with an immediate "gut punch" — matter
    // adjacent to the detonation gets accelerated outward this frame
    // instead of waiting for the wave's annulus to sweep over it.
    // Pressure falls off linearly with squared-distance / r².
    fn inject_pressure_disc(&mut self, cx: i32, cy: i32, radius: i32, amount: i16) {
        let r2 = (radius * radius).max(1);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist2 = dx * dx + dy * dy;
                if dist2 > r2 { continue; }
                let x = cx + dx;
                let y = cy + dy;
                if !Self::in_bounds(x, y) { continue; }
                let i = Self::idx(x, y);
                let falloff = 1.0 - (dist2 as f32 / r2 as f32);
                let add = (amount as f32 * falloff).round() as i32;
                let p = self.cells[i].pressure as i32 + add;
                self.cells[i].pressure =
                    p.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
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
            // Scan the annulus. Use a bounding box for efficiency.
            let r_out = new_r.ceil() as i32 + 1;
            let cx = s.cx as i32;
            let cy = s.cy as i32;
            for dy in -r_out..=r_out {
                for dx in -r_out..=r_out {
                    let d2 = (dx * dx + dy * dy) as f32;
                    let d = d2.sqrt();
                    if d < old_r || d > new_r { continue; }
                    let x = cx + dx;
                    let y = cy + dy;
                    if !Self::in_bounds(x, y) { continue; }
                    self.apply_shockwave_at(x, y, dx, dy, magnitude);
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
            Cell { el: Element::Stone, derived_id: 0, life: 0, seed: 0, flag: 1, temp: 20, moisture: 0, moisture_el: Element::Empty, burn: 0, pressure: 0, solute_el: Element::Empty, solute_amt: 0, solute_derived_id: 0 }
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
    // 3×3 neighborhood always counts (returns 1.0+); otherwise the
    // ambient atmosphere can supply O — but only if there's actual
    // air contact (an Empty cell in 3×3). Cells fully surrounded by
    // water/stone/etc. are sealed off from the atmosphere and get 0.
    // This is what makes white-P storage under water actually work
    // (submerged P sees no O₂ → ignite gate fails) and prevents
    // cell-deep buried materials from spontaneously combusting.
    fn oxygen_available(&self, x: i32, y: i32) -> f32 {
        let mut has_air = false;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x + dx;
                let ny = y + dy;
                if !Self::in_bounds(nx, ny) { continue; }
                let n = self.cells[Self::idx(nx, ny)].el;
                if n == Element::O {
                    return 1.0f32.max(self.ambient_oxygen);
                }
                if n == Element::Empty {
                    has_air = true;
                }
            }
        }
        if has_air {
            self.ambient_oxygen
        } else {
            // Buried — interior O₂ scales by the cell's own
            // air_permeability (porosity when packed against itself).
            // Loose materials (leaves, gunpowder) still get most of
            // the ambient supply; dense ones (wood interior, P, oil)
            // get little or none. Matches "leaves pile burns through
            // fast, log chars from the outside in".
            let porosity = self.cells[Self::idx(x, y)]
                .el.thermal().air_permeability;
            self.ambient_oxygen * porosity
        }
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

    // Generic radioactive decay for any atom with non-zero
    // half_life_frames. Each radioactive cell gets a per-frame
    // probability of decaying ≈ ln(2) / half_life_frames; on a hit
    // the cell transmutes to its decay_product and dumps decay_heat
    // into itself + cardinal neighbors. Long-half-life elements
    // (Tc, U, Th) tick rarely; short-half-life elements (Fr, Rn,
    // Po) tick fast. Bonded radioactive atoms inside derived
    // compounds don't tick — only free atomic cells do, since the
    // generic decay path can't easily express "Cl₂ now contains
    // a Tc-decay-product" mid-bond. Acceptable simplification.
    fn radioactive_decay(&mut self) {
        for i in 0..self.cells.len() {
            let c = self.cells[i];
            if c.is_updated() { continue; }
            let Some(profile) = atom_profile_for(c.el) else { continue; };
            let half_life = profile.half_life_frames;
            if half_life == 0 { continue; }
            let p = (std::f64::consts::LN_2 / half_life as f64) as f32;
            if rand::gen_range::<f32>(0.0, 1.0) > p { continue; }
            // Decay event — replace with daughter, dump decay heat.
            let decay_product = profile.decay_product;
            let decay_heat = profile.decay_heat;
            let prev_temp = c.temp;
            let mut daughter = Cell::new(decay_product);
            daughter.temp = prev_temp.saturating_add(decay_heat);
            daughter.flag |= Cell::FLAG_UPDATED;
            self.cells[i] = daughter;
            if decay_heat > 0 {
                let x = (i % W) as i32;
                let y = (i / W) as i32;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    let bump = decay_heat / 2;
                    self.cells[ni].temp = self.cells[ni].temp.saturating_add(bump);
                }
            }
        }
    }

    fn decay(&mut self) {
        if !self.has(Element::U) { return; }
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
        // 0.003 (down from 0.015) — plating per cathode-cell per
        // adjacent-brine per frame compounds across the whole electrode
        // surface, so even small per-frame odds build up to fast visible
        // deposition. Real electroplating is gradual (a clean layer in
        // minutes, not seconds); 1/5 the prior rate slows the visible
        // build to match.
        const PLATE_P: f32 = 0.001;
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
        // Galvanic cells are real-world weak current sources — a Cu/Zn
        // cell pushes ~tens of mA, not the amps a paint-on battery
        // does. Without limiting, V² × resistance heats brine fast
        // enough to boil the electrolyte in seconds (especially for
        // wider EN gaps like Zn/Au at 71V). Real galvanic experiments
        // run for hours on the same beaker without warming visibly.
        // Scale heating by 0.05× when galvanic is the active source so
        // the brine survives long enough to plate / show solute tints
        // without boiling away.
        let galvanic_mode = self.galvanic_voltage > 0.0;
        let mode_factor: f32 = if galvanic_mode { 0.05 } else { 1.0 };
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
            let delta = v2 * resistance * K * factor * mode_factor;
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
        // No battery and no possible galvanic cell — nothing on the
        // grid can drive current, so all the seed/BFS scans are a
        // no-op. Reset state cheaply and bail. (Galvanic needs at
        // least 2 different metals on Water; we approximate with
        // metals + Water both present, then let the inner logic
        // confirm. With sand-only neither is true.)
        let battery = self.has(Element::BattPos) || self.has(Element::BattNeg);
        let galvanic_possible = self.has(Element::Water)
            && self.atomic_metal_count() >= 2;
        if !battery && !galvanic_possible {
            for v in self.energized.iter_mut() { *v = false; }
            for v in self.cathode_mask.iter_mut() { *v = false; }
            for v in self.anode_mask.iter_mut() { *v = false; }
            self.galvanic_voltage = 0.0;
            self.active_emf = 0.0;
            self.galvanic_cathode_el = None;
            self.galvanic_anode_el = None;
            return;
        }
        for v in self.energized.iter_mut() { *v = false; }
        for v in self.cathode_mask.iter_mut() { *v = false; }
        for v in self.anode_mask.iter_mut() { *v = false; }
        self.galvanic_voltage = 0.0;
        self.active_emf = 0.0;
        self.galvanic_cathode_el = None;
        self.galvanic_anode_el = None;
        // Seed lists. Battery and galvanic seeds are tracked separately
        // so the two circuit types verify and energize independently —
        // a battery prefab in one chamber won't poison the galvanic
        // detection in another chamber, and vice versa.
        let mut batt_pos: Vec<(i32, i32)> = Vec::new();
        let mut batt_neg: Vec<(i32, i32)> = Vec::new();
        for i in 0..self.cells.len() {
            match self.cells[i].el {
                Element::BattPos => batt_pos.push(((i % W) as i32, (i / W) as i32)),
                Element::BattNeg => batt_neg.push(((i % W) as i32, (i / W) as i32)),
                _ => {}
            }
        }
        // Galvanic seeds are produced inside the galvanic block below.
        // Default empty so we can fall through if galvanic doesn't fire.
        let mut galvanic_seeds: (Vec<(i32, i32)>, Vec<(i32, i32)>) = (Vec::new(), Vec::new());
        // Galvanic cell detection. Two distinct metals touching the same
        // electrolyte drive a flood from the more-reactive (lowest EN →
        // anode / BattNeg) to the less-reactive (highest EN → cathode /
        // BattPos). Voltage scales with the EN gap so Cu/Zn-ish pairs
        // give ~1V while Cu/Na-ish pairs give more.
        //
        // Closed-loop check runs ONLY against galvanic seeds, not any
        // battery seeds that may be elsewhere on the grid. Without
        // this isolation, a battery prefab's seeds would mix with the
        // galvanic seeds in the same BFS check and a wonky battery
        // loop could fail-then-reset galvanic_voltage even though
        // the galvanic circuit is fine on its own. The two circuit
        // types must verify themselves independently.
        if galvanic_possible {
            // (element, (x, y), electronegativity). Any metal cell — loose
            // paint or a frozen wire — with a brine neighbor qualifies.
            let mut candidates: Vec<(Element, (i32, i32), f32)> = Vec::new();
            let n = W * H;
            let mut candidate_at: Vec<Option<usize>> = vec![None; n];
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
                    let nidx = Self::idx(nx, ny);
                    let nn = self.cells[nidx];
                    if nn.el == Element::Water && nn.solute_amt > 20 {
                        on_brine = true;
                        break;
                    }
                }
                if !on_brine { continue; }
                let en = atom_profile_for(c.el).map(|a| a.electronegativity).unwrap_or(0.0);
                candidate_at[i] = Some(candidates.len());
                candidates.push((c.el, (x, y), en));
            }
            // Per-connected-component galvanic evaluation. Candidates in
            // separate electrolyte pools (e.g. galvanic cell in one
            // chamber, battery prefab metals in another) are physically
            // unconnected — their EN values must NOT be compared as
            // potential electrode pairs. The previous global-min/max
            // approach silently mixed metals across chambers, producing
            // phantom electrode pairs that then failed the closed-loop
            // check and reset galvanic_voltage to 0. By BFS-walking
            // each connected component (external conductor path only,
            // skipping brine) and evaluating min/max EN within each,
            // unrelated chambers can't contaminate each other.
            let mut gal_pos: Vec<(i32, i32)> = Vec::new();
            let mut gal_neg: Vec<(i32, i32)> = Vec::new();
            let mut best_gap: f32 = 0.0;
            let mut visited_global = vec![false; n];
            for start_idx in 0..candidates.len() {
                let (_, start_pos, _) = candidates[start_idx];
                let start_cell_idx = Self::idx(start_pos.0, start_pos.1);
                if visited_global[start_cell_idx] { continue; }
                // BFS the connected component reachable from this
                // candidate via external conductors (skip Water).
                // Collect any candidates encountered in the component.
                let mut comp_candidates: Vec<usize> = Vec::new();
                self.energized_queue.clear();
                visited_global[start_cell_idx] = true;
                self.energized_queue.push(start_pos);
                while let Some((cx, cy)) = self.energized_queue.pop() {
                    let here_idx = Self::idx(cx, cy);
                    if let Some(ci) = candidate_at[here_idx] {
                        comp_candidates.push(ci);
                    }
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if !Self::in_bounds(nx, ny) { continue; }
                        let ni = Self::idx(nx, ny);
                        if visited_global[ni] { continue; }
                        let nc = self.cells[ni];
                        if nc.el == Element::Water { continue; }
                        if nc.conductivity() > 0.02
                            || nc.el.electrical().glow_color.is_some()
                        {
                            visited_global[ni] = true;
                            self.energized_queue.push((nx, ny));
                        }
                    }
                }
                if comp_candidates.len() < 2 { continue; }
                // Min/max EN within this component only.
                let lo = comp_candidates.iter()
                    .map(|&i| candidates[i].2)
                    .fold(f32::INFINITY, f32::min);
                let hi = comp_candidates.iter()
                    .map(|&i| candidates[i].2)
                    .fold(f32::NEG_INFINITY, f32::max);
                let gap = hi - lo;
                if gap <= 0.05 { continue; }
                if gap <= best_gap { continue; }
                // This component beats the current best — adopt its
                // electrodes as the active galvanic pair.
                best_gap = gap;
                gal_pos.clear();
                gal_neg.clear();
                let mut cathode_el: Option<Element> = None;
                let mut anode_el: Option<Element> = None;
                for &ci in &comp_candidates {
                    let (el, pos, en) = candidates[ci];
                    if (en - lo).abs() < 1e-3 {
                        gal_neg.push(pos);
                        anode_el = Some(el);
                    }
                    if (en - hi).abs() < 1e-3 {
                        gal_pos.push(pos);
                        cathode_el = Some(el);
                    }
                }
                self.galvanic_cathode_el = cathode_el;
                self.galvanic_anode_el = anode_el;
            }
            if best_gap > 0.0 {
                // Scale EN gap (Pauling units, typically 0.2–2.0) into a
                // usable voltage. Cap to avoid melting wires the instant
                // Na/Au touches brine.
                self.galvanic_voltage = (best_gap * 80.0).clamp(10.0, 250.0);
            }
            galvanic_seeds = (gal_pos, gal_neg);
        }
        // Pick the active circuit source. Galvanic wins if it has a
        // verified closed loop (galvanic_voltage > 0); otherwise the
        // battery (if present) drives. Battery and galvanic don't
        // combine — a battery prefab in one chamber doesn't bleed
        // current into a galvanic cell in another, and vice versa.
        // (Future: per-circuit isolated passes for both to coexist
        // independently. For now, single active source.)
        let (pos_seeds, neg_seeds) = if self.galvanic_voltage > 0.0 {
            self.active_emf = self.galvanic_voltage;
            galvanic_seeds
        } else if !batt_pos.is_empty() && !batt_neg.is_empty() {
            self.active_emf = self.battery_voltage;
            (batt_pos, batt_neg)
        } else {
            return;
        };
        if pos_seeds.is_empty() || neg_seeds.is_empty() { return; }
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
        self.frame = self.frame.wrapping_add(1);
        // Clear per-frame UPDATED bit; preserve FROZEN + phase.
        for c in self.cells.iter_mut() { c.flag &= !Cell::FLAG_UPDATED; }
        // Refresh per-element presence so each system can gate on
        // "is my required element on the grid" before scanning.
        self.refresh_presence();
        if wind.length_squared() > 0.0001 {
            self.compute_wind_exposure();
        }
        self.compute_energized();
        self.joule_heating();
        self.electrolysis();
        self.decay();
        self.radioactive_decay();
        self.tree_support_check();
        // Thermite runs BEFORE thermal so it can claim Rust+Al pairs
        // before the thermal pass decomposes Rust back to Fe + O at
        // 1538°C. Without this ordering, the user's ignition heat
        // melts Rust into free Fe, which then alloys with the Al
        // (AlFe) instead of running the thermite redox.
        self.thermite();
        // Magnesium combustion runs alongside thermite — Mg burns
        // brilliant white in air and is the canonical thermite fuse.
        self.magnesium_burn();
        // F + Glass etching — fluorine eats glass into SiF + O.
        // Bespoke because exposing Glass to the general chemistry
        // engine would also make it react with O and metals.
        self.glass_etching();
        // F + Water hydrolysis — F splits water into HF + O. Dedicated
        // pass because Water's chemistry face (O-valence 6) blocks the
        // generic engine from running F-as-acceptor here.
        self.fluorine_hydrolysis();
        self.metal_hydrogen_absorption();
        // Halogen displacement — F kicks Cl out of chloride salts.
        self.halogen_displacement();
        // Hg amalgamation — Hg dissolves Au/Ag/Cu/Na/etc. into a
        // liquid amalgam alloy. Bypasses the both-cells-liquid gate
        // of regular alloy_formation since the dissolved metal is
        // typically solid at room temp.
        self.hg_amalgamation();
        // Flame-test emission — heated metal salts emit colored flame.
        // Runs BEFORE color_fires so the new emitted Fire cells (which
        // are spawned with their solute_el already set) don't need
        // re-coloring this tick.
        self.flame_test_emission();
        // Flame-color inheritance — Fire cells adjacent to metal salts
        // pick up the metal's characteristic flame color.
        self.color_fires();
        self.thermal();
        self.chemical_reactions();
        self.flush_chem_blast();
        self.acid_displacement();
        self.alloy_acid_leach();
        self.base_neutralization();
        self.alloy_formation();
        self.dissolve();
        self.diffuse_solute();
        self.reactions();
        for y in (0..H as i32).rev() {
            let lr = self.frame % 2 == 0;
            for i in 0..W as i32 {
                let x = if lr { i } else { W as i32 - 1 - i };
                self.update_cell(x, y, wind);
            }
        }
        self.pressure_sources();
        self.pressure();
        self.tick_shockwaves();
        self.snapshot();
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
        // 1) Diffusion + ambient (double-buffered, parallelized over rows).
        // Pure read of `cells` + write to disjoint `temp_scratch` slots —
        // race-free under rayon. Stochastic rounding uses a deterministic
        // per-cell-and-frame hash instead of a thread-local RNG so the
        // result is reproducible (rewind-determinism tests rely on it)
        // and fast (no atomic-RNG contention across worker threads).
        use rayon::prelude::*;
        let cells = &self.cells;
        let ambient_offset = self.ambient_offset;
        let frame = self.frame;
        self.temp_scratch.par_chunks_mut(W).enumerate().for_each(|(yu, row)| {
            let y = yu as i32;
            for xu in 0..W {
                let x = xu as i32;
                let i = yu * W + xu;
                let c = cells[i];
                let my_k = c.el.thermal().conductivity;
                let mut delta: f32 = 0.0;
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || nx >= W as i32 || ny < 0 || ny >= H as i32 { continue; }
                    let ni = (ny as usize) * W + nx as usize;
                    let n = cells[ni];
                    let k = my_k.min(n.el.thermal().conductivity);
                    delta += k * (n.temp as f32 - c.temp as f32);
                }
                // Exposure: how many neighbors are a *different* material.
                // A cell fully surrounded by its own kind is insulated from
                // the environment — interior of a lava pool keeps its heat,
                // only boundary cells actually radiate.
                let exposure = if matches!(c.el, Element::Fire | Element::Empty) {
                    1.0
                } else {
                    let mut diff = 0.0f32;
                    for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || nx >= W as i32 || ny < 0 || ny >= H as i32 {
                            diff += 1.0;
                            continue;
                        }
                        let ni = (ny as usize) * W + nx as usize;
                        if cells[ni].el != c.el { diff += 1.0; }
                    }
                    diff / 4.0
                };
                // 10% baseline even for fully-interior cells — nothing is
                // perfectly insulated, and it lets slow gradients develop.
                let amb_factor = 0.10 + 0.90 * exposure;
                let ambient_t = c.el.thermal().ambient_temp as i32 + ambient_offset as i32;
                delta += c.el.thermal().ambient_rate * amb_factor
                    * (ambient_t as f32 - c.temp as f32);
                // Stochastic rounding: temp is an integer, but per-frame
                // changes can be well under 1°. Round up with probability
                // equal to the fractional part. Using a (frame, i) hash
                // instead of rand::gen_range to keep the closure thread-
                // safe and deterministic.
                let exact = c.temp as f32 + delta / c.el.thermal().heat_capacity;
                let floor = exact.floor();
                let frac = exact - floor;
                let h = frame.wrapping_mul(0x9E3779B1)
                    .wrapping_add((i as u32).wrapping_mul(0x85EBCA77));
                let h = h ^ (h >> 13);
                let roll = (h & 0xFFFF) as f32 / 65536.0;
                let stepped = if roll < frac { floor + 1.0 } else { floor };
                // Absolute zero is -273°C — the physical floor. Upper limit
                // stays at 4000°C, comfortably above any metal's boiling
                // point we'd plausibly simulate.
                let new_t = stepped.clamp(-273.0, 4000.0) as i16;
                row[xu] = new_t;
            }
        });
        for i in 0..(W * H) {
            self.cells[i].temp = self.temp_scratch[i];
        }

        // 2) Moisture dynamics — wetting from water contact, heat-driven and
        // passive evaporation *only on cells touching air* (surface-first).
        //
        // Derived compound cells in solid phases (Gravel/Powder/Solid) absorb
        // and wick moisture too — without this, hydration-shifting compounds
        // (CoCl₂, CuCl) sit in water with moisture=0 and never trigger their
        // wet-color shift. Liquid/gas-phase derived cells skip absorption
        // because molten salts and HCl gas shouldn't behave like wet sand.
        // We use the static MoistureProfile (which is_sink=false for the
        // Element::Derived stub) as the base, then opt-in solid derived
        // cells via runtime kind check.
        let cell_absorbs = |cell: Cell| -> bool {
            if cell.el.moisture().is_sink { return true; }
            if cell.el == Element::Derived {
                let kind = cell_physics(cell).kind;
                return matches!(kind, Kind::Solid | Kind::Gravel | Kind::Powder);
            }
            false
        };
        let cell_moisture_conductivity = |cell: Cell| -> f32 {
            let k = cell.el.moisture().conductivity;
            if k > 0.0 { return k; }
            // Default conductivity for derived solids — matches Sand's
            // 0.08 ballpark so a CuCl pile in water saturates over a
            // few seconds rather than only the surface layer.
            if cell.el == Element::Derived {
                let kind = cell_physics(cell).kind;
                if matches!(kind, Kind::Solid | Kind::Gravel | Kind::Powder) {
                    return 0.06;
                }
            }
            0.0
        };
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el == Element::Water || c.el == Element::Empty { continue; }

                // Absorption from any adjacent liquid. Sources tag the
                // cell's `moisture_el` so reactions downstream (halogen
                // displacement, future acid-on-metal) can read what kind
                // of liquid is in the cell. Standard sources (water, ice,
                // mud) tag Water; atomic halogen liquids tag themselves.
                // Sources are NOT consumed — water remains the magical
                // infinite puddle convention; gradient-based saturation
                // (below) is what creates a visible permeation limit.
                if c.moisture < 250 && cell_absorbs(c) {
                    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let n = self.get(x + dx, y + dy);
                        let absorb_as: Option<Element> =
                            if n.el.moisture().is_source {
                                Some(Element::Water)
                            } else if matches!(n.el, Element::Br) {
                                Some(Element::Br)
                            } else {
                                None
                            };
                        let Some(liquid_id) = absorb_as else { continue; };
                        // Type-match gate: a cell already wet with one
                        // liquid won't absorb a different liquid until
                        // it dries out. Prevents incoherent mixed-liquid
                        // semantics for displacement reactions.
                        if c.moisture_el != Element::Empty
                            && c.moisture_el != liquid_id
                        {
                            continue;
                        }
                        self.cells[i].moisture =
                            self.cells[i].moisture.saturating_add(5);
                        self.cells[i].moisture_el = liquid_id;
                        break;
                    }
                }

                // Wicking: gradient-driven diffusion through solids/powders.
                // Moisture shares with drier neighbors, scaled by conductivity.
                // The 25-unit gradient threshold (was 3) is what creates the
                // visible permeation limit the user wants — adjacent cells
                // equilibrate at a ~25-step gap, so a long chain decays
                // moisture roughly 25 per cell of distance from the source.
                // After ~10 cells the moisture trails to nothing and Br
                // (or water) doesn't propagate any deeper — excess liquid
                // piles up externally rather than soaking infinitely.
                //
                // Wicking also propagates moisture_el to dry recipients so
                // a Br-soaked surface drives a Br-soaked interior. Refuses
                // type-mismatched flow so a water-wet column can't bleed
                // its moisture_el over into a Br-wet column next door.
                let c = self.cells[i];
                let my_k = cell_moisture_conductivity(c);
                if c.moisture > 5 && my_k > 0.0 && cell_absorbs(c) {
                    for (dx, dy) in [(1i32, 0), (-1, 0), (0, 1), (0, -1)] {
                        if !Self::in_bounds(x + dx, y + dy) { continue; }
                        let nidx = Self::idx(x + dx, y + dy);
                        let n = self.cells[nidx];
                        if !cell_absorbs(n) { continue; }
                        // Don't pump moisture INTO cells past boiling — water
                        // doesn't travel into a cell that's actively evaporating,
                        // otherwise wet neighbors shield hot ones indefinitely.
                        if n.temp > 100 { continue; }
                        // Type-match gate.
                        if n.moisture_el != Element::Empty
                            && n.moisture_el != c.moisture_el
                        {
                            continue;
                        }
                        let k = my_k.min(cell_moisture_conductivity(n));
                        if k <= 0.0 { continue; }
                        let cm = self.cells[i].moisture as i16;
                        let nm = n.moisture as i16;
                        let gradient = cm - nm;
                        if gradient > 25 {
                            let flow = (k * gradient as f32).round().max(1.0) as i16;
                            let amt = flow
                                .min(cm)
                                .min(255 - nm)
                                .max(0) as u8;
                            if amt > 0 {
                                self.cells[i].moisture = self.cells[i].moisture.saturating_sub(amt);
                                self.cells[nidx].moisture = n.moisture.saturating_add(amt);
                                if n.moisture_el == Element::Empty {
                                    self.cells[nidx].moisture_el = c.moisture_el;
                                }
                                if self.cells[i].moisture == 0 {
                                    self.cells[i].moisture_el = Element::Empty;
                                }
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
                        if self.cells[i].moisture == 0 {
                            self.cells[i].moisture_el = Element::Empty;
                        }
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
                    if self.cells[i].moisture == 0 {
                        self.cells[i].moisture_el = Element::Empty;
                    }
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
                        // Porous flammables (leaves at 0.95, seed at 0.55,
                        // gunpowder at 0.60) chain-ignite via flame jumping
                        // through air gaps — the thermal-conduction model
                        // is too slow to capture this. Without explicit
                        // chain, a leaves pile only ignites cells in
                        // direct torch contact, then those snuff out
                        // before neighbors heat past ignite_above. Result
                        // looks like "leaves won't ignite" even though
                        // each contact cell flashes briefly. Threshold
                        // 0.5 includes leaves/seed/gunpowder but excludes
                        // wood (0.15) / oil (0.05) / carbon (0.20) / P
                        // (0.05) which legitimately propagate by heat.
                        if c.el.thermal().air_permeability >= 0.5 {
                            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el == c.el
                                    && self.cells[ni].burn == 0
                                {
                                    if let Some(dur) =
                                        c.el.thermal().burn_duration
                                    {
                                        self.cells[ni].burn = dur;
                                        self.cells[ni].moisture = 0;
                                        let t_self = self.cells[i].temp;
                                        if self.cells[ni].temp < t_self {
                                            self.cells[ni].temp = t_self;
                                        }
                                    }
                                }
                            }
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
                        // Francium — same chain detonation behavior as
                        // Cs, with all energy values bumped 5% to reflect
                        // its position as the most reactive alkali.
                        // Shockwave 945 (vs Cs 900), thermal burn temp
                        // 1470 (vs Cs 1400). Ignition threshold lower
                        // (28 vs 50) is in the thermal profile, so a Fr
                        // pile pops harder and earlier than a Cs pile.
                        if c.el == Element::Fr {
                            self.spawn_shockwave(x, y, 945.0);
                            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el == Element::Fr
                                    && self.cells[ni].burn == 0
                                {
                                    if let Some(dur) =
                                        Element::Fr.thermal().burn_duration
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
                            // Fuel consumed. Wood leaves carbon residue
                            // ~30% of the time (a real fire drops ash/
                            // char, which is mostly elemental carbon);
                            // otherwise the cell becomes hot smoke.
                            // Other flammables (leaves/oil/seed) vanish
                            // to smoke only — no substantial solid left.
                            let leaves_char =
                                c.el == Element::Wood && rand::gen_range::<u8>(0, 10) < 3;
                            if leaves_char {
                                let mut ch = Cell::new(Element::C);
                                ch.temp = 450;
                                self.cells[i] = ch;
                            } else if c.el == Element::H {
                                // Hydrogen combustion produces water
                                // (2H₂ + O₂ → 2H₂O), NOT CO2 — that's
                                // a carbon-combustion fallback that
                                // doesn't apply here. The bespoke
                                // (H, O) → Water chemistry path is
                                // blocked during burn by the burn-skip
                                // gate, so without this explicit
                                // override the H burn-end falls
                                // through to the generic CO2 cell and
                                // a hot H cloud "burns to smoke"
                                // instead of leaving steam behind.
                                let mut w = Cell::new(Element::Water);
                                w.temp = 500;
                                self.cells[i] = w;
                            } else if c.el == Element::P {
                                // Phosphorus burns to P₂O₅ (white smoke
                                // in real life). Use the derived
                                // registry entry so it's the same
                                // compound the chemistry path would
                                // produce — unified identity.
                                if let Some(p2o5_id) = derive_or_lookup(
                                    Element::P, Element::O)
                                {
                                    let mut p2o5 = Cell::new(Element::Derived);
                                    p2o5.derived_id = p2o5_id;
                                    p2o5.temp = 500;
                                    self.cells[i] = p2o5;
                                } else {
                                    let mut sm = Cell::new(Element::CO2);
                                    sm.temp = 500;
                                    self.cells[i] = sm;
                                }
                            } else if c.el == Element::S {
                                // Sulfur burns to SO₃ (well, mostly SO₂
                                // in real life — but our valence math
                                // produces 1:3 stoichiometry for S+O).
                                // Either way, oxide of sulfur, not
                                // CO₂. Real S flame is the iconic blue
                                // we just added a flame_color for.
                                if let Some(sox_id) = derive_or_lookup(
                                    Element::S, Element::O)
                                {
                                    let mut sox = Cell::new(Element::Derived);
                                    sox.derived_id = sox_id;
                                    sox.temp = 500;
                                    self.cells[i] = sox;
                                } else {
                                    let mut sm = Cell::new(Element::CO2);
                                    sm.temp = 500;
                                    self.cells[i] = sm;
                                }
                            } else if is_atomic_metal(c.el) {
                                // Burning metals leave their oxide as
                                // residue (Ca → CaO, Li → Li₂O, etc.)
                                // — NOT CO₂. Bulk transition metals
                                // (Ti, Be, Sc) don't ignite; oxide
                                // formation for those goes through the
                                // chemistry pass instead, see
                                // oxide_decomposition_threshold.
                                //
                                // Alkali metals: combustion in air is
                                // violent and produces an aerosol fume
                                // (Cs₂O / K₂O / Na₂O smoke) that drifts
                                // away rather than settling as a solid
                                // pile — visible as a quick puff in
                                // real life, then mostly nothing left.
                                // Without a dispersal pass, every
                                // single burning Cs cell creates a
                                // persistent Cs₂O cell, leaving an
                                // ungodly amount of oxide residue
                                // where there should be a small pile
                                // and a puff of smoke. Tier dispersal
                                // by reactivity: the hyperreactive
                                // alkalis (Cs/Fr) lose ~95% to
                                // aerosol; K/Rb ~85%; Na ~70%; Li
                                // ~50% (less violent burn, more
                                // residue plausible).
                                let disperse_p: f32 = match c.el {
                                    // Alkali. Real combustion is mostly
                                    // aerosol — visible smoke cloud,
                                    // not a solid oxide pile. Heavy
                                    // dispersal across the tier.
                                    Element::Cs | Element::Fr => 0.99,
                                    Element::K  | Element::Rb => 0.975,
                                    Element::Na               => 0.96,
                                    Element::Li               => 0.925,
                                    // Alkaline earth. Mg ribbon
                                    // photoflash vaporizes nearly
                                    // completely; Sr/Ba/Ra firework
                                    // burns produce a hot fume cloud
                                    // with a fraction of glowing
                                    // oxide particles suspended in it
                                    // (the firework color comes from
                                    // the metal in the gas phase, not
                                    // settled solid). Ca slightly
                                    // less violent. Be bulk-non-
                                    // combustible already.
                                    Element::Mg                       => 0.975,
                                    Element::Ca                       => 0.90,
                                    Element::Sr | Element::Ba
                                    | Element::Ra                     => 0.925,
                                    _                                 => 0.0,
                                };
                                let dispersed = disperse_p > 0.0
                                    && rand::gen_range::<f32>(0.0, 1.0) < disperse_p;
                                if dispersed {
                                    // Visible aerosol/fume: real alkali combustion
                                    // produces a gray-white smoke cloud, not a
                                    // void. Spawn hot CO2 (our gas-kind smoke
                                    // element) so the dispersal reads as a
                                    // billowing cloud rather than empty space.
                                    let mut puff = Cell::new(Element::CO2);
                                    puff.temp = 500;
                                    self.cells[i] = puff;
                                } else if let Some(oxide_id) = derive_or_lookup(
                                    c.el, Element::O)
                                {
                                    let mut ox = Cell::new(Element::Derived);
                                    ox.derived_id = oxide_id;
                                    ox.temp = 500;
                                    self.cells[i] = ox;
                                } else {
                                    let mut sm = Cell::new(Element::CO2);
                                    sm.temp = 500;
                                    self.cells[i] = sm;
                                }
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
                        // Emit visible flame. Default sparse upward column
                        // (1/10 per frame, only Empty above). Porous
                        // flammables emit more often and can spread to
                        // a side cell (not all four at once — that
                        // produced a wall of fire). One Fire cell per
                        // frame at most, picking a random Empty cardinal.
                        let porosity = c.el.thermal().air_permeability;
                        if porosity >= 0.5
                            && rand::gen_range::<f32>(0.0, 1.0) < 0.40
                        {
                            let order = rand::gen_range::<u8>(0, 4);
                            for k in 0..4 {
                                let (dx, dy) = match (order + k) % 4 {
                                    0 => (0i32, -1i32),
                                    1 => (1, 0),
                                    2 => (-1, 0),
                                    _ => (0, 1),
                                };
                                let fx = x + dx;
                                let fy = y + dy;
                                if !Self::in_bounds(fx, fy) { continue; }
                                let fi = Self::idx(fx, fy);
                                if self.cells[fi].el == Element::Empty {
                                    self.cells[fi] = Cell::new(Element::Fire);
                                    break;
                                }
                            }
                        } else if self.get(x, y - 1).el == Element::Empty
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
                                // Supersaturated boil: the water leaves as
                                // steam, the solute crystallizes. Prefer
                                // crystal-in-place + steam-in-neighbor (the
                                // crystal stays where the precipitation
                                // happens, matching real salt-pan behavior
                                // where dried-out spots leave visible
                                // residue). If no empty neighbor for the
                                // steam, fall back to crystal-in-place
                                // alone (steam escapes implicitly). Old
                                // behavior was the inverse — steam in
                                // place, crystal in neighbor — which lost
                                // the precipitate when the pool was packed
                                // and read as "evaporation lost the salt."
                                let mut crystal = Cell::new(solute_el);
                                crystal.derived_id = solute_did;
                                crystal.temp = p.threshold;
                                let mut steam_placed = false;
                                for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                                    if !Self::in_bounds(x + dx, y + dy) { continue; }
                                    let ni = Self::idx(x + dx, y + dy);
                                    if self.cells[ni].el != Element::Empty { continue; }
                                    let mut steam = Cell::new(p.target);
                                    steam.temp = p.threshold + 15;
                                    self.cells[ni] = steam;
                                    steam_placed = true;
                                    break;
                                }
                                self.cells[i] = crystal;
                                let _ = steam_placed;
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
                            // Both products inherit the decomp temperature
                            // (the visible "molten/vapor metal + glowing
                            // halogen/O"). The thermodynamic gate in the
                            // chemistry pass keeps them from re-forming the
                            // parent compound while they're hot — at temps
                            // ≥ decomp threshold, formation rate is 0; just
                            // below, it scales smoothly with how cool they
                            // are. So the falling metal cools through the
                            // threshold and only starts tarnishing again
                            // once it's well below — naturally giving it
                            // time to separate from the released byproduct.
                            //
                            // Emerge-temp floor at mp + 100°C: the heat
                            // source driving decomposition keeps pumping
                            // energy in, so the metal emerges at least
                            // molten rather than instantly solid. This is
                            // why Rust → Fe looks dramatic (Rust threshold
                            // 3500°C is far above Fe mp 1538°C, so Fe
                            // erupts as liquid and cascades down). Halide
                            // thresholds (500-1500°C) are sometimes below
                            // the donor's mp — without this clamp, decomp
                            // produces an instant solid hunk and skips
                            // the molten cascade. The clamp unifies the
                            // visible behavior: every decomp eruption
                            // looks like Rust does.
                            let donor_mp = atom_profile_for(donor_el)
                                .map(|a| a.melting_point)
                                .unwrap_or(0);
                            let emerge_t = (t as i16).max(donor_mp + 100);
                            let mut d = Cell::new(donor_el);
                            d.temp = emerge_t;
                            d.flag |= Cell::FLAG_UPDATED;
                            self.cells[i] = d;
                            // Emit one byproduct (gas atom) into an adjacent
                            // empty cell. The visible "O₂ evolution" / "I₂
                            // vapor" when oxides or heavy halides break
                            // apart. Stoichiometry is lossy at single-cell
                            // granularity — that's OK for a sim. Byproduct
                            // inherits the same emerge_t so the released
                            // gas is hot enough to rise convincingly even
                            // when decomp threshold is low.
                            for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                                let nx = x + dx;
                                let ny = y + dy;
                                if !Self::in_bounds(nx, ny) { continue; }
                                let ni = Self::idx(nx, ny);
                                if self.cells[ni].el != Element::Empty { continue; }
                                let mut g = Cell::new(byproduct_el);
                                g.temp = emerge_t;
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
    fn pressure_sources(&mut self) {
        if self.pressure_scratch.len() != W * H {
            self.pressure_scratch = vec![0; W * H];
        }

        // Compute per-cell TARGET pressure into the scratch buffer.
        // Pass 1: thermal component. cell_physics is phase-aware so a
        // derived gas cell (HCl, HF, NH₃) is correctly recognized as
        // Gas-kind here — `cell.el.physics()` would return the static
        // Powder stub for Element::Derived and zero out the thermal
        // contribution.
        for i in 0..self.cells.len() {
            let cell = self.cells[i];
            let kind = cell_physics(cell).kind;
            let thermal = if matches!(kind, Kind::Gas | Kind::Fire) {
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
                    // ANY frozen cell is structural, regardless of element
                    // kind. Iron walls are Kind::Gravel, stone is Kind::Solid,
                    // sand walls are Kind::Powder — when frozen they're all
                    // rigid and bear load through contact, not fluid column
                    // pressure. The kind-based check missed iron entirely.
                    let is_wall = cell.is_frozen()
                        && !matches!(
                            cell_physics(cell).kind,
                            Kind::Empty | Kind::Gas | Kind::Fire | Kind::Liquid,
                        );
                    if is_wall {
                        // Structural walls break the hydrostatic column.
                        // Iron's density (79) would otherwise saturate col_p
                        // to 4000 within a few cells, leaving every cell
                        // below the wall targeting max pressure — which made
                        // wall cells themselves read 4000 and ambient cells
                        // below them sit in the thousands. Walls also
                        // shouldn't hold hydrostatic pressure of their own,
                        // so we skip updating their target here (it stays at
                        // thermal = 0 from pass 1, and the blend will decay
                        // any stuck pressure they inherited).
                        col_p = 0.0;
                    } else {
                        // Only RESTING solids contribute hydrostatic
                        // weight. A grain in free-fall isn't pressing on
                        // the column — even an unbroken stream of
                        // touching falling grains is in zero-internal-
                        // stress free-fall (a falling rod feels no
                        // weight along its length). Two checks:
                        //   1. is_updated() — the cell moved THIS frame
                        //      via the motion pass that ran just before
                        //      pressure_sources. A moving cell is by
                        //      definition not in static equilibrium.
                        //   2. directly-below cell is rigid — backstop
                        //      for grains that happened not to move
                        //      this frame (probabilistic gravity etc.)
                        //      but are still resting on something solid.
                        let kind = cell_physics(cell).kind;
                        let is_solid_mass = matches!(
                            kind,
                            Kind::Solid | Kind::Gravel | Kind::Powder,
                        );
                        let count_weight = if is_solid_mass {
                            if cell.is_updated() {
                                false
                            } else {
                                let by = y + 1;
                                if by >= H as i32 {
                                    true
                                } else {
                                    let below = self.cells[Self::idx(x, by)];
                                    below.is_frozen() || matches!(
                                        cell_physics(below).kind,
                                        Kind::Solid | Kind::Gravel | Kind::Powder,
                                    )
                                }
                            }
                        } else {
                            true
                        };
                        if count_weight {
                            col_p += Self::cell_weight(cell.el) * g;
                        }
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
            // Phase-aware kind: el.physics() returns Kind::Powder for
            // Element::Derived, which would mark a derived gas (HCl,
            // HF, NH₃) as non-pressurizable and blend its overpressure
            // away each frame — capping the user's hold-paint stack
            // at the 4000 clamp instead of letting it accumulate.
            let k = cell_physics(self.cells[i]).kind;
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
        // Each iteration's inner pass is parallelized over rows — pure
        // read of cells + write to disjoint pressure_scratch slots, so
        // race-free under rayon.
        use rayon::prelude::*;
        const DIFF_SCALE: i32 = 2048;
        const ITERS: usize = 6;
        for _iter in 0..ITERS {
            let cells = &self.cells;
            self.pressure_scratch.par_chunks_mut(W).enumerate().for_each(|(yu, row)| {
                let y = yu as i32;
                for xu in 0..W {
                    let x = xu as i32;
                    let i = yu * W + xu;
                    let me_perm = cell_pressure_p(cells[i]).permeability as i32;
                    let me_p = cells[i].pressure as i32;
                    // Walls (perm=0) can't diffuse — no neighbor flux would
                    // pass min_perm=0 gate anyway. Skip the 4-neighbor
                    // scan entirely and just carry pressure through.
                    if me_perm == 0 {
                        row[xu] = me_p as i16;
                        continue;
                    }
                    let mut new_p = me_p;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx;
                        let ny = y + dy;
                        // Boundary conditions:
                        //   - Horizontal out-of-bounds → open to implied
                        //     infinite atmosphere at P=0 with max permeability.
                        //     Pressure leaks out the sides; prevents the
                        //     play space from acting as a sealed box that
                        //     permanently accumulates everything painted.
                        //   - Vertical out-of-bounds → sealed (ceiling / floor).
                        let in_bounds = nx >= 0 && nx < W as i32 && ny >= 0 && ny < H as i32;
                        let (n_p, n_perm): (i32, i32) = if in_bounds {
                            let ni = (ny as usize) * W + nx as usize;
                            (
                                cells[ni].pressure as i32,
                                cell_pressure_p(cells[ni]).permeability as i32,
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
                    row[xu] = new_p as i16;
                }
            });
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
        if !self.has_reactive_chem() { return; }
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
                // Skip cells in active combustion. The burn cascade is
                // already consuming this cell (sustaining heat, emitting
                // Fire, eventually transmuting via burn-end). Letting
                // the chemistry pass also fire per-frame produces a
                // per-adjacent-O oxide cell per frame for the entire
                // burn duration — a tsunami of oxide residue (worst on
                // alkali metals where activation drops below ambient
                // and rate is high). Combustion in real chemistry is
                // a hot rapid process producing aerosol, not a per-
                // frame deposit. Burn-end handles the residue.
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
                    // Burn-skip: when either cell is in active combustion
                    // AND the pair would form a metal oxide (acceptor is
                    // O / water), skip — the burn cascade owns oxide
                    // formation in this case (alkali combustion was
                    // producing per-frame oxide tsunamis from chemistry
                    // running alongside burn-end transmute). For other
                    // products (halogen chemistry, e.g. burning S + Cl
                    // → SCl₂; burning P + Br₂ flash → PBr₃), the
                    // chemistry path SHOULD fire — those are real
                    // combustion-environment reactions distinct from
                    // oxide aerosol.
                    let either_burning = c.burn > 0 || self.cells[ni].burn > 0;
                    let acceptor_is_oxide_class = matches!(c.el,
                            Element::O | Element::Water | Element::Ice | Element::Steam)
                        || matches!(self.cells[ni].el,
                            Element::O | Element::Water | Element::Ice | Element::Steam);
                    if either_burning && acceptor_is_oxide_class { continue; }
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
                    //
                    // Hydrogen is excluded: real H₂ in air is inert at any
                    // temperature reachable by ambient drift (needs a literal
                    // spark), and the cubic temp-ramp wasn't enough to
                    // contain the cascade — once even a single trace water
                    // formed from virtual O, its delta_temp heated neighbors
                    // past the 318°C full-rate threshold and the whole H
                    // cloud detonated. The user has to paint explicit O for
                    // H+O to react.
                    let virtual_o = gas_mix_distance == 0
                        && n.el == Element::Empty
                        && ambient_o > 0.0
                        && c.el != Element::H
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
                                let cell = self.cells[pi];
                                let e = cell.el;
                                if matches!(e, Element::Water | Element::Ice
                                    | Element::Steam | Element::Salt)
                                    && cat_count < catalysts.len()
                                {
                                    catalysts[cat_count] = e;
                                    cat_count += 1;
                                }
                                // Dissolved NaCl in water counts as Salt
                                // catalyst — without this, salt water is
                                // chemically identical to plain water
                                // (since salt is stored in solute_el of
                                // a Water cell, not as a Salt cell). 32
                                // threshold filters out trace dissolved
                                // amounts so only meaningfully-salty
                                // brine triggers the salt path.
                                if e == Element::Water
                                    && cell.solute_amt >= 32
                                    && cell.solute_el == Element::Salt
                                    && cat_count < catalysts.len()
                                {
                                    catalysts[cat_count] = Element::Salt;
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
                    //
                    // Surface-oxidation override: when the hot solid is
                    // hotter than the global ambient, treat the virtual-O
                    // n_temp as the SOLID's temperature. Models the real
                    // physics where atmospheric oxygen molecules contact
                    // a hot surface and react there — the reaction site is
                    // the surface, not the bulk atmosphere. Without this,
                    // locally-heating a Ru pile to 700°C in 20°C air never
                    // fires because virtual-O reads 20°C and the activation
                    // gate min(700, 20) blocks at any threshold above 20.
                    let n_temp = if virtual_o {
                        let amb = 20i16.saturating_add(self.ambient_offset);
                        amb.max(c.temp)
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
                    //   * Ignited H+O override: once both cells are hot
                    //     enough to be in the detonation regime, drop the
                    //     gas-distance damping so the flame front rips
                    //     through the mixed cloud in 1-2 frames instead
                    //     of crawling at 1/(d+1) and stretching the
                    //     reaction across many frames (= ring stack).
                    let water_product = matches!(r.products[0].el, Element::Water)
                        || matches!(r.products[1].el, Element::Water);
                    let hot_h_o = water_product
                        && c.temp.max(n_temp) >= 350;
                    // Passivating-metal pairs (Al/Cr/Ti/V/Sc/Be/Ni/Cu/
                    // Zn/Ga/Ge/As/Se/Sr/Y/Zr/Nb/Mo/Tc/Ru + O at the
                    // 0.002 cap) bypass the virtual_o ×0.1 slowdown.
                    // The cap is already the intended final atmospheric
                    // rate; multiplying again collapses it 10× and is
                    // exactly the cap-compounding pattern the user
                    // complained about. Detect via the ProductSpec
                    // result + acceptor — if eff_n_el is O AND rate
                    // came back at ~0.002 (passivation cap), we treat
                    // virtual_o as already-handled.
                    let passivating = eff_n_el == Element::O
                        && r.rate <= 0.0021
                        && matches!(r.products[0].el, Element::Derived);
                    let mut eff_rate = if virtual_o && passivating {
                        // Already the final atmospheric rate; don't
                        // double-tap with 0.1×.
                        r.rate
                    } else if virtual_o {
                        r.rate * 0.1
                    } else if hot_h_o {
                        (r.rate * 0.75).max(0.50)
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
                    // Thermodynamic gate + rate scaling. If a derived
                    // product would form, look up its decomposition
                    // threshold. Above the threshold, the forward
                    // reaction is unfavorable — equilibrium has
                    // flipped toward decomposition (real chemistry:
                    // that's what "decomposes at 2400°C" means).
                    // Just below the threshold, formation is slow;
                    // far below, it runs at full rate. Multiplying
                    // eff_rate by a (thr - max_t)/thr factor gives
                    // smooth hysteresis around equilibrium with no
                    // arbitrary cooldown timer or special flag —
                    // hot decomp products refuse to re-form their
                    // parent compound because the math says so.
                    let max_reactant_t = c.temp.max(n.temp);
                    let mut thermo_factor: f32 = 1.0;
                    for prod in [r.products[0], r.products[1]] {
                        if prod.el != Element::Derived { continue; }
                        let thr_opt = derived_hot(prod.derived_id)
                            .and_then(|h| h.decomposes_above);
                        let Some(thr) = thr_opt else { continue; };
                        if max_reactant_t as i32 >= thr as i32 {
                            thermo_factor = 0.0;
                            break;
                        }
                        let f = (thr as f32 - max_reactant_t as f32)
                            / (thr as f32).max(1.0);
                        if f < thermo_factor { thermo_factor = f; }
                    }
                    eff_rate *= thermo_factor;
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
                    // Solute-rescue: if we're about to overwrite a water
                    // cell that's carrying solute (from prior dissolution
                    // events), offload that solute to an adjacent water
                    // cell with room before the chemistry consumes the
                    // cell into Steam / a fresh product. Without this
                    // rescue, every Mn+Water → MnO+Steam reaction
                    // destroys the solute the consumed water cell had
                    // accumulated, so saturation can never climb past
                    // the rate at which fresh residue replaces it.
                    // Same logic as boil_above's offload — preserves
                    // dissolved species across cell consumption.
                    let acceptor_was_water = matches!(
                        self.cells[ni].el,
                        Element::Water | Element::Ice | Element::Steam
                    );
                    let new_acceptor_carries_solute = matches!(
                        pb.el,
                        Element::Water | Element::Ice | Element::Steam
                    );
                    if acceptor_was_water && !new_acceptor_carries_solute {
                        let saved_amt = self.cells[ni].solute_amt;
                        let saved_el  = self.cells[ni].solute_el;
                        let saved_did = self.cells[ni].solute_derived_id;
                        if saved_amt > 0 {
                            let mut remaining = saved_amt;
                            for (sx, sy) in [(0i32, 1i32), (-1, 0), (1, 0), (0, -1)] {
                                if remaining == 0 { break; }
                                let snx = nx + sx;
                                let sny = ny + sy;
                                if !Self::in_bounds(snx, sny) { continue; }
                                let si = Self::idx(snx, sny);
                                if si == i || si == ni { continue; }
                                let sn = self.cells[si];
                                if !matches!(sn.el, Element::Water | Element::Ice) {
                                    continue;
                                }
                                if sn.solute_amt > 0
                                    && (sn.solute_el != saved_el
                                        || sn.solute_derived_id != saved_did)
                                {
                                    continue;
                                }
                                let room = 255u8.saturating_sub(sn.solute_amt);
                                let take = remaining.min(room);
                                if take == 0 { continue; }
                                self.cells[si].solute_el = saved_el;
                                self.cells[si].solute_derived_id = saved_did;
                                self.cells[si].solute_amt = sn.solute_amt + take;
                                remaining -= take;
                            }
                        }
                    }
                    // Virtual-O path: don't materialize a product in Empty.
                    if !virtual_o {
                        let acceptor_preserved = pb.el == n.el
                            && pb.derived_id == n.derived_id;
                        let dt = if acceptor_preserved { 0 } else { r.delta_temp as i32 };
                        let mut cb = Cell::new(pb.el);
                        cb.derived_id = pb.derived_id;
                        cb.temp = (n.temp as i32 + dt).clamp(-273, 5000) as i16;
                        cb.flag |= Cell::FLAG_UPDATED;
                        self.cells[ni] = cb;
                    }
                    // Optional byproduct gas — emitted into a third
                    // neighbor when the reaction defines one (slow
                    // hydrolysis releases H₂ while preserving the water
                    // cell). Strategy: prefer an Empty cardinal (above-
                    // surface or air-gap reaction); if none, walk
                    // upward through water until we find Empty air
                    // above the pool surface and emit there. Models
                    // gas bubbles rising out of solution to break at
                    // the surface. Crucially, this emits into Empty,
                    // not into Water — the water column is fully
                    // preserved, only Empty above the surface is
                    // consumed. If no path to surface (e.g. sealed
                    // container completely full of water), the H is
                    // released implicitly (lost as if vented through
                    // some unmodeled path); accepting that loss is
                    // better than consuming water.
                    if let Some(byp_el) = r.byproduct {
                        let mut emit_at: Option<usize> = None;
                        for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                            let bx = x + dx;
                            let by = y + dy;
                            if !Self::in_bounds(bx, by) { continue; }
                            let bi = Self::idx(bx, by);
                            if self.cells[bi].el == Element::Empty {
                                emit_at = Some(bi);
                                break;
                            }
                        }
                        if emit_at.is_none() {
                            // Walk upward through water/ice/steam looking
                            // for air above the pool surface. Stop on
                            // anything else (wall, metal, gas bubble).
                            let mut yy = y - 1;
                            while yy >= 0 {
                                let ti = Self::idx(x, yy);
                                let t_el = self.cells[ti].el;
                                if t_el == Element::Empty {
                                    emit_at = Some(ti);
                                    break;
                                }
                                if !matches!(t_el, Element::Water | Element::Ice | Element::Steam) {
                                    break;
                                }
                                yy -= 1;
                            }
                        }
                        if let Some(bi) = emit_at {
                            let mut g = Cell::new(byp_el);
                            g.temp = c.temp;
                            g.flag |= Cell::FLAG_UPDATED;
                            self.cells[bi] = g;
                        }
                    }
                    // Slow-hydrolysis simmer: occasional Steam puff so a
                    // metal pile reacting in water reads as visibly
                    // simmering even though we preserve the water cell
                    // (water → Steam in the products would have
                    // destroyed solvent and prevented saturation).
                    // Probabilistic, not stoichiometric — the Steam
                    // is a visual tag, not actual conserved water.
                    let is_slow_hydrolysis_pair = matches!(c.el,
                        Element::Mg | Element::Ca | Element::Sr
                        | Element::Ba | Element::Ra
                        | Element::Sc | Element::Mn)
                        && matches!(n.el, Element::Water | Element::Ice | Element::Steam);
                    if is_slow_hydrolysis_pair
                        && rand::gen_range::<f32>(0.0, 1.0) < 0.02
                    {
                        for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                            let fx = x + dx;
                            let fy = y + dy;
                            if !Self::in_bounds(fx, fy) { continue; }
                            let fi = Self::idx(fx, fy);
                            if self.cells[fi].el == Element::Empty {
                                let mut s = Cell::new(Element::Steam);
                                s.temp = 110;
                                s.flag |= Cell::FLAG_UPDATED;
                                self.cells[fi] = s;
                                break;
                            }
                        }
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
                    // Highly-exothermic reactions feed an accumulator
                    // instead of spawning their own shockwave. One blast
                    // gets emitted at end-of-pass at the energy-weighted
                    // centroid, so a chained H+O cloud produces ONE big
                    // ring rather than a ring per ignition cell.
                    // Threshold is set high enough that ordinary combustion
                    // (C+O at ΔT=900) DOESN'T contribute; only detonation-
                    // class chemistry like H+O (1800) clears the bar.
                    if r.delta_temp as i32 >= 1200 {
                        // All detonation-class reactions feed the global
                        // blast accumulator so K+F / Cs+F / Fr+F still
                        // detonate. The violent-tier check upstream
                        // (donor_e < 0.85) excludes Li/Na so their salt
                        // formation never reaches this delta_temp and
                        // they just glow hot without detonating —
                        // letting the user actually see the LiF salt.
                        let water_product = matches!(pa.el, Element::Water)
                            || matches!(pb.el, Element::Water);
                        let e = if water_product {
                            // 2.5× pressure multiplier on a lower 300
                            // baseline — H+O punches harder than its
                            // 1800°C heat would suggest.
                            (r.delta_temp as f32 - 300.0).max(0.0) * 2.5
                        } else {
                            (r.delta_temp as f32 - 400.0).max(0.0)
                        };
                        self.add_chem_blast(x, y, e);
                        if water_product {
                            self.flash_ignite_h_o_neighbors(x, y, 3);
                            self.inject_pressure_disc(x, y, 4, 1200);
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
        if self.atomic_metal_count() == 0 { return; }
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
                    // Strict > so Co (EN 1.88) lands inside the
                    // cutoff while Cu (1.90) still sits above it.
                    // Co does react with dilute HCl in real life
                    // (above Cu in the activity series), just slowly;
                    // the original >= cutoff put Co exactly on the
                    // boundary and silently excluded it.
                    if ap.electronegativity > METAL_E_CUTOFF { continue; }
                    let metal_reactivity = (METAL_E_CUTOFF - ap.electronegativity).max(0.01);
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
        if !self.has(Element::Rust) || !self.has(Element::Al) { return; }
        // 60 frames (~1s) per cell, doubled from 30 — slower visible
        // burn-through with more time per cell at incandescence.
        const BURN_DURATION: u8 = 60;
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
                        // Transmute. Real thermite stoichiometry:
                        // 2 Al + Fe₂O₃ → Al₂O₃ + 2 Fe + heat. Per
                        // cell at our granularity, "every Al becomes
                        // a full Al₂O₃ cell" produces a slag pile
                        // roughly equal to the original reactant
                        // mass — wrong by volume (real Al₂O₃ slag
                        // is denser than its Al + O₂ inputs by a
                        // bit, but in our visual cell economy that
                        // 1:1 conversion reads as a giant ash pile
                        // that stifles further reaction).
                        //
                        // Better cell economy: Rust cells transmute
                        // to Fe AND consume up to 2 nearby Al cells
                        // (those Al cells "burn into" the reaction
                        // as Fire instead of becoming separate slag
                        // cells). Slag spawns probabilistically near
                        // the Rust ignition site. Al cells that
                        // finish their independent burn (no Rust
                        // partner found) mostly become Fire too,
                        // with a small chance of leaving a slag
                        // cell behind. Net: per thermite event,
                        // ~3 reactant cells → 1 Fe + ~0.3 Al₂O₃ +
                        // Fire flames. Cascade has room to breathe
                        // because products don't tile every cell.
                        let new_cell = if c.el == Element::Rust {
                            // Consume up to 2 adjacent Al cells —
                            // they vanish into the heat as Fire.
                            let mut consumed = 0;
                            for ddy in -1..=1i32 {
                                for ddx in -1..=1i32 {
                                    if ddx == 0 && ddy == 0 { continue; }
                                    if consumed >= 2 { break; }
                                    let cx = x + ddx;
                                    let cy = y + ddy;
                                    if !Self::in_bounds(cx, cy) { continue; }
                                    let ci = Self::idx(cx, cy);
                                    let nc = self.cells[ci];
                                    if nc.el != Element::Al { continue; }
                                    let mut fire = Cell::new(Element::Fire);
                                    fire.temp = FINAL_TEMP;
                                    fire.flag |= Cell::FLAG_UPDATED;
                                    self.cells[ci] = fire;
                                    consumed += 1;
                                }
                                if consumed >= 2 { break; }
                            }
                            // Spawn one Al₂O₃ slag cell near the
                            // Rust location, ~30% of the time, in
                            // an Empty/Fire cardinal. Sparse slag
                            // matches real thermite visuals (you
                            // see molten Fe, not a uniform ash).
                            if consumed > 0
                                && rand::gen_range::<f32>(0.0, 1.0) < 0.30
                            {
                                if let Some(id) = derive_or_lookup(
                                    Element::Al, Element::O)
                                {
                                    for (dx, dy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                                        let sx = x + dx;
                                        let sy = y + dy;
                                        if !Self::in_bounds(sx, sy) { continue; }
                                        let si = Self::idx(sx, sy);
                                        let se = self.cells[si].el;
                                        if se != Element::Empty
                                            && se != Element::Fire
                                        { continue; }
                                        let mut s = Cell::new(Element::Derived);
                                        s.derived_id = id;
                                        s.temp = FINAL_TEMP;
                                        s.flag |= Cell::FLAG_UPDATED;
                                        self.cells[si] = s;
                                        break;
                                    }
                                }
                            }
                            // Half the Rust cells become Fe; the
                            // other half "burn off" as Fire. Per-cell
                            // 1:1 Fe yield was producing visible chunks
                            // of solid/liquid iron in the middle of
                            // the reactant pile (real thermite drains
                            // molten Fe to a small puddle below the
                            // slag, not a uniform tile of solid metal).
                            // 50% Fe / 50% Fire keeps a visible Fe
                            // residue that sinks proportionally to
                            // its real-world volumetric share without
                            // overwhelming the visible result.
                            if rand::gen_range::<f32>(0.0, 1.0) < 0.50 {
                                let mut f = Cell::new(Element::Fe);
                                f.temp = FINAL_TEMP;
                                f.flag |= Cell::FLAG_UPDATED;
                                f
                            } else {
                                let mut fire = Cell::new(Element::Fire);
                                fire.temp = FINAL_TEMP;
                                fire.flag |= Cell::FLAG_UPDATED;
                                fire
                            }
                        } else {
                            // Al that completed its burn without
                            // being consumed by a paired Rust event
                            // mostly becomes Fire too — only a small
                            // chance of leaving slag.
                            if rand::gen_range::<f32>(0.0, 1.0) < 0.15 {
                                let slag = derive_or_lookup(
                                    Element::Al, Element::O);
                                if let Some(id) = slag {
                                    let mut s = Cell::new(Element::Derived);
                                    s.derived_id = id;
                                    s.temp = FINAL_TEMP;
                                    s.flag |= Cell::FLAG_UPDATED;
                                    s
                                } else {
                                    let mut fire = Cell::new(Element::Fire);
                                    fire.temp = FINAL_TEMP;
                                    fire.flag |= Cell::FLAG_UPDATED;
                                    fire
                                }
                            } else {
                                let mut fire = Cell::new(Element::Fire);
                                fire.temp = FINAL_TEMP;
                                fire.flag |= Cell::FLAG_UPDATED;
                                fire
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
        // This function CREATES Fire cells from hot metal salts —
        // a flame is the OUTPUT, not a precondition. So we gate on
        // "any flame-coloring source is present" (the elements
        // flame_color() returns Some for), not on has(Fire). The
        // earlier has(Fire) gate broke "Cu on lava" — Cu sitting in
        // a flame-less hot environment should still emit its own
        // green flame, but the gate skipped the function until some
        // OTHER fire happened to exist.
        if !self.has(Element::Cu)
            && !self.has(Element::Na)
            && !self.has(Element::K)
            && !self.has(Element::Ca)
            && !self.has(Element::Mg)
            && !self.has(Element::B)
            && !self.has(Element::Salt)
        { return; }
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.is_updated() { continue; }
                if c.burn > 0 { continue; }
                if c.temp < 600 { continue; }
                if flame_color(c.el).is_none() { continue; }
                // If this cell can ignite and is currently above its
                // ignition threshold, the burn cascade is about to take
                // over — emitting Fire here too would double up with
                // the burn cascade's emission and produce visible
                // spam. Most noticeable for elements with a big gap
                // between ignite_above and the 600°C flame_test floor
                // (S: 232 vs 600). Skip the cell and let burn handle.
                if let Some(ig) = c.el.thermal().ignite_above {
                    if c.temp > ig { continue; }
                }
                // Flame-color emission isn't combustion — it's the
                // tint hot metal vapor adds to an existing flame. So
                // gate it on either real oxygen presence (a flame
                // could exist here) or an already-burning Fire cell
                // adjacent (a flame already exists here). At 0%
                // ambient O₂ in vacuum, hot Na should glow but not
                // spawn Fire cells — that was producing visible
                // "flames" with no oxidizer present.
                let local_o2 = self.oxygen_available(x, y);
                let adjacent_fire = [
                    (1i32, 0i32), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                ].iter().any(|&(dx, dy)| {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { return false; }
                    self.cells[Self::idx(nx, ny)].el == Element::Fire
                });
                if local_o2 <= 0.01 && !adjacent_fire { continue; }
                // Emission probability scales with oxygen. With an
                // adjacent flame, the floor is 10% so the tint is
                // visible even in low-O₂ pockets near an existing
                // fire. Without adjacent flame, fully proportional
                // to local O₂.
                let o2_factor = local_o2.clamp(0.0, 1.0);
                let emit_p = if adjacent_fire {
                    0.10 + 0.30 * o2_factor
                } else {
                    0.40 * o2_factor
                };
                if rand::gen_range::<f32>(0.0, 1.0) > emit_p { continue; }
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
        if !self.has(Element::Fire) { return; }
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
        if !self.has(Element::Mg) { return; }
        const BURN_DURATION: u8 = 50;
        const BURN_TEMP: i16 = 3000;
        const FINAL_TEMP: i16 = 1700;
        const IGNITION: i16 = 470;
        // 80°C/frame to neighbors keeps the iconic photoflash chain
        // alive — without it Mg ribbon ignites only the cell in direct
        // torch contact, then snuffs before its neighbors heat past
        // ignition (the conductivity-only path is too slow for the
        // 470°C ignite threshold and the surface-area-driven real-
        // world flash).
        const HEAT_BROADCAST_DELTA: i16 = 80;
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
                    // H-solute amplifier: if this Mg cell is loaded
                    // with absorbed H (from the metal_hydrogen_absorption
                    // pass), each burn tick drains some H and dumps
                    // extra heat + spawns an extra Fire cell. Real
                    // MgH₂ in air burns more vigorously than pure Mg
                    // because the H₂ release combusts on top of the
                    // Mg combustion. Without this, the Mg burn just
                    // looks like normal Mg combustion despite the H
                    // load; the H gets silently consumed when the cell
                    // transmutes to MgO.
                    if self.cells[i].solute_el == Element::H
                        && self.cells[i].solute_amt > 0
                    {
                        let drain = 8u8;
                        let new_amt = self.cells[i].solute_amt.saturating_sub(drain);
                        self.cells[i].solute_amt = new_amt;
                        if new_amt == 0 {
                            self.cells[i].solute_el = Element::Empty;
                        }
                        // Bonus heat — boost temp / sustain burn.
                        let bonus = (self.cells[i].temp as i32 + 300).min(5000) as i16;
                        self.cells[i].temp = bonus;
                        // Visible flame spawn: spread fire into a
                        // nearby empty cell. Larger H reserves keep
                        // throwing fire.
                        for (fx, fy) in [(0i32, -1), (1, 0), (-1, 0), (0, 1)] {
                            let cx = x + fx;
                            let cy = y + fy;
                            if !Self::in_bounds(cx, cy) { continue; }
                            let ci = Self::idx(cx, cy);
                            if self.cells[ci].el != Element::Empty { continue; }
                            let mut f = Cell::new(Element::Fire);
                            f.temp = 2200;
                            f.flag |= Cell::FLAG_UPDATED;
                            self.cells[ci] = f;
                            break;
                        }
                    }
                    let c = self.cells[i]; // re-read after possible burn refresh
                    let new_burn = c.burn - 1;
                    if new_burn == 0 {
                        // Transmute. Real Mg ribbon photoflash vaporizes
                        // nearly completely — the iconic puff of white
                        // smoke is the visible aerosol, with only a
                        // sliver of MgO settling as visible powder.
                        // Mirror the alkali/alkaline-earth dispersal
                        // table from the generic burn-end transmute:
                        // 97.5% disperse to hot CO2 puff, 2.5% retain
                        // as MgO. Without this, a Mg pile fully burns
                        // to a solid MgO pile of the same size,
                        // missing the smoke-cloud aesthetic.
                        let dispersed = rand::gen_range::<f32>(0.0, 1.0) < 0.975;
                        let new_cell = if dispersed {
                            let mut puff = Cell::new(Element::CO2);
                            puff.temp = FINAL_TEMP;
                            puff.flag |= Cell::FLAG_UPDATED;
                            puff
                        } else {
                            match derive_or_lookup(Element::Mg, Element::O) {
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
        if !self.has(Element::Hg) { return; }
        // Real Hg amalgamation (especially with Cu) is gradual — a
        // copper coin in mercury takes hours to visibly dissolve. The
        // previous 0.05 rate consumed solid metal on contact, which
        // read as "Hg eats your metal cell-by-cell every frame." 0.005
        // drops the visible behavior to a slow creeping infiltration
        // — the contact line shows amalgam forming, but the parent
        // metal pile persists across many seconds. Au remains the
        // exception (real Hg sucks up Au fast); we don't differentiate
        // per-metal yet but could.
        const RATE: f32 = 0.005;
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
        // Reactivity series F > Cl > Br > I (by electronegativity).
        // A more-electronegative halogen displaces any less-EN halogen
        // bound in a metal salt: F kicks out Cl/Br/I, Cl kicks out
        // Br/I, Br kicks out I. Heat decreases up the series, so
        // F-displacement is the most exothermic.
        let any_attacker = self.has(Element::F)
            || self.has(Element::Cl)
            || self.has(Element::Br)
            || self.has(Element::I);
        if !any_attacker { return; }
        // Returns the halogen's rank (4 = F, ..., 1 = I); higher is
        // more reactive. Non-halogens return 0 so any halogen wins.
        let rank = |el: Element| -> u8 {
            match el {
                Element::F  => 4,
                Element::Cl => 3,
                Element::Br => 2,
                Element::I  => 1,
                _ => 0,
            }
        };
        const REACTION_HEAT_F: i16 = 200;
        const REACTION_HEAT_CL: i16 = 120;
        const REACTION_HEAT_BR: i16 = 60;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                let attacker_rank = rank(c.el);
                if attacker_rank == 0 { continue; }
                // Per-halogen base rate. F is the most aggressive
                // (real F₂ displaces every other halogen on contact);
                // Br is the least, only displacing I.
                let base_rate: f32 = match c.el {
                    Element::F  => 0.30,
                    Element::Cl => 0.20,
                    Element::Br => 0.12,
                    _ => continue,
                };
                let reaction_heat = match c.el {
                    Element::F  => REACTION_HEAT_F,
                    Element::Cl => REACTION_HEAT_CL,
                    Element::Br => REACTION_HEAT_BR,
                    _ => 0,
                };
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    // Identify (metal, bound halogen) in the salt.
                    let (metal_el, bound_halogen): (Element, Element) = match n.el {
                        Element::Salt => (Element::Na, Element::Cl),
                        Element::Derived => {
                            let m = {
                                let reg = DERIVED_COMPOUNDS.read();
                                reg.get(n.derived_id as usize).and_then(|cd| {
                                    if cd.constituents.len() != 2 { return None; }
                                    let (e0, _) = cd.constituents[0];
                                    let (e1, _) = cd.constituents[1];
                                    if rank(e0) > 0 { Some((e1, e0)) }
                                    else if rank(e1) > 0 { Some((e0, e1)) }
                                    else { None }
                                })
                            };
                            match m {
                                Some(pair) => pair,
                                None => continue,
                            }
                        }
                        _ => continue,
                    };
                    // Only displace if the "metal" is actually a metal —
                    // otherwise interhalogen compounds (ClF₇, BrCl, etc.)
                    // get mis-detected as halides and the displacement
                    // runs derive_or_lookup(F, F) → garbage. Halogen-on-
                    // halogen displacement isn't real chemistry anyway.
                    if !is_atomic_metal(metal_el) { continue; }
                    // Attacker must outrank the bound halogen.
                    if attacker_rank <= rank(bound_halogen) { continue; }
                    let mut rate = base_rate;
                    if n.is_frozen() { rate *= 0.02; }
                    if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                    let Some(new_salt_id) = derive_or_lookup(metal_el, c.el)
                        else { continue; };
                    // Halide cell → new metal halide (with the attacker).
                    let mut new_salt = Cell::new(Element::Derived);
                    new_salt.derived_id = new_salt_id;
                    new_salt.temp = (n.temp as i32 + reaction_heat as i32).min(5000) as i16;
                    new_salt.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = new_salt;
                    // Attacker cell → bound halogen (displaced, freed
                    // back to its elemental form).
                    let mut freed = Cell::new(bound_halogen);
                    freed.temp = (c.temp as i32 + reaction_heat as i32).min(5000) as i16;
                    freed.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = freed;
                    break;
                }
            }
        }
        // Internal displacement via absorbed-halogen moisture. A halide
        // salt cell whose moisture_el is a more-electronegative halogen
        // than what's bound (e.g. NaCl deep in a Br-soaked pile) fires
        // displacement against itself — moisture IS the attacker. Rate
        // scales with moisture amount: barely-wet cells fire rarely,
        // soaked cells fire often. The natural moisture gradient (~25
        // per cell of depth) means cells deep in the pile see less
        // moisture and fire less, giving a depth-limited reaction zone
        // that matches the visible wet zone.
        //
        // Internal events with no empty neighbor for the freed halogen
        // roll 50/50 between making the cell the new salt OR the freed
        // halogen. Across many events both products end up represented
        // through the pile (KBr AND I appear, not just KBr).
        const INTERNAL_BASE_RATE: f32 = 0.10;
        const INTERNAL_MOISTURE_REQ: u8 = 32;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                if self.cells[i].is_updated() { continue; }
                let c = self.cells[i];
                let soak_rank = rank(c.moisture_el);
                if soak_rank == 0 { continue; }
                if c.moisture < INTERNAL_MOISTURE_REQ { continue; }
                let (metal_el, bound_halogen): (Element, Element) = match c.el {
                    Element::Salt => (Element::Na, Element::Cl),
                    Element::Derived => {
                        let m = {
                            let reg = DERIVED_COMPOUNDS.read();
                            reg.get(c.derived_id as usize).and_then(|cd| {
                                if cd.constituents.len() != 2 { return None; }
                                let (e0, _) = cd.constituents[0];
                                let (e1, _) = cd.constituents[1];
                                if rank(e0) > 0 { Some((e1, e0)) }
                                else if rank(e1) > 0 { Some((e0, e1)) }
                                else { None }
                            })
                        };
                        match m {
                            Some(pair) => pair,
                            None => continue,
                        }
                    }
                    _ => continue,
                };
                if !is_atomic_metal(metal_el) { continue; }
                if soak_rank <= rank(bound_halogen) { continue; }
                // Rate scales with moisture (cap at 250 → full base rate).
                // The "moisture requirement" the user mentioned is just
                // this proportional gate — drier cells are less likely
                // to fire, so the reaction front falls off with depth
                // along the moisture gradient.
                let moisture_factor = (c.moisture as f32 / 250.0).min(1.0);
                let mut rate = INTERNAL_BASE_RATE * moisture_factor;
                if c.is_frozen() { rate *= 0.02; }
                if rand::gen_range::<f32>(0.0, 1.0) > rate { continue; }
                let Some(new_salt_id) = derive_or_lookup(metal_el, c.moisture_el)
                    else { continue; };
                let attacker_heat = match c.moisture_el {
                    Element::F  => REACTION_HEAT_F,
                    Element::Cl => REACTION_HEAT_CL,
                    Element::Br => REACTION_HEAT_BR,
                    _ => 0,
                };
                let drained_moisture = c.moisture.saturating_sub(INTERNAL_MOISTURE_REQ);
                let next_moisture_el = if drained_moisture > 0 {
                    c.moisture_el
                } else {
                    Element::Empty
                };
                let mut empty_neighbor: Option<usize> = None;
                for (dx, dy) in [(0i32, -1), (1, 0), (-1, 0), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].el == Element::Empty {
                        empty_neighbor = Some(ni);
                        break;
                    }
                }
                let make_new_salt = || -> Cell {
                    let mut s = Cell::new(Element::Derived);
                    s.derived_id = new_salt_id;
                    s.temp = (c.temp as i32 + attacker_heat as i32).min(5000) as i16;
                    s.moisture = drained_moisture;
                    s.moisture_el = next_moisture_el;
                    s.flag |= Cell::FLAG_UPDATED;
                    s
                };
                let make_freed = |xtemp: i16| -> Cell {
                    let mut f = Cell::new(bound_halogen);
                    f.temp = (xtemp as i32 + attacker_heat as i32).min(5000) as i16;
                    f.flag |= Cell::FLAG_UPDATED;
                    f
                };
                match empty_neighbor {
                    Some(ni) => {
                        self.cells[i]  = make_new_salt();
                        self.cells[ni] = make_freed(c.temp);
                    }
                    None => {
                        if rand::gen_range::<u8>(0, 2) == 0 {
                            self.cells[i] = make_new_salt();
                        } else {
                            let mut f = make_freed(c.temp);
                            f.moisture = drained_moisture;
                            f.moisture_el = next_moisture_el;
                            self.cells[i] = f;
                        }
                    }
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
        if !self.has(Element::F) || !self.has(Element::Glass) { return; }
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

    // F + Water hydrolysis. Real reaction: 2 F₂ + 2 H₂O → 4 HF + O₂.
    // We can't run this through the normal chemistry engine because
    // Water's chemistry face (O at valence 6) makes Water-as-donor fail
    // Metal-hydrogen absorption — Pd, Ti, and Mg form metal hydrides
    // by literally absorbing H gas into their lattice. Pd-H is the
    // textbook example (real Pd soaks 900× its own volume of H₂ at
    // STP) and is the basis of hydrogen storage tech. Ti and Mg do
    // the same but need heat (Ti above ~200°C, Mg above ~300°C).
    //
    // Storage uses the existing `solute_el` / `solute_amt` slots,
    // which are otherwise unused for solid cells. Per absorption
    // event the metal soaks up one H neighbor (cell vanishes) and
    // gains 40 of "absorbed H" capacity, saturating at 255 (~6 H
    // cells per metal cell — sandbox abstraction of real lattice
    // saturation).
    //
    // Strict whitelist: only Pd / Ti / Mg metals, only H gas. Other
    // solids do NOT suddenly absorb gases — this pass is bespoke
    // for the metal-hydride chemistry, not a generic capability.
    fn metal_hydrogen_absorption(&mut self) {
        let any_absorber = self.has(Element::Pd)
            || self.has(Element::Ti)
            || self.has(Element::Mg);
        if !any_absorber { return; }
        // Note: NOT early-outing on `!has(H)` — Pd/Ti/Mg cells can
        // already be loaded (solute_el=H) from a previous frame, and
        // the catalytic combustion + wicking sub-passes need to run
        // even when no free H gas exists in the world.
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.is_updated() { continue; }
                // Per-metal temperature gate. Pd absorbs at room temp
                // (real-life behavior); Ti and Mg need heat to break
                // through their oxide skins and reach the bulk metal.
                let absorbs = match c.el {
                    Element::Pd => true,
                    Element::Ti => c.temp >= 200,
                    Element::Mg => c.temp >= 300,
                    _ => false,
                };
                if !absorbs { continue; }
                // Saturation cap. Already-loaded cells need to dehydride
                // (release H back) before they can take more.
                if c.solute_amt >= 250 { continue; }
                if c.solute_el != Element::Empty
                    && c.solute_el != Element::H
                {
                    continue;
                }
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    let n = self.cells[ni];
                    if n.el != Element::H { continue; }
                    if n.is_updated() { continue; }
                    // Absorb: H cell vanishes into the lattice.
                    self.cells[i].solute_el = Element::H;
                    self.cells[i].solute_amt =
                        self.cells[i].solute_amt.saturating_add(40);
                    self.cells[ni] = Cell::EMPTY;
                    break;
                }
            }
        }
        // Catalytic ignition: a loaded metal-hydride cell adjacent to
        // O (real spawned cell or virtual ambient O₂) catalyzes
        // H + ½O₂ → H₂O on its surface. Drains one H unit, dumps the
        // bespoke H+O exotherm (1800°C) into both cells, spawns Water
        // where the O was. Pd is catalytic at any temp (real Pd-H is
        // pyrophoric in air); Ti needs ≥400°C, Mg ≥250°C to break
        // their oxide skins. The flash_ignite seeds the burn cascade
        // through nearby unabsorbed H/O so the rest of the chamber
        // chain-combusts via the existing H+O bespoke path (which
        // also feeds chem_blast for the shockwave).
        let ambient_o = self.ambient_oxygen;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.is_updated() { continue; }
                if !matches!(c.el, Element::Pd | Element::Ti | Element::Mg) { continue; }
                if c.solute_el != Element::H || c.solute_amt == 0 { continue; }
                let active = match c.el {
                    Element::Pd => true,
                    Element::Ti => c.temp >= 350,
                    Element::Mg => c.temp >= 250,
                    _ => false,
                };
                if !active { continue; }
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    if self.cells[ni].is_updated() { continue; }
                    let n = self.cells[ni];
                    let is_real_o = n.el == Element::O;
                    let is_virtual_o = n.el == Element::Empty
                        && ambient_o > 0.0
                        && rand::gen_range::<f32>(0.0, 1.0) < ambient_o;
                    if !is_real_o && !is_virtual_o { continue; }
                    const HEAT: i32 = 1800;
                    const H_DRAIN: u8 = 40;
                    // Drain one H unit; reset solute_el if fully unloaded.
                    let new_amt = self.cells[i].solute_amt.saturating_sub(H_DRAIN);
                    self.cells[i].solute_amt = new_amt;
                    if new_amt == 0 {
                        self.cells[i].solute_el = Element::Empty;
                    }
                    self.cells[i].temp =
                        (c.temp as i32 + HEAT).clamp(-273, 5000) as i16;
                    self.cells[i].flag |= Cell::FLAG_UPDATED;
                    if is_real_o {
                        // Heat the real O cell rather than consume it.
                        // Real Pd surface catalysis turns over fast —
                        // water forms briefly at the surface and
                        // diffuses away, fresh O₂ keeps moving in.
                        // Consuming the cell each event made spawned-
                        // O reactions die immediately because the hot
                        // Pd's thermal pressure shoves replacement O
                        // away faster than it could drift back. Now
                        // O cell stays put + gets hot, water product
                        // vents into a separate empty neighbor (or
                        // implicitly if no empty around).
                        self.cells[ni].temp =
                            (n.temp as i32 + HEAT).clamp(-273, 5000) as i16;
                        self.cells[ni].flag |= Cell::FLAG_UPDATED;
                        // Vent water into a different empty neighbor
                        // for visual product representation.
                        for (vx, vy) in [(0i32, -1i32), (1, 0), (-1, 0), (0, 1)] {
                            let wx = x + vx;
                            let wy = y + vy;
                            if !Self::in_bounds(wx, wy) { continue; }
                            let wi = Self::idx(wx, wy);
                            if wi == ni { continue; }
                            if self.cells[wi].el != Element::Empty { continue; }
                            let mut w = Cell::new(Element::Water);
                            w.temp = (n.temp as i32 + HEAT).clamp(-273, 5000) as i16;
                            w.flag |= Cell::FLAG_UPDATED;
                            self.cells[wi] = w;
                            break;
                        }
                    }
                    // Seed the cascade: heat nearby H/O so the rest of
                    // the chamber chain-combusts via the chemistry
                    // pass instead of crawling cell-by-cell. The
                    // existing H+O bespoke path then feeds chem_blast
                    // for the shockwave.
                    self.flash_ignite_h_o_neighbors(x, y, 3);
                    break;
                }
            }
        }
        // Wicking pass: absorbed H spreads inward from contact-loaded
        // surface cells to drier interior cells of the same metal.
        // Same gradient-flow shape as the moisture wicking pipeline,
        // so a Pd pile in H atmosphere shows a visible loaded-front
        // propagating inward rather than only the outermost shell
        // saturating. Matches real Pd-H lattice diffusion (which is
        // famously fast) and gives the user a gradient cue per the
        // color shift.
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if !matches!(c.el, Element::Pd | Element::Ti | Element::Mg) { continue; }
                if c.solute_el != Element::H { continue; }
                if c.solute_amt < 20 { continue; }
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    let n = self.cells[ni];
                    if n.el != c.el { continue; }
                    if n.solute_el != Element::Empty
                        && n.solute_el != Element::H
                    {
                        continue;
                    }
                    let gradient = c.solute_amt as i16 - n.solute_amt as i16;
                    if gradient < 16 { continue; }
                    let amt = (gradient / 4).max(1) as u8;
                    self.cells[i].solute_amt =
                        self.cells[i].solute_amt.saturating_sub(amt);
                    self.cells[ni].solute_el = Element::H;
                    self.cells[ni].solute_amt =
                        self.cells[ni].solute_amt.saturating_add(amt);
                }
            }
        }
    }

    // the donor_v ≤ 4 check. Dedicated pass: F adjacent to Water/Ice/
    // Steam → F cell becomes derived HF, water cell becomes O. Strongly
    // exothermic in real life; we add 600°C to both products.
    fn fluorine_hydrolysis(&mut self) {
        if !self.has(Element::F)
            || !(self.has(Element::Water)
                || self.has(Element::Ice)
                || self.has(Element::Steam))
        { return; }
        const RATE: f32 = 0.50;
        const REACTION_HEAT: i16 = 600;
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
                    if !matches!(n.el, Element::Water | Element::Ice | Element::Steam) { continue; }
                    if rand::gen_range::<f32>(0.0, 1.0) > RATE { continue; }
                    let Some(hf_id) = derive_or_lookup(Element::H, Element::F)
                        else { continue; };
                    // F cell → HF (derived).
                    let mut hf = Cell::new(Element::Derived);
                    hf.derived_id = hf_id;
                    hf.temp = (c.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    hf.flag |= Cell::FLAG_UPDATED;
                    self.cells[i] = hf;
                    // Water cell → O.
                    let mut o = Cell::new(Element::O);
                    o.temp = (n.temp as i32 + REACTION_HEAT as i32).min(5000) as i16;
                    o.flag |= Cell::FLAG_UPDATED;
                    self.cells[ni] = o;
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
        if self.atomic_metal_count() == 0 { return; }
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
        if self.atomic_metal_count() < 2 { return; }
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
        if self.atomic_metal_count() == 0 { return; }
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
        if !self.has(Element::Water) { return; }
        // Per-solute saturation cap and per-event fill step. Different
        // compound classes have wildly different real-world solubility:
        // NaCl saturates water at ~360 g/L (basically unlimited cap),
        // but Mn(OH)₂ has Ksp ~10⁻¹³ (effectively insoluble, only a
        // tiny equilibrium concentration). Treating both with the same
        // 255-fill-on-contact rate makes hydrolysis residues vanish
        // like salt does, which is wrong-physics and gameplay-bad
        // (Mn metal "dissolves before hitting bottom" because its
        // residue immediately disappears into solute).
        //
        // The per-solute split: salts get the original fast-saturating
        // behavior, hydrolysis residues get a low cap and small step
        // so they accumulate slowly and reach saturation early — fresh
        // residue then piles up as visible powder instead of being
        // continually absorbed. Mixed wastewater pools (multi-species
        // contamination) get a slightly higher shared cap since
        // multiple species can independently contribute.
        // Gate: when water's solute_amt is below this threshold, it
        // has "room" to absorb more of this solute. Fill: how much
        // gets added per successful dissolution event. Separating
        // these lets salts behave as before (gate at the historical
        // 192 ABSORB_THRESHOLD, fill 255 = one-shot to saturated)
        // while hydrolysis residues use a low gate AND a small fill
        // step so they accumulate gradually and saturate early —
        // matching real Mn(OH)₂ / Mg(OH)₂ low solubility (Ksp ~10⁻¹³).
        fn solute_gate_for(n: Cell) -> u8 {
            if n.el == Element::Salt { return 192; }
            if n.el == Element::Derived {
                if derived_is_soluble_salt(n.derived_id) { return 192; }
                // Hydrolysis residue gate raised to 160 (was 64). Real
                // Mn(OH)₂/Mg(OH)₂ are much less soluble than salts, but
                // a 64 gate is too low to read visibly after diffusion
                // spreads the solute across a pool: dump thousands of
                // Mn cells, see ~23 saturation per cell because the
                // small per-cell cap diffuses to a vanishing average.
                // 160 leaves a clear visible-vs-fully-saturated gap
                // below salt's 192 (so brine still reads as more
                // saturated than hydrolyzed-metal water) while giving
                // residue solute enough headroom to be obvious.
                if derived_is_hydrolysis_residue(n.derived_id) { return 160; }
            }
            0
        }
        fn solute_fill_for(n: Cell) -> u8 {
            if n.el == Element::Salt { return 255; }
            if n.el == Element::Derived {
                if derived_is_soluble_salt(n.derived_id) { return 255; }
                // Match fill to gate so each residue cell saturates one
                // water cell in a single dissolution event. Without
                // this, dissolution per cell per frame (≈ rate × fill)
                // gets outpaced by diffusion (~8 units per cell per
                // frame), so local concentration at the metal-water
                // interface never builds — solute spreads across the
                // whole pool and dilutes to invisibility.
                if derived_is_hydrolysis_residue(n.derived_id) { return 160; }
            }
            0
        }
        const TRY_P: f32 = 0.20;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el != Element::Water { continue; }
                if rand::gen_range::<f32>(0.0, 1.0) > TRY_P { continue; }
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if !Self::in_bounds(nx, ny) { continue; }
                    let ni = Self::idx(nx, ny);
                    let n = self.cells[ni];
                    if n.is_frozen() { continue; }
                    let base_gate = solute_gate_for(n);
                    if base_gate == 0 { continue; }
                    // Mixed-wastewater gate: when this water carries a
                    // different solute, the pool can accumulate beyond
                    // either single-species gate (independent Ksp
                    // contributions stack). Bumped to 96 vs single-
                    // species 64 for hydrolysis residue, but still
                    // capped well below salts' 192 since we're
                    // assuming the dominant species is residue-tier.
                    let same_solute = c.solute_el == n.el
                        && c.solute_derived_id == n.derived_id;
                    let mixed = c.solute_amt > 0 && !same_solute;
                    let gate = if mixed { base_gate.max(96) } else { base_gate };
                    if c.solute_amt >= gate { continue; }
                    let fill = solute_fill_for(n);
                    // Saturate at u8::MAX, not at gate — salts fill to
                    // 255 in one shot (preserves the "saltwater = full
                    // brine" visual), residues fill incrementally up
                    // to gate and stop.
                    let add = fill.min(255_u8.saturating_sub(c.solute_amt));
                    if add == 0 { continue; }
                    // Identity update: empty water adopts the new solute,
                    // matching water keeps it, mismatched contaminates to
                    // the Mixed wastewater sentinel (solute_el = Empty +
                    // solute_amt > 0). Identity is lost in mixed pools
                    // but saturation still works.
                    if c.solute_amt == 0 || same_solute {
                        self.cells[i].solute_el = n.el;
                        self.cells[i].solute_derived_id = n.derived_id;
                    } else {
                        self.cells[i].solute_el = Element::Empty;
                        self.cells[i].solute_derived_id = 0;
                    }
                    self.cells[i].solute_amt =
                        c.solute_amt.saturating_add(add);
                    self.cells[ni] = Cell::EMPTY;
                    break;
                }
            }
        }
    }

    fn diffuse_solute(&mut self) {
        if !self.has(Element::Water) { return; }
        // Solute_amt equalizes between adjacent water cells carrying the
        // same solute (concentration gradient → diffusion). This is what
        // breaks the interface-saturation stall: without it the layer of
        // water touching a salt pile saturates and never mixes away, and
        // no fresh water ever reaches the remaining salt.
        //
        // Picks a random neighbor per cell per frame. Transfers half the
        // gap each event, capped at *_MAX so visual mixing looks
        // gradual rather than snapping to uniform in a frame.
        //
        // Per-solute split: salts (NaCl, AuCl, MgCl₂, …) diffuse fast
        // for smooth visible mixing — same as the historical 0.35 × 24
        // rates. Hydrolysis residues (MgO, CaO, MnO, ScO₃, …) diffuse
        // slowly so local concentration at a metal-water interface
        // can build to visible saturation before spreading outward
        // (matches real liquid-phase ion diffusion which is much
        // slower than visible mixing speeds anyway). Without the
        // slow tier for residues, the small-amount-per-event from
        // their dissolution gets out-paced by spread and the pool
        // never visibly saturates.
        const SALT_DIFFUSE_P: f32 = 0.35;
        const SALT_DIFFUSE_MAX: u8 = 24;
        const RESIDUE_DIFFUSE_P: f32 = 0.05;
        const RESIDUE_DIFFUSE_MAX: u8 = 4;
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let i = Self::idx(x, y);
                let c = self.cells[i];
                if c.el != Element::Water { continue; }
                if c.solute_amt == 0 { continue; }
                let is_residue = c.solute_el == Element::Derived
                    && derived_is_hydrolysis_residue(c.solute_derived_id);
                let (diffuse_p, diffuse_max) = if is_residue {
                    (RESIDUE_DIFFUSE_P, RESIDUE_DIFFUSE_MAX)
                } else {
                    (SALT_DIFFUSE_P, SALT_DIFFUSE_MAX)
                };
                if rand::gen_range::<f32>(0.0, 1.0) > diffuse_p { continue; }
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
                // Mismatched solutes — different species adjacent. Don't
                // refuse the diffusion; instead contaminate both cells to
                // the Mixed sentinel (solute_el → Empty). The two species
                // are now lost as a pair, but solute_amt remains additive
                // and the pool saturates correctly. Same Mixed-tier rule
                // used by dissolve(): identity sacrificed, saturation cap
                // preserved.
                let mismatched = n.solute_amt > 0
                    && (n.solute_el != c.solute_el
                        || n.solute_derived_id != c.solute_derived_id);
                if mismatched {
                    self.cells[i].solute_el = Element::Empty;
                    self.cells[i].solute_derived_id = 0;
                    self.cells[ni].solute_el = Element::Empty;
                    self.cells[ni].solute_derived_id = 0;
                    continue;
                }
                if n.solute_amt >= c.solute_amt { continue; }
                let gap = c.solute_amt - n.solute_amt;
                let transfer = (gap / 2).min(diffuse_max).max(1);
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
        if !self.has(Element::Water) { return; }
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
        //
        // Steam exception: when overpressure (pressure above the local
        // hydrostatic baseline) is significant, the steam is contained
        // (sealed pressure cooker, hot pipe, etc.) and should persist
        // indefinitely the way real steam does in a closed vessel above
        // its saturation pressure. Open-air steam still dissipates
        // because its overpressure equilibrates to ~0 within frames as
        // the gas spreads.
        //
        // Hydrostatic baseline is approximated as a linear gradient
        // from 0 at the top of the play space to ~160 at the bottom
        // (matches the column-weight integration the pressure pass
        // produces in pure atmosphere). Without this subtraction, deep
        // open-air steam would read as "contained" simply by depth.
        if matches!(me, Element::Fire | Element::Steam) {
            let contained = me == Element::Steam && {
                let baseline = (160i32 * y) / H as i32;
                let overpressure = (self.cells[i].pressure as i32 - baseline).max(0);
                overpressure > 50
            };
            if !contained {
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
        let my_compl = cell_pressure_p(me_cell).compliance as i32;
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
        // cell_physics is phase-aware AND looks through the derived
        // registry — `me.physics()` (the static fallback) returns the
        // Element::Derived stub with molar_mass=0, killing buoyancy
        // entirely for every derived gas (HCl, HF, NH₃, etc.).
        let my_mass = cell_physics(me_cell).molar_mass;
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
        let compliance = cell_pressure_p(me_cell).compliance as i32;
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
        // Per-kind canonical bounding box. Line is a horizontal bar
        // (length × thickness) before rotation; Circle is a square
        // bbox using max(w, h) so the ring is round-ish; the rest
        // use the user-provided width × height directly.
        let (canon_w, canon_h) = match kind {
            PrefabKind::Line => (w.max(1), thickness.max(1)),
            PrefabKind::Circle => {
                let d = w.max(h).max(2);
                (d, d)
            }
            _ => (w, h),
        };
        let (bw, bh) = if rot == 1 || rot == 3 { (canon_h, canon_w) } else { (canon_w, canon_h) };
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
                    PrefabKind::Line => {
                        // Solid bar — fill the entire bbox. Rotation
                        // already produced the right bw/bh, so every
                        // cell in the box is part of the line.
                        stamp(self, x, y, el);
                    }
                    PrefabKind::Circle => {
                        // Hollow ring of `thickness` width inside the
                        // square bbox. Use canonical (rotation-invariant)
                        // distance from centre — circles look the same
                        // at any rotation, so rotation doesn't affect
                        // the stamping.
                        let lx = x - x0;
                        let ly = y - y0;
                        let cxc = bw / 2;
                        let cyc = bh / 2;
                        let dx = (lx - cxc) as f32;
                        let dy = (ly - cyc) as f32;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let outer_r = (bw.min(bh) as f32) * 0.5;
                        let inner_r = (outer_r - thickness as f32).max(0.0);
                        if dist <= outer_r && dist >= inner_r {
                            stamp(self, x, y, el);
                        }
                    }
                    PrefabKind::Bowl => {
                        // Lower half of an ellipse. Open at the top
                        // edge of the bbox. Wall is `thickness` cells
                        // thick. Rotation rotates the open face — a
                        // bowl rotated 180° opens downward.
                        let lx = x - x0;
                        let ly = y - y0;
                        // Map screen-relative position to canonical
                        // (top-open) bowl coordinates so rotation works.
                        let (u_pos, v_pos, u_max, v_max) = match rot {
                            0 => (lx, ly,           bw, bh),  // canonical: open top
                            1 => (ly, bw - 1 - lx, bh, bw),  // open right (rotated 90° CW)
                            2 => (lx, bh - 1 - ly, bw, bh),  // open bottom
                            3 => (ly, lx,           bh, bw),  // open left
                            _ => (lx, ly,           bw, bh),
                        };
                        // u along the open edge (0..u_max), v from
                        // open edge inward (0..v_max). Lower half of
                        // ellipse has v >= 0 from top-centre.
                        let cu = (u_max / 2) as f32;
                        let du = u_pos as f32 - cu;
                        let dv = v_pos as f32;
                        let a_outer = (u_max as f32 * 0.5).max(1.0);
                        let b_outer = (v_max as f32).max(1.0);
                        let outer = (du / a_outer).powi(2) + (dv / b_outer).powi(2);
                        if dv >= 0.0 && outer <= 1.0 {
                            let a_inner = (a_outer - thickness as f32).max(0.5);
                            let b_inner = (b_outer - thickness as f32).max(0.5);
                            let inner = (du / a_inner).powi(2) + (dv / b_inner).powi(2);
                            if inner > 1.0 {
                                stamp(self, x, y, el);
                            }
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
        // Phase-aware kind resolution: el.physics() returns the static
        // table, which for Element::Derived is the (Kind::Powder, 0)
        // stub. So all the per-kind paint logic below — sparsity for
        // liquids, overpaint-pressure for gases — silently treated
        // every derived compound as a powder. Result: holding paint
        // on derived gases (HCl, HF, NH₃) added zero stacking pressure
        // → no rapid expansion, no smoke ring, just a flat blob.
        let resolved_kind = if el == Element::Derived {
            derived_hot(derived_id)
                .map(|h| h.physics.kind)
                .unwrap_or(Kind::Powder)
        } else {
            el.physics().kind
        };
        let painting_solid = matches!(resolved_kind, Kind::Solid | Kind::Gravel);
        let sparsity: u16 = if frozen {
            1
        } else {
            match resolved_kind {
                Kind::Liquid => 50,
                _ => 1,
            }
        };
        let overpaint_pressure: i16 = match resolved_kind {
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
                    // Re-set formation_pressure here too — Cell::new
                    // pulled it from el.pressure_p() which for
                    // Element::Derived is the static (perm=0, form=0)
                    // fallback. Phase-aware cell_pressure_p gives a
                    // derived gas the same ~30 spawn pressure that
                    // atomic gases get, so it actually puffs out
                    // instead of sitting in a rigid blob.
                    if el == Element::Derived {
                        c.derived_id = derived_id;
                        c.pressure = cell_pressure_p(c).formation_pressure;
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
    // Hydration-shifting compounds (CoCl₂ blue↔pink) interpolate
    // between their dry and wet colors based on cell.moisture.
    let mut hydration_overrides_wet_tint = false;
    let (r, g, b) = if c.el == Element::Derived {
        let base = derived_color_of(c.derived_id);
        if let Some(hot) = derived_hot(c.derived_id) {
            if let Some(wet) = hot.hydration_color {
                let m = (c.moisture as f32 / 255.0).clamp(0.0, 1.0);
                let mix = |a: u8, b: u8| -> u8 {
                    ((a as f32) * (1.0 - m) + (b as f32) * m) as u8
                };
                hydration_overrides_wet_tint = true;
                (mix(base.0, wet.0), mix(base.1, wet.1), mix(base.2, wet.2))
            } else {
                base
            }
        } else {
            base
        }
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
    let is_noble_gas = atom_profile_for(c.el)
        .map_or(false, |p| matches!(p.category, AtomCategory::NobleGas));
    if c.moisture > 20
        && c.el != Element::Water
        && c.el != Element::Empty
        && !is_noble_gas
        && !hydration_overrides_wet_tint
    {
        let wet = ((c.moisture as f32 - 20.0) / 235.0).clamp(0.0, 1.0) * 0.55;
        // Tint toward the absorbed liquid's identity color — Br soaks
        // in as red-brown, water as the cool blue legacy tint, future
        // liquids carry their own color through. Falls back to the
        // legacy blue tint when moisture_el is Empty (cell got
        // moisture from a path that didn't tag the liquid identity).
        let (tr, tg, tb): (f32, f32, f32) = if c.moisture_el != Element::Empty
            && c.moisture_el != Element::Water
        {
            let (rr, gg, bb) = c.moisture_el.base_color();
            (rr as f32, gg as f32, bb as f32)
        } else {
            // Cool wet-blue (matches the original hardcoded tint).
            (40.0, 60.0, 130.0)
        };
        r = ((r as f32) * (1.0 - wet) + tr * wet) as u8;
        g = ((g as f32) * (1.0 - wet) + tg * wet) as u8;
        b = ((b as f32) * (1.0 - wet) + tb * wet) as u8;
    }
    // Metal-hydride loading tint — Pd, Ti, Mg with absorbed H (stored
    // in solute_el/solute_amt) get a cool blue-grey shift proportional
    // to load. Matches real Pd-H going from silver to matte blue-grey
    // as it saturates, and gives the user a visible gradient through
    // a partly-loaded pile (surface cells dark blue, interior pale).
    if matches!(c.el, Element::Pd | Element::Ti | Element::Mg)
        && c.solute_el == Element::H
        && c.solute_amt > 0
    {
        let load = (c.solute_amt as f32 / 255.0).clamp(0.0, 1.0) * 0.55;
        let (tr, tg, tb) = (90.0, 110.0, 160.0);
        r = ((r as f32) * (1.0 - load) + tr * load) as u8;
        g = ((g as f32) * (1.0 - load) + tg * load) as u8;
        b = ((b as f32) * (1.0 - load) + tb * load) as u8;
    }
    // Dissolved solute tints the water toward the solute's own color. Mixes
    // up to ~0.75 at saturation — enough to read (saltwater pales, FeCl
    // water yellows, CuCl turns cyan, MnO water browns) without losing
    // the water identity. For hydrolysis residues (derived oxide of a
    // slow-tier metal) we tint toward the *donor metal's* color rather
    // than the averaged compound color: the dissolved species is
    // effectively the metal cation, and averaging in the O contribution
    // dilutes the result toward water's own blue and reads as no tint.
    // Mixed-solute wastewater (solute_el == Empty + solute_amt > 0) uses
    // a generic gray-brown tint, signaling the pool is contaminated with
    // multiple unrecoverable species.
    if c.el == Element::Water && c.solute_amt > 0 {
        let (sr, sg, sb) = if c.solute_el == Element::Empty {
            (110, 95, 75) // wastewater — muted gray-brown
        } else if c.solute_el == Element::Derived {
            // Hydration-shifting solutes (CoCl₂) are always at "fully
            // wet" when dissolved in water, so use their hydration
            // color directly — produces the iconic pink CoCl₂
            // solution. Hydrolysis residues use the donor metal's
            // color (dissolved species is the metal cation, averaging
            // in O dilutes toward water blue and reads invisibly).
            // Other salts/halides use the registered compound color.
            let hot = derived_hot(c.solute_derived_id);
            if let Some(wet) = hot.and_then(|h| h.hydration_color) {
                wet
            } else if derived_is_hydrolysis_residue(c.solute_derived_id) {
                let reg = DERIVED_COMPOUNDS.read();
                reg.get(c.solute_derived_id as usize)
                    .and_then(|cd| cd.constituents.first().map(|(el, _)| el.base_color()))
                    .unwrap_or_else(|| derived_color_of(c.solute_derived_id))
            } else {
                derived_color_of(c.solute_derived_id)
            }
        } else {
            c.solute_el.base_color()
        };
        let t = (c.solute_amt as f32 / 255.0) * 0.75;
        r = ((r as f32) * (1.0 - t) + (sr as f32) * t) as u8;
        g = ((g as f32) * (1.0 - t) + (sg as f32) * t) as u8;
        b = ((b as f32) * (1.0 - t) + (sb as f32) * t) as u8;
    }
    // Atomic metals stay silvery longer than the generic 250°C ramp
    // suggests — real molten Al at 660°C is bright silver, real
    // molten Au at 1064°C is shiny gold, etc. Visible incandescent
    // glow only kicks in around 800-1000°C+. Without this offset,
    // freshly-melted metal puddles read as orange-tinted instead of
    // their proper liquid-metal sheen.
    let is_atomic_metal = atom_profile_for(c.el).map_or(false, |p|
        matches!(p.category,
            AtomCategory::AlkaliMetal
            | AtomCategory::AlkalineEarth
            | AtomCategory::TransitionMetal
            | AtomCategory::PostTransition
            | AtomCategory::Lanthanide
            | AtomCategory::Actinide));
    let warm_start: i16 = if is_atomic_metal { 800 } else { 250 };
    if c.temp > warm_start && c.el != Element::Fire {
        // Stage 1: cool → red → orange → yellow. Models iron glowing
        // red at ~700°C, orange at ~1100°C, yellow at ~1500°C.
        let warm_heat = ((c.temp - warm_start) as f32 / 1500.0).clamp(0.0, 1.0);
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
    // element's hand-tuned color with no tint. Noble gases skip this
    // entirely — real liquid/solid Ne, Ar, Kr, Xe are essentially
    // colorless, and the blue-shifted tint turns warm-base noble gases
    // (orange Ne, red Rn) into dusty purple, which doesn't read as
    // "frozen colorless" at all.
    if !is_noble_gas {
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
    }
    [r, g, b]
}


// Retained as a zero-height constant so existing code that offsets by
// TOP_BAR still compiles while the top bar itself is gone. The panel
// uses a small internal padding (PANEL_TOP_PAD) instead.
const TOP_BAR: f32 = 0.0;
const PANEL_TOP_PAD: f32 = 10.0;

// Paintable compound materials. MoltenGlass is intentionally omitted —
// you get it by melting sand. Fire is intentionally omitted — it
// emerges from combustion, driven by the Heat tool raising a fuel's
// temperature past its ignition threshold. Wood was originally
// omitted (trees grow wood) but added back as a paintable for
// audit/testing — comparing burn dynamics across Wood vs Leaves vs
// C is awkward when one of them requires growing a tree first.
// These live in the periodic-table overlay under the atom grid.
const COMPOUND_PALETTE: [Element; 19] = [
    Element::Sand, Element::Water, Element::Stone,
    Element::Wood,
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
    if open { 240.0 } else { 0.0 }
}

// Width-row label varies by shape — "Width" for box-style shapes,
// "Length" for line, "Diameter" for circle.
fn param_width_label(kind: PrefabKind) -> &'static str {
    match kind {
        PrefabKind::Line => "Length",
        PrefabKind::Circle => "Diameter",
        _ => "Width",
    }
}

// Tiny shape glyph for the kind selector and option list. Drawn with
// macroquad primitives at ~16-18 px so it reads at panel size without
// needing a sprite sheet.
fn draw_prefab_icon(kind: PrefabKind, x: f32, y: f32, size: f32) {
    let stroke = (size * 0.10).max(1.5);
    let col = Color::from_rgba(220, 220, 230, 255);
    match kind {
        PrefabKind::Beaker => {
            // U-shape: left, right, bottom.
            draw_line(x, y, x, y + size, stroke, col);
            draw_line(x + size, y, x + size, y + size, stroke, col);
            draw_line(x, y + size, x + size, y + size, stroke, col);
        }
        PrefabKind::Box => {
            draw_rectangle_lines(x, y, size, size, stroke, col);
        }
        PrefabKind::Battery => {
            let body_y = y + size * 0.15;
            let body_h = size * 0.70;
            draw_rectangle_lines(x, body_y, size, body_h, stroke, col);
            // + terminal bump on top
            let nub = size * 0.25;
            draw_rectangle(
                x + size * 0.5 - nub * 0.5, y, nub, size * 0.15, col,
            );
            // − band on bottom
            draw_rectangle(
                x + size * 0.2, y + size * 0.92, size * 0.6, size * 0.08, col,
            );
        }
        PrefabKind::Line => {
            // Diagonal line so it reads as a line and not a rect side.
            draw_line(x, y + size, x + size, y, stroke, col);
        }
        PrefabKind::Circle => {
            draw_circle_lines(
                x + size * 0.5, y + size * 0.5, size * 0.45, stroke, col,
            );
        }
        PrefabKind::Bowl => {
            // Approximate U-curve with a polyline.
            let cx = x + size * 0.5;
            let cy = y;
            let r = size * 0.5;
            let segs = 14;
            let mut prev = (x, y);
            for i in 0..=segs {
                let t = i as f32 / segs as f32;
                let ang = std::f32::consts::PI * t;
                let px = cx - r * ang.cos();
                let py = cy + r * ang.sin();
                if i > 0 {
                    draw_line(prev.0, prev.1, px, py, stroke, col);
                }
                prev = (px, py);
            }
        }
    }
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

// Ambient control rows — Temp, O₂, Gravity, EMF readout. Positioned
// below the element readout with a SIMULATION section header above
// them. First three are hit-targets for scroll-on-hover adjustment.
// EMF is read-only — shows active galvanic / battery voltage when a
// circuit is detected, or "—" when nothing is energized. Lets the
// user immediately see whether their galvanic-cell setup is firing.
const PANEL_AMBIENT_COUNT: usize = 4;
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
// the Build button when Prefab is the active tool. Layout: kind-picker
// dropdown selector at the top, then T/W/H (and Voltage for Battery)
// hover-scroll rows, then the Material picker.
const PREFAB_ROW_COUNT: usize = 4;
const PREFAB_KIND_OPTION_HEIGHT: f32 = 26.0;
fn prefab_kind_selector_rect() -> (f32, f32, f32, f32) {
    let prefab_btn = panel_button_rects(false, false)[4];
    let y = prefab_btn.1 + prefab_btn.3 + 10.0;
    (prefab_btn.0, y, prefab_btn.2, 30.0)
}
fn prefab_kind_option_rects() -> [(f32, f32, f32, f32); PREFAB_KIND_COUNT] {
    let sel = prefab_kind_selector_rect();
    let mut out = [(0.0, 0.0, 0.0, 0.0); PREFAB_KIND_COUNT];
    let mut y = sel.1 + sel.3 + 2.0;
    for i in 0..PREFAB_KIND_COUNT {
        out[i] = (sel.0, y, sel.2, PREFAB_KIND_OPTION_HEIGHT);
        y += PREFAB_KIND_OPTION_HEIGHT + 2.0;
    }
    out
}
fn prefab_slider_rects() -> [(f32, f32, f32, f32); PREFAB_ROW_COUNT] {
    let sel = prefab_kind_selector_rect();
    let x = sel.0;
    let w = sel.2;
    let h = 26.0;
    let gap = 5.0;
    let mut y = sel.1 + sel.3 + 10.0;
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

    draw_ui_text(&format!("{}    (compound)", el.display_label()), px, py, 24.0, WHITE);
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
        phase_bits.push(format!("freeze → {} below {}°C", p.target.display_label(), p.threshold));
    }
    if let Some(p) = therm.melt_above {
        phase_bits.push(format!("melt → {} at {}°C", p.target.display_label(), p.threshold));
    }
    if let Some(p) = therm.boil_above {
        phase_bits.push(format!("boil → {} at {}°C", p.target.display_label(), p.threshold));
    }
    if let Some(p) = therm.condense_below {
        phase_bits.push(format!("condense → {} below {}°C", p.target.display_label(), p.threshold));
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
        moist_bits.push(format!("→ {} at moisture {}", t.display_label(), m));
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

// ---- GPU post-process shaders ----
// Bloom + gas-cloud blur live on the GPU as four separable triangle-blur
// passes (h+v for each effect) feeding a composite shader that adds them
// onto the base sim texture. Replacing the CPU rayon path here is what
// makes the idle frame budget viable — the CPU was spending 10+ ms/frame
// blurring 100k pixels with 37-tap and 17-tap kernels every frame even
// when there was nothing to blur.

const POST_VERT_SRC: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;
varying lowp vec2 uv;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
}
"#;

// Triangle-kernel separable blur, hardcoded radius 18 (37 taps).
// Direction = (1,0) for horizontal pass, (0,1) for vertical.
const BLUR_BLOOM_FRAG_SRC: &str = r#"#version 100
precision highp float;
varying lowp vec2 uv;
uniform sampler2D Texture;
uniform vec2 inv_size;
uniform vec2 direction;
void main() {
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int k = -18; k <= 18; k++) {
        float w = 19.0 - abs(float(k));
        vec2 off = direction * float(k) * inv_size;
        sum += texture2D(Texture, uv + off) * w;
        total += w;
    }
    gl_FragColor = sum / total;
}
"#;

// Same kernel structure, radius 8 (17 taps). Smaller halo for gas
// clouds — tighter than bloom's giant glow.
const BLUR_GAS_FRAG_SRC: &str = r#"#version 100
precision highp float;
varying lowp vec2 uv;
uniform sampler2D Texture;
uniform vec2 inv_size;
uniform vec2 direction;
void main() {
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int k = -8; k <= 8; k++) {
        float w = 9.0 - abs(float(k));
        vec2 off = direction * float(k) * inv_size;
        sum += texture2D(Texture, uv + off) * w;
        total += w;
    }
    gl_FragColor = sum / total;
}
"#;

// Final composite + per-material visual treatments. The sim grid is
// 320×315 cells but renders to a much larger output region (≈3.25× per
// axis at the panel's normal layout), so each cell has multiple output
// pixels — enough room for sub-cell shading like edge highlights, grain
// noise, glyphs, and transparency.
//
// kind_tex packs per-cell metadata RGBA:
//   R = element id (0..54)
//   G = physics Kind (0=Empty, 1=Solid, 2=Gravel, 3=Powder, 4=Liquid,
//                     5=Gas, 6=Fire) — phase-aware via cell_physics()
//   B = frozen flag (0 or 1)
//   A = unused (255)
// Sampled with FilterMode::Nearest so cell boundaries are crisp.
//
// bloom_tex / gas_tex come from TWO render-target passes (each pass
// y-flips, so they end up at normal orientation). Gas density lives
// in alpha and is amped 8× to counteract two-pass triangle-blur
// attenuation.
const COMPOSITE_FRAG_SRC: &str = r#"#version 100
precision highp float;
varying lowp vec2 uv;
uniform sampler2D Texture;
uniform sampler2D bloom_tex;
uniform sampler2D gas_tex;
uniform sampler2D kind_tex;
uniform float u_time;

const float GRID_W = 320.0;
const float GRID_H = 315.0;

// Element IDs we treat specifically.
const float K_SAND     = 1.0;
const float K_LAVA     = 8.0;
const float K_ICE      = 14.0;
const float K_GLASS    = 16.0;
const float K_HG       = 37.0;
const float K_BATTPOS  = 46.0;
const float K_BATTNEG  = 47.0;

// Physics Kind discriminants (must match enum order).
const float P_SOLID  = 1.0;
const float P_GRAVEL = 2.0;
const float P_POWDER = 3.0;
const float P_GAS    = 5.0;

// Float-equality helper. Element IDs are small integers stored in u8
// then unpacked to [0..255]; tolerance hides any FP rounding.
bool eq(float a, float b) { return abs(a - b) < 0.5; }

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec3 base = texture2D(Texture, uv).rgb;
    vec4 ki = texture2D(kind_tex, uv);
    float k      = ki.r * 255.0;     // element id
    float pkind  = ki.g * 255.0;     // physics kind
    float frozen = ki.b * 255.0;     // 0 or 1

    vec2 texel = vec2(1.0 / GRID_W, 1.0 / GRID_H);
    vec2 cell_uv = uv * vec2(GRID_W, GRID_H);
    vec2 within = fract(cell_uv);

    // Neighbor element IDs (for edge detection + ambient-occlusion).
    float k_t  = texture2D(kind_tex, uv + vec2( 0.0, -texel.y)).r * 255.0;
    float k_b  = texture2D(kind_tex, uv + vec2( 0.0,  texel.y)).r * 255.0;
    float k_l  = texture2D(kind_tex, uv + vec2(-texel.x, 0.0)).r * 255.0;
    float k_r  = texture2D(kind_tex, uv + vec2( texel.x, 0.0)).r * 255.0;
    float k_tl = texture2D(kind_tex, uv + vec2(-texel.x, -texel.y)).r * 255.0;
    float k_tr = texture2D(kind_tex, uv + vec2( texel.x, -texel.y)).r * 255.0;
    float k_bl = texture2D(kind_tex, uv + vec2(-texel.x,  texel.y)).r * 255.0;
    float k_br = texture2D(kind_tex, uv + vec2( texel.x,  texel.y)).r * 255.0;

    // Detect SOLID-phase atomic metals only. Liquid metals (molten
    // Hg, molten Au, boiled-then-cooled vapor) fall through to the
    // CPU liquid styling; gas-phase boiled metals fall through to
    // the volumetric gas pipeline — both look bad with the
    // surface specular below.
    bool atomic_metal_id =
           eq(k, 25.0) || eq(k, 26.0) || eq(k, 27.0) || eq(k, 32.0)
        || eq(k, 33.0) || eq(k, 34.0) || eq(k, 35.0) || eq(k, 36.0)
        || eq(k, 37.0) || eq(k, 38.0) || eq(k, 48.0) || eq(k, 49.0)
        || eq(k, 50.0) || eq(k, 51.0) || eq(k, 53.0) || eq(k, 54.0);
    bool atomic_metal = atomic_metal_id
        && (eq(pkind, P_SOLID) || eq(pkind, P_GRAVEL));

    // ---- Edge depth shading (non-metal blocks) ----
    // Brighten the top + left band where the cell borders a different
    // element; darken the bottom + right band. Universal "fake light
    // from upper-left" cue for static blocks. Skipped for atomic
    // metals — they get a stronger custom version that pushes harder
    // toward white-shine on the lit faces.
    bool is_block = (eq(pkind, P_SOLID) || eq(pkind, P_GRAVEL)
        || eq(pkind, P_POWDER) || frozen > 0.5) && !atomic_metal;
    if (is_block) {
        float ew = 0.30;
        float lit = 0.0;
        float shd = 0.0;
        if (!eq(k_t, k) && within.y < ew) lit += (ew - within.y) / ew;
        if (!eq(k_l, k) && within.x < ew) lit += (ew - within.x) / ew;
        if (!eq(k_b, k) && within.y > (1.0 - ew)) shd += (within.y - (1.0 - ew)) / ew;
        if (!eq(k_r, k) && within.x > (1.0 - ew)) shd += (within.x - (1.0 - ew)) / ew;
        base = mix(base, base * 1.45, clamp(lit * 0.5, 0.0, 0.45));
        base = mix(base, base * 0.55, clamp(shd * 0.5, 0.0, 0.45));
    }

    // ---- Material shading per kind ----
    vec2 fp = gl_FragCoord.xy;

    if (atomic_metal) {
        // Surface-aware metallic shading + light directional AO.
        // Real metal bodies don't go dark in the middle — they
        // stay full-bright with hot specular on lit edges and
        // shadow on back edges. AO is kept subtle and biased
        // upward (cells with metal above them dim more than
        // cells with metal below them) so a pile reads as
        // "lit from above" instead of "ambient-occluded into mud".
        bool face_t = !eq(k_t, k);
        bool face_b = !eq(k_b, k);
        bool face_l = !eq(k_l, k);
        bool face_r = !eq(k_r, k);

        // Specular: push toward pure white at 90 % mix on the very
        // edge. Narrow band (ew = 0.35) so it reads as a hot
        // highlight rather than a soft glow.
        float ew = 0.35;
        float spec = 0.0;
        if (face_t) spec = max(spec, smoothstep(ew, 0.0, within.y));
        if (face_l) spec = max(spec, smoothstep(ew, 0.0, within.x) * 0.85);
        base = mix(base, vec3(1.0), spec * 0.90);

        // Back-edge shadow — toned down (0.25 → 0.45 floor) so
        // the metal still reads as bright on the unlit side.
        float shd = 0.0;
        if (face_b) shd = max(shd, smoothstep(ew, 0.0, 1.0 - within.y));
        if (face_r) shd = max(shd, smoothstep(ew, 0.0, 1.0 - within.x) * 0.85);
        base *= mix(1.0, 0.45, shd);

        // Directional AO: upper neighbors weighted more so cells
        // BURIED beneath other metal (deep in a pile) get shaded
        // while cells with metal below stay bright. Total weight 8;
        // floor lifted from 0.55 → 0.80 so bodies stay clearly lit.
        float same_m = 0.0;
        if (eq(k_t,  k)) same_m += 1.8;
        if (eq(k_tl, k)) same_m += 1.4;
        if (eq(k_tr, k)) same_m += 1.4;
        if (eq(k_l,  k)) same_m += 0.8;
        if (eq(k_r,  k)) same_m += 0.8;
        if (eq(k_b,  k)) same_m += 0.5;
        if (eq(k_bl, k)) same_m += 0.65;
        if (eq(k_br, k)) same_m += 0.65;
        float ao_m = same_m / 8.0;
        base *= mix(1.0, 0.80, ao_m);

        // Subtle within-cell vertical gradient — top of each cell
        // a touch brighter than the bottom, reinforces "lit from
        // above" for the body interior.
        base *= 1.0 - within.y * 0.08;
    } else if (eq(pkind, P_POWDER)) {
        // Powders: bump-mapped lighting + speckle. Loose particles
        // catching light from above.
        float h_c = hash21(fp);
        float h_r = hash21(fp + vec2(1.0, 0.0));
        float h_d = hash21(fp + vec2(0.0, 1.0));
        float bump = (h_r - h_c) + (h_d - h_c);
        base *= 1.0 + bump * 0.45;
        base += (h_c - 0.5) * 0.18 * (base * 0.5 + vec3(0.04));
        float bright = max(0.0, h_c - 0.94) * 16.7;
        base = mix(base, base * 1.55, clamp(bright * 0.45, 0.0, 0.45));
        float dark = max(0.0, 0.06 - h_c) * 16.7;
        base = mix(base, base * 0.42, clamp(dark * 0.50, 0.0, 0.50));
    } else if (eq(pkind, P_GRAVEL)) {
        // Coarse tonal patches for natural rock variation.
        vec2 q = floor(fp * 0.25);
        float h_patch = hash21(q);
        base *= mix(0.92, 1.08, h_patch);

        // Convex ambient-occlusion: count same-kind among the 8
        // neighbors. Surface cells (few same-kind around) stay
        // bright; interior cells (most/all same-kind around) get
        // darkened, so a stone pile reads as a 3D dome instead of
        // a flat tonal field. AO falls off into the rim, so the
        // pile looks lit from the outside.
        float same = 0.0;
        if (eq(k_t,  k)) same += 1.0;
        if (eq(k_b,  k)) same += 1.0;
        if (eq(k_l,  k)) same += 1.0;
        if (eq(k_r,  k)) same += 1.0;
        if (eq(k_tl, k)) same += 1.0;
        if (eq(k_tr, k)) same += 1.0;
        if (eq(k_bl, k)) same += 1.0;
        if (eq(k_br, k)) same += 1.0;
        float ao = same / 8.0;
        base *= mix(1.10, 0.68, ao);
    } else if (eq(pkind, P_SOLID)) {
        // Non-metal solids (Wood/Quartz/Firebrick/Seed): per-cell
        // stable tint at very low amplitude.
        vec2 ci = floor(cell_uv);
        float pc = hash21(ci);
        base *= mix(0.94, 1.06, pc);
    }

    // ---- Glass / Ice transparency ----
    // Blend with what's a few pixels "behind" (offset down) plus a
    // bright edge band. Glass tints neutral; Ice tints cool blue.
    if (eq(k, K_GLASS) || eq(k, K_ICE)) {
        vec3 behind = texture2D(Texture, uv + vec2(0.0, texel.y * 4.0)).rgb;
        float opacity = eq(k, K_GLASS) ? 0.50 : 0.55;
        base = mix(behind, base, opacity);
        bool top_e = !eq(k_t, k) && within.y < 0.18;
        bool bot_e = !eq(k_b, k) && within.y > 0.82;
        bool lft_e = !eq(k_l, k) && within.x < 0.18;
        bool rgt_e = !eq(k_r, k) && within.x > 0.82;
        if (top_e || bot_e || lft_e || rgt_e) {
            vec3 edge_color = eq(k, K_ICE) ? vec3(0.85, 0.95, 1.0) : vec3(1.0);
            base = mix(base, edge_color, 0.4);
        }
    }

    // ---- Mercury shimmer ----
    if (eq(k, K_HG)) {
        float wave = sin(uv.y * GRID_H * 0.4 + u_time * 1.5) * 0.5 + 0.5;
        base = mix(base, vec3(1.0), wave * 0.20);
    }

    // ---- Lava pulse ----
    if (eq(k, K_LAVA)) {
        float pulse = sin(u_time * 3.0 + (cell_uv.x + cell_uv.y) * 0.3) * 0.5 + 0.5;
        base = mix(base, base * 1.35, pulse * 0.35);
    }

    // ---- Battery terminal caps ----
    // Render BattPos / BattNeg cells so a CONTIGUOUS RUN of them reads
    // as a single polished metal cap, not as a checkerboard of dome
    // gradients. The trick: highlights only appear on the OUTER edge
    // of the cap (where the terminal cell borders something that ISN'T
    // a terminal) — interior cells get plain metal. Adjacent terminal
    // cells in the same row/column then look like one continuous bar
    // because they share the same color in their interiors.
    //
    // BattPos: copper gold. BattNeg: brushed silver. Both get a slow
    // animated specular streak so the metal feels polished, not flat.
    if (eq(k, K_BATTPOS) || eq(k, K_BATTNEG)) {
        // Toned-down metallic palette — was reading too "wet/chrome";
        // these values land closer to oxidized copper and brushed
        // steel than mirror-polished metal.
        vec3 metal = eq(k, K_BATTPOS)
            ? vec3(0.70, 0.40, 0.16)
            : vec3(0.62, 0.66, 0.72);
        vec3 hi_col = eq(k, K_BATTPOS)
            ? vec3(0.90, 0.68, 0.38)
            : vec3(0.85, 0.88, 0.92);
        vec3 sh_col = eq(k, K_BATTPOS)
            ? vec3(0.32, 0.15, 0.05)
            : vec3(0.22, 0.24, 0.28);

        bool face_t = !eq(k_t, k);
        bool face_b = !eq(k_b, k);
        bool face_l = !eq(k_l, k);
        bool face_r = !eq(k_r, k);

        // Subtler outer-edge band — was 0.55, now 0.32.
        float outer = 0.0;
        if (face_t) outer = max(outer, smoothstep(0.50, 0.0, within.y));
        if (face_b) outer = max(outer, smoothstep(0.50, 1.0, within.y));
        if (face_l) outer = max(outer, smoothstep(0.30, 0.0, within.x));
        if (face_r) outer = max(outer, smoothstep(0.30, 1.0, within.x));

        float inner = 0.0;
        if (face_t) inner = max(inner, smoothstep(0.55, 1.0, within.y));
        if (face_b) inner = max(inner, smoothstep(0.55, 0.0, within.y));

        vec3 lit = metal;
        lit = mix(lit, hi_col, outer * 0.32);
        lit = mix(lit, sh_col, inner * 0.30);

        // Animated sheen kept but much weaker — 0.18 → 0.06. Reads
        // as "light catching the metal occasionally" instead of
        // "wet plastic shimmer".
        float t = u_time * 0.4;
        float band = sin((cell_uv.x + cell_uv.y) * 0.6 + t * 3.0);
        float sheen = pow(max(0.0, band), 6.0);
        lit += hi_col * sheen * 0.06;

        base = lit;
    }

    // ---- Composite bloom + gas (existing) ----
    vec3 bloom = texture2D(bloom_tex, uv).rgb;
    vec4 gas = texture2D(gas_tex, uv);
    float gd = clamp(gas.a * 8.0, 0.0, 1.0);
    vec3 gas_color = gas.rgb * gd;

    vec3 final_col = base + bloom + gas_color;
    gl_FragColor = vec4(min(final_col, vec3(1.0)), 1.0);
}
"#;

pub async fn run_game() {
    init_ui_font();
    let mut world = World::new();
    let mut image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let texture = Texture2D::from_image(&image);
    texture.set_filter(FilterMode::Nearest);

    // Sidecar images for the GPU post-process. bright_image holds the
    // emission-tinted color per cell (driven by temp / Fire / Lava /
    // energized noble gases) — fed into the bloom blur. gas_image holds
    // the gas atom's color in RGB and per-atom density in alpha — fed
    // into the gas-cloud blur. Both are zero-cleared each frame and
    // populated alongside the base sim image.
    let mut bright_image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let bright_texture = Texture2D::from_image(&bright_image);
    bright_texture.set_filter(FilterMode::Nearest);
    let mut gas_image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let gas_texture = Texture2D::from_image(&gas_image);
    gas_texture.set_filter(FilterMode::Nearest);

    // Per-cell metadata for the per-material composite shader. RGBA:
    //   R = element id  (0..54)
    //   G = physics Kind (phase-aware via cell_physics)
    //   B = frozen flag (0 or 1)
    //   A = unused (255)
    // Always uploaded each frame — the shader uses it for edge shading,
    // grain noise, transparency, glyphs, and per-element specials, so
    // the data has to track current state.
    let mut kind_image = Image::gen_image_color(W as u16, H as u16, BLACK);
    let kind_texture = Texture2D::from_image(&kind_image);
    kind_texture.set_filter(FilterMode::Nearest);

    // Render targets for the four blur passes. Each is W×H (the sim
    // resolution), filtered Linear so the bilinear samples blend the
    // taps smoothly across edges.
    let bloom_h_target = render_target(W as u32, H as u32);
    bloom_h_target.texture.set_filter(FilterMode::Linear);
    let bloom_v_target = render_target(W as u32, H as u32);
    bloom_v_target.texture.set_filter(FilterMode::Linear);
    let gas_h_target = render_target(W as u32, H as u32);
    gas_h_target.texture.set_filter(FilterMode::Linear);
    let gas_v_target = render_target(W as u32, H as u32);
    gas_v_target.texture.set_filter(FilterMode::Linear);

    let blur_bloom_material = load_material(
        ShaderSource::Glsl {
            vertex: POST_VERT_SRC,
            fragment: BLUR_BLOOM_FRAG_SRC,
        },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("inv_size", UniformType::Float2),
                UniformDesc::new("direction", UniformType::Float2),
            ],
            ..Default::default()
        },
    ).expect("bloom blur shader compile");
    let blur_gas_material = load_material(
        ShaderSource::Glsl {
            vertex: POST_VERT_SRC,
            fragment: BLUR_GAS_FRAG_SRC,
        },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("inv_size", UniformType::Float2),
                UniformDesc::new("direction", UniformType::Float2),
            ],
            ..Default::default()
        },
    ).expect("gas blur shader compile");
    let composite_material = load_material(
        ShaderSource::Glsl {
            vertex: POST_VERT_SRC,
            fragment: COMPOSITE_FRAG_SRC,
        },
        MaterialParams {
            textures: vec![
                "bloom_tex".to_string(),
                "gas_tex".to_string(),
                "kind_tex".to_string(),
            ],
            uniforms: vec![
                UniformDesc::new("u_time", UniformType::Float1),
            ],
            ..Default::default()
        },
    ).expect("composite shader compile");

    // Offscreen camera for blur passes — one display rect at sim
    // resolution, reused across passes by reassigning render_target.
    let inv_size = vec2(1.0 / W as f32, 1.0 / H as f32);

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
    const PIPET_CAPACITY: usize = 20_000;
    let mut pipet_target: Option<(Element, u8)> = None;
    let mut pipet_bucket: Vec<Cell> = Vec::new();
    // Frames remaining on the "empty pipet first" warning flash. Shown
    // when the user tries to change target while bucket is non-empty.
    let mut pipet_warning_frames: u32 = 0;
    // Toast overlay — short status message centered at the top of the
    // sim area. Used for save/load feedback and could be reused for
    // any "operation succeeded/failed" notification.
    let mut toast_msg: String = String::new();
    let mut toast_color: Color = WHITE;
    let mut toast_frames: u32 = 0;
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
    // Per-kind config stashes, indexed by `PrefabKind as usize`. The
    // "live" prefab_* values above mirror whichever kind is currently
    // selected. Switching kinds saves the current values into the
    // OLD kind's stash and loads the NEW kind's stash — so each
    // shape remembers its own customizations.
    // Tuple layout: (thickness, width, height, material).
    let mut prefab_stashes: [(i32, i32, i32, Element); PREFAB_KIND_COUNT] = [
        (10, 145, 200, Element::Glass),    // Beaker
        (10, 145, 200, Element::Glass),    // Box
        (10,  30,  40, Element::Quartz),   // Battery
        ( 4,  80,   1, Element::Stone),    // Line  (W = length, T = bar thickness)
        ( 4,  80,  80, Element::Stone),    // Circle (W = diameter, T = ring thickness)
        ( 8, 100,  60, Element::Glass),    // Bowl  (W × H ellipse outline)
    ];
    // Whether the kind-picker dropdown inside the prefab sub-panel is
    // expanded. While open it covers the parameter rows; clicking an
    // option (or anywhere outside) closes it and shows the chosen
    // shape's parameters.
    let mut prefab_kind_dropdown_open: bool = false;
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

    // Per-section frame timing — accumulates µs across each second and
    // emits an averaged report so we can see where the budget actually
    // goes (sim step vs. render-build/post-process vs. UI/panel).
    let mut perf_frames: u32 = 0;
    let mut perf_t_step_us: u64 = 0;
    let mut perf_t_render_us: u64 = 0;
    let mut perf_t_ui_us: u64 = 0;
    let mut perf_last_print = std::time::Instant::now();

    // Cross-frame tracking for the render-skip gate. When emission or
    // gas disappears, we still run the full pipeline ONE more frame so
    // the corresponding render target is cleared to all-zeros. After
    // that, we can skip the upload + GPU passes until the content
    // reappears.
    let mut last_had_emission: bool = true;
    let mut last_had_gas: bool = true;

    loop {
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
                    // 5°C fine / 50°C shift gives access to chemistry-
                    // relevant thresholds (e.g. 80°C activation for
                    // slow-hydrolysis) without forcing big jumps.
                    let step: i16 = if shift_scroll { 50 } else { 5 };
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
            // -273..=5000°C actual. 5°C fine step lets you hit chemistry
            // thresholds precisely (the 80°C slow-hydrolysis activation
            // is reachable with twelve ticks from 20°C ambient); shift
            // bumps to 50°C for sweeping the full range.
            let step: i16 = if shift_scroll { 50 } else { 5 };
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
            // Tab is the "I want to pick something to paint" key. Always
            // open the table targeting the Paint slot — even if a side-
            // panel material button (Prefab Material / Wire Material)
            // had previously redirected pt_target, Tab overrides that.
            // The PtTarget::Paint branch in the picker also flips
            // tool_mode back to Paint, so Tab → pick → paint Just Works.
            if periodic_open {
                periodic_open = false;
            } else {
                periodic_open = true;
                pt_target = PtTarget::Paint;
            }
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

        // F11 saves the current cell grid + ambient offset to
        // alembic_state.save in the working dir; F12 reloads it. Lets
        // the user build a testing scene once and reload it without
        // rebuilding by hand each session.
        if is_key_pressed(KeyCode::F11) {
            match world.save_state("alembic_state.save") {
                Ok(_) => {
                    toast_msg = "State saved".to_string();
                    toast_color = Color::from_rgba(140, 220, 140, 255);
                }
                Err(e) => {
                    toast_msg = format!("Save failed: {}", e);
                    toast_color = Color::from_rgba(240, 130, 130, 255);
                }
            }
            toast_frames = 120;
        }
        if is_key_pressed(KeyCode::F12) {
            match world.load_state("alembic_state.save") {
                Ok(_) => {
                    toast_msg = "State loaded".to_string();
                    toast_color = Color::from_rgba(140, 220, 140, 255);
                }
                Err(e) => {
                    toast_msg = format!("Load failed: {}", e);
                    toast_color = Color::from_rgba(240, 130, 130, 255);
                }
            }
            toast_frames = 120;
        }

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
                let hit_k = |r: (f32, f32, f32, f32)|
                    mx >= r.0 && mx < r.0 + r.2
                    && my >= r.1 && my < r.1 + r.3;
                // Kind-picker dropdown: clicking the selector toggles
                // the dropdown open. While open, clicking an option
                // commits that kind and closes the dropdown. Live
                // values stash/restore per kind so customizations
                // survive switching.
                let sel = prefab_kind_selector_rect();
                if hit_k(sel) {
                    prefab_kind_dropdown_open = !prefab_kind_dropdown_open;
                    consume_stroke = true;
                }
                if prefab_kind_dropdown_open {
                    let opt_rects = prefab_kind_option_rects();
                    for (i, opt_kind) in PREFAB_KINDS.iter().enumerate() {
                        if hit_k(opt_rects[i]) {
                            if *opt_kind != prefab_kind {
                                prefab_stashes[prefab_kind as usize] =
                                    (prefab_thickness, prefab_width, prefab_height, prefab_material);
                                let next = prefab_stashes[*opt_kind as usize];
                                prefab_thickness = next.0;
                                prefab_width     = next.1;
                                prefab_height    = next.2;
                                prefab_material  = next.3;
                                prefab_kind      = *opt_kind;
                            }
                            prefab_kind_dropdown_open = false;
                            consume_stroke = true;
                            break;
                        }
                    }
                }
                // Material click is suppressed if the dropdown logic
                // already consumed this click — selecting "Bowl" (the
                // last option) sits visually above the Material button
                // and was double-firing both. consume_stroke is set by
                // the dropdown handler whenever it eats a click.
                if !consume_stroke {
                    let mr = prefab_material_rect();
                    if hit_k(mr) {
                        pt_target = PtTarget::PrefabMaterial;
                        periodic_open = true;
                        consume_stroke = true;
                    }
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
        if toast_frames > 0 { toast_frames -= 1; }

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
        let perf_t_step_start = std::time::Instant::now();
        if !paused { world.step(wind); }
        perf_t_step_us += perf_t_step_start.elapsed().as_micros() as u64;

        let perf_t_render_start = std::time::Instant::now();
        // --- render sim ---
        // Single parallel pass that fills all three sim textures (base,
        // bright, gas) at once. Each pixel touches its own slot in each
        // image, so the three images can be zipped under rayon and
        // updated concurrently — no shared writes, no double-pass over
        // the same cell data.
        {
            use rayon::prelude::*;
            let cells_ro = &world.cells;
            let energized_ro = &world.energized;
            image.bytes
                .par_chunks_exact_mut(4)
                .zip(bright_image.bytes.par_chunks_exact_mut(4))
                .zip(gas_image.bytes.par_chunks_exact_mut(4))
                .zip(kind_image.bytes.par_chunks_exact_mut(4))
                .enumerate()
                .for_each(|(i, (((base_pix, bright_pix), gas_pix), kind_pix))| {
                    let c = cells_ro[i];
                    // Per-cell metadata for the composite shader. Use
                    // cell_physics() so phase-aware Kind (boiled-metal
                    // gas, frozen Hg, etc.) is what the shader sees.
                    let phys = cell_physics(c);
                    kind_pix[0] = c.el as u8;
                    kind_pix[1] = phys.kind as u8;
                    kind_pix[2] = if c.is_frozen() { 1 } else { 0 };
                    kind_pix[3] = 255;
                    let [mut r, mut g, mut b] = color_rgb(c);
                    // Radioactive halo — Empty cells within 2 cells of
                    // any radioactive atom pick up a greenish-yellow
                    // glow, like radium dial paint or a Cherenkov-ish
                    // visual cue. Atoms themselves render normally so
                    // a Tc pile looks like Tc, but the air around it
                    // gleams. Distance-weighted: immediate neighbors
                    // contribute full intensity, 2-cell-away neighbors
                    // half intensity. Falls off naturally with distance.
                    if c.el == Element::Empty {
                        let cx = (i % W) as i32;
                        let cy = (i / W) as i32;
                        let mut weighted: f32 = 0.0;
                        for dy in -2..=2i32 {
                            for dx in -2..=2i32 {
                                if dx == 0 && dy == 0 { continue; }
                                let nx = cx + dx;
                                let ny = cy + dy;
                                if nx < 0 || nx >= W as i32 || ny < 0 || ny >= H as i32 {
                                    continue;
                                }
                                let ni = (ny as usize) * W + nx as usize;
                                let nc = cells_ro[ni];
                                let is_rad = if nc.el == Element::Derived {
                                    derived_hot(nc.derived_id)
                                        .map_or(false, |h| h.is_radioactive)
                                } else {
                                    atom_profile_for(nc.el)
                                        .map_or(false, |p| p.half_life_frames > 0)
                                };
                                if !is_rad { continue; }
                                // Chebyshev distance: 1 = adjacent (full),
                                // 2 = corner of 5x5 (half).
                                let d = dx.abs().max(dy.abs());
                                let w = if d <= 1 { 1.0 } else { 0.5 };
                                weighted += w;
                            }
                        }
                        if weighted > 0.0 {
                            // Saturating cap: ~6 effective neighbors
                            // already maxes out; full pile interior
                            // boundary saturates the glow.
                            let glow = (weighted / 6.0).min(1.0) * 0.85;
                            r = ((r as f32) * (1.0 - glow * 0.5) + 180.0 * glow * 0.5) as u8;
                            g = ((g as f32) * (1.0 - glow * 0.5) + 250.0 * glow * 0.5) as u8;
                            b = ((b as f32) * (1.0 - glow * 0.7) + 80.0 * glow * 0.7) as u8;
                        }
                    }
                    // Liquefied noble gases are essentially colorless in
                    // real life — liquid Ne, Ar, Kr, Xe are clear/pale
                    // with at most a faint tint. Desaturate toward
                    // neutral pale when a noble-gas cell is in Liquid
                    // phase so it reads as near-transparent rather than
                    // a vivid colored puddle. Gas-phase keeps its full
                    // base color (visible cloud) and the energized
                    // override below replaces with the discharge glow.
                    if cell_physics(c).kind == Kind::Liquid {
                        if let Some(profile) = atom_profile_for(c.el) {
                            if matches!(profile.category, AtomCategory::NobleGas) {
                                r = ((r as u16 + 220) / 2) as u8;
                                g = ((g as u16 + 220) / 2) as u8;
                                b = ((b as u16 + 220) / 2) as u8;
                            }
                        }
                    }
                    // Energized cells get their electrical glow color (noble
                    // gases light up; conducting metals stay their normal
                    // color since glow_color is None for them).
                    if energized_ro[i] {
                        if let Some((gr, gg, gb)) = c.el.electrical().glow_color {
                            r = gr; g = gg; b = gb;
                        }
                    }
                    // Liquid styling — surface highlight + depth shading.
                    // All effects scale the cell's existing color, so dark
                    // liquids stay dark and bright liquids stay bright.
                    // Phase-aware via cell_physics so molten metals get
                    // the same treatment.
                    if cell_physics(c).kind == Kind::Liquid {
                        let cx = (i % W) as i32;
                        let cy = (i / W) as i32;
                        let is_top = |x: i32, y: i32| -> bool {
                            if y <= 0 || x < 0 || x >= W as i32 { return false; }
                            let here = (y as usize) * W + x as usize;
                            if cell_physics(cells_ro[here]).kind != Kind::Liquid { return false; }
                            let above = ((y - 1) as usize) * W + x as usize;
                            cell_physics(cells_ro[above]).kind != Kind::Liquid
                        };
                        let self_top = is_top(cx, cy);
                        let neigh_top = is_top(cx - 1, cy) as i32
                            + is_top(cx + 1, cy) as i32;
                        let on_surface = self_top && neigh_top >= 1;
                        if on_surface {
                            r = ((r as u32 * 122 / 100).min(255)) as u8;
                            g = ((g as u32 * 122 / 100).min(255)) as u8;
                            b = ((b as u32 * 122 / 100).min(255)) as u8;
                        }
                        let col_depth = |x: i32| -> i32 {
                            if x < 0 || x >= W as i32 { return 0; }
                            let mut d = 0i32;
                            for dy in 1..=8 {
                                let py = cy - dy;
                                if py < 0 { break; }
                                let pi = py as usize * W + x as usize;
                                if cell_physics(cells_ro[pi]).kind == Kind::Liquid { d += 1; }
                                else { break; }
                            }
                            d
                        };
                        let depth = col_depth(cx)
                            .min(col_depth(cx - 1))
                            .min(col_depth(cx + 1));
                        if depth > 0 && !on_surface {
                            let darken = 100 - (depth * 3).min(24);
                            r = (r as u32 * darken as u32 / 100) as u8;
                            g = (g as u32 * darken as u32 / 100) as u8;
                            b = (b as u32 * darken as u32 / 100) as u8;
                        }
                    }
                    // Compute emission early so gas sidecar can use it to
                    // bake heat into the smooth density blur instead of
                    // relying on per-cell bloom (which produces stippled
                    // grids when adjacent gas cells saturate the blur
                    // kernel).
                    let mut emission: u32 = if c.temp > 500 {
                        (((c.temp - 500) as i32 * 255 / 2000).clamp(0, 255)) as u32
                    } else { 0 };
                    if matches!(c.el, Element::Fire | Element::Lava) {
                        emission = emission.max(220);
                    }
                    if energized_ro[i] && c.el.electrical().glow_color.is_some() {
                        emission = emission.max(180);
                    }
                    let is_gas = matches!(cell_physics(c).kind, Kind::Gas);
                    // Gas sidecar: cloud color + density. Hot gas blends
                    // toward white so the blurred density pass naturally
                    // carries the glow — a hot N₂ cloud reads as a bright
                    // pink-warm haze, not pixel-discrete bright cores
                    // separated by darker gaps. The blend factor is capped
                    // at 0.75 so the gas keeps some of its base hue even
                    // at maximum emission.
                    // Skip warmify for energized cells — the glow color
                    // is the intended visual; warmifying dilutes vivid
                    // discharge colors (Ne orange, Ar purple, etc.)
                    // toward dull peach. Only apply warmify for thermal
                    // emission (hot gases) where the white-shift is
                    // physically meaningful.
                    let (gas_r, gas_g, gas_b) = if is_gas && emission > 0 && !energized_ro[i] {
                        let warmth = (emission as f32 / 255.0 * 0.75).min(0.75);
                        let toward_white = |c: u8, w: f32| -> u8 {
                            (c as f32 + (255.0 - c as f32) * w).round() as u8
                        };
                        (
                            toward_white(r, warmth),
                            toward_white(g, warmth),
                            toward_white(b, warmth),
                        )
                    } else {
                        (r, g, b)
                    };
                    if is_gas {
                        gas_pix[0] = gas_r;
                        gas_pix[1] = gas_g;
                        gas_pix[2] = gas_b;
                        gas_pix[3] = 255;
                        r = 0; g = 0; b = 0;
                    } else {
                        gas_pix[0] = 0;
                        gas_pix[1] = 0;
                        gas_pix[2] = 0;
                        gas_pix[3] = 0;
                    }
                    base_pix[0] = r;
                    base_pix[1] = g;
                    base_pix[2] = b;
                    base_pix[3] = 255;
                    // Bright sidecar: emission-tinted color, inherits cell hue.
                    if emission == 0 {
                        bright_pix[0] = 0;
                        bright_pix[1] = 0;
                        bright_pix[2] = 0;
                        bright_pix[3] = 0;
                    } else {
                        // Source-color for bloom is pre-gas-mask (gas cells
                        // have their atom pixel hidden but should still
                        // glow if hot enough). Gas cells write a damped
                        // bloom contribution (1/4) since the warmified gas
                        // sidecar already carries most of the temperature
                        // signal — full per-cell bloom would re-introduce
                        // the stippled-grid look this option is fixing.
                        let (br, bg, bb, gas_atten) = if is_gas {
                            (gas_r as u32, gas_g as u32, gas_b as u32, 4u32)
                        } else {
                            (r as u32, g as u32, b as u32, 1u32)
                        };
                        bright_pix[0] = ((br * emission) / (255 * gas_atten)).min(255) as u8;
                        bright_pix[1] = ((bg * emission) / (255 * gas_atten)).min(255) as u8;
                        bright_pix[2] = ((bb * emission) / (255 * gas_atten)).min(255) as u8;
                        bright_pix[3] = 255;
                    }
                });
        }
        texture.update(&image);
        kind_texture.update(&kind_image);

        // Gate the bright/gas pipeline on actual content. When the grid
        // has no emissive cells (no Fire/Lava/hot/energized) we skip the
        // upload + 2 bloom blur passes entirely — that's ~1.5 ms saved
        // per frame for sand-only scenes. The transition frame still
        // runs the full pipeline so bloom_v_target ends up cleared to
        // zeros; subsequent frames sample the stale (zero) target
        // safely until emission returns.
        let has_emission_now = world.has_emission();
        let has_gas_now = world.has_any_gas();
        let do_bloom = has_emission_now || last_had_emission;
        let do_gas = has_gas_now || last_had_gas;
        last_had_emission = has_emission_now;
        last_had_gas = has_gas_now;

        let pass_camera = |target: &RenderTarget| -> Camera2D {
            let mut cam = Camera2D::from_display_rect(
                Rect::new(0.0, 0.0, W as f32, H as f32),
            );
            cam.render_target = Some(target.clone());
            cam
        };
        let dest_full = || DrawTextureParams {
            dest_size: Some(vec2(W as f32, H as f32)),
            ..Default::default()
        };

        if do_bloom {
            bright_texture.update(&bright_image);
            // Bloom horizontal — bright_texture → bloom_h_target.
            set_camera(&pass_camera(&bloom_h_target));
            clear_background(BLACK);
            blur_bloom_material.set_uniform("inv_size", inv_size);
            blur_bloom_material.set_uniform("direction", vec2(1.0, 0.0));
            gl_use_material(&blur_bloom_material);
            draw_texture_ex(&bright_texture, 0.0, 0.0, WHITE, dest_full());
            gl_use_default_material();
            // Bloom vertical — bloom_h_target → bloom_v_target.
            set_camera(&pass_camera(&bloom_v_target));
            clear_background(BLACK);
            blur_bloom_material.set_uniform("inv_size", inv_size);
            blur_bloom_material.set_uniform("direction", vec2(0.0, 1.0));
            gl_use_material(&blur_bloom_material);
            draw_texture_ex(&bloom_h_target.texture, 0.0, 0.0, WHITE, dest_full());
            gl_use_default_material();
        }

        if do_gas {
            gas_texture.update(&gas_image);
            // Gas horizontal — gas_texture → gas_h_target.
            set_camera(&pass_camera(&gas_h_target));
            clear_background(BLACK);
            blur_gas_material.set_uniform("inv_size", inv_size);
            blur_gas_material.set_uniform("direction", vec2(1.0, 0.0));
            gl_use_material(&blur_gas_material);
            draw_texture_ex(&gas_texture, 0.0, 0.0, WHITE, dest_full());
            gl_use_default_material();
            // Gas vertical — gas_h_target → gas_v_target.
            set_camera(&pass_camera(&gas_v_target));
            clear_background(BLACK);
            blur_gas_material.set_uniform("inv_size", inv_size);
            blur_gas_material.set_uniform("direction", vec2(0.0, 1.0));
            gl_use_material(&blur_gas_material);
            draw_texture_ex(&gas_h_target.texture, 0.0, 0.0, WHITE, dest_full());
            gl_use_default_material();
        }

        // Back to screen for the composite + UI.
        set_default_camera();
        clear_background(panel_bg());
        composite_material.set_texture("bloom_tex", bloom_v_target.texture.clone());
        composite_material.set_texture("gas_tex", gas_v_target.texture.clone());
        composite_material.set_texture("kind_tex", kind_texture.clone());
        composite_material.set_uniform("u_time", get_time() as f32);
        gl_use_material(&composite_material);
        draw_texture_ex(
            &texture, sim_x, sim_y, WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(sim_w, sim_h)),
                ..Default::default()
            },
        );
        gl_use_default_material();
        perf_t_render_us += perf_t_render_start.elapsed().as_micros() as u64;

        let perf_t_ui_start = std::time::Instant::now();
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
            let emf_label = if world.active_emf > 0.0 {
                format!("{:.0}V", world.active_emf)
            } else {
                "—".to_string()
            };
            let rows = [
                ("Temp", format!("{:+}°C", ambient_actual)),
                ("O₂",   format!("{:.0}%", world.ambient_oxygen * 100.0)),
                ("Grav", format!("{:.1}×", world.gravity)),
                ("EMF",  emf_label),
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

            // Prefab sub-panel — kind dropdown + parameter rows + material.
            if tool_mode == ToolMode::Prefab {
                let sel = prefab_kind_selector_rect();
                draw_ui_text(
                    "PREFAB",
                    sel.0 + 2.0, sel.1 - 10.0, 11.0,
                    Color::from_rgba(130, 130, 150, 255),
                );
                let hit_k = |r: (f32, f32, f32, f32)|
                    mx >= r.0 && mx < r.0 + r.2
                    && my >= r.1 && my < r.1 + r.3;
                // Kind selector — shows current kind's icon + name +
                // a chevron suggesting it expands.
                let sel_hov = hit_k(sel);
                let sel_bg = if sel_hov {
                    Color::from_rgba(45, 50, 70, 255)
                } else {
                    Color::from_rgba(28, 30, 42, 255)
                };
                draw_rectangle(sel.0, sel.1, sel.2, sel.3, sel_bg);
                draw_rectangle_lines(
                    sel.0, sel.1, sel.2, sel.3, 1.0,
                    Color::from_rgba(70, 75, 95, 255),
                );
                let icon_size = 18.0;
                let icon_x = sel.0 + 10.0;
                let icon_y = sel.1 + (sel.3 - icon_size) * 0.5;
                draw_prefab_icon(prefab_kind, icon_x, icon_y, icon_size);
                draw_ui_text(
                    prefab_kind.label(),
                    sel.0 + 10.0 + icon_size + 8.0,
                    sel.1 + sel.3 * 0.5 + 5.0,
                    14.0,
                    Color::from_rgba(220, 220, 230, 255),
                );
                let chev = if prefab_kind_dropdown_open { "▲" } else { "▼" };
                let cd = measure_ui_text(chev, 12);
                draw_ui_text(
                    chev,
                    sel.0 + sel.2 - cd.width - 10.0,
                    sel.1 + sel.3 * 0.5 + 4.0,
                    12.0,
                    Color::from_rgba(160, 160, 180, 255),
                );
                if prefab_kind_dropdown_open {
                    // Option list — covers the parameter / material area.
                    let opt = prefab_kind_option_rects();
                    for (i, opt_kind) in PREFAB_KINDS.iter().enumerate() {
                        let r = opt[i];
                        let active = *opt_kind == prefab_kind;
                        let hov = hit_k(r);
                        let bg = if active {
                            Color::from_rgba(50, 80, 130, 255)
                        } else if hov {
                            Color::from_rgba(38, 40, 52, 255)
                        } else {
                            Color::from_rgba(22, 24, 32, 255)
                        };
                        draw_rectangle(r.0, r.1, r.2, r.3, bg);
                        draw_rectangle_lines(
                            r.0, r.1, r.2, r.3, 1.0,
                            Color::from_rgba(50, 52, 64, 255),
                        );
                        let oi_size = 16.0;
                        draw_prefab_icon(*opt_kind, r.0 + 10.0,
                            r.1 + (r.3 - oi_size) * 0.5, oi_size);
                        draw_ui_text(
                            opt_kind.label(),
                            r.0 + 10.0 + oi_size + 8.0,
                            r.1 + r.3 * 0.5 + 5.0,
                            13.0,
                            Color::from_rgba(220, 220, 230, 255),
                        );
                    }
                } else {
                    // Parameter rows. Voltage is Battery-only.
                    let row_count = if prefab_kind == PrefabKind::Battery { 4 } else { 3 };
                    let labels: [(&str, String); 4] = [
                        ("Thickness", prefab_thickness.to_string()),
                        (param_width_label(prefab_kind), prefab_width.to_string()),
                        ("Height",     prefab_height.to_string()),
                        ("Voltage",    format!("{} V", prefab_voltage)),
                    ];
                    let sr = prefab_slider_rects();
                    for i in 0..row_count {
                        let r = sr[i];
                        let hov = hit_k(r);
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
                    let mr = prefab_material_rect();
                    let mr_hov = hit_k(mr);
                    draw_panel_button(
                        mr,
                        &format!("Material: {}", prefab_material.display_label()),
                        false, mr_hov,
                    );
                    draw_ui_text(
                        "click in sim to place",
                        mr.0 + 2.0, mr.1 + mr.3 + 14.0, 11.0,
                        Color::from_rgba(130, 130, 150, 255),
                    );
                }
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
                    &format!("Material: {}", wire_material.display_label()),
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
                    } else { el.display_label() },
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
                        } else { key.0.display_label() };
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
                        } else { el.display_label() };
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
                    } else { selected.display_label() },
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
                cell.el.display_label()
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
            // Mixed-species sentinel (solute_el == Empty + solute_amt > 0)
            // shows as "Wastewater"; Empty would otherwise fall through
            // to its "Erase" tool label which reads as a UI bug.
            if cell.solute_amt > 0 {
                let label = if cell.solute_el == Element::Empty {
                    "Wastewater".to_string()
                } else if cell.solute_el == Element::Derived {
                    derived_formula_of(cell.solute_derived_id)
                } else {
                    cell.solute_el.display_label()
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

        // Toast overlay — fades over the last ~30 frames of its window.
        // Drawn after everything else so it sits on top of UI and any
        // open overlays. Centered at the top of the sim area (left of
        // the side panel). Captured by the screenshot below.
        if toast_frames > 0 && !toast_msg.is_empty() {
            let fade = if toast_frames < 30 {
                toast_frames as f32 / 30.0
            } else { 1.0 };
            let alpha = (fade * 240.0) as u8;
            let dim = measure_ui_text(&toast_msg, 16);
            let sim_w = (W as f32) * 3.25;
            let bx = (sim_w - dim.width) * 0.5 - 12.0;
            let by = 24.0;
            draw_rectangle(
                bx, by, dim.width + 24.0, 28.0,
                Color::from_rgba(20, 20, 28, alpha.min(220)),
            );
            draw_rectangle_lines(
                bx, by, dim.width + 24.0, 28.0, 2.0,
                Color::from_rgba(
                    (toast_color.r * 255.0) as u8,
                    (toast_color.g * 255.0) as u8,
                    (toast_color.b * 255.0) as u8,
                    alpha,
                ),
            );
            let mut c = toast_color;
            c.a = fade;
            draw_ui_text(
                &toast_msg, bx + 12.0, by + 20.0, 16.0, c,
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

        perf_t_ui_us += perf_t_ui_start.elapsed().as_micros() as u64;

        perf_frames += 1;
        if perf_last_print.elapsed().as_secs_f32() >= 1.0 {
            let f = perf_frames as f32;
            let to_ms = |us: u64| us as f32 / f / 1000.0;
            let step = to_ms(perf_t_step_us);
            let render = to_ms(perf_t_render_us);
            let ui = to_ms(perf_t_ui_us);
            println!(
                "[perf] fps={} cpu_total={:.2}ms step={:.2}ms render={:.2}ms ui={:.2}ms",
                get_fps(),
                step + render + ui,
                step,
                render,
                ui,
            );
            perf_t_step_us = 0;
            perf_t_render_us = 0;
            perf_t_ui_us = 0;
            perf_frames = 0;
            perf_last_print = std::time::Instant::now();
        }

        next_frame().await
    }
}
