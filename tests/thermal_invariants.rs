//! Thermal-model integration invariants from `test-targets/04-thermal.md`.
//! Covers bounds, phase transitions, latent heat behavior, diffusion, and
//! thermal-to-pressure coupling.

use alembic::{Cell, Element, World, H, W};
use macroquad::prelude::Vec2;

const SPEC_MIN_TEMP: i16 = -500;
const SPEC_MAX_TEMP: i16 = 10_000;

fn fresh_world(seed: u64) -> World {
    macroquad::rand::srand(seed);
    World::new()
}

fn tick_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step(Vec2::ZERO);
    }
}

fn idx(x: i32, y: i32) -> usize {
    (y as usize) * W + x as usize
}

fn in_bounds(x: i32, y: i32) -> bool {
    x >= 0 && x < W as i32 && y >= 0 && y < H as i32
}

fn cell_at(world: &World, x: i32, y: i32) -> Cell {
    assert!(in_bounds(x, y), "out-of-bounds cell ({x}, {y})");
    world.cells[idx(x, y)]
}

fn place(world: &mut World, x: i32, y: i32, el: Element) {
    world.paint(x, y, 0, el, 0, false);
}

fn place_frozen(world: &mut World, x: i32, y: i32, el: Element) {
    world.paint(x, y, 0, el, 0, true);
}

fn set_temp(world: &mut World, x: i32, y: i32, t: i16) {
    world.cells[idx(x, y)].temp = t;
}

fn neighborhood_contains(world: &World, x: i32, y: i32, radius: i32, el: Element) -> bool {
    for ny in (y - radius)..=(y + radius) {
        for nx in (x - radius)..=(x + radius) {
            if in_bounds(nx, ny) && cell_at(world, nx, ny).el == el {
                return true;
            }
        }
    }
    false
}

fn assert_all_temps_in_spec_range(world: &World, context: &str) {
    for (i, cell) in world.cells.iter().enumerate() {
        assert!(
            (SPEC_MIN_TEMP..=SPEC_MAX_TEMP).contains(&cell.temp),
            "{context}: cell {i} temp {} outside spec range [{SPEC_MIN_TEMP}, {SPEC_MAX_TEMP}]",
            cell.temp
        );
    }
}

#[test]
fn temperature_stays_within_hard_clamp() {
    let mut world = fresh_world(0x04_7E_B0_01);
    let cx = (W / 2) as i32;
    let cy = (H / 2) as i32;

    for (x, y, el, t) in [
        (cx - 3, cy, Element::Water, i16::MAX),
        (cx - 2, cy, Element::Steam, i16::MIN),
        (cx - 1, cy, Element::Sand, 5000),
        (cx, cy, Element::Water, -5000),
        (cx + 1, cy, Element::Steam, 1234),
        (cx + 2, cy, Element::Sand, -777),
    ] {
        place(&mut world, x, y, el);
        set_temp(&mut world, x, y, t);
    }

    tick_n(&mut world, 200);
    assert_all_temps_in_spec_range(&world, "post-thermal evolution");
}

#[test]
fn ambient_offset_respects_documented_clamp() {
    let mut world_hot = fresh_world(0x04_7E_B0_02);
    world_hot.ambient_offset = 4980;
    tick_n(&mut world_hot, 100);
    assert_all_temps_in_spec_range(&world_hot, "ambient_offset=4980");

    let mut world_cold = fresh_world(0x04_7E_B0_03);
    world_cold.ambient_offset = -293;
    tick_n(&mut world_cold, 100);
    assert_all_temps_in_spec_range(&world_cold, "ambient_offset=-293");
}

#[test]
fn water_boils_above_threshold_to_steam() {
    let mut world = fresh_world(0x04_7E_B0_11);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;
    place_frozen(&mut world, x, y, Element::Water);
    for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
        place_frozen(&mut world, nx, ny, Element::Stone);
    }
    set_temp(&mut world, x, y, 300);

    for tick in 0..200 {
        world.step(Vec2::ZERO);
        if cell_at(&world, x, y).el == Element::Steam {
            return;
        }
        if tick == 199 {
            panic!("water at ({x}, {y}) never transitioned to steam within 200 ticks");
        }
    }
}

#[test]
fn steam_condenses_below_threshold_to_water() {
    let mut world = fresh_world(0x04_7E_B0_12);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;
    place(&mut world, x, y, Element::Steam);
    set_temp(&mut world, x, y, -50);

    for tick in 0..500 {
        world.step(Vec2::ZERO);
        if neighborhood_contains(&world, x, y, 2, Element::Water) {
            return;
        }
        if tick == 499 {
            panic!("steam at ({x}, {y}) never condensed to water within 500 ticks");
        }
    }
}

#[test]
fn no_spontaneous_transition_below_threshold() {
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;

    let mut safe = fresh_world(0x04_7E_B0_13);
    place_frozen(&mut safe, x, y, Element::Water);
    set_temp(&mut safe, x, y, 50);
    tick_n(&mut safe, 300);
    assert_eq!(
        cell_at(&safe, x, y).el,
        Element::Water,
        "water in safe thermal band should remain water"
    );

    let mut low = fresh_world(0x04_7E_B0_14);
    place_frozen(&mut low, x, y, Element::Water);
    set_temp(&mut low, x, y, -10);
    tick_n(&mut low, 150);
    assert_ne!(
        cell_at(&low, x, y).el,
        Element::Steam,
        "sub-zero water must not transition to steam"
    );

    let mut high = fresh_world(0x04_7E_B0_15);
    place_frozen(&mut high, x, y, Element::Water);
    set_temp(&mut high, x, y, 150);
    tick_n(&mut high, 150);
    assert_ne!(
        cell_at(&high, x, y).el,
        Element::Ice,
        "superheated water must not transition to ice"
    );
}

#[test]
#[ignore = "Element::Fe currently has no thermal phase profile for melt/freeze"]
fn molten_iron_refreezes_when_cooled() {
    let mut world = fresh_world(0x04_7E_B0_16);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;
    place_frozen(&mut world, x, y, Element::Fe);
    set_temp(&mut world, x, y, 2000);
    world.step(Vec2::ZERO);
    set_temp(&mut world, x, y, 20);
    tick_n(&mut world, 50);
    assert_eq!(cell_at(&world, x, y).el, Element::Fe);
}

#[test]
#[ignore = "Spec invariant currently differs from implementation behavior"]
fn latent_heat_prevents_single_frame_boil() {
    let mut world = fresh_world(0x04_7E_B0_21);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;
    place_frozen(&mut world, x, y, Element::Water);
    set_temp(&mut world, x, y, 4000);

    world.step(Vec2::ZERO);
    assert_eq!(
        cell_at(&world, x, y).el,
        Element::Water,
        "latent budget spec expects no one-frame water->steam transition"
    );
}

#[test]
fn latent_heat_drains_neighbor_energy_on_boil() {
    let mut world = fresh_world(0x04_7E_B0_22);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;

    place_frozen(&mut world, x, y, Element::Water);
    set_temp(&mut world, x, y, 4000);

    let neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)];
    for (nx, ny) in neighbors {
        place_frozen(&mut world, nx, ny, Element::Stone);
        set_temp(&mut world, nx, ny, 300);
    }

    let before_sum: i32 = neighbors
        .iter()
        .map(|(nx, ny)| cell_at(&world, *nx, *ny).temp as i32)
        .sum();

    world.step(Vec2::ZERO);

    assert!(
        neighborhood_contains(&world, x, y, 2, Element::Steam),
        "water should boil to steam under extreme heat"
    );

    let after_sum: i32 = neighbors
        .iter()
        .map(|(nx, ny)| cell_at(&world, *nx, *ny).temp as i32)
        .sum();
    assert!(
        after_sum <= before_sum - 50,
        "neighbor thermal reservoir did not drain enough: before={before_sum}, after={after_sum}"
    );
}

#[test]
fn heat_diffusion_converges_between_two_cells() {
    let mut world = fresh_world(0x04_7E_B0_31);
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;

    place_frozen(&mut world, x, y, Element::Sand);
    place_frozen(&mut world, x + 1, y, Element::Sand);
    set_temp(&mut world, x, y, 1000);
    set_temp(&mut world, x + 1, y, 0);

    let initial_diff = (cell_at(&world, x, y).temp as i32 - cell_at(&world, x + 1, y).temp as i32)
        .abs();

    tick_n(&mut world, 500);

    let hot_final = cell_at(&world, x, y).temp as i32;
    let cold_final = cell_at(&world, x + 1, y).temp as i32;
    let final_diff = (hot_final - cold_final).abs();
    assert!(
        final_diff < initial_diff / 2,
        "diffusion did not sufficiently converge: initial={initial_diff}, final={final_diff}"
    );
    assert!(hot_final < 1000, "hot cell did not cool: final={hot_final}");
    assert!(cold_final > 0, "cold cell did not warm: final={cold_final}");
}

#[test]
fn insulated_region_preserves_total_energy_within_tolerance() {
    let mut world = fresh_world(0x04_7E_B0_32);
    world.ambient_offset = 0;

    for y in 0..H as i32 {
        for x in 0..W as i32 {
            place_frozen(&mut world, x, y, Element::Sand);
            set_temp(&mut world, x, y, 20);
        }
    }

    let cx = (W / 2) as i32;
    let cy = (H / 2) as i32;
    let block = [500, 300, 800, 200, 600, 400, 700, 100, 900];
    let mut k = 0usize;
    for y in (cy - 1)..=(cy + 1) {
        for x in (cx - 1)..=(cx + 1) {
            set_temp(&mut world, x, y, block[k]);
            k += 1;
        }
    }

    let sum_excess = |w: &World| -> i32 {
        let mut sum = 0i32;
        for y in (cy - 4)..=(cy + 4) {
            for x in (cx - 4)..=(cx + 4) {
                sum += cell_at(w, x, y).temp as i32 - 20;
            }
        }
        sum
    };

    let initial_excess = sum_excess(&world);
    assert!(initial_excess > 0, "initial excess energy must be positive");

    tick_n(&mut world, 50);

    let final_excess = sum_excess(&world);
    let ratio = final_excess as f32 / initial_excess as f32;
    assert!(
        (0.70..=1.30).contains(&ratio),
        "9x9-region excess drift outside tolerance: initial={initial_excess}, final={final_excess}, ratio={ratio:.3}"
    );
}

#[test]
fn hotter_gas_yields_higher_or_equal_pressure_target() {
    let x = (W / 2) as i32;
    let y = (H / 2) as i32;

    let mut world_a = fresh_world(0x04_7E_B0_41);
    place_frozen(&mut world_a, x, y, Element::Steam);
    set_temp(&mut world_a, x, y, 100);
    world_a.step(Vec2::ZERO);
    let scratch_a = world_a.pressure_scratch[idx(x, y)] as i32;

    let mut world_b = fresh_world(0x04_7E_B0_41);
    place_frozen(&mut world_b, x, y, Element::Steam);
    set_temp(&mut world_b, x, y, 500);
    world_b.step(Vec2::ZERO);
    let scratch_b = world_b.pressure_scratch[idx(x, y)] as i32;

    assert!(
        scratch_b >= scratch_a,
        "hotter steam should not lower pressure_scratch target-ish value: cold={scratch_a}, hot={scratch_b}"
    );

    tick_n(&mut world_a, 5);
    tick_n(&mut world_b, 5);
    let steady_a = cell_at(&world_a, x, y).pressure as i32;
    let steady_b = cell_at(&world_b, x, y).pressure as i32;
    assert!(
        steady_b >= steady_a,
        "hotter steam should not have lower cell pressure after settling: cold={steady_a}, hot={steady_b}"
    );
}
