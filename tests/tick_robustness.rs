use std::time::{Duration, Instant};

use alembic::{Cell, Element, World, H, W};
use macroquad::prelude::Vec2;

const FLAG_UPDATED: u8 = 0x01;

const ALL_ELEMENTS: &[Element] = &[
    Element::Empty,
    Element::Sand,
    Element::Water,
    Element::Stone,
    Element::Wood,
    Element::Fire,
    Element::CO2,
    Element::Steam,
    Element::Lava,
    Element::Obsidian,
    Element::Seed,
    Element::Mud,
    Element::Leaves,
    Element::Oil,
    Element::Ice,
    Element::MoltenGlass,
    Element::Glass,
    Element::Charcoal,
    Element::H,
    Element::He,
    Element::C,
    Element::N,
    Element::O,
    Element::F,
    Element::Ne,
    Element::Na,
    Element::Mg,
    Element::Al,
    Element::Si,
    Element::P,
    Element::S,
    Element::Cl,
    Element::K,
    Element::Ca,
    Element::Fe,
    Element::Cu,
    Element::Au,
    Element::Hg,
    Element::U,
    Element::Rust,
    Element::Salt,
    Element::Derived,
    Element::Gunpowder,
    Element::Quartz,
    Element::Firebrick,
    Element::Ar,
    Element::BattPos,
    Element::BattNeg,
    Element::Zn,
    Element::Ag,
    Element::Ni,
    Element::Pb,
    Element::B,
    Element::Ra,
    Element::Cs,
];

#[inline]
fn random_element() -> Element {
    let idx = macroquad::rand::gen_range(0, ALL_ELEMENTS.len() as i32) as usize;
    ALL_ELEMENTS[idx]
}

#[inline]
fn random_i32_inclusive(min: i32, max: i32) -> i32 {
    macroquad::rand::gen_range(min, max + 1)
}

#[inline]
fn random_u8() -> u8 {
    macroquad::rand::gen_range::<u8>(0, u8::MAX)
}

#[inline]
fn random_bool() -> bool {
    macroquad::rand::gen_range::<u8>(0, 2) == 0
}

fn assert_world_floats_finite(world: &World, tick: u32) {
    assert!(
        world.battery_voltage.is_finite(),
        "tick {tick}: world.battery_voltage non-finite: {}",
        world.battery_voltage
    );
    assert!(
        world.galvanic_voltage.is_finite(),
        "tick {tick}: world.galvanic_voltage non-finite: {}",
        world.galvanic_voltage
    );
    assert!(
        world.active_emf.is_finite(),
        "tick {tick}: world.active_emf non-finite: {}",
        world.active_emf
    );
    assert!(
        world.gravity.is_finite(),
        "tick {tick}: world.gravity non-finite: {}",
        world.gravity
    );
    assert!(
        world.ambient_oxygen.is_finite(),
        "tick {tick}: world.ambient_oxygen non-finite: {}",
        world.ambient_oxygen
    );

    for (i, sw) in world.shockwaves.iter().enumerate() {
        assert!(sw.cx.is_finite(), "tick {tick}: shockwave[{i}].cx non-finite: {}", sw.cx);
        assert!(sw.cy.is_finite(), "tick {tick}: shockwave[{i}].cy non-finite: {}", sw.cy);
        assert!(
            sw.radius.is_finite(),
            "tick {tick}: shockwave[{i}].radius non-finite: {}",
            sw.radius
        );
        assert!(
            sw.yield_p.is_finite(),
            "tick {tick}: shockwave[{i}].yield_p non-finite: {}",
            sw.yield_p
        );
    }

    for _cell in &world.cells {
        // TODO: Cell currently has no f32 fields; add scans here if any are introduced.
    }
}

#[test]
fn fuzz_paint_tick_no_panic_and_all_floats_finite() {
    macroquad::rand::srand(0x01_F022);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;

    for tick in 0..500 {
        let element = random_element();
        let cx = random_i32_inclusive(-10, w + 10);
        let cy = random_i32_inclusive(-10, h + 10);
        let radius = random_i32_inclusive(0, 20);
        let derived_id = random_u8();
        let frozen = random_bool();
        world.paint(cx, cy, radius, element, derived_id, frozen);

        if macroquad::rand::gen_range::<u32>(0, 100) < 5 {
            let sx = random_i32_inclusive(-10, w + 10);
            let sy = random_i32_inclusive(-10, h + 10);
            let yield_p = macroquad::rand::gen_range(0.0f32, 1000.0f32);
            world.spawn_shockwave_capped(sx, sy, yield_p, 5000.0);
        }

        let wind = Vec2::new(
            macroquad::rand::gen_range(-5.0f32, 5.0f32),
            macroquad::rand::gen_range(-5.0f32, 5.0f32),
        );
        world.step(wind);
        assert_world_floats_finite(&world, tick);
    }
}

#[test]
fn tick_completes_in_bounded_time_under_adversarial_paint() {
    macroquad::rand::srand(0x03_71AE);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;

    for tick in 0..50 {
        for _ in 0..6 {
            let element = random_element();
            let cx = random_i32_inclusive(-10, w + 10);
            let cy = random_i32_inclusive(-10, h + 10);
            let radius = random_i32_inclusive(16, 48);
            world.paint(cx, cy, radius, element, random_u8(), random_bool());
        }

        let shockwaves = random_i32_inclusive(0, 3);
        for _ in 0..shockwaves {
            let sx = random_i32_inclusive(-10, w + 10);
            let sy = random_i32_inclusive(-10, h + 10);
            let yield_p = macroquad::rand::gen_range(0.0f32, 1000.0f32);
            world.spawn_shockwave_capped(sx, sy, yield_p, 5000.0);
        }

        let wind = Vec2::new(
            macroquad::rand::gen_range(-5.0f32, 5.0f32),
            macroquad::rand::gen_range(-5.0f32, 5.0f32),
        );
        let start = Instant::now();
        world.step(wind);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "tick {tick} exceeded 2.0s bound: {elapsed:?}"
        );
        assert_world_floats_finite(&world, tick);
    }
}

#[test]
fn empty_world_stays_empty_for_many_ticks() {
    let mut world = World::new();
    for tick in 0..200 {
        world.step(Vec2::ZERO);
        for (i, cell) in world.cells.iter().enumerate() {
            assert_eq!(
                cell.el,
                Cell::EMPTY.el,
                "tick {tick}: expected empty at cell {i}, got {:?}",
                cell.el
            );
        }
        assert!(
            world.shockwaves.is_empty(),
            "tick {tick}: shockwaves should remain empty"
        );
    }
}

#[test]
fn repeated_world_new_produces_identical_initial_state() {
    macroquad::rand::srand(0x05_C702);
    let baseline = World::new();

    for case in 1..20 {
        macroquad::rand::srand(0x05_C702);
        let world = World::new();

        assert_eq!(
            world.cells.len(),
            baseline.cells.len(),
            "case {case}: cells length mismatch"
        );

        for (i, (actual, expected)) in world.cells.iter().zip(&baseline.cells).enumerate() {
            assert_eq!(
                actual.el, expected.el,
                "case {case}: cell {i} element mismatch"
            );
        }

        assert_eq!(
            world.battery_voltage.to_bits(),
            baseline.battery_voltage.to_bits(),
            "case {case}: battery_voltage mismatch"
        );
        assert_eq!(
            world.galvanic_voltage.to_bits(),
            baseline.galvanic_voltage.to_bits(),
            "case {case}: galvanic_voltage mismatch"
        );
        assert_eq!(
            world.active_emf.to_bits(),
            baseline.active_emf.to_bits(),
            "case {case}: active_emf mismatch"
        );
        assert_eq!(
            world.gravity.to_bits(),
            baseline.gravity.to_bits(),
            "case {case}: gravity mismatch"
        );
        assert_eq!(
            world.ambient_oxygen.to_bits(),
            baseline.ambient_oxygen.to_bits(),
            "case {case}: ambient_oxygen mismatch"
        );
    }
}

#[test]
fn flag_updated_cleared_at_tick_boundary() {
    macroquad::rand::srand(0x06_F1A6);
    let mut world = World::new();
    let cx = (W as i32) / 2;
    let sand_y = (H as i32) - 20;
    let water_y = (H as i32) - 30;

    world.paint(cx, sand_y, 8, Element::Sand, 0, false);
    world.paint(cx, water_y, 6, Element::Water, 0, false);

    world.step(Vec2::ZERO);
    let updated_count = world
        .cells
        .iter()
        .filter(|c| c.flag & FLAG_UPDATED != 0)
        .count();
    assert!(
        updated_count > 0,
        "expected at least one cell with FLAG_UPDATED set after activity tick"
    );

    for cell in &mut world.cells {
        cell.el = Cell::EMPTY.el;
    }
    world.shockwaves.clear();

    world.step(Vec2::ZERO);
    for (i, cell) in world.cells.iter().enumerate() {
        assert_eq!(
            cell.flag & FLAG_UPDATED,
            0,
            "cell {i} retained FLAG_UPDATED after no-op boundary tick; flag=0x{:02X}",
            cell.flag
        );
    }
}

#[test]
fn shockwaves_stay_finite_under_repeated_spawns() {
    macroquad::rand::srand(0x02_5A0A);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;

    for tick in 0..200 {
        let n = random_i32_inclusive(1, 3);
        for _ in 0..n {
            let sx = random_i32_inclusive(-10, w + 10);
            let sy = random_i32_inclusive(-10, h + 10);
            let yield_p = macroquad::rand::gen_range(0.0f32, 1000.0f32);
            world.spawn_shockwave_capped(sx, sy, yield_p, 5000.0);
        }
        world.step(Vec2::ZERO);

        for (i, sw) in world.shockwaves.iter().enumerate() {
            assert!(sw.cx.is_finite(), "tick {tick}: shockwave[{i}].cx non-finite");
            assert!(sw.cy.is_finite(), "tick {tick}: shockwave[{i}].cy non-finite");
            assert!(
                sw.radius.is_finite(),
                "tick {tick}: shockwave[{i}].radius non-finite"
            );
            assert!(
                sw.yield_p.is_finite(),
                "tick {tick}: shockwave[{i}].yield_p non-finite"
            );
        }
    }
}
