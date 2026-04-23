use alembic::{Cell, Element, H, W, World};
use macroquad::prelude::Vec2;
use serial_test::serial;

const TEST_SEED: u64 = 0xD3AD_B33F;

fn deterministic_paint_and_step(world: &mut World, n_steps: usize) {
    for step in 0..n_steps {
        let x = ((step * 37) % W) as i32;
        let y = ((step * 53) % H) as i32;
        let radius = (step % 5 + 1) as i32;
        let wind = match step % 4 {
            0 => Vec2::ZERO,
            1 => Vec2::new(0.25, 0.0),
            2 => Vec2::new(0.0, -0.25),
            _ => Vec2::new(-0.25, 0.25),
        };

        world.paint(x, y, radius, Element::Stone, 0, false);
        world.step(wind);
    }
}

fn cells_equal(left: &Cell, right: &Cell) -> bool {
    left.el == right.el
        && left.derived_id == right.derived_id
        && left.life == right.life
        && left.seed == right.seed
        && left.flag == right.flag
        && left.temp == right.temp
        && left.moisture == right.moisture
        && left.burn == right.burn
        && left.pressure == right.pressure
        && left.solute_el == right.solute_el
        && left.solute_amt == right.solute_amt
        && left.solute_derived_id == right.solute_derived_id
}

fn format_cell(cell: &Cell) -> String {
    format!(
        "Cell{{el:{:?},derived_id:{},life:{},seed:{},flag:{},temp:{},moisture:{},burn:{},pressure:{},solute_el:{:?},solute_amt:{},solute_derived_id:{}}}",
        cell.el,
        cell.derived_id,
        cell.life,
        cell.seed,
        cell.flag,
        cell.temp,
        cell.moisture,
        cell.burn,
        cell.pressure,
        cell.solute_el,
        cell.solute_amt,
        cell.solute_derived_id
    )
}

fn assert_cells_eq(a: &[Cell], b: &[Cell], context_msg: &str) {
    assert_eq!(a.len(), b.len(), "{context_msg}: different grid lengths");
    for (idx, (left, right)) in a.iter().zip(b.iter()).enumerate() {
        if !cells_equal(left, right) {
            panic!(
                "{context_msg}: first mismatch at index {idx}; left={}, right={}",
                format_cell(left),
                format_cell(right)
            );
        }
    }
}

#[test]
#[serial]
fn fixed_seed_produces_deterministic_cells() {
    macroquad::rand::srand(TEST_SEED);
    let mut a = World::new();
    deterministic_paint_and_step(&mut a, 128);

    macroquad::rand::srand(TEST_SEED);
    let mut b = World::new();
    deterministic_paint_and_step(&mut b, 128);

    assert_cells_eq(
        &a.cells,
        &b.cells,
        "same seed and same operations must produce identical cell grids",
    );
}
