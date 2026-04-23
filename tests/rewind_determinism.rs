use alembic::{Cell, Element, HISTORY_CAPACITY, H, W, World};
use macroquad::prelude::Vec2;
use serial_test::serial;

const TEST_SEED: u64 = 0xD3AD_B33F;

// NOTE: skipped query-purity invariants. `World::in_bounds` and private UI
// label helpers are not reachable from integration tests without exposing
// internals.
// NOTE: skipped rewind fidelity checks for shockwaves/frame/electrical state.
// Rewind scope here is cell-grid only per spec.
// NOTE: skipped fresh-world equivalence test. Covered by
// `repeated_world_new_produces_identical_initial_state` in
// `tests/tick_robustness.rs`.

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
        if left != right {
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
fn ring_buffer_bounds_hold_and_count_saturates() {
    macroquad::rand::srand(TEST_SEED);
    let mut world = World::new();
    let total_steps = HISTORY_CAPACITY * 2 + 17;

    for i in 0..total_steps {
        world.step(Vec2::ZERO);
        assert!(
            world.history_count <= HISTORY_CAPACITY,
            "history_count exceeded capacity at step {i}"
        );
        assert!(
            world.history_write < HISTORY_CAPACITY,
            "history_write out of range at step {i}"
        );
        assert!(
            world.rewind_offset <= world.history_count,
            "rewind_offset exceeded history_count at step {i}"
        );
    }

    assert_eq!(
        world.history_count, HISTORY_CAPACITY,
        "history_count should saturate at HISTORY_CAPACITY"
    );
    world.step(Vec2::ZERO);
    assert_eq!(
        world.history_count, HISTORY_CAPACITY,
        "history_count should remain saturated once full"
    );
}

#[test]
#[serial]
fn stepping_while_rewound_resets_rewind_offset() {
    macroquad::rand::srand(TEST_SEED + 1);
    let mut world = World::new();
    deterministic_paint_and_step(&mut world, 24);

    world.seek(5);
    assert!(world.rewind_offset > 0, "precondition: world must be rewound");

    // step() auto-snapshots at the end of each tick (lib.rs:3560), so
    // stepping while rewound triggers the "snapshot resets rewind_offset"
    // invariant.
    world.step(Vec2::new(0.4, -0.2));
    assert_eq!(
        world.rewind_offset, 0,
        "taking a new snapshot should return to live timeline"
    );
}

#[test]
#[serial]
fn snapshot_storage_is_deep_copy_not_alias() {
    macroquad::rand::srand(TEST_SEED + 2);
    let mut world = World::new();
    world.step(Vec2::ZERO);
    assert!(world.history_count > 0, "history must contain at least one frame");

    let latest_idx = (world.history_write + HISTORY_CAPACITY - 1) % HISTORY_CAPACITY;
    let snapshot_before_mutation = world.history[latest_idx].clone();

    world.cells[0].el = Element::Sand;
    world.cells[1].el = Element::Water;
    world.cells[W + 1].temp = world.cells[W + 1].temp.saturating_add(25);
    world.cells[2 * W + 2].moisture = world.cells[2 * W + 2].moisture.saturating_add(50);

    assert!(
        world.history[latest_idx] == snapshot_before_mutation,
        "mutating live cells must not mutate stored snapshot"
    );
}

#[test]
#[serial]
fn forward_then_rewind_returns_byte_identical_cells() {
    macroquad::rand::srand(TEST_SEED + 3);
    let mut world = World::new();

    world.step(Vec2::ZERO);
    let baseline = world.cells.clone();
    let rewind_steps = 80;

    deterministic_paint_and_step(&mut world, rewind_steps);
    world.seek(rewind_steps as i32);

    assert_cells_eq(
        &world.cells,
        &baseline,
        "forward N then rewind N must return to baseline snapshot",
    );
}

#[test]
#[serial]
fn rewind_is_idempotent_at_oldest_boundary() {
    macroquad::rand::srand(TEST_SEED + 4);
    let mut world = World::new();
    deterministic_paint_and_step(&mut world, HISTORY_CAPACITY + 32);

    world.seek((HISTORY_CAPACITY * 8) as i32);
    let boundary = world.history_count.saturating_sub(1);
    assert_eq!(
        world.rewind_offset, boundary,
        "seek should clamp to oldest available snapshot"
    );

    let before = world.rewind_offset;
    world.seek(1);
    assert_eq!(
        world.rewind_offset, before,
        "additional rewind at boundary should be a no-op"
    );
}

#[test]
#[serial]
fn fixed_seed_produces_deterministic_cells() {
    let steps = 8;
    macroquad::rand::srand(TEST_SEED + 5);
    let mut a = World::new();
    deterministic_paint_and_step(&mut a, steps);

    macroquad::rand::srand(TEST_SEED + 5);
    let mut b = World::new();
    deterministic_paint_and_step(&mut b, steps);

    assert_cells_eq(
        &a.cells,
        &b.cells,
        "same seed and same inputs should produce identical cell grids",
    );
}
