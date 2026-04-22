//! Smoke test for the headless simulation harness.
//!
//! Proves that `alembic::World` can be constructed, ticked, and inspected
//! without a macroquad window or any graphics context. Every other
//! integration test depends on this path working.

use alembic::{Cell, World, H, W};
use macroquad::prelude::Vec2;

#[test]
fn world_new_produces_empty_grid() {
    let world = World::new();
    assert_eq!(world.cells.len(), W * H, "grid must be W*H cells");
    for (i, cell) in world.cells.iter().enumerate() {
        assert_eq!(cell.el, Cell::EMPTY.el, "cell {i} not empty at construction");
    }
    assert_eq!(world.shockwaves.len(), 0, "no shockwaves at construction");
    assert_eq!(world.frame, 0, "frame counter starts at zero");
}

#[test]
fn tick_100_frames_zero_wind_is_stable() {
    macroquad::rand::srand(0xA1E_B1C);
    let mut world = World::new();
    for i in 0..100 {
        world.step(Vec2::ZERO);
        assert_eq!(world.cells.len(), W * H, "grid resized on tick {i}");
        for sw in &world.shockwaves {
            assert!(sw.cx.is_finite(), "shockwave cx non-finite on tick {i}");
            assert!(sw.cy.is_finite(), "shockwave cy non-finite on tick {i}");
            assert!(sw.radius.is_finite(), "shockwave radius non-finite on tick {i}");
            assert!(sw.yield_p.is_finite(), "shockwave yield non-finite on tick {i}");
        }
    }
}
