//! Shockwave invariants from `test-targets/07-shockwaves.md`.

use alembic::{Cell, World, H, W};
use macroquad::prelude::Vec2;
use serial_test::serial;

const SEED_PREFIX: u64 = 0x07_5700;

fn fresh_world(seed: u64) -> World {
    macroquad::rand::srand(seed);
    World::new()
}

fn tick_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step(Vec2::ZERO);
    }
}

#[inline]
fn idx(x: i32, y: i32) -> usize {
    (y as usize) * W + x as usize
}

#[inline]
fn in_bounds(x: i32, y: i32) -> bool {
    x >= 0 && x < W as i32 && y >= 0 && y < H as i32
}

#[inline]
fn cell_at(world: &World, x: i32, y: i32) -> Cell {
    assert!(in_bounds(x, y), "requested out-of-bounds cell ({x}, {y})");
    world.cells[idx(x, y)]
}

#[inline]
fn center() -> (i32, i32) {
    (W as i32 / 2, H as i32 / 2)
}

/// Invariant: `spawn_shockwave` clamps to the default cap.
#[test]
#[serial]
fn spawn_shockwave_clamps_to_default_cap() {
    let mut world = fresh_world(SEED_PREFIX + 0x01);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 1_000_000.0);

    assert_eq!(world.shockwaves.len(), 1, "spawn did not create exactly one wave");
    assert_eq!(world.shockwaves[0].yield_p, 50_000.0, "default cap should clamp to 50000");
}

/// Invariant: `spawn_shockwave_capped` clamps to caller-provided cap.
#[test]
#[serial]
fn spawn_shockwave_capped_respects_custom_cap() {
    let mut world = fresh_world(SEED_PREFIX + 0x02);
    let (cx, cy) = center();
    world.spawn_shockwave_capped(cx, cy, 9_999.0, 1_234.0);

    assert_eq!(world.shockwaves.len(), 1, "spawn did not create exactly one wave");
    assert_eq!(world.shockwaves[0].yield_p, 1_234.0, "custom cap should clamp yield");
}

/// Invariant: active shockwaves keep finite origin coordinates.
#[test]
#[serial]
fn active_shockwave_origins_are_finite() {
    let mut world = fresh_world(SEED_PREFIX + 0x03);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 10_000.0);

    for tick in 0..8 {
        world.step(Vec2::ZERO);
        for (i, sw) in world.shockwaves.iter().enumerate() {
            assert!(sw.cx.is_finite(), "wave {i} has non-finite cx on tick {tick}");
            assert!(sw.cy.is_finite(), "wave {i} has non-finite cy on tick {tick}");
        }
    }
}

/// Invariant: radius strictly increases each tick until retirement.
#[test]
#[serial]
fn radius_grows_monotonically_until_retired() {
    let mut world = fresh_world(SEED_PREFIX + 0x04);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 50_000.0);

    let mut previous = 0.0_f32;
    let mut observed = 0usize;
    while !world.shockwaves.is_empty() {
        world.step(Vec2::ZERO);
        if world.shockwaves.is_empty() {
            break;
        }
        let r = world.shockwaves[0].radius;
        assert!(r > previous, "radius did not strictly increase: prev={previous}, now={r}");
        previous = r;
        observed += 1;
        assert!(observed < 300, "wave did not retire in expected time");
    }
    assert!(observed > 0, "no radius updates observed");
}

/// Invariant: a typical wave retires within bounded, deterministic frames.
#[test]
#[serial]
fn typical_shockwave_retires_within_reasonable_frames() {
    let mut world = fresh_world(SEED_PREFIX + 0x05);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 10_000.0);

    let mut frames = 0usize;
    while !world.shockwaves.is_empty() {
        world.step(Vec2::ZERO);
        frames += 1;
        assert!(frames < 200, "wave did not retire before 200 frames");
    }

    assert!(frames > 0, "wave retired without advancing any frame");
}

/// Invariant: leading edge injects pressure where it crosses.
#[test]
#[serial]
fn leading_edge_increases_local_pressure() {
    let mut world = fresh_world(SEED_PREFIX + 0x06);
    let (cx, cy) = center();
    let sample_points = [(cx + 5, cy), (cx - 5, cy), (cx, cy + 5), (cx, cy - 5)];

    let mut baseline = [0_i16; 4];
    for (i, (x, y)) in sample_points.iter().enumerate() {
        baseline[i] = cell_at(&world, *x, *y).pressure;
    }

    world.spawn_shockwave(cx, cy, 6_000.0);
    world.step(Vec2::ZERO);

    let mut rose = false;
    for (i, (x, y)) in sample_points.iter().enumerate() {
        let now = cell_at(&world, *x, *y).pressure;
        if now > baseline[i] {
            rose = true;
        }
    }
    assert!(rose, "no sampled leading-edge cell pressure increased");
}

/// Invariant: effective magnitude follows inverse-square falloff over radius.
#[test]
#[serial]
fn effective_magnitude_decreases_with_radius() {
    let mut world = fresh_world(SEED_PREFIX + 0x07);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 40_000.0);

    let mut previous_radius = 0.0_f32;
    let mut previous_mag = f32::INFINITY;
    let mut samples = 0usize;

    while !world.shockwaves.is_empty() {
        world.step(Vec2::ZERO);
        if world.shockwaves.is_empty() {
            break;
        }

        let sw = world.shockwaves[0];
        let r_mid = (previous_radius + sw.radius) * 0.5;
        let decay = 1.0 + r_mid / 6.0;
        let mag = sw.yield_p / (decay * decay);
        assert!(mag < previous_mag, "falloff not decreasing: prev={previous_mag}, now={mag}");
        previous_mag = mag;
        previous_radius = sw.radius;
        samples += 1;
    }

    assert!(samples >= 2, "need at least two magnitude samples for falloff check");
}

/// Invariant: shockwave pressure injection does not recursively spawn more waves.
#[test]
#[serial]
fn shockwave_does_not_self_trigger_cascade() {
    let mut world = fresh_world(SEED_PREFIX + 0x08);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 30_000.0);

    let mut peak = 0usize;
    for _ in 0..80 {
        world.step(Vec2::ZERO);
        peak = peak.max(world.shockwaves.len());
    }

    assert!(peak <= 1, "single-wave run cascaded to {peak} concurrent waves");
    assert!(world.shockwaves.is_empty(), "single-wave run did not fully retire");
}

/// Invariant: nearby waves pool and never reduce the larger input yield.
#[test]
#[serial]
fn nearby_shockwaves_pool_monotonically() {
    let mut world = fresh_world(SEED_PREFIX + 0x09);
    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 4_000.0);
    world.spawn_shockwave(cx + 1, cy + 1, 5_000.0);

    assert_eq!(world.shockwaves.len(), 1, "nearby waves should have pooled");
    assert!(world.shockwaves[0].yield_p >= 5_000.0, "pooled yield fell below larger input");
}

/// Invariant: pooling path does not clobber yield when caller cap is lower.
#[test]
#[serial]
fn pooled_wave_never_decreases_with_lower_cap_caller() {
    let mut world = fresh_world(SEED_PREFIX + 0x0A);
    let (cx, cy) = center();
    world.spawn_shockwave_capped(cx, cy, 10_000.0, 20_000.0);
    let before = world.shockwaves[0].yield_p;
    world.spawn_shockwave_capped(cx + 1, cy, 10_000.0, 5_000.0);

    assert_eq!(world.shockwaves.len(), 1, "nearby waves should have pooled");
    assert!(
        world.shockwaves[0].yield_p >= before,
        "pooling reduced yield: before={before}, after={}",
        world.shockwaves[0].yield_p
    );
}

/// Invariant: shockwave-only activity never flips U central-blast flags.
#[test]
#[serial]
fn u_central_blast_flags_stay_false_for_manual_shockwaves() {
    let mut world = fresh_world(SEED_PREFIX + 0x0B);
    assert_eq!(world.u_central_blast_fired.len(), W * H, "flag array size mismatch");
    assert!(
        world.u_central_blast_fired.iter().all(|v| !*v),
        "new world must start with all central-blast flags false"
    );

    let (cx, cy) = center();
    world.spawn_shockwave(cx, cy, 20_000.0);
    tick_n(&mut world, 100);

    assert!(
        world.u_central_blast_fired.iter().all(|v| !*v),
        "manual shockwaves should not toggle U central-blast flags"
    );
}

#[test]
#[ignore = "TARGET: enforce and expose MAX_SHOCKWAVES hard ceiling under dense detonation"]
fn active_shockwave_count_has_hard_ceiling() {
    panic!("pending: needs public MAX_SHOCKWAVES contract or observable cap signal");
}

#[test]
#[ignore = "TARGET: validate NaN/inf rejection for spawn_shockwave and spawn_shockwave_capped"]
fn non_finite_shockwave_input_is_rejected_or_sanitized() {
    panic!("pending: needs explicit non-finite input policy in public API");
}
