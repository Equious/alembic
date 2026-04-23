use alembic::{Element, PrefabKind, World, HISTORY_CAPACITY, H, W};
use macroquad::prelude::Vec2;

// NOTE: omitted the shockwave-count-ceiling invariant because no public,
// documented ceiling constant exists yet to assert against.

fn assert_grid_sized_buffers(world: &World, context: &str) {
    let expected = W * H;
    assert_eq!(world.cells.len(), expected, "{context}: cells resized");
    assert_eq!(
        world.temp_scratch.len(),
        expected,
        "{context}: temp_scratch resized"
    );
    assert_eq!(
        world.pressure_scratch.len(),
        expected,
        "{context}: pressure_scratch resized"
    );
    assert_eq!(
        world.support_scratch.len(),
        expected,
        "{context}: support_scratch resized"
    );
    assert_eq!(
        world.vacuum_moved.len(),
        expected,
        "{context}: vacuum_moved resized"
    );
    assert_eq!(
        world.wind_exposed.len(),
        expected,
        "{context}: wind_exposed resized"
    );
    assert_eq!(
        world.energized.len(),
        expected,
        "{context}: energized resized"
    );
    assert_eq!(
        world.cathode_mask.len(),
        expected,
        "{context}: cathode_mask resized"
    );
    assert_eq!(
        world.anode_mask.len(),
        expected,
        "{context}: anode_mask resized"
    );
    assert_eq!(
        world.u_component_size.len(),
        expected,
        "{context}: u_component_size resized"
    );
    assert_eq!(
        world.u_burst_committed.len(),
        expected,
        "{context}: u_burst_committed resized"
    );
    assert_eq!(
        world.u_component_cx.len(),
        expected,
        "{context}: u_component_cx resized"
    );
    assert_eq!(
        world.u_component_cy.len(),
        expected,
        "{context}: u_component_cy resized"
    );
    assert_eq!(
        world.u_central_blast_fired.len(),
        expected,
        "{context}: u_central_blast_fired resized"
    );
}

fn assert_history_bounds(world: &World, context: &str) {
    assert!(
        world.history_count <= HISTORY_CAPACITY,
        "{context}: history_count exceeded capacity"
    );
    assert!(
        world.history_write < HISTORY_CAPACITY,
        "{context}: history_write out of range"
    );
    assert!(
        world.rewind_offset <= world.history_count,
        "{context}: rewind_offset out of range"
    );
}

#[test]
fn paint_clamps_to_grid_under_adversarial_inputs() {
    macroquad::rand::srand(0x02_B0_A1D5);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;
    let cases = [
        (-500, -500, 3),
        (500, 500, 3),
        (-1, -1, 1),
        (w, h, 1),
        (w + 1, h + 1, 4),
        (0, 0, 0),
        (0, 0, 1),
        (w - 1, h - 1, 1),
        (w - 1, 0, 3),
        (0, h - 1, 3),
        (w / 2, h / 2, 64),
        (w / 2, h / 2, 200),
        (-20, h / 2, 10),
        (w + 20, h / 2, 10),
        (w / 2, -20, 10),
        (w / 2, h + 20, 10),
    ];

    for (i, (cx, cy, radius)) in cases.iter().copied().enumerate() {
        world.paint(cx, cy, radius, Element::Stone, 0, false);
        assert_grid_sized_buffers(&world, &format!("paint case {i} ({cx}, {cy}, {radius})"));
    }
}

#[test]
fn place_prefab_clamps_to_grid_under_adversarial_inputs() {
    macroquad::rand::srand(0x02_B0_A1D6);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;
    let centers = [
        (-500, -500),
        (500, 500),
        (-1, -1),
        (w, h),
        (w + 1, h + 1),
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w / 2, h / 2),
        (-20, h / 2),
        (w + 20, h / 2),
        (w / 2, -20),
        (w / 2, h + 20),
        (w / 2, h + 1),
    ];

    for (i, (cx, cy)) in centers.iter().copied().enumerate() {
        world.place_prefab(cx, cy, PrefabKind::Beaker, Element::Stone, 2, 20, 30, 0);
        assert_grid_sized_buffers(&world, &format!("beaker case {i} ({cx}, {cy})"));
    }

    world.place_prefab(-500, h + 500, PrefabKind::Box, Element::Stone, 3, 24, 16, 1);
    assert_grid_sized_buffers(&world, "box extreme case");

    world.place_prefab(w + 500, -500, PrefabKind::Battery, Element::Stone, 2, 30, 20, 2);
    assert_grid_sized_buffers(&world, "battery extreme case");
}

#[test]
fn place_wire_line_clamps_to_grid_under_adversarial_inputs() {
    macroquad::rand::srand(0x02_B0_A1D7);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;
    let cases = [
        (-500, -500, 500, 500, 1),
        (w + 500, -500, -500, h + 500, 2),
        (-1, -1, w, h, 1),
        (0, 0, 0, h - 1, 0),
        (0, 0, w - 1, 0, 3),
        (0, h - 1, w - 1, h - 1, 3),
        (w - 1, 0, w - 1, h - 1, 3),
        (w / 2, h / 2, w / 2, h / 2, -5),
        (w + 10, h / 2, w + 40, h / 2, 4),
        (-40, h / 2, -10, h / 2, 4),
        (w / 2, -40, w / 2, -10, 4),
        (w / 2, h + 10, w / 2, h + 40, 4),
    ];

    for (i, (x0, y0, x1, y1, thickness)) in cases.iter().copied().enumerate() {
        world.place_wire_line(x0, y0, x1, y1, Element::Cu, thickness);
        assert_grid_sized_buffers(
            &world,
            &format!("wire case {i} ({x0}, {y0}) -> ({x1}, {y1}), t={thickness}"),
        );
    }
}

#[test]
fn shockwaves_handle_off_grid_origins_without_panics() {
    macroquad::rand::srand(0x02_B0_A1D8);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;
    let origins = [
        (-200, -200),
        (w + 200, -200),
        (-200, h + 200),
        (w + 200, h + 200),
        (-1, h / 2),
        (w, h / 2),
        (w / 2, -1),
        (w / 2, h),
    ];

    for (i, (cx, cy)) in origins.iter().copied().enumerate() {
        world.spawn_shockwave_capped(cx, cy, 50_000.0, 200_000.0);
        world.step(Vec2::new(0.0, 0.0));
        assert_grid_sized_buffers(&world, &format!("shockwave origin case {i}"));
    }

    for tick in 0..40 {
        world.step(Vec2::ZERO);
        assert_grid_sized_buffers(&world, &format!("shockwave drain tick {tick}"));
    }
}

#[test]
fn scratch_buffers_are_grid_sized_after_world_new() {
    macroquad::rand::srand(0x02_B0_A1D9);
    let world = World::new();
    assert_grid_sized_buffers(&world, "world new");
}

#[test]
fn history_ring_buffer_stays_bounded_over_long_run() {
    macroquad::rand::srand(0x02_B0_A1DA);
    let mut world = World::new();

    for tick in 0..(HISTORY_CAPACITY * 3 + 29) {
        world.step(Vec2::ZERO);
        assert_history_bounds(&world, &format!("history tick {tick}"));
    }

    assert_eq!(
        world.history_count, HISTORY_CAPACITY,
        "history_count should saturate at capacity"
    );
}

#[test]
fn mixed_workload_preserves_grid_and_ring_invariants() {
    macroquad::rand::srand(0x02_B0_A1DB);
    let mut world = World::new();
    let w = W as i32;
    let h = H as i32;

    for step in 0..360 {
        match step % 5 {
            0 => {
                let cx = ((step * 37) % (w + 80)) - 40;
                let cy = ((step * 53) % (h + 80)) - 40;
                let radius = (step % 12) as i32;
                world.paint(cx, cy, radius, Element::Stone, 0, false);
            }
            1 => {
                let cx = ((step * 31) % (w + 120)) - 60;
                let cy = ((step * 29) % (h + 120)) - 60;
                let kind = match step % 3 {
                    0 => PrefabKind::Beaker,
                    1 => PrefabKind::Box,
                    _ => PrefabKind::Battery,
                };
                world.place_prefab(cx, cy, kind, Element::Stone, 2, 20, 30, (step % 4) as u8);
            }
            2 => {
                let x0 = ((step * 11) % (w + 120)) - 60;
                let y0 = ((step * 13) % (h + 120)) - 60;
                let x1 = ((step * 17) % (w + 120)) - 60;
                let y1 = ((step * 19) % (h + 120)) - 60;
                let thickness = ((step % 7) as i32) - 2;
                world.place_wire_line(x0, y0, x1, y1, Element::Cu, thickness);
            }
            3 => {
                let cx = ((step * 23) % (w + 160)) - 80;
                let cy = ((step * 27) % (h + 160)) - 80;
                world.spawn_shockwave_capped(cx, cy, 10_000.0, 120_000.0);
            }
            _ => {}
        }

        let wind = match step % 4 {
            0 => Vec2::ZERO,
            1 => Vec2::new(0.25, 0.0),
            2 => Vec2::new(0.0, -0.25),
            _ => Vec2::new(-0.25, 0.25),
        };
        world.step(wind);

        assert_grid_sized_buffers(&world, &format!("mixed workload step {step}"));
        assert_history_bounds(&world, &format!("mixed workload step {step}"));
    }
}
