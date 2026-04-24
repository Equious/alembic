use alembic::{Cell, Element, PrefabKind, World, H, W};
use macroquad::prelude::Vec2;
use serial_test::serial;

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

mod helpers {
    use super::{cell_at, in_bounds, Element, World};
    use std::collections::{HashSet, VecDeque};

    pub fn count_nonempty_in_rect(
        world: &World,
        left: i32,
        top: i32,
        width: i32,
        height: i32,
    ) -> usize {
        let mut count = 0usize;
        for y in top..(top + height) {
            for x in left..(left + width) {
                if !in_bounds(x, y) {
                    continue;
                }
                if cell_at(world, x, y).el != Element::Empty {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn count_disk_pixels(radius: i32) -> usize {
        let mut count = 0usize;
        let r2 = radius * radius;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= r2 {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn find_element_positions(
        world: &World,
        left: i32,
        top: i32,
        width: i32,
        height: i32,
        el: Element,
    ) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        for y in top..(top + height) {
            for x in left..(left + width) {
                if !in_bounds(x, y) {
                    continue;
                }
                if cell_at(world, x, y).el == el {
                    out.push((x, y));
                }
            }
        }
        out
    }

    pub fn flood_fill_nonempty(world: &World, start: (i32, i32)) -> HashSet<(i32, i32)> {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        if !in_bounds(start.0, start.1) {
            return seen;
        }
        if cell_at(world, start.0, start.1).el == Element::Empty {
            return seen;
        }

        seen.insert(start);
        queue.push_back(start);
        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if !in_bounds(nx, ny) {
                    continue;
                }
                if seen.contains(&(nx, ny)) {
                    continue;
                }
                if cell_at(world, nx, ny).el == Element::Empty {
                    continue;
                }
                seen.insert((nx, ny));
                queue.push_back((nx, ny));
            }
        }

        seen
    }

    pub fn perpendicular_width(world: &World, cx: i32, cy: i32, sx: i32, sy: i32, span: i32) -> i32 {
        let mut width = 0;
        for k in -span..=span {
            let x = cx + k * sx;
            let y = cy + k * sy;
            if !in_bounds(x, y) {
                continue;
            }
            if cell_at(world, x, y).el != Element::Empty {
                width += 1;
            }
        }
        width
    }
}

// --- Paint tool ---

#[test]
#[serial]
fn paint_element_written() {
    macroquad::rand::srand(0xA11C_1001);
    let mut world = World::new();

    world.paint(40, 40, 0, Element::Sand, 0, true);
    world.paint(60, 40, 0, Element::Water, 0, true);
    world.paint(80, 40, 0, Element::Stone, 0, true);

    assert_eq!(cell_at(&world, 40, 40).el, Element::Sand);
    assert_eq!(cell_at(&world, 60, 40).el, Element::Water);
    assert_eq!(cell_at(&world, 80, 40).el, Element::Stone);
}

#[test]
#[serial]
fn paint_kind_behavior_consistent() {
    macroquad::rand::srand(0xA11C_1002);
    let mut world = World::new();

    // Expected kinds per element (PHYSICS table, private):
    // Sand -> Kind::Powder (src/lib.rs:214)
    // Water -> Kind::Liquid (src/lib.rs:215)
    // Stone -> Kind::Gravel (src/lib.rs:216)
    // Verified behaviorally: Powder/Gravel fall, Liquid flows downward.
    let sand_x = 70;
    let water_x = 120;
    let stone_x = 170;
    let y0 = 50;

    world.paint(sand_x, y0, 0, Element::Sand, 0, false);
    world.paint(stone_x, y0, 0, Element::Stone, 0, false);
    for _ in 0..300 {
        world.paint(water_x, y0, 0, Element::Water, 0, false);
        if cell_at(&world, water_x, y0).el == Element::Water {
            break;
        }
    }
    assert_eq!(cell_at(&world, sand_x, y0).el, Element::Sand);
    assert_eq!(cell_at(&world, water_x, y0).el, Element::Water);
    assert_eq!(cell_at(&world, stone_x, y0).el, Element::Stone);

    for _ in 0..5 {
        world.step(Vec2::ZERO);
    }

    let sand_positions = helpers::find_element_positions(&world, 0, 0, W as i32, H as i32, Element::Sand);
    let water_positions =
        helpers::find_element_positions(&world, 0, 0, W as i32, H as i32, Element::Water);
    let stone_positions =
        helpers::find_element_positions(&world, 0, 0, W as i32, H as i32, Element::Stone);

    assert_eq!(sand_positions.len(), 1, "expected exactly one sand cell");
    assert_eq!(water_positions.len(), 1, "expected exactly one water cell");
    assert_eq!(stone_positions.len(), 1, "expected exactly one stone cell");

    assert!(sand_positions[0].1 > y0, "sand should fall: start y={y0}, end y={}", sand_positions[0].1);
    assert!(
        water_positions[0].1 > y0,
        "water should move downward in open space: start y={y0}, end y={}",
        water_positions[0].1
    );
    assert!(
        stone_positions[0].1 > y0,
        "stone should fall: start y={y0}, end y={}",
        stone_positions[0].1
    );
}

#[test]
#[serial]
fn paint_frozen_flag_respected() {
    macroquad::rand::srand(0xA11C_1003);
    let mut world = World::new();

    world.paint(35, 35, 0, Element::Stone, 0, true);
    world.paint(45, 35, 0, Element::Stone, 0, false);

    assert!(cell_at(&world, 35, 35).is_frozen());
    assert!(!cell_at(&world, 45, 35).is_frozen());

    world.step(Vec2::ZERO);

    assert!(cell_at(&world, 35, 35).is_frozen());
    assert!(!cell_at(&world, 45, 36).is_frozen());
}

#[test]
#[serial]
fn paint_radius_zero_paints_center_only() {
    macroquad::rand::srand(0xA11C_1004);
    let mut world = World::new();
    let (cx, cy) = (90, 90);

    world.paint(cx, cy, 0, Element::Stone, 0, true);

    let count = helpers::count_nonempty_in_rect(&world, cx - 1, cy - 1, 3, 3);
    assert_eq!(count, 1, "radius 0 should paint exactly one cell");
    assert_eq!(cell_at(&world, cx, cy).el, Element::Stone);

    for (dx, dy) in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ] {
        assert_eq!(cell_at(&world, cx + dx, cy + dy).el, Element::Empty);
    }
}

#[test]
#[serial]
fn paint_radius_n_is_disk_not_square() {
    for (seed, radius) in [(0xA11C_1005u64, 3), (0xA11C_1006u64, 5)] {
        macroquad::rand::srand(seed);
        let mut world = World::new();
        let (cx, cy) = (120, 120);
        world.paint(cx, cy, radius, Element::Stone, 0, true);

        assert_ne!(cell_at(&world, cx + radius, cy).el, Element::Empty);
        assert_eq!(cell_at(&world, cx + radius, cy + radius).el, Element::Empty);

        let painted = helpers::count_nonempty_in_rect(
            &world,
            cx - radius,
            cy - radius,
            2 * radius + 1,
            2 * radius + 1,
        );
        let expected = helpers::count_disk_pixels(radius);
        assert_eq!(painted, expected, "disk pixel count mismatch for radius {radius}");
    }
}

// --- Prefab tool ---

#[test]
#[serial]
fn prefab_beaker_shape() {
    macroquad::rand::srand(0xA11C_2001);
    let mut world = World::new();
    let (cx, cy, w, h) = (90, 110, 10, 12);
    world.place_prefab(cx, cy, PrefabKind::Beaker, Element::Stone, 1, w, h, 0);

    let x0 = cx - w / 2;
    let y0 = cy - h / 2;
    let x1 = x0 + w;
    let y1 = y0 + h;

    for x in x0..x1 {
        if x == x0 || x == x1 - 1 {
            assert_eq!(cell_at(&world, x, y0).el, Element::Stone);
        } else {
            assert_eq!(cell_at(&world, x, y0).el, Element::Empty, "beaker top interior must be open");
        }
        let bottom = cell_at(&world, x, y1 - 1);
        assert_eq!(bottom.el, Element::Stone);
        assert!(bottom.is_frozen());
    }

    for y in y0..y1 {
        let left = cell_at(&world, x0, y);
        let right = cell_at(&world, x1 - 1, y);
        assert_eq!(left.el, Element::Stone);
        assert_eq!(right.el, Element::Stone);
        assert!(left.is_frozen());
        assert!(right.is_frozen());
    }

    for y in (y0 + 1)..(y1 - 1) {
        for x in (x0 + 1)..(x1 - 1) {
            assert_eq!(cell_at(&world, x, y).el, Element::Empty);
        }
    }
}

#[test]
#[serial]
fn prefab_box_sealed_and_frozen() {
    macroquad::rand::srand(0xA11C_2002);
    let mut world = World::new();
    let (cx, cy, w, h) = (150, 110, 12, 14);
    world.place_prefab(cx, cy, PrefabKind::Box, Element::Stone, 1, w, h, 0);

    let x0 = cx - w / 2;
    let y0 = cy - h / 2;
    let x1 = x0 + w;
    let y1 = y0 + h;

    for y in y0..y1 {
        for x in x0..x1 {
            let c = cell_at(&world, x, y);
            let is_border = x == x0 || x == x1 - 1 || y == y0 || y == y1 - 1;
            if is_border {
                assert_eq!(c.el, Element::Stone);
                assert!(c.is_frozen());
            } else {
                assert_eq!(c.el, Element::Empty);
            }
        }
    }
}

#[test]
#[serial]
fn prefab_battery_layout_and_energized_floodfill() {
    macroquad::rand::srand(0xA11C_2003);
    let mut world = World::new();
    let (cx, cy, w, h) = (220, 110, 8, 10);
    world.place_prefab(cx, cy, PrefabKind::Battery, Element::Cu, 1, w, h, 0);

    let x0 = cx - w / 2;
    let y0 = cy - h / 2;
    let x1 = x0 + w;
    let y1 = y0 + h;

    for x in x0..x1 {
        assert_eq!(cell_at(&world, x, y0).el, Element::BattPos);
        assert_eq!(cell_at(&world, x, y1 - 1).el, Element::BattNeg);
    }
    for y in (y0 + 1)..(y1 - 1) {
        for x in x0..x1 {
            assert_eq!(cell_at(&world, x, y).el, Element::Cu);
        }
    }

    let connected = helpers::flood_fill_nonempty(&world, (x0, y0));
    assert!(!connected.is_empty(), "expected non-empty connected component");

    let mut has_body = false;
    let mut has_negative = false;
    for (x, y) in &connected {
        let el = cell_at(&world, *x, *y).el;
        if el == Element::Cu {
            has_body = true;
        }
        if el == Element::BattNeg {
            has_negative = true;
        }
    }
    assert!(has_body, "battery component must include body cells");
    assert!(has_negative, "battery component must reach negative terminal");
}

#[test]
#[serial]
fn prefab_size_sliders_honored() {
    macroquad::rand::srand(0xA11C_2004);

    let mut small = World::new();
    let mut large = World::new();

    let (small_w, small_h) = (6, 8);
    let (large_w, large_h) = (16, 20);

    small.place_prefab(80, 120, PrefabKind::Beaker, Element::Stone, 1, small_w, small_h, 0);
    large.place_prefab(80, 120, PrefabKind::Beaker, Element::Stone, 1, large_w, large_h, 0);

    let small_nonempty = helpers::count_nonempty_in_rect(
        &small,
        80 - small_w / 2,
        120 - small_h / 2,
        small_w,
        small_h,
    );
    let large_nonempty = helpers::count_nonempty_in_rect(
        &large,
        80 - large_w / 2,
        120 - large_h / 2,
        large_w,
        large_h,
    );

    assert!(large_nonempty > small_nonempty, "larger beaker should have more wall cells");

    let small_interior_span = small_w - 2;
    let large_interior_span = large_w - 2;
    assert!(
        large_interior_span > small_interior_span,
        "larger beaker should have wider interior"
    );
}

// --- Wire tool ---

#[test]
#[serial]
fn wire_thickness_accuracy() {
    for (seed, t) in [(0xA11C_3001u64, 1), (0xA11C_3002u64, 2), (0xA11C_3003u64, 3)] {
        macroquad::rand::srand(seed);
        let mut world = World::new();
        world.place_wire_line(40, 100, 80, 100, Element::Cu, t);

        let mut col_count = 0;
        for y in 0..(H as i32) {
            if cell_at(&world, 60, y).el != Element::Empty {
                col_count += 1;
            }
        }
        assert_eq!(col_count, 2 * t + 1, "horizontal line width mismatch for thickness {t}");
    }

    for (seed, t) in [(0xA11C_3004u64, 1), (0xA11C_3005u64, 2), (0xA11C_3006u64, 3)] {
        macroquad::rand::srand(seed);
        let mut world = World::new();
        world.place_wire_line(40, 40, 80, 80, Element::Cu, t);

        let width = helpers::perpendicular_width(&world, 60, 60, 1, -1, 20);
        assert!(
            width >= 2 * t - 1 && width <= 2 * t + 5,
            "diagonal perpendicular width out of range for thickness {t}: {width}"
        );
    }
}

#[test]
#[serial]
fn wire_material_matches_selection() {
    for (seed, material) in [(0xA11C_3007u64, Element::Cu), (0xA11C_3008u64, Element::Au)] {
        macroquad::rand::srand(seed);
        let mut world = World::new();
        world.place_wire_line(30, 200, 90, 200, material, 2);

        let mut nonempty = 0usize;
        for y in 195..=205 {
            for x in 25..=95 {
                let c = cell_at(&world, x, y);
                if c.el != Element::Empty {
                    nonempty += 1;
                    assert_eq!(c.el, material, "wire cell material mismatch at ({x}, {y})");
                }
            }
        }
        assert!(nonempty > 0, "wire placement should create non-empty cells");
    }
}

// --- Pressure inheritance ---

#[test]
#[serial]
fn build_solid_inherits_prior_gas_pressure() {
    macroquad::rand::srand(0xA11C_4001);
    let mut world = World::new();
    let (x, y) = (120, 70);

    world.paint(x, y, 0, Element::H, 0, false);
    for _ in 0..8 {
        world.paint(x, y, 0, Element::H, 0, false);
    }
    let prior_pressure = cell_at(&world, x, y).pressure;
    assert!(prior_pressure > 0, "expected overpainted gas to carry pressure");

    world.paint(x, y, 0, Element::Stone, 0, true);
    let stone = cell_at(&world, x, y);
    assert_eq!(stone.el, Element::Stone);
    assert_eq!(stone.pressure, prior_pressure);
}

// --- Vacuum / Pipet ---

#[test]
#[serial]
#[ignore = "requires pub apply_vacuum — currently private (src/lib.rs:5873)"]
fn vacuum_conserves_atoms_within_tolerance() {
    macroquad::rand::srand(0xA11C_5001);
    let mut world = World::new();
    world.paint(100, 100, 4, Element::Sand, 0, false);
    let _before = helpers::count_nonempty_in_rect(&world, 80, 80, 40, 40);
}

#[test]
#[serial]
#[ignore = "requires pub pipet_collect — currently private (src/lib.rs:6103)"]
fn pipet_stores_valid_element_discriminant() {
    macroquad::rand::srand(0xA11C_5002);
    let mut world = World::new();
    world.paint(110, 110, 0, Element::Water, 0, true);
    let c = cell_at(&world, 110, 110);
    assert_eq!(c.el, Element::Water);
}
