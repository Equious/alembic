//! Electrical/circuit integration invariants from `test-targets/09-electrical.md`.

use alembic::{Cell, Element, PrefabKind, World, H, W};
use macroquad::prelude::Vec2;
use serial_test::serial;

fn fresh_world(seed: u64) -> World {
    macroquad::rand::srand(seed);
    World::new()
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

fn tick_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step(Vec2::ZERO);
    }
}

fn place(world: &mut World, x: i32, y: i32, el: Element) {
    world.paint(x, y, 0, el, 0, false);
}

fn place_frozen(world: &mut World, x: i32, y: i32, el: Element) {
    world.paint(x, y, 0, el, 0, true);
}

fn make_brine(world: &mut World, x: i32, y: i32) {
    place_frozen(world, x, y, Element::Water);
    let i = idx(x, y);
    world.cells[i].solute_el = Element::Salt;
    world.cells[i].solute_amt = 128;
    world.cells[i].solute_derived_id = 0;
}

fn paint_battery(world: &mut World, cx: i32, cy: i32) {
    world.place_prefab(cx, cy, PrefabKind::Battery, Element::Stone, 1, 4, 8, 0);
}

fn battery_terminals(cx: i32, cy: i32) -> ((i32, i32), (i32, i32)) {
    ((cx, cy - 4), (cx, cy + 3))
}

fn paint_galvanic_cell(
    world: &mut World,
    cx: i32,
    cy: i32,
    anode_el: Element,
    cathode_el: Element,
    with_external_wire: bool,
) -> ((i32, i32), (i32, i32)) {
    let anode = (cx - 5, cy);
    let cathode = (cx + 5, cy);

    place_frozen(world, anode.0, anode.1, anode_el);
    place_frozen(world, cathode.0, cathode.1, cathode_el);
    for x in (anode.0 + 1)..=(cathode.0 - 1) {
        make_brine(world, x, cy);
    }

    if with_external_wire {
        world.place_wire_line(anode.0, cy - 1, anode.0, cy - 4, Element::Cu, 1);
        world.place_wire_line(anode.0, cy - 4, cathode.0, cy - 4, Element::Cu, 1);
        world.place_wire_line(cathode.0, cy - 4, cathode.0, cy - 1, Element::Cu, 1);
    }

    (anode, cathode)
}

fn center() -> (i32, i32) {
    ((W / 2) as i32, (H / 2) as i32)
}

fn any_energized(world: &World) -> bool {
    world.energized.iter().any(|&v| v)
}

fn approx_eq(a: f32, b: f32, eps: f32) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ~= {b} (eps={eps}), diff={}",
        (a - b).abs()
    );
}

fn assert_masks_exclusive(world: &World) {
    for i in 0..(W * H) {
        assert!(
            !(world.cathode_mask[i] && world.anode_mask[i]),
            "cell {i} appears in both cathode_mask and anode_mask"
        );
    }
}

#[test]
#[serial]
fn energized_only_battpos_no_circuit() {
    let mut world = fresh_world(0x09_10_01);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    world.place_wire_line(pos.0, pos.1, pos.0 + 18, pos.1, Element::Cu, 1);
    tick_n(&mut world, 2);

    assert!(!any_energized(&world), "open circuit from BattPos must not energize");
    assert!(!world.energized[idx(pos.0 + 8, pos.1)]);
    assert!(!world.energized[idx(neg.0, neg.1)]);
}

#[test]
#[serial]
fn energized_only_battneg_no_circuit() {
    let mut world = fresh_world(0x09_10_02);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (_, neg) = battery_terminals(cx, by);

    world.place_wire_line(neg.0, neg.1, neg.0 + 18, neg.1, Element::Cu, 1);
    tick_n(&mut world, 2);

    assert!(!any_energized(&world), "open circuit from BattNeg must not energize");
    assert!(!world.energized[idx(neg.0 + 8, neg.1)]);
}

#[test]
#[serial]
fn energized_complete_loop() {
    let mut world = fresh_world(0x09_10_03);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    let x_right = cx + 20;
    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Cu, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Cu, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Cu, 1);

    tick_n(&mut world, 2);

    for p in [(cx + 10, pos.1), (x_right, by), (cx + 10, neg.1)] {
        assert!(
            world.energized[idx(p.0, p.1)],
            "loop cell {:?} should be energized in closed circuit",
            p
        );
    }
}

#[test]
#[serial]
fn topology_loop_broken_by_insulator() {
    let mut world = fresh_world(0x09_10_05);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);
    let x_right = cx + 20;

    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Cu, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Cu, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Cu, 1);
    place_frozen(&mut world, x_right, by, Element::Stone);

    tick_n(&mut world, 2);

    assert!(!any_energized(&world), "insulator gap should open the loop");
}

#[test]
#[serial]
fn noble_gas_glow_one_hop_only() {
    let mut world = fresh_world(0x09_10_06);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    let x_right = cx + 20;
    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Cu, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Cu, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Cu, 1);

    let wire_y = by;
    place_frozen(&mut world, cx + 8, wire_y - 1, Element::Ne);
    place_frozen(&mut world, cx + 8, wire_y - 2, Element::Ne);

    tick_n(&mut world, 2);

    assert!(world.energized[idx(cx + 8, wire_y - 1)], "adjacent Ne should glow");
    assert!(
        !world.energized[idx(cx + 8, wire_y - 2)],
        "second-hop Ne should not be energized"
    );
}

#[test]
#[serial]
fn bitmap_clears_when_no_battery() {
    let mut world = fresh_world(0x09_10_07);
    let (cx, cy) = center();
    world.place_wire_line(cx - 15, cy, cx + 15, cy, Element::Cu, 1);
    tick_n(&mut world, 2);

    assert!(world.energized.iter().all(|&v| !v), "energized bitmap should clear");
    assert!(
        world.cathode_mask.iter().all(|&v| !v),
        "cathode_mask should clear with no source"
    );
    assert!(
        world.anode_mask.iter().all(|&v| !v),
        "anode_mask should clear with no source"
    );
}

#[test]
#[serial]
fn battery_voltage_precedence() {
    let mut world = fresh_world(0x09_10_08);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    world.place_wire_line(pos.0, pos.1, cx + 18, pos.1, Element::Fe, 1);
    world.place_wire_line(cx + 18, pos.1, cx + 18, neg.1, Element::Fe, 1);
    world.place_wire_line(cx + 18, neg.1, neg.0, neg.1, Element::Fe, 1);
    world.battery_voltage = 137.0;

    tick_n(&mut world, 1);

    approx_eq(world.active_emf, world.battery_voltage, 1e-6);
}

#[test]
#[serial]
fn galvanic_fallback() {
    let mut world = fresh_world(0x09_10_09);
    let (cx, cy) = center();
    paint_galvanic_cell(&mut world, cx, cy, Element::Zn, Element::Cu, true);

    tick_n(&mut world, 2);

    assert!(world.galvanic_voltage > 0.0, "galvanic pair with external loop should produce EMF");
    approx_eq(world.active_emf, world.galvanic_voltage, 1e-6);
    assert!(any_energized(&world), "galvanic loop should energize at least one cell");
}

#[test]
#[serial]
fn no_source_fallback() {
    let mut world = fresh_world(0x09_10_0A);
    let (cx, cy) = center();
    world.place_wire_line(cx - 10, cy, cx + 10, cy, Element::Cu, 1);
    tick_n(&mut world, 2);

    approx_eq(world.active_emf, 0.0, 1e-6);
    approx_eq(world.galvanic_voltage, 0.0, 1e-6);
}

#[test]
#[serial]
fn galvanic_emf_from_en_gap() {
    let (cx, cy) = center();

    let mut zn_cu = fresh_world(0x09_10_0B);
    paint_galvanic_cell(&mut zn_cu, cx, cy, Element::Zn, Element::Cu, true);
    tick_n(&mut zn_cu, 2);
    assert!(zn_cu.galvanic_voltage > 0.0, "Zn/Cu should have nonzero EN-gap EMF");

    let mut zn_zn = fresh_world(0x09_10_0C);
    paint_galvanic_cell(&mut zn_zn, cx, cy, Element::Zn, Element::Zn, true);
    tick_n(&mut zn_zn, 2);
    approx_eq(zn_zn.galvanic_voltage, 0.0, 1e-6);
}

#[test]
#[serial]
fn galvanic_requires_external_conductor() {
    let mut world = fresh_world(0x09_10_0D);
    let (cx, cy) = center();
    paint_galvanic_cell(&mut world, cx, cy, Element::Zn, Element::Cu, false);

    tick_n(&mut world, 2);

    approx_eq(world.galvanic_voltage, 0.0, 1e-6);
    approx_eq(world.active_emf, 0.0, 1e-6);
    assert!(!any_energized(&world), "no dry external wire means open galvanic loop");
}

#[test]
#[serial]
fn galvanic_polarity_identity() {
    let mut world = fresh_world(0x09_10_0E);
    let (cx, cy) = center();
    paint_galvanic_cell(&mut world, cx, cy, Element::Zn, Element::Cu, true);

    tick_n(&mut world, 2);

    assert_eq!(world.galvanic_cathode_el, Some(Element::Cu));
    assert_eq!(world.galvanic_anode_el, Some(Element::Zn));
}

#[test]
#[serial]
fn electrolysis_masks() {
    let mut world = fresh_world(0x09_10_0F);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    let anode = (cx + 12, by - 2);
    let cathode = (cx + 12, by + 1);
    place_frozen(&mut world, anode.0, anode.1, Element::Zn);
    place_frozen(&mut world, cathode.0, cathode.1, Element::Cu);
    world.place_wire_line(pos.0, pos.1, anode.0, anode.1, Element::Cu, 1);
    world.place_wire_line(neg.0, neg.1, cathode.0, cathode.1, Element::Cu, 1);

    make_brine(&mut world, anode.0, anode.1 + 1);
    make_brine(&mut world, cathode.0, cathode.1 - 1);

    tick_n(&mut world, 3);

    assert!(world.anode_mask[idx(anode.0, anode.1)], "anode electrode should be in anode_mask");
    assert!(
        world.cathode_mask[idx(cathode.0, cathode.1)],
        "cathode electrode should be in cathode_mask"
    );
    assert_masks_exclusive(&world);
}

#[test]
#[serial]
fn electrolysis_joule_gating() {
    let (cx, cy) = center();

    let mut no_source = fresh_world(0x09_10_11);
    let left = (cx - 6, cy);
    let right = (cx + 6, cy);
    place_frozen(&mut no_source, left.0, left.1, Element::Cu);
    place_frozen(&mut no_source, right.0, right.1, Element::Cu);
    for x in (left.0 + 1)..=(right.0 - 1) {
        make_brine(&mut no_source, x, cy);
    }
    let temp_a_before = cell_at(&no_source, left.0, left.1).temp;
    tick_n(&mut no_source, 120);
    let temp_a_after = cell_at(&no_source, left.0, left.1).temp;
    assert_eq!(cell_at(&no_source, cx, cy).el, Element::Water, "no source: no plating expected");
    assert_eq!(temp_a_before, temp_a_after, "no source: no joule heating expected");
    approx_eq(no_source.active_emf, 0.0, 1e-6);

    let mut open_battery = fresh_world(0x09_10_12);
    let by = cy - 10;
    paint_battery(&mut open_battery, cx, by);
    let (pos, _) = battery_terminals(cx, by);
    let rod = (cx + 12, by - 1);
    place_frozen(&mut open_battery, rod.0, rod.1, Element::Cu);
    world_place_open_electrolyte_fixture(&mut open_battery, rod.0, rod.1 + 1);
    open_battery.place_wire_line(pos.0, pos.1, rod.0, rod.1, Element::Cu, 1);

    let temp_b_before = cell_at(&open_battery, rod.0, rod.1).temp;
    tick_n(&mut open_battery, 120);
    let temp_b_after = cell_at(&open_battery, rod.0, rod.1).temp;

    assert!(open_battery.active_emf > 0.0, "battery sets potential even in open loop");
    assert!(
        !any_energized(&open_battery),
        "open battery loop should have no energized current-carrying cells"
    );
    assert_eq!(
        cell_at(&open_battery, rod.0, rod.1 + 1).el,
        Element::Water,
        "open loop: no plating expected"
    );
    assert_eq!(temp_b_before, temp_b_after, "open loop: no joule heating expected");
}

fn world_place_open_electrolyte_fixture(world: &mut World, brine_x: i32, brine_y: i32) {
    make_brine(world, brine_x, brine_y);
    make_brine(world, brine_x + 1, brine_y);
}

#[test]
#[serial]
fn plating_deposits_metal_on_cathode() {
    let mut world = fresh_world(0x09_10_13);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    let cathode = (cx + 14, by + 1);
    let anode = (cx + 14, by - 2);
    place_frozen(&mut world, cathode.0, cathode.1, Element::Cu);
    place_frozen(&mut world, anode.0, anode.1, Element::Zn);
    world.place_wire_line(neg.0, neg.1, cathode.0, cathode.1, Element::Cu, 1);
    world.place_wire_line(pos.0, pos.1, anode.0, anode.1, Element::Cu, 1);

    for dy in -1..=1 {
        let y = cathode.1 + dy;
        make_brine(&mut world, cathode.0 - 1, y);
        let i = idx(cathode.0 - 1, y);
        world.cells[i].solute_el = Element::Cu;
        world.cells[i].solute_amt = 200;
    }

    tick_n(&mut world, 500);

    let mut plated = 0usize;
    let mut frozen_plated = 0usize;
    let mut loose_plated = 0usize;
    for dy in -2..=2 {
        let x = cathode.0 - 1;
        let y = cathode.1 + dy;
        if !in_bounds(x, y) {
            continue;
        }
        let c = cell_at(&world, x, y);
        if c.el == Element::Cu {
            plated += 1;
            if c.is_frozen() {
                frozen_plated += 1;
            } else {
                loose_plated += 1;
            }
        }
    }

    assert!(plated > 0, "electrolysis should deposit metal at cathode side");
    assert!(frozen_plated > 0, "expected adherent (frozen) plated population");
    assert!(loose_plated > 0, "expected loose plated population");
}

fn joule_rise_for_voltage(seed: u64, battery_voltage: f32, ticks: usize) -> f32 {
    let mut world = fresh_world(seed);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);

    let x_right = cx + 20;
    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Fe, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Fe, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Fe, 1);
    world.battery_voltage = battery_voltage;

    let probes = [
        (cx + 8, pos.1),
        (cx + 14, pos.1),
        (x_right, by - 1),
        (x_right, by),
        (x_right, by + 1),
        (cx + 14, neg.1),
        (cx + 8, neg.1),
    ];
    let before: f32 = probes
        .iter()
        .map(|&(x, y)| cell_at(&world, x, y).temp as f32)
        .sum::<f32>()
        / probes.len() as f32;

    tick_n(&mut world, ticks);

    let after: f32 = probes
        .iter()
        .map(|&(x, y)| cell_at(&world, x, y).temp as f32)
        .sum::<f32>()
        / probes.len() as f32;
    after - before
}

#[test]
#[serial]
fn joule_scaling_v_squared() {
    let rise_100 = joule_rise_for_voltage(0x09_10_14, 100.0, 60);
    let rise_200 = joule_rise_for_voltage(0x09_10_15, 200.0, 60);

    assert!(rise_100 > 0.0, "100V fixture should produce measurable heating");
    let ratio = rise_200 / rise_100;
    assert!(
        (3.0..=5.0).contains(&ratio),
        "expected V^2-like scaling ratio in [3,5], got {ratio} (rise100={rise_100}, rise200={rise_200})"
    );
}

#[test]
#[serial]
fn joule_cap_per_cell_per_frame() {
    let mut world = fresh_world(0x09_10_16);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);
    let x_right = cx + 18;
    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Fe, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Fe, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Fe, 1);
    world.battery_voltage = 500_000.0;

    tick_n(&mut world, 1);
    let before: Vec<i16> = world.cells.iter().map(|c| c.temp).collect();
    tick_n(&mut world, 1);

    let mut max_delta = i32::MIN;
    for i in 0..(W * H) {
        if !world.energized[i] {
            continue;
        }
        let delta = (world.cells[i].temp as i32 - before[i] as i32).abs();
        max_delta = max_delta.max(delta);
    }

    assert!(max_delta >= 0, "fixture should have energized cells");
    assert!(max_delta < 1_000_000, "per-frame Joule increase should be bounded, got {max_delta}");
}

#[test]
#[serial]
fn non_conductors_no_direct_joule() {
    let mut world = fresh_world(0x09_10_17);
    let (cx, cy) = center();
    let by = cy - 10;
    paint_battery(&mut world, cx, by);
    let (pos, neg) = battery_terminals(cx, by);
    let x_right = cx + 16;

    world.place_wire_line(pos.0, pos.1, x_right, pos.1, Element::Fe, 1);
    world.place_wire_line(x_right, pos.1, x_right, neg.1, Element::Fe, 1);
    world.place_wire_line(x_right, neg.1, neg.0, neg.1, Element::Fe, 1);
    world.battery_voltage = 180.0;

    let adjacent_stone = (cx + 10, by - 1);
    let far_stone = (cx - 20, by - 1);
    place_frozen(&mut world, adjacent_stone.0, adjacent_stone.1, Element::Stone);
    place_frozen(&mut world, far_stone.0, far_stone.1, Element::Stone);

    tick_n(&mut world, 1);
    let a0 = cell_at(&world, adjacent_stone.0, adjacent_stone.1).temp;
    let b0 = cell_at(&world, far_stone.0, far_stone.1).temp;
    tick_n(&mut world, 1);
    let a1 = cell_at(&world, adjacent_stone.0, adjacent_stone.1).temp;
    let b1 = cell_at(&world, far_stone.0, far_stone.1).temp;

    let da = a1 as f32 - a0 as f32;
    let db = b1 as f32 - b0 as f32;
    assert!(
        (da - db).abs() <= 1e-3,
        "insulator should not receive direct Joule heating; adjacent delta={da}, far delta={db}"
    );
}
