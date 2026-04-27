//! wgpu rendering backend for Alembic — v0.3 migration.
//!
//! Replaces macroquad's GL ES 2.0 stack with modern wgpu so we can
//! use real compute shaders and storage buffers for the simulation
//! physics. This module owns the wgpu device, queue, and surface;
//! holds rendering pipelines; and drives the per-frame render loop.
//!
//! Sim state and chemistry rules continue to live in `lib.rs` (the
//! `World`, `Cell`, `Element` types and all reaction logic). What
//! changes here is purely the GPU driver: how we get pixels on screen
//! and how compute dispatches will eventually replace CPU sim passes.
//!
//! Current scope: minimal "hello, wgpu" — opens a window and clears
//! to a recognizable gradient. Once validated, we layer on a texture
//! for cell colors, then a compute pipeline for the sim itself.

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use egui_wgpu::ScreenDescriptor;

/// Perceptual brightness (0..255) — used to pick black vs white text
/// when overlaying a label on an element-color tile.
fn luminance(r: u8, g: u8, b: u8) -> u32 {
    (r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000
}

use crate::{color_rgb, motion_props, pressure_source_props, thermal_profile_vec4, Element, World, H, W};
// Reference motion_props from a const so the `crate::motion_props`
// path inside the motion ctx body doesn't trip the unused-import
// lint when the only use is via the inner `crate::` path.
#[allow(dead_code)]
const _USE_MOTION_PROPS: fn(u8) -> [f32; 4] = motion_props;

const THERMAL_COMPUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    ambient_offset: i32,
    frame: u32,
    pass_id: u32,            // 0=extract, 1=diffuse, 2=writeback
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> temp_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> temp_out: array<i32>;
@group(0) @binding(3) var<storage, read_write> cells: array<vec4<u32>>;
// Per-element thermal profile, indexed by el extracted from cells[i].
//   x = conductivity, y = ambient_temp, z = ambient_rate, w = heat_capacity
@group(0) @binding(4) var<uniform> profiles: array<vec4<f32>, 96>;

const FIRE_ID: u32 = 5u;
const EMPTY_ID: u32 = 0u;

fn hash_random(i: u32, frame: u32) -> f32 {
    var h: u32 = i * 2654435761u;
    h ^= frame * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    h *= 3266489917u;
    h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

fn cell_el_thermal(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
// Extract i16 from bits 16-31 of `word` and sign-extend to i32.
fn extract_i16_thermal(word: u32) -> i32 {
    let raw = (word >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    if (x >= w_i || y >= h_i) { return; }
    let i = u32(y * w_i + x);

    if (u.pass_id == 0u) {
        // Extract: copy current temp (i16 at bits 16-31 of cells[i].y)
        // into temp_out (acts as scratch_a for the first diffuse iter).
        temp_out[i] = extract_i16_thermal(cells[i].y);
        return;
    }
    if (u.pass_id == 2u) {
        // Writeback: temp_in holds the final diffused temp. Clamp to
        // i16, write back to bits 16-31 of cells[i].y while preserving
        // seed (bits 0-7) and flag (bits 8-15).
        let final_t = clamp(temp_in[i], -273, 4000);
        let raw = u32(final_t) & 0xFFFFu;
        let lo = cells[i].y & 0x0000FFFFu;
        cells[i].y = lo | (raw << 16u);
        return;
    }

    // pass_id == 1: diffusion iteration.
    let me_cell = cells[i];
    let me_el = cell_el_thermal(me_cell);
    let me_props = profiles[me_el];
    let my_k = me_props.x;
    let me_t = f32(temp_in[i]);

    var delta: f32 = 0.0;
    var diff_neighbors: f32 = 0.0;
    var oob_neighbors: f32 = 0.0;
    if (x > 0) {
        let ni = u32(y * w_i + (x - 1));
        let n_el = cell_el_thermal(cells[ni]);
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (x < w_i - 1) {
        let ni = u32(y * w_i + (x + 1));
        let n_el = cell_el_thermal(cells[ni]);
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (y > 0) {
        let ni = u32((y - 1) * w_i + x);
        let n_el = cell_el_thermal(cells[ni]);
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (y < h_i - 1) {
        let ni = u32((y + 1) * w_i + x);
        let n_el = cell_el_thermal(cells[ni]);
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }

    var exposure: f32;
    if (me_el == FIRE_ID || me_el == EMPTY_ID) {
        exposure = 1.0;
    } else {
        exposure = (diff_neighbors + oob_neighbors) / 4.0;
    }
    let amb_factor = 0.10 + 0.90 * exposure;
    let ambient_t = me_props.y + f32(u.ambient_offset);
    delta += me_props.z * amb_factor * (ambient_t - me_t);
    let heat_cap = max(me_props.w, 0.0001);
    let exact = me_t + delta / heat_cap;
    let floor_v = floor(exact);
    let frac = exact - floor_v;
    let roll = hash_random(i, u.frame);
    var stepped: f32;
    if (roll < frac) { stepped = floor_v + 1.0; }
    else { stepped = floor_v; }
    let new_t = clamp(stepped, -273.0, 4000.0);
    temp_out[i] = i32(new_t);
}
"#;

const PRESSURE_COMPUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0=extract, 1=diffuse, 2=writeback
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pressure_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> pressure_out: array<i32>;
@group(0) @binding(3) var<storage, read_write> cells: array<vec4<u32>>;
// Per-element permeability LUT: 96 elements packed 4 per vec4.
@group(0) @binding(4) var<uniform> perm_lut: array<vec4<u32>, 24>;

const DIFF_SCALE: i32 = 2048;

fn cell_el_pcompute(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cell_perm_pcompute(c: vec4<u32>) -> u32 {
    let id = cell_el_pcompute(c);
    return perm_lut[id / 4u][id % 4u];
}
// Extract i16 from bits 16-31 of `word` and sign-extend to i32.
fn extract_i16_pcompute(word: u32) -> i32 {
    let raw = (word >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    if (x >= w_i || y >= h_i) { return; }
    let i = u32(y * w_i + x);

    if (u.pass_id == 0u) {
        // Extract: copy current pressure (i16 at bits 16-31 of cells[i].z)
        // into pressure_out (acts as scratch_a for the first diffuse iter).
        pressure_out[i] = extract_i16_pcompute(cells[i].z);
        return;
    }
    if (u.pass_id == 2u) {
        // Writeback: pressure_in holds the final diffused pressure.
        // Clamp to i16, write back to bits 16-31 of cells[i].z while
        // preserving moisture (bits 0-7) and burn (bits 8-15).
        let final_p = clamp(pressure_in[i], -4000, 4000);
        let raw = u32(final_p) & 0xFFFFu;
        let lo = cells[i].z & 0x0000FFFFu;
        cells[i].z = lo | (raw << 16u);
        return;
    }

    // pass_id == 1: diffusion iteration (existing logic, perm via LUT).
    let me_cell = cells[i];
    let me_perm = i32(cell_perm_pcompute(me_cell));
    let me_p = pressure_in[i];
    if (me_perm == 0) {
        pressure_out[i] = me_p;
        return;
    }
    var new_p = me_p;
    // LEFT — horizontal OOB acts as open atmosphere (P=0, perm=255).
    if (x > 0) {
        let ni = u32(y * w_i + (x - 1));
        let n_p = pressure_in[ni];
        let n_perm = i32(cell_perm_pcompute(cells[ni]));
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    } else {
        new_p += (-me_p) * min(me_perm, 255) / DIFF_SCALE;
    }
    // RIGHT
    if (x < w_i - 1) {
        let ni = u32(y * w_i + (x + 1));
        let n_p = pressure_in[ni];
        let n_perm = i32(cell_perm_pcompute(cells[ni]));
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    } else {
        new_p += (-me_p) * min(me_perm, 255) / DIFF_SCALE;
    }
    // UP — vertical OOB sealed.
    if (y > 0) {
        let ni = u32((y - 1) * w_i + x);
        let n_p = pressure_in[ni];
        let n_perm = i32(cell_perm_pcompute(cells[ni]));
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    }
    // DOWN
    if (y < h_i - 1) {
        let ni = u32((y + 1) * w_i + x);
        let n_p = pressure_in[ni];
        let n_perm = i32(cell_perm_pcompute(cells[ni]));
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    }
    pressure_out[i] = new_p;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ComputeUniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PressurePassUniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0=extract, 1=diffuse, 2=writeback
    _pad: u32,
}

/// GPU compute pipeline for pressure diffusion. All cell state lives in
/// `cells_buf` (shared across motion/pressure/thermal/PS) — no per-cell
/// CPU stage loops, no separate readback buffer. Three pass-modes share
/// one pipeline:
///   pass 0 (extract):   reads pressure i16 out of `cells_buf` into pressure_a.
///   pass 1 (diffuse):   3 iterations of 4-neighbor flux, ping-pong a↔b,
///                       perm read from cells_buf via perm_lut LUT.
///   pass 2 (writeback): clamps + writes the final scratch back into
///                       cells_buf bits 16-31 of `.z`.
struct PressureComputeCtx {
    pipeline: wgpu::ComputePipeline,
    pressure_a: wgpu::Buffer,
    pressure_b: wgpu::Buffer,
    /// Pre-baked uniform buffers, one per pass id. queue.write_buffer
    /// collapses to last-write-wins inside a single submit, so each
    /// pass-mode needs its own immutable uniform.
    u_extract: wgpu::Buffer,
    u_diffuse: wgpu::Buffer,
    u_writeback: wgpu::Buffer,
    perm_lut_buf: wgpu::Buffer,
    bg_extract: wgpu::BindGroup,
    bg_a_to_b: wgpu::BindGroup,
    bg_b_to_a: wgpu::BindGroup,
    bg_writeback: wgpu::BindGroup,
    iters: u32,
}

impl PressureComputeCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let cell_count = W * H;
        let scratch_bytes = (cell_count * 4) as wgpu::BufferAddress;

        let mk_uniform = |label: &str, pass_id: u32| {
            let u = PressurePassUniforms {
                width: W as u32,
                height: H as u32,
                pass_id,
                _pad: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let u_extract   = mk_uniform("alembic-pressure-uniforms-extract",   0);
        let u_diffuse   = mk_uniform("alembic-pressure-uniforms-diffuse",   1);
        let u_writeback = mk_uniform("alembic-pressure-uniforms-writeback", 2);
        let _ = queue;

        // Per-element permeability LUT — 96 u32s packed 4 per vec4.
        let mut perm_lut: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0..96u32 {
            let p = crate::pressure_perm_props(el_id as u8);
            perm_lut[(el_id / 4) as usize][(el_id % 4) as usize] = p[0];
        }
        let perm_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-pressure-perm-lut"),
            contents: bytemuck::cast_slice(&perm_lut),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: scratch_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let pressure_a = make_storage("alembic-pressure-a");
        let pressure_b = make_storage("alembic-pressure-b");

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-pressure-compute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        // Bind group for extract: scratch_a is bound to pressure_in
        // (unused by the shader in this pass) AND pressure_out (the
        // target). The shader only writes pressure_out in pass 0.
        let bg_extract = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bg-extract"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_extract.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: perm_lut_buf.as_entire_binding() },
            ],
        });
        let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bg-a-to-b"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_diffuse.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: perm_lut_buf.as_entire_binding() },
            ],
        });
        let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bg-b-to-a"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_diffuse.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: perm_lut_buf.as_entire_binding() },
            ],
        });
        // Writeback reads the final scratch (pressure_b after 3 iters
        // starting from a→b: a→b, b→a, a→b → result in b).
        let bg_writeback = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bg-writeback"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_writeback.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: perm_lut_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-pressure-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(PRESSURE_COMPUTE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-pressure-compute-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-pressure-compute-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        PressureComputeCtx {
            pipeline,
            pressure_a,
            pressure_b,
            u_extract,
            u_diffuse,
            u_writeback,
            perm_lut_buf,
            bg_extract,
            bg_a_to_b,
            bg_b_to_a,
            bg_writeback,
            iters: 3,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 15) / 16;
        let wg_y = (H as u32 + 15) / 16;
        // Pass 0: extract pressure from cells_buf → pressure_a.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-pressure-extract"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_extract, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Pass 1: diffuse — 3 iterations, ping-pong.
        for iter in 0..self.iters {
            let bind = if iter % 2 == 0 { &self.bg_a_to_b } else { &self.bg_b_to_a };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-pressure-diffuse"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Pass 2: writeback final scratch → cells_buf pressure bits.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-pressure-writeback"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_writeback, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        let _ = (&self.u_extract, &self.u_diffuse, &self.u_writeback, &self.perm_lut_buf);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ThermalUniforms {
    width: u32,
    height: u32,
    ambient_offset: i32,
    frame: u32,
    pass_id: u32,            // 0=extract, 1=diffuse, 2=writeback
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU compute pipeline for thermal diffusion (heat exchange + ambient
/// blend). Reads temp + el directly out of the shared `cells_buf`
/// using extract → diffuse (3 iters) → writeback. No per-cell CPU
/// stage loops, no separate readback path.
struct ThermalComputeCtx {
    pipeline: wgpu::ComputePipeline,
    /// Per-frame uniform — only `frame`, `ambient_offset`, and
    /// `pass_id` change. Each pass has its own pre-baked uniform
    /// buffer because queue.write_buffer collapses to last-write-wins
    /// inside a submit.
    u_extract: wgpu::Buffer,
    u_diffuse: wgpu::Buffer,
    u_writeback: wgpu::Buffer,
    #[allow(dead_code)]
    profiles_buf: wgpu::Buffer,
    temp_a: wgpu::Buffer,
    temp_b: wgpu::Buffer,
    bg_extract: wgpu::BindGroup,
    bg_a_to_b: wgpu::BindGroup,
    bg_b_to_a: wgpu::BindGroup,
    bg_writeback: wgpu::BindGroup,
    iters: u32,
}

impl ThermalComputeCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let cell_count = W * H;
        let scratch_bytes = (cell_count * 4) as wgpu::BufferAddress;

        let mut profile_data: Vec<[f32; 4]> = vec![[0.0, 20.0, 0.0, 1.0]; 96];
        for i in 0..96 {
            profile_data[i] = thermal_profile_vec4(i as u8);
        }
        let profiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-thermal-profiles"),
            contents: bytemuck::cast_slice(&profile_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mk_uniform = |label: &str, pass_id: u32| {
            let u = ThermalUniforms {
                width: W as u32,
                height: H as u32,
                ambient_offset: 0,
                frame: 0,
                pass_id,
                _pad0: 0, _pad1: 0, _pad2: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let u_extract   = mk_uniform("alembic-thermal-uniforms-extract",   0);
        let u_diffuse   = mk_uniform("alembic-thermal-uniforms-diffuse",   1);
        let u_writeback = mk_uniform("alembic-thermal-uniforms-writeback", 2);
        let _ = queue;

        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: scratch_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let temp_a = make_storage("alembic-temp-a");
        let temp_b = make_storage("alembic-temp-b");

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-thermal-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bg_extract = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bg-extract"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_extract.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bg-a-to-b"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_diffuse.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bg-b-to-a"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_diffuse.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let bg_writeback = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bg-writeback"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_writeback.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-thermal-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(THERMAL_COMPUTE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-thermal-compute-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-thermal-compute-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        ThermalComputeCtx {
            pipeline,
            u_extract,
            u_diffuse,
            u_writeback,
            profiles_buf,
            temp_a,
            temp_b,
            bg_extract,
            bg_a_to_b,
            bg_b_to_a,
            bg_writeback,
            iters: 1,
        }
    }

    /// Update the per-frame fields (frame counter + ambient offset) in
    /// each pass uniform. Cell field reads come straight from
    /// cells_buf — no per-cell staging.
    fn update_frame(&self, queue: &wgpu::Queue, frame: u32, ambient_offset: i16) {
        // Layout: width(4), height(4), ambient_offset(4), frame(4),
        // pass_id(4), pad(4)*3. Update bytes 8-15 (ambient_offset, frame).
        let bytes: [u8; 8] = bytemuck::cast([(ambient_offset as i32), frame as i32]);
        for buf in [&self.u_extract, &self.u_diffuse, &self.u_writeback] {
            queue.write_buffer(buf, 8, &bytes);
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 15) / 16;
        let wg_y = (H as u32 + 15) / 16;
        // Pass 0: extract.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-thermal-extract"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_extract, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Pass 1: diffuse iters (1 by default — keeps the original
        // single-iter behavior of thermal_diffuse).
        for iter in 0..self.iters {
            let bind = if iter % 2 == 0 { &self.bg_a_to_b } else { &self.bg_b_to_a };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-thermal-diffuse"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Pass 2: writeback. With iters=1 (a→b), final is in temp_b,
        // which is what bg_writeback's pressure_in points to.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-thermal-writeback"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_writeback, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const PRESSURE_SOURCES_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    gravity_present: u32,
    gy: i32,
    gravity_mag: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
// vec4 per element: x = kind_id, y = weight, z = _, w = _
@group(0) @binding(2) var<uniform> profiles: array<vec4<f32>, 96>;

const FLAG_FROZEN: u32 = 0x02u;
const KIND_EMPTY: u32  = 0u;
const KIND_LIQUID: u32 = 4u;
const KIND_GAS: u32    = 5u;
const KIND_FIRE: u32   = 6u;

fn cell_el_ps(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cell_flag_ps(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn extract_temp_ps(c: vec4<u32>) -> i32 {
    let raw = (c.y >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}
fn extract_pressure_ps(c: vec4<u32>) -> i32 {
    let raw = (c.z >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}

// One thread per column. Walks vertically computing the column-
// integrated hydrostatic pressure plus per-cell thermal target,
// then blends current pressure → target with the asymmetric rule
// (gas/fire only blend up; everything else blends both ways).
// Reads + writes pressure directly in cells_buf — no per-field
// staging. Per-column dispatch keeps writes race-free.
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= u.width) { return; }

    let h = i32(u.height);
    let w = u.width;
    var col_p: f32 = 0.0;

    var y_start: i32;
    var y_end: i32;
    var step: i32;
    if (u.gy >= 0) {
        y_start = 0;
        y_end = h;
        step = 1;
    } else {
        y_start = h - 1;
        y_end = -1;
        step = -1;
    }

    var y = y_start;
    loop {
        if (y == y_end) { break; }
        let i = w * u32(y) + x;
        let c = cells[i];
        let id = cell_el_ps(c);
        let prof = profiles[id];
        let kind_id = u32(prof.x);
        let weight = prof.y;

        let is_pressurizable = (kind_id == KIND_GAS || kind_id == KIND_FIRE);
        let is_wallable = (kind_id != KIND_EMPTY
            && kind_id != KIND_LIQUID
            && kind_id != KIND_GAS
            && kind_id != KIND_FIRE);
        let is_frozen = (cell_flag_ps(c) & FLAG_FROZEN) != 0u;

        var tgt: i32 = 0;
        if (is_pressurizable) {
            let t = (extract_temp_ps(c) - 20) * 5;
            tgt = clamp(t, -300, 4000);
        }

        if (u.gravity_present != 0u && u.gy != 0) {
            if (is_frozen && is_wallable) {
                col_p = 0.0;
            } else {
                col_p = col_p + weight * u.gravity_mag;
                let p_c = i32(clamp(col_p, -4000.0, 4000.0));
                tgt = clamp(tgt + p_c, -4000, 4000);
            }
        }

        let current = extract_pressure_ps(c);
        let delta = tgt - current;
        var new_p = current;
        if (delta > 0) {
            var stp = (delta * 5) / 100;
            if (stp < 1) { stp = 1; }
            new_p = clamp(current + stp, -4000, 4000);
        } else if (delta < 0 && !is_pressurizable) {
            var stp = (delta * 2) / 100;
            if (stp > -1) { stp = -1; }
            new_p = clamp(current + stp, -4000, 4000);
        }
        // Write pressure back to cells[i].z bits 16-31, preserving
        // moisture (bits 0-7) and burn (bits 8-15).
        let raw = u32(new_p) & 0xFFFFu;
        let lo = cells[i].z & 0x0000FFFFu;
        cells[i].z = lo | (raw << 16u);

        y = y + step;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PressureSourcesUniforms {
    width: u32,
    height: u32,
    gravity_present: u32,
    gy: i32,
    gravity_mag: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU compute pipeline for the hydrostatic + thermal pressure-target
/// pass. Reads cell fields directly from the shared `cells_buf` and
/// writes new pressure straight back into it. Per-column dispatch
/// keeps writes race-free (each thread owns its column entirely).
struct PressureSourcesCtx {
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    profiles_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PressureSourcesCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        // Per-element profile uniform (96 vec4s = 1536 bytes).
        let mut profile_data: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 0.0]; 96];
        for i in 0..96 {
            profile_data[i] = pressure_source_props(i as u8);
        }
        let profiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-ps-profiles"),
            contents: bytemuck::cast_slice(&profile_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let init_uniforms = PressureSourcesUniforms {
            width: W as u32,
            height: H as u32,
            gravity_present: 1,
            gy: 1,
            gravity_mag: 1.0,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-ps-uniforms"),
            contents: bytemuck::cast_slice(&[init_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let _ = queue;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-ps-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-ps-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-ps-shader"),
            source: wgpu::ShaderSource::Wgsl(PRESSURE_SOURCES_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-ps-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-ps-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        PressureSourcesCtx {
            pipeline,
            uniform_buf,
            profiles_buf,
            bind_group,
        }
    }

    /// Update gravity uniform (only thing that varies across frames).
    fn update_frame(&self, queue: &wgpu::Queue, world: &World) {
        let uniforms = PressureSourcesUniforms {
            width: W as u32,
            height: H as u32,
            gravity_present: if world.gravity > 0.0 { 1 } else { 0 },
            gy: 1,
            gravity_mag: world.gravity,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 63) / 64;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-ps-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, 1, 1);
    }
}

const MOTION_COMPUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,         // 0 = vertical fall, 1 = liquid spread
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
// vec4 per element: x = kind_id, y = density (signed),
//                   z = viscosity, w = molar_mass
@group(0) @binding(2) var<uniform> motion_props: array<vec4<f32>, 96>;
// Derived compound mirror: same layout as motion_props, indexed by
// cell.derived_id when cell.el == Element::Derived (41).
@group(0) @binding(3) var<uniform> derived_phys: array<vec4<f32>, 256>;

const KIND_EMPTY: u32  = 0u;
const KIND_SOLID: u32  = 1u;
const KIND_GRAVEL: u32 = 2u;
const KIND_POWDER: u32 = 3u;
const KIND_LIQUID: u32 = 4u;
const KIND_GAS: u32    = 5u;
const KIND_FIRE: u32   = 6u;

const FLAG_UPDATED: u32 = 0x01u;
const FLAG_FROZEN:  u32 = 0x02u;
const EL_DERIVED:   u32 = 41u;

fn cell_idx(x: u32, y: u32) -> u32 { return y * u.width + x; }
fn cell_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cell_derived_id(c: vec4<u32>) -> u32 { return (c.x >> 8u) & 0xFFu; }
fn cell_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn cell_props(c: vec4<u32>) -> vec4<f32> {
    let el = cell_el(c);
    if (el == EL_DERIVED) {
        return derived_phys[cell_derived_id(c)];
    }
    return motion_props[el];
}
fn cell_kind(c: vec4<u32>) -> u32 { return u32(cell_props(c).x); }
fn cell_density(c: vec4<u32>) -> f32 { return cell_props(c).y; }
fn cell_viscosity(c: vec4<u32>) -> f32 { return cell_props(c).z; }
fn cell_molar_mass(c: vec4<u32>) -> f32 { return cell_props(c).w; }
fn cell_frozen(c: vec4<u32>) -> bool { return (cell_flag(c) & FLAG_FROZEN) != 0u; }
fn cell_updated(c: vec4<u32>) -> bool { return (cell_flag(c) & FLAG_UPDATED) != 0u; }
fn mark_updated(c: vec4<u32>) -> vec4<u32> {
    return vec4<u32>(c.x, c.y | 0x100u, c.z, c.w);
}

fn kind_is_rigid(k: u32) -> bool {
    return k == KIND_SOLID || k == KIND_GRAVEL || k == KIND_POWDER;
}

// Faithful port of `World::can_enter(src, tx, ty, dy)` from lib.rs.
// Returns true iff `src_cell` can swap with the cell at (tx, ty)
// when moving in direction dy (-1=up, 0=horizontal, 1=down).
fn can_enter(src_cell: vec4<u32>, tx: i32, ty: i32, dy: i32) -> bool {
    if (tx < 0 || tx >= i32(u.width) || ty < 0 || ty >= i32(u.height)) {
        return false;
    }
    let i_t = u32(ty) * u.width + u32(tx);
    let tgt = cells[i_t];
    if (cell_updated(tgt)) { return false; }
    let sk = cell_kind(src_cell);
    let tk = cell_kind(tgt);
    if (tk == KIND_EMPTY) { return true; }
    if (kind_is_rigid(tk)) { return false; }
    // Rigid src into viscous liquid (lava crust) — can't sink.
    if (kind_is_rigid(sk) && cell_viscosity(tgt) > 100.0) { return false; }
    // Gas-gas mixing: any gas/fire can swap with any other gas/fire.
    let s_gasy = (sk == KIND_GAS || sk == KIND_FIRE);
    let t_gasy = (tk == KIND_GAS || tk == KIND_FIRE);
    if (s_gasy && t_gasy) { return true; }
    // Density-direction check (matches CPU exactly).
    let sd = cell_density(src_cell);
    let td = cell_density(tgt);
    if (dy > 0) { return sd > td; }
    if (dy < 0) { return sd < td; }
    return sd > td;
}

// Pass 0 — vertical fall: one thread per column, bottom-up walk.
// When sand at row Y swaps with empty at Y+1, the next iteration
// (row Y-1) sees the now-empty Y and can fall into it. Cascading is
// automatic, so a column shifts down by one per frame without
// fragmenting (which is what Margolus block updates produced).
//
const EL_WOOD: u32 = 4u;
const EL_SEED: u32 = 10u;
const EL_LEAVES: u32 = 12u;

// Pass 0 — vertical fall. Faithful port of update_powder, update_gravel,
// update_liquid's straight-down step PLUS the bespoke wood-life-fall
// branch from update_cell, the unrooted-seed fall from update_seed,
// and the unsupported-leaves fall from update_leaves. Per-column
// thread, walks bottom-up so cascading happens within one pass.
fn vertical_fall(x: u32) {
    var y = i32(u.height) - 2;
    loop {
        if (y < 0) { break; }
        let i_here = cell_idx(x, u32(y));
        let c = cells[i_here];
        if (cell_updated(c) || cell_frozen(c)) { y = y - 1; continue; }
        let k = cell_kind(c);
        let el = cell_el(c);
        let life = (c.x >> 16u) & 0xFFFFu;

        var tried = false;

        // Powder / Gravel / Liquid → standard straight-down via can_enter.
        if (k == KIND_POWDER || k == KIND_GRAVEL || k == KIND_LIQUID) {
            if (can_enter(c, i32(x), y + 1, 1)) {
                let i_below = cell_idx(x, u32(y + 1));
                let c_below = cells[i_below];
                cells[i_here]  = c_below;
                cells[i_below] = mark_updated(c);
                tried = true;
            }
        }

        // Wood with life > 0 → falling unsupported wood. Per CPU's
        // update_cell wood branch: only falls into Empty (not gas/fire).
        if (!tried && el == EL_WOOD && life > 0u) {
            let i_below = cell_idx(x, u32(y + 1));
            let c_below = cells[i_below];
            if (cell_kind(c_below) == KIND_EMPTY
                && !cell_updated(c_below) && !cell_frozen(c_below)) {
                cells[i_here]  = c_below;
                cells[i_below] = mark_updated(c);
                tried = true;
            } else {
                // Landed: clear falling flag (life = 0). Per CPU.
                cells[i_here] = vec4<u32>(c.x & 0xFFFFu, c.y, c.z, c.w);
            }
        }

        // Seed with life == 0 → unrooted seed. Falls into Empty only
        // (CPU update_seed). Diagonals tried in dslide.
        if (!tried && el == EL_SEED && life == 0u) {
            let i_below = cell_idx(x, u32(y + 1));
            let c_below = cells[i_below];
            if (cell_kind(c_below) == KIND_EMPTY
                && !cell_updated(c_below) && !cell_frozen(c_below)) {
                cells[i_here]  = c_below;
                cells[i_below] = mark_updated(c);
                tried = true;
            }
        }

        // Leaves → unsupported fall. CPU update_leaves checks 5×5
        // for any Wood/Seed; if found, supported (do nothing).
        // Otherwise, 25% chance per frame to fall via can_enter.
        if (!tried && el == EL_LEAVES) {
            // Probability gate (matches CPU's 1-in-4 + should_fall=1 at g=1).
            let r = hash_u32_motion(i_here, u.frame);
            if ((r & 3u) == 0u) {
                // Check 5×5 neighborhood for Wood/Seed.
                var supported = false;
                for (var dy: i32 = -2; dy <= 2 && !supported; dy = dy + 1) {
                    for (var dx: i32 = -2; dx <= 2 && !supported; dx = dx + 1) {
                        if (dx == 0 && dy == 0) { continue; }
                        let nx = i32(x) + dx;
                        let ny = i32(y) + dy;
                        if (nx < 0 || nx >= i32(u.width) || ny < 0 || ny >= i32(u.height)) { continue; }
                        let n_el = cell_el(cells[cell_idx(u32(nx), u32(ny))]);
                        if (n_el == EL_WOOD || n_el == EL_SEED) { supported = true; }
                    }
                }
                if (!supported) {
                    if (can_enter(c, i32(x), y + 1, 1)) {
                        let i_below = cell_idx(x, u32(y + 1));
                        let c_below = cells[i_below];
                        cells[i_here]  = c_below;
                        cells[i_below] = mark_updated(c);
                    }
                }
            }
        }
        y = y - 1;
    }
}

// Pass 3/4 — diagonal slide for Powder and Liquid. Faithful port of
// update_powder and update_liquid's diagonal step: only fires when
// straight-down is blocked. Tries one diagonal first (chosen by
// frame parity to avoid drift), then the other if the first is blocked.
// Gravel does NOT slide diagonally (only Powder + Liquid do per CPU).
//
// Per-column threads with even/odd column parity sub-passes — the
// diagonal target column is x±1, and in each sub-pass only every-other
// column is active, so writes to the target column don't race.
fn diagonal_slide(x: u32, parity: u32) {
    if ((x & 1u) != parity) { return; }
    let frame_lr = (u.frame & 1u) == 0u;
    var dx_first: i32 = 1;
    var dx_second: i32 = -1;
    if (frame_lr) { dx_first = -1; dx_second = 1; }

    var y = i32(u.height) - 2;
    loop {
        if (y < 0) { break; }
        let i_here = cell_idx(x, u32(y));
        let c = cells[i_here];
        if (cell_updated(c) || cell_frozen(c)) { y = y - 1; continue; }
        let k = cell_kind(c);
        if (k != KIND_POWDER && k != KIND_LIQUID) { y = y - 1; continue; }
        // If straight-down was open, vfall would have moved us. If it
        // was blocked, try diagonals now.
        if (can_enter(c, i32(x), y + 1, 1)) { y = y - 1; continue; }

        // Try first diagonal.
        let nx1 = i32(x) + dx_first;
        if (can_enter(c, nx1, y + 1, 1)) {
            let i_t = cell_idx(u32(nx1), u32(y + 1));
            let c_t = cells[i_t];
            cells[i_here] = c_t;
            cells[i_t]    = mark_updated(c);
        } else {
            let nx2 = i32(x) + dx_second;
            if (can_enter(c, nx2, y + 1, 1)) {
                let i_t = cell_idx(u32(nx2), u32(y + 1));
                let c_t = cells[i_t];
                cells[i_here] = c_t;
                cells[i_t]    = mark_updated(c);
            }
        }
        y = y - 1;
    }
}

// Pass 1/2 — liquid horizontal spread. Each thread handles one row,
// Pass 1/2 — liquid horizontal spread. Faithful port of update_liquid's
// horizontal step: only fires when straight-down AND both diagonals
// below are blocked (otherwise vfall + dslide handle it). Viscosity
// throttles the spread rate (lava oozes, water runs flat).
//
// Per-row threads with even/odd row parity sub-passes keep the
// support-check read of row y+1 race-free.
fn liquid_spread(y: u32, parity: u32) {
    if ((y & 1u) != parity) { return; }
    let frame_lr = (u.frame & 1u) == 0u;
    var dx_first: i32 = 1;
    var dx_second: i32 = -1;
    if (frame_lr) { dx_first = -1; dx_second = 1; }
    let w_i = i32(u.width);

    var x = 0i;
    loop {
        if (x >= w_i) { break; }
        let i_here = cell_idx(u32(x), y);
        let c = cells[i_here];
        if (cell_updated(c) || cell_frozen(c)) { x = x + 1; continue; }
        if (cell_kind(c) != KIND_LIQUID) { x = x + 1; continue; }
        // Only spread sideways if straight-down + both diagonals are
        // blocked — that's the CPU's fall priority order.
        if (can_enter(c, x, i32(y) + 1, 1)) { x = x + 1; continue; }
        if (can_enter(c, x + dx_first,  i32(y) + 1, 1)) { x = x + 1; continue; }
        if (can_enter(c, x + dx_second, i32(y) + 1, 1)) { x = x + 1; continue; }
        // Viscosity throttle: visc 0 always spreads, visc 400 never.
        let visc = cell_viscosity(c);
        let r = hash_u32_motion(u32(i_here), u.frame);
        if (visc > 0.0 && (f32(r & 0xFFFu) / 4096.0) * 400.0 < visc) {
            x = x + 1; continue;
        }
        // Try first horizontal direction, then second.
        if (can_enter(c, x + dx_first, i32(y), 0)) {
            let i_t = cell_idx(u32(x + dx_first), y);
            let c_t = cells[i_t];
            cells[i_here] = c_t;
            cells[i_t]    = mark_updated(c);
        } else if (can_enter(c, x + dx_second, i32(y), 0)) {
            let i_t = cell_idx(u32(x + dx_second), y);
            let c_t = cells[i_t];
            cells[i_here] = c_t;
            cells[i_t]    = mark_updated(c);
        }
        x = x + 1;
    }
}

// Pass 5/6 — gas turbulence: every gas/fire cell, with PRNG-driven
// probability, swaps with an empty horizontal neighbor (left or right
// chosen by hash). Combined with vertical_fall's density-based rise
// (when the gas is lighter than empty), the result is a turbulent
// cloud that slowly drifts up or down based on the gas's signed
// density — not a column shooting to the ceiling.
//
// Even-column vs odd-column sub-passes keep writes race-free: an
// even-x thread writes its own column AND the odd column next door;
// in that sub-pass no odd-x thread is running.
fn hash_u32_motion(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    h *= 3266489917u;
    h ^= h >> 16u;
    return h;
}

fn gas_dynamics(x: u32, parity: u32) {
    if ((x & 1u) != parity) { return; }
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    var y = 0i;
    loop {
        if (y >= h_i) { break; }
        let i_here = cell_idx(x, u32(y));
        let c = cells[i_here];
        if (cell_updated(c) || cell_frozen(c)) { y = y + 1; continue; }
        let k = cell_kind(c);
        if (k != KIND_GAS && k != KIND_FIRE) { y = y + 1; continue; }

        // Faithful port of update_gas's "empty expansion" — the
        // primary drive of gas motion in CPU sim. Tries 4 cardinal
        // directions in randomized order, 50% probability each;
        // first successful swap with empty wins.
        let r = hash_u32_motion(i_here, u.frame);
        let start = r & 3u;
        var moved = false;
        // Unrolled 4-direction try with rotated start.
        for (var k4 = 0u; k4 < 4u && !moved; k4 = k4 + 1u) {
            let pick = (start + k4) & 3u;
            var dx: i32 = 0;
            var dy: i32 = 0;
            if (pick == 0u) { dx = -1; }
            else if (pick == 1u) { dx = 1; }
            else if (pick == 2u) { dy = -1; }
            else { dy = 1; }
            let nx = i32(x) + dx;
            let ny = y + dy;
            if (nx < 0 || nx >= w_i || ny < 0 || ny >= h_i) { continue; }
            let i_t = cell_idx(u32(nx), u32(ny));
            let c_t = cells[i_t];
            if (cell_kind(c_t) != KIND_EMPTY) { continue; }
            if (cell_updated(c_t) || cell_frozen(c_t)) { continue; }
            // 50% per-direction probability (CPU baseline).
            let prob_bit = (r >> (8u + k4 * 2u)) & 1u;
            if (prob_bit == 0u) { continue; }
            cells[i_here] = mark_updated(c_t);
            cells[i_t]    = mark_updated(c);
            moved = true;
        }
        // Buoyancy fallback: if no lateral/vertical empty swap fired,
        // try molar-mass-driven rise/sink. Light gases (mass < 29 ~
        // AMBIENT_AIR) bias upward; heavy gases bias downward.
        if (!moved) {
            let m = cell_molar_mass(c);
            if (m > 0.0) {
                let air_mass: f32 = 29.0;
                let bias = (air_mass - m) / air_mass;
                let dir_y: i32 = select(1, -1, bias > 0.0);
                let bias_abs = abs(bias);
                let prob_byte = (r >> 16u) & 0xFFu;
                if (f32(prob_byte) / 255.0 < bias_abs) {
                    let nx = i32(x);
                    let ny = y + dir_y;
                    if (ny >= 0 && ny < h_i) {
                        if (can_enter(c, nx, ny, dir_y)) {
                            let i_t = cell_idx(u32(nx), u32(ny));
                            let c_t = cells[i_t];
                            cells[i_here] = c_t;
                            cells[i_t]    = mark_updated(c);
                        }
                    }
                }
            }
        }
        y = y + 1;
    }
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (u.pass_id == 0u) {
        let x = gid.x;
        if (x >= u.width) { return; }
        vertical_fall(x);
    } else if (u.pass_id == 1u) {
        let y = gid.x;
        if (y >= u.height - 1u) { return; }
        liquid_spread(y, 0u);
    } else if (u.pass_id == 2u) {
        let y = gid.x;
        if (y >= u.height - 1u) { return; }
        liquid_spread(y, 1u);
    } else if (u.pass_id == 3u) {
        let x = gid.x;
        if (x >= u.width) { return; }
        diagonal_slide(x, 0u);
    } else if (u.pass_id == 4u) {
        let x = gid.x;
        if (x >= u.width) { return; }
        diagonal_slide(x, 1u);
    } else if (u.pass_id == 5u) {
        let x = gid.x;
        if (x >= u.width) { return; }
        gas_dynamics(x, 0u);
    } else if (u.pass_id == 6u) {
        let x = gid.x;
        if (x >= u.width) { return; }
        gas_dynamics(x, 1u);
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MotionUniforms {
    width: u32,
    height: u32,
    pass_id: u32,         // 0 = vertical fall, 1 = liquid spread
    frame: u32,
}

/// GPU motion compute pipeline. Two passes per frame:
///   pass 0: vertical fall (1 thread per column, bottom-up walk).
///           Cells fall through empty space, cascading within the
///           same thread. Adjacent columns are independent.
///   pass 1: liquid spread (1 thread per row, frame-parity-biased
///           direction). Liquids resting on solid floor spread
///           horizontally into adjacent empty/gas.
/// Each pass uses its own pre-baked uniform buffer + bind group
/// because queue.write_buffer collapses to last-write-wins within
/// a single submit (so a shared uniform with in-loop writes makes
/// every dispatch see the same value).
struct MotionComputeCtx {
    pipeline: wgpu::ComputePipeline,
    pass_uniform_bufs: [wgpu::Buffer; 7],
    #[allow(dead_code)]
    motion_props_buf: wgpu::Buffer,
    readback_bufs: [wgpu::Buffer; 2],
    pass_bind_groups: [wgpu::BindGroup; 7],
    write_idx: usize,
    has_data: [bool; 2],
}

impl MotionComputeCtx {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cells_buf: &wgpu::Buffer,
        derived_phys_buf: &wgpu::Buffer,
    ) -> Self {
        let cell_count = W * H;
        let cell_bytes = (cell_count * std::mem::size_of::<crate::Cell>()) as wgpu::BufferAddress;

        // Per-element motion props: kind_id, density, falling, is_liquid.
        let mut props_data: Vec<[f32; 4]> = vec![[0.0; 4]; 96];
        for i in 0..96 {
            props_data[i] = crate::motion_props(i as u8);
        }
        let motion_props_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-motion-props"),
            contents: bytemuck::cast_slice(&props_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 2 pass uniform buffers, each pre-baked with a different
        // pass_id. Frame is updated every frame on both.
        let mk_pass_uniform = |pass_id: u32, label: &str| {
            let u = MotionUniforms { width: W as u32, height: H as u32, pass_id, frame: 0 };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pass_uniform_bufs: [wgpu::Buffer; 7] = [
            mk_pass_uniform(0, "alembic-motion-uniforms-vfall"),
            mk_pass_uniform(1, "alembic-motion-uniforms-lspread-even"),
            mk_pass_uniform(2, "alembic-motion-uniforms-lspread-odd"),
            mk_pass_uniform(3, "alembic-motion-uniforms-dslide-even"),
            mk_pass_uniform(4, "alembic-motion-uniforms-dslide-odd"),
            mk_pass_uniform(5, "alembic-motion-uniforms-gdrift-even"),
            mk_pass_uniform(6, "alembic-motion-uniforms-gdrift-odd"),
        ];
        let _ = queue;

        let make_readback = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cell_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let readback_bufs = [
            make_readback("alembic-motion-readback-0"),
            make_readback("alembic-motion-readback-1"),
        ];

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-motion-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-motion-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: motion_props_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: derived_phys_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 7] = [
            mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3),
            mk_bind(4), mk_bind(5), mk_bind(6),
        ];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-motion-shader"),
            source: wgpu::ShaderSource::Wgsl(MOTION_COMPUTE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-motion-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-motion-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let _ = cell_count;
        MotionComputeCtx {
            pipeline,
            pass_uniform_bufs,
            motion_props_buf,
            readback_bufs,
            pass_bind_groups,
            write_idx: 0,
            has_data: [false; 2],
        }
    }

    /// Encode all 7 motion passes for one frame, then snapshot the
    /// shared `cells_buf` into the readback buffer for next-frame
    /// CPU sync.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32, cells_buf: &wgpu::Buffer) {
        let cell_count = W * H;
        let frame_arr = [frame];
        let frame_bytes = bytemuck::cast_slice(&frame_arr);
        for i in 0..7 {
            queue.write_buffer(&self.pass_uniform_bufs[i], 12, frame_bytes);
        }
        let wg_col = (W as u32 + 63) / 64;
        let wg_row = (H as u32 + 63) / 64;
        let labels_dispatches = [
            ("alembic-motion-vfall",         wg_col, 0usize),
            ("alembic-motion-lspread-even",  wg_row, 1usize),
            ("alembic-motion-lspread-odd",   wg_row, 2usize),
            ("alembic-motion-dslide-even",   wg_col, 3usize),
            ("alembic-motion-dslide-odd",    wg_col, 4usize),
            ("alembic-motion-gdrift-even",   wg_col, 5usize),
            ("alembic-motion-gdrift-odd",    wg_col, 6usize),
        ];
        for (label, wg, idx) in labels_dispatches {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.pass_bind_groups[idx], &[]);
            cpass.dispatch_workgroups(wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            cells_buf, 0, &self.readback_bufs[self.write_idx], 0,
            (cell_count * std::mem::size_of::<crate::Cell>()) as wgpu::BufferAddress,
        );
    }

    fn start_map(&mut self) {
        self.readback_bufs[self.write_idx]
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.has_data[self.write_idx] = true;
    }

    fn read_back_prev_into(&mut self, world: &mut World) {
        let read_idx = 1 - self.write_idx;
        if !self.has_data[read_idx] { return; }
        let slice = self.readback_bufs[read_idx].slice(..);
        {
            let data = slice.get_mapped_range();
            // Zero-copy: GPU buffer is exactly Cell-shaped bytes.
            crate::cells_copy_from_bytes(&mut world.cells, &data);
        }
        self.readback_bufs[read_idx].unmap();
        self.has_data[read_idx] = false;
    }

    fn advance_frame(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

// ============================================================
// Phase 2 framework passes on GPU
// ============================================================
//
// Element-data-driven framework primitives:
//   * clear_flags         — per-cell write to one bit only
//   * lifecycle           — generic life-decrement + decay-to-product
//                           for any element flagged ephemeral (Fire,
//                           Steam, etc.). One LUT row per element.
//   * color_fires         — per-cell write to one byte (cell.w lo)
//   * flame_test_emission — Margolus 2x2, 4 phases, neighbor write
//                           lives entirely within the same block

const THERMAL_POST_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    frame: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
// Per-element combustion data:
//   x = ignite_above (NO_THRESHOLD if not flammable)
//   y = burn_duration   z = burn_temp   w = self_oxidizing
@group(0) @binding(2) var<uniform> burn_data: array<vec4<f32>, 96>;
// Low-side phase transitions: x=freeze_thr, y=freeze_target, z=condense_thr, w=condense_target
@group(0) @binding(3) var<uniform> phase_lo: array<vec4<f32>, 96>;
// High-side phase transitions: x=melt_thr, y=melt_target, z=boil_thr, w=boil_target
@group(0) @binding(4) var<uniform> phase_hi: array<vec4<f32>, 96>;
// Per-element burn decay product:
//   .x byte0 = primary el (CO2), byte1 = secondary el (Charcoal for Wood),
//      byte2 = secondary prob /16 (3 = ~30%)
@group(0) @binding(5) var<uniform> burn_decay: array<vec4<u32>, 24>;

const NO_THRESHOLD: f32 = -32768.0;

fn tp_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn tp_temp(c: vec4<u32>) -> i32 {
    let raw = (c.y >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}
fn tp_moisture(c: vec4<u32>) -> u32 { return c.z & 0xFFu; }
fn tp_burn(c: vec4<u32>) -> u32 { return (c.z >> 8u) & 0xFFu; }

fn tp_set_temp(c: vec4<u32>, t: i32) -> vec4<u32> {
    let clamped = clamp(t, -273, 5000);
    let raw = u32(clamped) & 0xFFFFu;
    let lo_y = c.y & 0xFFFFu;
    return vec4<u32>(c.x, lo_y | (raw << 16u), c.z, c.w);
}
fn tp_set_burn(c: vec4<u32>, b: u32) -> vec4<u32> {
    let z = c.z & 0xFFFF00FFu;
    return vec4<u32>(c.x, c.y, z | ((b & 0xFFu) << 8u), c.w);
}
fn tp_set_moisture(c: vec4<u32>, m: u32) -> vec4<u32> {
    let z = c.z & 0xFFFFFF00u;
    return vec4<u32>(c.x, c.y, z | (m & 0xFFu), c.w);
}
fn tp_set_el(c: vec4<u32>, el: u32) -> vec4<u32> {
    let x = (c.x & 0xFFFFFF00u) | (el & 0xFFu);
    return vec4<u32>(x, c.y, c.z, c.w);
}

fn tp_hash(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    return h;
}

// Faithful port of the per-cell parts of `World::thermal_post` —
// ignition, burn-out, and phase changes. The bespoke gunpowder/Cs
// detonation path and latent-heat neighbor temperature transfers
// are NOT yet ported (those need shockwave/Margolus support).
//
// All data is element-driven via burn_data + phase_lo + phase_hi
// LUTs — no element-specific code in the shader, just table lookups.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let i = y * u.width + x;
    var c = cells[i];
    let el = tp_el(c);
    if (el == 0u) { return; } // Empty cells are inert.

    let t = tp_temp(c);
    let moisture = tp_moisture(c);
    let burn_in = tp_burn(c);
    let bd = burn_data[el];
    let ignite_thr = bd.x;
    let burn_dur = u32(bd.y);
    let burn_temp = i32(bd.z);
    let self_ox = bd.w > 0.5;
    let has_ignite = ignite_thr > -1000.0;

    // ---- Combustion ignition ----
    if (has_ignite && burn_in == 0u) {
        let normal = (f32(t) > ignite_thr) && (moisture < 20u);
        let flash = f32(t) > ignite_thr + 300.0;
        // Simplified O2: assume always available. Once electrolysis +
        // explicit O2 are on GPU we can read the real oxygen mask.
        let has_o2 = true;
        if ((normal || flash) && has_o2 && burn_dur > 0u) {
            c = tp_set_burn(c, burn_dur);
            c = tp_set_moisture(c, 0u);
        }
    }

    // ---- Burn-out tick ----
    let burn_now = tp_burn(c);
    if (burn_now > 0u) {
        // Drowning.
        if (moisture > 150u) {
            c = tp_set_burn(c, 0u);
            if (has_ignite && f32(t) > ignite_thr - 20.0) {
                c = tp_set_temp(c, i32(ignite_thr - 20.0));
            }
        } else {
            // Burn decrement: 1 per frame baseline (we don't have
            // per-cell oxygen on GPU yet).
            let next_burn = burn_now - 1u;
            c = tp_set_burn(c, next_burn);
            if (next_burn == 0u) {
                // Decay product: CO2 by default, Charcoal probabilistic for Wood.
                let bdec = burn_decay[el / 4u][el % 4u];
                let primary_el = bdec & 0xFFu;
                let secondary_el = (bdec >> 8u) & 0xFFu;
                let sec_prob = (bdec >> 16u) & 0xFFu;
                var decay_el = primary_el;
                if (secondary_el != 0u && sec_prob > 0u) {
                    let r = tp_hash(i, u.frame);
                    if ((r & 0xFu) < sec_prob) { decay_el = secondary_el; }
                }
                // Hot smoke / charcoal residue: temp ~500.
                c = vec4<u32>(decay_el, 0u, 0u, 0u);
                c = tp_set_temp(c, 500);
            } else {
                // Sustain combustion temperature.
                if (i32(burn_temp) > t) {
                    c = tp_set_temp(c, burn_temp);
                }
            }
        }
        // (Skip flame emission above — would need parity sub-pass for
        // the multi-cell write. Add later.)
    }

    // ---- Phase changes (only if combustion didn't already replace cell) ----
    let cur_el = tp_el(c);
    if (cur_el == el && tp_burn(c) == 0u) {
        let cur_t = tp_temp(c);
        let plo = phase_lo[el];
        let phi = phase_hi[el];
        // Try freeze.
        if (plo.x > -1000.0 && f32(cur_t) < plo.x) {
            let target_el = u32(plo.y);
            c = vec4<u32>(target_el, c.y & 0xFFFF0000u, 0u, 0u);
            c = tp_set_temp(c, cur_t);
        }
        // Try condense.
        else if (plo.z > -1000.0 && f32(cur_t) < plo.z) {
            let target_el = u32(plo.w);
            c = vec4<u32>(target_el, c.y & 0xFFFF0000u, 0u, 0u);
            c = tp_set_temp(c, cur_t);
        }
        // Try melt.
        else if (phi.x > -1000.0 && f32(cur_t) > phi.x) {
            let target_el = u32(phi.y);
            c = vec4<u32>(target_el, c.y & 0xFFFF0000u, 0u, 0u);
            c = tp_set_temp(c, cur_t);
        }
        // Try boil.
        else if (phi.z > -1000.0 && f32(cur_t) > phi.z) {
            let target_el = u32(phi.w);
            c = vec4<u32>(target_el, c.y & 0xFFFF0000u, 0u, 0u);
            // CPU adds 15° gas overshoot.
            c = tp_set_temp(c, i32(phi.z) + 15);
        }
    }

    cells[i] = c;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ThermalPostUniforms {
    width: u32,
    height: u32,
    frame: u32,
    _pad: u32,
}

/// GPU port of the per-cell core of `World::thermal_post`: combustion
/// ignition, burn-out tick, and phase changes (freeze/melt/boil/condense).
/// Latent heat with neighbor temp transfer and bespoke gunpowder/Cs
/// detonations are NOT yet on GPU — those need additional infrastructure
/// (Margolus or shockwave queues). CPU thermal_post still handles those
/// pieces when run.
struct ThermalPostCtx {
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    burn_data_buf: wgpu::Buffer,
    #[allow(dead_code)]
    phase_lo_buf: wgpu::Buffer,
    #[allow(dead_code)]
    phase_hi_buf: wgpu::Buffer,
    #[allow(dead_code)]
    burn_decay_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ThermalPostCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let uniforms = ThermalPostUniforms {
            width: W as u32, height: H as u32, frame: 0, _pad: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-tpost-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mk_lut_f32 = |label: &str, f: fn(u8) -> [f32; 4]| {
            let mut data: Vec<[f32; 4]> = vec![[0.0; 4]; 96];
            for el_id in 0u32..96u32 {
                data[el_id as usize] = f(el_id as u8);
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let burn_data_buf = mk_lut_f32("alembic-tpost-burn", crate::thermal_burn_props);
        let phase_lo_buf  = mk_lut_f32("alembic-tpost-phase-lo", crate::thermal_phase_lo_props);
        let phase_hi_buf  = mk_lut_f32("alembic-tpost-phase-hi", crate::thermal_phase_hi_props);
        // Burn decay: 96 u32 packed in 24 vec4<u32>.
        let mut decay: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0u32..96u32 {
            let p = crate::burn_decay_props(el_id as u8);
            decay[(el_id / 4) as usize][(el_id % 4) as usize] = p[0];
        }
        let burn_decay_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-tpost-burn-decay"),
            contents: bytemuck::cast_slice(&decay),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-tpost-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-tpost-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: burn_data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: phase_lo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: phase_hi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: burn_decay_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-tpost-shader"),
            source: wgpu::ShaderSource::Wgsl(THERMAL_POST_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-tpost-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-tpost-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        ThermalPostCtx {
            pipeline,
            uniform_buf,
            burn_data_buf,
            phase_lo_buf,
            phase_hi_buf,
            burn_decay_buf,
            bind_group,
        }
    }

    fn update_frame(&self, queue: &wgpu::Queue, frame: u32) {
        // Update only `frame` field at offset 8 (after width, height).
        let arr = [frame];
        queue.write_buffer(&self.uniform_buf, 8, bytemuck::cast_slice(&arr));
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-tpost-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

const FIRE_EMIT_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    parity: u32,            // 0 = even y emits, 1 = odd y emits
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;

const EL_EMPTY: u32 = 0u;
const EL_FIRE:  u32 = 5u;

fn fe_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn fe_burn(c: vec4<u32>) -> u32 { return (c.z >> 8u) & 0xFFu; }

fn fe_hash(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    return h;
}

// Fire emission from burning cells. CPU's thermal_post emits a Fire
// cell into the empty cell above with 1/10 probability while a cell
// is burning. Multi-cell write (we write to (x, y-1)), so split into
// even-y vs odd-y sub-passes: a cell at row Y writes to Y-1, which
// is the opposite parity, and the opposite-parity row isn't being
// processed in this sub-pass — race-free.
//
// Fire cell template matches Cell::new(Element::Fire): life ≈ 60,
// temp = 20 (initial, will be heated by thermal_diffuse next frame),
// other fields zero.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    if (y == 0u) { return; }
    if ((y & 1u) != u.parity) { return; }
    let i = y * u.width + x;
    let c = cells[i];
    if (fe_burn(c) == 0u) { return; }
    let i_above = (y - 1u) * u.width + x;
    let above = cells[i_above];
    if (fe_el(above) != EL_EMPTY) { return; }
    let r = fe_hash(i, u.frame);
    if ((r % 10u) != 0u) { return; }
    // Build a Fire cell. life = 60 (mid-range of CPU's 40..80 random),
    // temp = 20.
    let pack0 = 5u | (60u << 16u);
    let pack1 = 20u << 16u;
    let pack2 = 0u;
    let pack3 = 0u;
    cells[i_above] = vec4<u32>(pack0, pack1, pack2, pack3);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FireEmitUniforms {
    width: u32,
    height: u32,
    parity: u32,
    frame: u32,
}

/// GPU port of the "emit Fire above burning cell" branch of CPU
/// thermal_post. Two parity sub-passes (even-y rows, odd-y rows)
/// keep multi-cell writes race-free: row Y writes to row Y-1, which
/// is opposite parity and not being processed in the same sub-pass.
struct FireEmitCtx {
    pipeline: wgpu::ComputePipeline,
    pass_uniform_bufs: [wgpu::Buffer; 2],
    pass_bind_groups: [wgpu::BindGroup; 2],
}

impl FireEmitCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let mk_uniform = |label: &str, parity: u32| {
            let u = FireEmitUniforms { width: W as u32, height: H as u32, parity, frame: 0 };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pass_uniform_bufs: [wgpu::Buffer; 2] = [
            mk_uniform("alembic-fireemit-uniforms-even", 0),
            mk_uniform("alembic-fireemit-uniforms-odd", 1),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-fireemit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-fireemit-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 2] = [mk_bind(0), mk_bind(1)];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-fireemit-shader"),
            source: wgpu::ShaderSource::Wgsl(FIRE_EMIT_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-fireemit-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-fireemit-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        FireEmitCtx { pipeline, pass_uniform_bufs, pass_bind_groups }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32) {
        let arr = [frame];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        for i in 0..2 {
            queue.write_buffer(&self.pass_uniform_bufs[i], 12, bytes);
        }
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        for i in 0..2 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-fireemit-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.pass_bind_groups[i], &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const SOLUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0..3 = dissolve phase, 4..7 = diffuse phase
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;

const EL_EMPTY: u32 = 0u;
const EL_WATER: u32 = 2u;
const EL_SALT:  u32 = 40u;

const FLAG_FROZEN: u32 = 0x02u;

fn s_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn s_derived_id(c: vec4<u32>) -> u32 { return (c.x >> 8u) & 0xFFu; }
fn s_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn s_frozen(c: vec4<u32>) -> bool { return (s_flag(c) & FLAG_FROZEN) != 0u; }
fn s_solute_el(c: vec4<u32>) -> u32 { return c.w & 0xFFu; }
fn s_solute_amt(c: vec4<u32>) -> u32 { return (c.w >> 8u) & 0xFFu; }
fn s_solute_did(c: vec4<u32>) -> u32 { return (c.w >> 16u) & 0xFFu; }

fn s_set_solute(c: vec4<u32>, el: u32, amt: u32, did: u32) -> vec4<u32> {
    let pad = c.w & 0xFF000000u;
    let w = pad | (el & 0xFFu) | ((amt & 0xFFu) << 8u) | ((did & 0xFFu) << 16u);
    return vec4<u32>(c.x, c.y, c.z, w);
}

fn s_hash(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    return h;
}

const ABSORB_THRESHOLD: u32 = 192u;
const DIFFUSE_MAX: u32 = 24u;

// Faithful port of `World::dissolve` and `World::diffuse_solute`,
// Margolus-2x2 4-phase encoded:
//
//   pass_id  0..3: dissolve, 4 phase alignments. Within each 2x2
//   block, scan for a Water cell with solute_amt < ABSORB_THRESHOLD
//   adjacent to a Salt cell; consume the Salt (becomes Empty) and
//   set the water's solute_amt = 255.
//
//   pass_id  4..7: diffuse_solute, 4 phase alignments. Within each
//   2x2 block, find Water-Water pair with matching solute element
//   and a concentration gap > 1; transfer half the gap (capped at
//   DIFFUSE_MAX).
//
// Margolus 2x2 keeps writes block-local and race-free per phase.
// Note: the CPU code's Derived-soluble-salt path (FeCl, KCl, …) is
// NOT yet ported because the Derived registry is dynamic. Salt-only
// for now; CPU still handles Derived dissolution when active.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let phase = u.pass_id & 3u;
    let off_x = phase & 1u;
    let off_y = (phase >> 1u) & 1u;
    let bx = gid.x * 2u + off_x;
    let by = gid.y * 2u + off_y;
    if (bx + 1u >= u.width || by + 1u >= u.height) { return; }

    let i00 = by * u.width + bx;
    let i10 = by * u.width + bx + 1u;
    let i01 = (by + 1u) * u.width + bx;
    let i11 = (by + 1u) * u.width + bx + 1u;
    var c00 = cells[i00];
    var c10 = cells[i10];
    var c01 = cells[i01];
    var c11 = cells[i11];

    let r = s_hash(by * u.width + bx, u.frame);

    if (u.pass_id < 4u) {
        // ---- dissolve ----
        // 4 cells × 4 neighbor pairs per block. Try each Water → Salt
        // pair; on first match within probability gate, consume Salt.
        // 20% probability per attempt.
        if ((r & 0xFFu) > 51u) { return; } // ~20%

        // Helper to attempt dissolve(water, salt) → mark both updated.
        // Returns updated cells via early exits.
        // c00 ↔ c10 (horizontal, top row)
        if (s_el(c00) == EL_WATER && s_solute_amt(c00) < ABSORB_THRESHOLD
            && s_el(c10) == EL_SALT && !s_frozen(c10)) {
            c00 = s_set_solute(c00, EL_SALT, 255u, 0u);
            c10 = vec4<u32>(0u, 0u, 0u, 0u);
        } else if (s_el(c10) == EL_WATER && s_solute_amt(c10) < ABSORB_THRESHOLD
            && s_el(c00) == EL_SALT && !s_frozen(c00)) {
            c10 = s_set_solute(c10, EL_SALT, 255u, 0u);
            c00 = vec4<u32>(0u, 0u, 0u, 0u);
        }
        // c01 ↔ c11 (horizontal, bottom row)
        else if (s_el(c01) == EL_WATER && s_solute_amt(c01) < ABSORB_THRESHOLD
            && s_el(c11) == EL_SALT && !s_frozen(c11)) {
            c01 = s_set_solute(c01, EL_SALT, 255u, 0u);
            c11 = vec4<u32>(0u, 0u, 0u, 0u);
        } else if (s_el(c11) == EL_WATER && s_solute_amt(c11) < ABSORB_THRESHOLD
            && s_el(c01) == EL_SALT && !s_frozen(c01)) {
            c11 = s_set_solute(c11, EL_SALT, 255u, 0u);
            c01 = vec4<u32>(0u, 0u, 0u, 0u);
        }
        // c00 ↔ c01 (vertical, left col)
        else if (s_el(c00) == EL_WATER && s_solute_amt(c00) < ABSORB_THRESHOLD
            && s_el(c01) == EL_SALT && !s_frozen(c01)) {
            c00 = s_set_solute(c00, EL_SALT, 255u, 0u);
            c01 = vec4<u32>(0u, 0u, 0u, 0u);
        } else if (s_el(c01) == EL_WATER && s_solute_amt(c01) < ABSORB_THRESHOLD
            && s_el(c00) == EL_SALT && !s_frozen(c00)) {
            c01 = s_set_solute(c01, EL_SALT, 255u, 0u);
            c00 = vec4<u32>(0u, 0u, 0u, 0u);
        }
        // c10 ↔ c11 (vertical, right col)
        else if (s_el(c10) == EL_WATER && s_solute_amt(c10) < ABSORB_THRESHOLD
            && s_el(c11) == EL_SALT && !s_frozen(c11)) {
            c10 = s_set_solute(c10, EL_SALT, 255u, 0u);
            c11 = vec4<u32>(0u, 0u, 0u, 0u);
        } else if (s_el(c11) == EL_WATER && s_solute_amt(c11) < ABSORB_THRESHOLD
            && s_el(c10) == EL_SALT && !s_frozen(c10)) {
            c11 = s_set_solute(c11, EL_SALT, 255u, 0u);
            c10 = vec4<u32>(0u, 0u, 0u, 0u);
        }
    } else {
        // ---- diffuse_solute ----
        // 35% probability per attempt.
        if ((r & 0xFFu) > 89u) { return; } // ~35%
        // For each pair, transfer solute from higher to lower if same element.
        // c00 ↔ c10
        if (s_el(c00) == EL_WATER && s_el(c10) == EL_WATER) {
            let s0 = s_solute_amt(c00);
            let s1 = s_solute_amt(c10);
            let same = (s_solute_el(c00) == s_solute_el(c10) || s0 == 0u || s1 == 0u)
                && (s_solute_did(c00) == s_solute_did(c10) || s0 == 0u || s1 == 0u);
            if (same && s0 > s1 && (s0 - s1) > 1u) {
                let gap = s0 - s1;
                var transfer = (gap / 2u);
                if (transfer > DIFFUSE_MAX) { transfer = DIFFUSE_MAX; }
                if (transfer < 1u) { transfer = 1u; }
                let donor_el = s_solute_el(c00);
                let donor_did = s_solute_did(c00);
                let new_s0 = s0 - transfer;
                let new_s1 = min(s1 + transfer, 255u);
                let final_el0 = select(donor_el, 0u, new_s0 == 0u);
                let final_did0 = select(donor_did, 0u, new_s0 == 0u);
                c00 = s_set_solute(c00, final_el0, new_s0, final_did0);
                c10 = s_set_solute(c10, donor_el, new_s1, donor_did);
            } else if (same && s1 > s0 && (s1 - s0) > 1u) {
                let gap = s1 - s0;
                var transfer = (gap / 2u);
                if (transfer > DIFFUSE_MAX) { transfer = DIFFUSE_MAX; }
                if (transfer < 1u) { transfer = 1u; }
                let donor_el = s_solute_el(c10);
                let donor_did = s_solute_did(c10);
                let new_s1 = s1 - transfer;
                let new_s0 = min(s0 + transfer, 255u);
                let final_el1 = select(donor_el, 0u, new_s1 == 0u);
                let final_did1 = select(donor_did, 0u, new_s1 == 0u);
                c10 = s_set_solute(c10, final_el1, new_s1, final_did1);
                c00 = s_set_solute(c00, donor_el, new_s0, donor_did);
            }
        }
        // c01 ↔ c11 (analogous)
        if (s_el(c01) == EL_WATER && s_el(c11) == EL_WATER) {
            let s0 = s_solute_amt(c01);
            let s1 = s_solute_amt(c11);
            let same = (s_solute_el(c01) == s_solute_el(c11) || s0 == 0u || s1 == 0u)
                && (s_solute_did(c01) == s_solute_did(c11) || s0 == 0u || s1 == 0u);
            if (same && s0 > s1 && (s0 - s1) > 1u) {
                let gap = s0 - s1;
                var transfer = (gap / 2u);
                if (transfer > DIFFUSE_MAX) { transfer = DIFFUSE_MAX; }
                if (transfer < 1u) { transfer = 1u; }
                let donor_el = s_solute_el(c01);
                let donor_did = s_solute_did(c01);
                let new_s0 = s0 - transfer;
                let new_s1 = min(s1 + transfer, 255u);
                let final_el0 = select(donor_el, 0u, new_s0 == 0u);
                let final_did0 = select(donor_did, 0u, new_s0 == 0u);
                c01 = s_set_solute(c01, final_el0, new_s0, final_did0);
                c11 = s_set_solute(c11, donor_el, new_s1, donor_did);
            } else if (same && s1 > s0 && (s1 - s0) > 1u) {
                let gap = s1 - s0;
                var transfer = (gap / 2u);
                if (transfer > DIFFUSE_MAX) { transfer = DIFFUSE_MAX; }
                if (transfer < 1u) { transfer = 1u; }
                let donor_el = s_solute_el(c11);
                let donor_did = s_solute_did(c11);
                let new_s1 = s1 - transfer;
                let new_s0 = min(s0 + transfer, 255u);
                let final_el1 = select(donor_el, 0u, new_s1 == 0u);
                let final_did1 = select(donor_did, 0u, new_s1 == 0u);
                c11 = s_set_solute(c11, final_el1, new_s1, final_did1);
                c01 = s_set_solute(c01, donor_el, new_s0, donor_did);
            }
        }
        // c00 ↔ c01 (vertical pairs handled by next phase alignment)
        // Skipping vertical here — phase rotation covers all pairs.
    }

    cells[i00] = c00;
    cells[i10] = c10;
    cells[i01] = c01;
    cells[i11] = c11;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SoluteUniforms {
    width: u32,
    height: u32,
    pass_id: u32,
    frame: u32,
}

/// GPU port of `World::dissolve` + `World::diffuse_solute`. Both are
/// neighbor-pair operations on Water cells, ported via Margolus 2x2
/// 4-phase encoding (race-free per phase). 8 dispatches total per
/// frame: 4 dissolve phases + 4 diffuse phases. The Derived-soluble
/// salt branch isn't yet ported; only Element::Salt is recognized.
struct SoluteCtx {
    pipeline: wgpu::ComputePipeline,
    pass_uniform_bufs: [wgpu::Buffer; 8],
    pass_bind_groups: [wgpu::BindGroup; 8],
}

impl SoluteCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let mk_uniform = |label: &str, pass_id: u32| {
            let u = SoluteUniforms { width: W as u32, height: H as u32, pass_id, frame: 0 };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pass_uniform_bufs: [wgpu::Buffer; 8] = [
            mk_uniform("alembic-solute-u-dis0", 0),
            mk_uniform("alembic-solute-u-dis1", 1),
            mk_uniform("alembic-solute-u-dis2", 2),
            mk_uniform("alembic-solute-u-dis3", 3),
            mk_uniform("alembic-solute-u-dif0", 4),
            mk_uniform("alembic-solute-u-dif1", 5),
            mk_uniform("alembic-solute-u-dif2", 6),
            mk_uniform("alembic-solute-u-dif3", 7),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-solute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-solute-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 8] = [
            mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3),
            mk_bind(4), mk_bind(5), mk_bind(6), mk_bind(7),
        ];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-solute-shader"),
            source: wgpu::ShaderSource::Wgsl(SOLUTE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-solute-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-solute-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        SoluteCtx { pipeline, pass_uniform_bufs, pass_bind_groups }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32) {
        let arr = [frame];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        for i in 0..8 {
            queue.write_buffer(&self.pass_uniform_bufs[i], 12, bytes);
        }
        let blocks_x = (W as u32 + 1) / 2;
        let blocks_y = (H as u32 + 1) / 2;
        let wg_x = (blocks_x + 7) / 8;
        let wg_y = (blocks_y + 7) / 8;
        for i in 0..8 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-solute-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.pass_bind_groups[i], &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const WATER_SAND_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0..3 = Margolus phase
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;

const EL_EMPTY: u32 = 0u;
const EL_SAND:  u32 = 1u;
const EL_WATER: u32 = 2u;
const EL_MUD:   u32 = 11u;

const FLAG_FROZEN: u32 = 0x02u;

fn ws_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn ws_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn ws_frozen(c: vec4<u32>) -> bool { return (ws_flag(c) & FLAG_FROZEN) != 0u; }

fn ws_hash(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    return h;
}

// Build an Empty / Mud cell from scratch — discards solute, derived,
// moisture, etc., matching CPU `Cell::EMPTY` / `Cell::new(Element::Mud)`.
fn make_empty() -> vec4<u32> { return vec4<u32>(0u, 0u, 0u, 0u); }
fn make_mud() -> vec4<u32> {
    // el = 11; temp at byte 6-7 of c.y → packed in c.y high 16 bits.
    // Cell::new uses temp=20.
    return vec4<u32>(EL_MUD, 20u << 16u, 0u, 0u);
}

// Faithful port of `World::reactions` (moisture chemistry only):
//
//   * water above sand  → 1/200 chance: water → empty, sand → mud
//                          (CPU's "water-pass" with mud column = 0)
//   * sand with water as a cardinal neighbor (N/E/W) → 1/60 chance:
//                          sand → mud, water → empty
//
// Margolus 2x2 4-phase — within a phase, the four cells of a block
// are written by a single thread, race-free. The mud column
// percolation (water dripping through a mud layer to soak sand below)
// is not preserved; that requires a sequential downward walk and is a
// rare, slow effect not visible per-frame.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let phase = u.pass_id & 3u;
    let off_x = phase & 1u;
    let off_y = (phase >> 1u) & 1u;
    let bx = gid.x * 2u + off_x;
    let by = gid.y * 2u + off_y;
    if (bx + 1u >= u.width || by + 1u >= u.height) { return; }

    let i00 = by * u.width + bx;
    let i10 = by * u.width + bx + 1u;
    let i01 = (by + 1u) * u.width + bx;
    let i11 = (by + 1u) * u.width + bx + 1u;
    var c00 = cells[i00];
    var c10 = cells[i10];
    var c01 = cells[i01];
    var c11 = cells[i11];

    let r0 = ws_hash(by * u.width + bx, u.frame);
    let r1 = ws_hash(by * u.width + bx + 7u, u.frame ^ 0xA5A5A5A5u);
    let r2 = ws_hash(by * u.width + bx + 13u, u.frame ^ 0x5A5A5A5Au);
    let r3 = ws_hash(by * u.width + bx + 23u, u.frame ^ 0xC3C3C3C3u);

    // Vertical pairs — water above sand. Combined CPU rate ≈ 1/45
    // (water-pass 1/200 ∪ sand-pass-N 1/60). Use 6/256 ≈ 1/43.
    // c00 (top) / c01 (bottom)
    if (ws_el(c00) == EL_WATER && ws_el(c01) == EL_SAND
        && !ws_frozen(c00) && !ws_frozen(c01)
        && (r0 & 0xFFu) < 6u) {
        c00 = make_empty();
        c01 = make_mud();
    } else if (ws_el(c00) == EL_SAND && ws_el(c01) == EL_WATER
        && !ws_frozen(c00) && !ws_frozen(c01)
        && (r0 & 0xFFu) < 4u) {
        // Sand above water — CPU sand-pass checks N (above) for water,
        // but here water is BELOW sand. CPU sand only checks N/E/W,
        // not S. So sand-above-water is NOT a reaction. Skip.
    }
    // c10 (top) / c11 (bottom)
    if (ws_el(c10) == EL_WATER && ws_el(c11) == EL_SAND
        && !ws_frozen(c10) && !ws_frozen(c11)
        && (r1 & 0xFFu) < 6u) {
        c10 = make_empty();
        c11 = make_mud();
    }

    // Horizontal pairs — sand absorbs water at sides. Rate 1/60 →
    // 4/256 ≈ 1/64.
    // c00 (left) / c10 (right)
    if (ws_el(c00) == EL_SAND && ws_el(c10) == EL_WATER
        && !ws_frozen(c00) && !ws_frozen(c10)
        && (r2 & 0xFFu) < 4u) {
        c00 = make_mud();
        c10 = make_empty();
    } else if (ws_el(c10) == EL_SAND && ws_el(c00) == EL_WATER
        && !ws_frozen(c10) && !ws_frozen(c00)
        && (r2 & 0xFFu) < 4u) {
        c10 = make_mud();
        c00 = make_empty();
    }
    // c01 (left) / c11 (right)
    if (ws_el(c01) == EL_SAND && ws_el(c11) == EL_WATER
        && !ws_frozen(c01) && !ws_frozen(c11)
        && (r3 & 0xFFu) < 4u) {
        c01 = make_mud();
        c11 = make_empty();
    } else if (ws_el(c11) == EL_SAND && ws_el(c01) == EL_WATER
        && !ws_frozen(c11) && !ws_frozen(c01)
        && (r3 & 0xFFu) < 4u) {
        c11 = make_mud();
        c01 = make_empty();
    }

    cells[i00] = c00;
    cells[i10] = c10;
    cells[i01] = c01;
    cells[i11] = c11;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterSandUniforms {
    width: u32,
    height: u32,
    pass_id: u32,
    frame: u32,
}

/// GPU port of `World::reactions` — Water+Sand→Mud moisture chemistry.
/// Margolus 2x2 4-phase, race-free per phase. Ports the local-pair
/// behavior; the mud-column percolation (water dripping through 1-30
/// mud cells to soak sand at the bottom) is lost — those conversions
/// were rare (1/200 per frame per cell) and barely visible.
struct WaterSandCtx {
    pipeline: wgpu::ComputePipeline,
    pass_uniform_bufs: [wgpu::Buffer; 4],
    pass_bind_groups: [wgpu::BindGroup; 4],
}

impl WaterSandCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let mk_uniform = |label: &str, pass_id: u32| {
            let u = WaterSandUniforms { width: W as u32, height: H as u32, pass_id, frame: 0 };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pass_uniform_bufs: [wgpu::Buffer; 4] = [
            mk_uniform("alembic-watersand-u-0", 0),
            mk_uniform("alembic-watersand-u-1", 1),
            mk_uniform("alembic-watersand-u-2", 2),
            mk_uniform("alembic-watersand-u-3", 3),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-watersand-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-watersand-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 4] = [mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3)];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-watersand-shader"),
            source: wgpu::ShaderSource::Wgsl(WATER_SAND_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-watersand-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-watersand-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        WaterSandCtx { pipeline, pass_uniform_bufs, pass_bind_groups }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32) {
        let arr = [frame];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        for i in 0..4 {
            queue.write_buffer(&self.pass_uniform_bufs[i], 12, bytes);
        }
        let blocks_x = (W as u32 + 1) / 2;
        let blocks_y = (H as u32 + 1) / 2;
        let wg_x = (blocks_x + 7) / 8;
        let wg_y = (blocks_y + 7) / 8;
        for i in 0..4 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-watersand-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.pass_bind_groups[i], &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const GLASS_ETCH_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0..3 = Margolus phase
    frame: u32,
    sif_derived_id: u32,     // pre-registered SiF compound id
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;

const EL_O:           u32 = 22u;
const EL_F:           u32 = 23u;
const EL_GLASS:       u32 = 16u;
const EL_MOLTENGLASS: u32 = 15u;
const EL_DERIVED:     u32 = 41u;

const FLAG_FROZEN: u32 = 0x02u;

fn ge_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn ge_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn ge_frozen(c: vec4<u32>) -> bool { return (ge_flag(c) & FLAG_FROZEN) != 0u; }
fn ge_temp(c: vec4<u32>) -> i32 {
    let raw = (c.y >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}

fn ge_hash(a: u32, b: u32) -> u32 {
    var h: u32 = a * 2654435761u;
    h ^= b * 1597334677u;
    h ^= h >> 16u;
    h *= 2246822519u;
    h ^= h >> 13u;
    return h;
}

// Build a Cell with given el, derived_id, temp; clears all other fields
// (matches `Cell::new(el)` then `derived_id=`/`temp=` overrides).
fn make_cell(el: u32, did: u32, temp: i32) -> vec4<u32> {
    let clamped = clamp(temp, -273, 5000);
    let traw = u32(clamped) & 0xFFFFu;
    let x = (el & 0xFFu) | ((did & 0xFFu) << 8u);
    let y = traw << 16u;
    return vec4<u32>(x, y, 0u, 0u);
}

// Faithful port of `World::glass_etching`:
//
//   F + Glass/MoltenGlass  →  SiF (Derived) + O
//
//   Rate 0.20 / cell-pair (frozen Glass: 0.02× → 0.004 effective).
//   Both products gain +800°C exotherm.
//
// Margolus 2x2 4-phase makes the multi-cell write race-free per phase.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let phase = u.pass_id & 3u;
    let off_x = phase & 1u;
    let off_y = (phase >> 1u) & 1u;
    let bx = gid.x * 2u + off_x;
    let by = gid.y * 2u + off_y;
    if (bx + 1u >= u.width || by + 1u >= u.height) { return; }

    let i00 = by * u.width + bx;
    let i10 = by * u.width + bx + 1u;
    let i01 = (by + 1u) * u.width + bx;
    let i11 = (by + 1u) * u.width + bx + 1u;
    var c00 = cells[i00];
    var c10 = cells[i10];
    var c01 = cells[i01];
    var c11 = cells[i11];

    let r00 = ge_hash(by * u.width + bx, u.frame);
    let r10 = ge_hash(by * u.width + bx + 7u, u.frame ^ 0xA5A5A5A5u);
    let r01 = ge_hash(by * u.width + bx + 13u, u.frame ^ 0x5A5A5A5Au);
    let r11 = ge_hash(by * u.width + bx + 23u, u.frame ^ 0xC3C3C3C3u);

    let sif = u.sif_derived_id;

    // Rate 0.20 → 51/256. Frozen Glass: 0.004 → 1/256.
    // Try every (F, Glass-or-MoltenGlass) ordered neighbor pair within
    // the 2x2 block and apply with the correct rate. Returns after the
    // first conversion in the block (matches CPU's `break` on first hit
    // per F cell).
    var done = false;

    // Helper macro is awkward in WGSL; inline pairs below.
    // 4 cell positions × 2 horiz/vert neighbors = 4 unique unordered
    // pairs in the block: (00,10), (01,11), (00,01), (10,11).

    // Pair c00 ↔ c10
    if (!done) {
        if (ge_el(c00) == EL_F
            && (ge_el(c10) == EL_GLASS || ge_el(c10) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c10);
            let thresh = select(51u, 1u, frozen);
            if ((r00 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c10);
                let t_f = ge_temp(c00);
                c10 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c00 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        } else if (ge_el(c10) == EL_F
            && (ge_el(c00) == EL_GLASS || ge_el(c00) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c00);
            let thresh = select(51u, 1u, frozen);
            if ((r00 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c00);
                let t_f = ge_temp(c10);
                c00 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c10 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        }
    }
    // Pair c01 ↔ c11
    if (!done) {
        if (ge_el(c01) == EL_F
            && (ge_el(c11) == EL_GLASS || ge_el(c11) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c11);
            let thresh = select(51u, 1u, frozen);
            if ((r10 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c11);
                let t_f = ge_temp(c01);
                c11 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c01 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        } else if (ge_el(c11) == EL_F
            && (ge_el(c01) == EL_GLASS || ge_el(c01) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c01);
            let thresh = select(51u, 1u, frozen);
            if ((r10 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c01);
                let t_f = ge_temp(c11);
                c01 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c11 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        }
    }
    // Pair c00 ↔ c01 (vertical, left)
    if (!done) {
        if (ge_el(c00) == EL_F
            && (ge_el(c01) == EL_GLASS || ge_el(c01) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c01);
            let thresh = select(51u, 1u, frozen);
            if ((r01 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c01);
                let t_f = ge_temp(c00);
                c01 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c00 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        } else if (ge_el(c01) == EL_F
            && (ge_el(c00) == EL_GLASS || ge_el(c00) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c00);
            let thresh = select(51u, 1u, frozen);
            if ((r01 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c00);
                let t_f = ge_temp(c01);
                c00 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c01 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        }
    }
    // Pair c10 ↔ c11 (vertical, right)
    if (!done) {
        if (ge_el(c10) == EL_F
            && (ge_el(c11) == EL_GLASS || ge_el(c11) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c11);
            let thresh = select(51u, 1u, frozen);
            if ((r11 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c11);
                let t_f = ge_temp(c10);
                c11 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c10 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        } else if (ge_el(c11) == EL_F
            && (ge_el(c10) == EL_GLASS || ge_el(c10) == EL_MOLTENGLASS)) {
            let frozen = ge_frozen(c10);
            let thresh = select(51u, 1u, frozen);
            if ((r11 & 0xFFu) < thresh) {
                let t_glass = ge_temp(c10);
                let t_f = ge_temp(c11);
                c10 = make_cell(EL_DERIVED, sif, t_glass + 800);
                c11 = make_cell(EL_O, 0u, t_f + 800);
                done = true;
            }
        }
    }

    cells[i00] = c00;
    cells[i10] = c10;
    cells[i01] = c01;
    cells[i11] = c11;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlassEtchUniforms {
    width: u32,
    height: u32,
    pass_id: u32,
    frame: u32,
    sif_derived_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU port of `World::glass_etching`. Margolus 2x2 4-phase, race-free
/// per phase. Pre-registers the SiF derived compound at startup so
/// the shader can write its known derived_id directly. The reaction
/// rate (0.20, or 0.004 for frozen Glass) and exotherm (+800°C) match
/// the CPU implementation.
struct GlassEtchCtx {
    pipeline: wgpu::ComputePipeline,
    pass_uniform_bufs: [wgpu::Buffer; 4],
    pass_bind_groups: [wgpu::BindGroup; 4],
}

impl GlassEtchCtx {
    fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        cells_buf: &wgpu::Buffer,
        sif_derived_id: u32,
    ) -> Self {
        let mk_uniform = |label: &str, pass_id: u32| {
            let u = GlassEtchUniforms {
                width: W as u32, height: H as u32, pass_id, frame: 0,
                sif_derived_id, _pad0: 0, _pad1: 0, _pad2: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let pass_uniform_bufs: [wgpu::Buffer; 4] = [
            mk_uniform("alembic-glassetch-u-0", 0),
            mk_uniform("alembic-glassetch-u-1", 1),
            mk_uniform("alembic-glassetch-u-2", 2),
            mk_uniform("alembic-glassetch-u-3", 3),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-glassetch-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-glassetch-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 4] = [mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3)];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-glassetch-shader"),
            source: wgpu::ShaderSource::Wgsl(GLASS_ETCH_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-glassetch-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-glassetch-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        GlassEtchCtx { pipeline, pass_uniform_bufs, pass_bind_groups }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32) {
        let arr = [frame];
        let bytes: &[u8] = bytemuck::cast_slice(&arr);
        for i in 0..4 {
            queue.write_buffer(&self.pass_uniform_bufs[i], 12, bytes);
        }
        let blocks_x = (W as u32 + 1) / 2;
        let blocks_y = (H as u32 + 1) / 2;
        let wg_x = (blocks_x + 7) / 8;
        let wg_y = (blocks_y + 7) / 8;
        for i in 0..4 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-glassetch-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.pass_bind_groups[i], &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const TREE_SUPPORT_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    pass_id: u32,            // 0 = anchor pass, 1 = propagate iter
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> motion_props: array<vec4<f32>, 96>;

const EL_WOOD: u32 = 4u;
const KIND_SOLID: u32 = 1u;
const KIND_GRAVEL: u32 = 2u;
const KIND_POWDER: u32 = 3u;
const FLAG_FROZEN: u32 = 0x02u;

fn ts_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn ts_kind(c: vec4<u32>) -> u32 { return u32(motion_props[ts_el(c)].x); }
fn ts_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn ts_frozen(c: vec4<u32>) -> bool { return (ts_flag(c) & FLAG_FROZEN) != 0u; }
fn ts_life(c: vec4<u32>) -> u32 { return (c.x >> 16u) & 0xFFFFu; }
fn ts_set_life(c: vec4<u32>, life: u32) -> vec4<u32> {
    return vec4<u32>((c.x & 0xFFFFu) | (life << 16u), c.y, c.z, c.w);
}

// Faithful port of `World::tree_support_check`. Two pass modes:
//
// pass 0 (anchor): For every Wood cell, decide whether it's
//   directly grounded:
//     * frozen wood → always anchor
//     * y == H-1     → on floor → anchor
//     * cell directly below is non-Wood Solid/Gravel/Powder → anchor
//   Anchors get life=0, all other wood cells get life=1.
//
// pass 1 (propagate, run K times): For every Wood cell with life=1,
//   if any 4-neighbor is Wood with life=0 (supported), inherit
//   life=0. After K iterations, all wood connected to an anchor
//   has been marked supported.
//
// life=1 cells fall via the wood-fall branch in vertical_fall.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let i = y * u.width + x;
    let c = cells[i];
    if (ts_el(c) != EL_WOOD) { return; }

    if (u.pass_id == 0u) {
        // Anchor pass.
        var anchor = false;
        if (ts_frozen(c)) {
            anchor = true;
        } else if (y == u.height - 1u) {
            anchor = true;
        } else {
            let i_below = (y + 1u) * u.width + x;
            let cb = cells[i_below];
            let kb = ts_kind(cb);
            let eb = ts_el(cb);
            if (eb != EL_WOOD && (kb == KIND_SOLID || kb == KIND_GRAVEL || kb == KIND_POWDER)) {
                anchor = true;
            }
        }
        if (anchor) {
            cells[i] = ts_set_life(c, 0u);
        } else {
            cells[i] = ts_set_life(c, 1u);
        }
        return;
    }

    // pass 1: propagate. Only act on cells currently marked unsupported.
    if (ts_life(c) == 0u) { return; }
    // Check 4 neighbors for supported wood.
    var supported = false;
    if (x > 0u) {
        let n = cells[y * u.width + (x - 1u)];
        if (ts_el(n) == EL_WOOD && ts_life(n) == 0u) { supported = true; }
    }
    if (!supported && x + 1u < u.width) {
        let n = cells[y * u.width + (x + 1u)];
        if (ts_el(n) == EL_WOOD && ts_life(n) == 0u) { supported = true; }
    }
    if (!supported && y > 0u) {
        let n = cells[(y - 1u) * u.width + x];
        if (ts_el(n) == EL_WOOD && ts_life(n) == 0u) { supported = true; }
    }
    if (!supported && y + 1u < u.height) {
        let n = cells[(y + 1u) * u.width + x];
        if (ts_el(n) == EL_WOOD && ts_life(n) == 0u) { supported = true; }
    }
    if (supported) {
        cells[i] = ts_set_life(c, 0u);
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TreeSupportUniforms {
    width: u32,
    height: u32,
    pass_id: u32,
    _pad: u32,
}

/// GPU port of `World::tree_support_check`. Runs every 30 frames
/// (matches CPU cadence). Two pass modes share a pipeline; we
/// dispatch one anchor pass + many propagate iterations to push
/// "supported" flags outward through connected wood.
struct TreeSupportCtx {
    pipeline: wgpu::ComputePipeline,
    u_anchor: wgpu::Buffer,
    u_propagate: wgpu::Buffer,
    #[allow(dead_code)]
    motion_props_buf: wgpu::Buffer,
    bg_anchor: wgpu::BindGroup,
    bg_propagate: wgpu::BindGroup,
    iters: u32,
}

impl TreeSupportCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let mk_uniform = |label: &str, pass_id: u32| {
            let u = TreeSupportUniforms {
                width: W as u32,
                height: H as u32,
                pass_id,
                _pad: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let u_anchor = mk_uniform("alembic-treesup-uniforms-anchor", 0);
        let u_propagate = mk_uniform("alembic-treesup-uniforms-propagate", 1);

        let mut props_data: Vec<[f32; 4]> = vec![[0.0; 4]; 96];
        for i in 0..96 {
            props_data[i] = crate::motion_props(i as u8);
        }
        let motion_props_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-treesup-motion-props"),
            contents: bytemuck::cast_slice(&props_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-treesup-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bg_anchor = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-treesup-bind-anchor"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_anchor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: motion_props_buf.as_entire_binding() },
            ],
        });
        let bg_propagate = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-treesup-bind-propagate"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_propagate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: motion_props_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-treesup-shader"),
            source: wgpu::ShaderSource::Wgsl(TREE_SUPPORT_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-treesup-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-treesup-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        TreeSupportCtx {
            pipeline,
            u_anchor,
            u_propagate,
            motion_props_buf,
            bg_anchor,
            bg_propagate,
            iters: 50,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        // Anchor pass.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-treesup-anchor"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_anchor, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Propagation iterations.
        for _ in 0..self.iters {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-treesup-propagate"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bg_propagate, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        let _ = (&self.u_anchor, &self.u_propagate);
    }
}

const LIFECYCLE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
// Per-element lifecycle data (96 elements packed 4 per vec4<u32>):
//   each u32 packs: ephemeral(8) | decay_product_el(8) | preserve_state(8) | _(8)
@group(0) @binding(2) var<uniform> lifecycle: array<vec4<u32>, 24>;

fn lc_lookup(el_id: u32) -> u32 {
    let safe = min(el_id, 95u);
    return lifecycle[safe / 4u][safe % 4u];
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let i = y * u.width + x;
    let c = cells[i];
    let el = c.x & 0xFFu;
    let lc = lc_lookup(el);
    let ephemeral = (lc & 0xFFu) != 0u;
    if (!ephemeral) { return; }

    let life = (c.x >> 16u) & 0xFFFFu;
    if (life == 0u) {
        // Decay: replace this cell with its decay product. Bytes 1
        // (derived_id) and the rest of cell.w (solute_*) reset to
        // zero; preserve_state controls whether temp + pressure
        // carry over (Steam → Water keeps the boiling heat so the
        // condensation is visibly hot, not 0°C).
        let decay_el = (lc >> 8u) & 0xFFu;
        let preserve = ((lc >> 16u) & 0xFFu) != 0u;
        if (preserve) {
            // Keep temp (cells[i].y bits 16-31) and pressure
            // (cells[i].z bits 16-31). Reset moisture, burn, life,
            // derived_id, seed, flag, solute.
            let pack0 = decay_el; // el only, no derived_id, no life
            let pack1 = c.y & 0xFFFF0000u;            // keep temp, zero seed/flag
            let pack2 = c.z & 0xFFFF0000u;            // keep pressure, zero moisture/burn
            let pack3 = 0u;
            cells[i] = vec4<u32>(pack0, pack1, pack2, pack3);
        } else {
            // Clean transition: decay product as a fresh cell. Used
            // for Fire → Empty, where leftover heat would re-ignite
            // anything underneath.
            cells[i] = vec4<u32>(decay_el, 0u, 0u, 0u);
        }
    } else {
        // Tick down: life - 1, leave everything else alone.
        let new_life = life - 1u;
        cells[i].x = (c.x & 0xFFFFu) | (new_life << 16u);
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LifecycleUniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU lifecycle pass — element-data-driven life decrement and decay.
/// Generic across all elements: per-cell read, look up the element's
/// lifecycle row, tick or decay accordingly. Adding a new ephemeral
/// element is just a row in `lifecycle_props`.
struct LifecycleCtx {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    lifecycle_lut_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl LifecycleCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let uniforms = LifecycleUniforms { width: W as u32, height: H as u32, _pad0: 0, _pad1: 0 };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-lifecycle-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Pack lifecycle data: each element gets 1 u32 with ephemeral|decay|preserve.
        let mut lut: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0u32..96u32 {
            let p = crate::lifecycle_props(el_id as u8);
            let packed = (p[0] & 0xFFu32)
                | ((p[1] & 0xFFu32) << 8)
                | ((p[2] & 0xFFu32) << 16);
            lut[(el_id / 4) as usize][(el_id % 4) as usize] = packed;
        }
        let lifecycle_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-lifecycle-lut"),
            contents: bytemuck::cast_slice(&lut),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-lifecycle-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-lifecycle-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lifecycle_lut_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-lifecycle-shader"),
            source: wgpu::ShaderSource::Wgsl(LIFECYCLE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-lifecycle-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-lifecycle-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        LifecycleCtx { pipeline, uniform_buf, lifecycle_lut_buf, bind_group }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-lifecycle-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}



const CLEAR_FLAGS_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;

// FLAG_UPDATED = bit 0 of the flag byte. The flag byte sits in
// bits 8-15 of cells[i].y.
const FLAG_UPDATED_BIT_IN_Y: u32 = 0x100u;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let i = y * u.width + x;
    cells[i].y = cells[i].y & ~FLAG_UPDATED_BIT_IN_Y;
}
"#;

const COLOR_FIRES_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
// Per-element flame-color flag: 1 if `flame_color()` returns Some.
// 96 elements packed 4 per vec4<u32>.
@group(0) @binding(2) var<uniform> has_flame: array<vec4<u32>, 24>;

const EL_FIRE: u32 = 5u;
const EL_WATER: u32 = 2u;

fn cf_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cf_solute_el(c: vec4<u32>) -> u32 { return c.w & 0xFFu; }
fn cf_has_flame(el_id: u32) -> bool {
    let safe_id = min(el_id, 95u);
    return has_flame[safe_id / 4u][safe_id % 4u] != 0u;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    if (x >= w_i || y >= h_i) { return; }
    let i = u32(y * w_i + x);
    let c = cells[i];
    if (cf_el(c) != EL_FIRE) { return; }
    if (cf_solute_el(c) != 0u) { return; }

    var picked: u32 = 0u;
    // 8 neighbors, fully unrolled. Inlining avoids array<vec2,8>
    // literal in let bindings — some shader compilers spill those
    // awkwardly and previous attempt crashed on Vulkan/NV.
    // (1, 0)
    if (x + 1 < w_i) {
        let n = cells[u32(y * w_i + (x + 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (-1, 0)
    if (picked == 0u && x > 0) {
        let n = cells[u32(y * w_i + (x - 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (0, 1)
    if (picked == 0u && y + 1 < h_i) {
        let n = cells[u32((y + 1) * w_i + x)];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (0, -1)
    if (picked == 0u && y > 0) {
        let n = cells[u32((y - 1) * w_i + x)];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (1, 1)
    if (picked == 0u && x + 1 < w_i && y + 1 < h_i) {
        let n = cells[u32((y + 1) * w_i + (x + 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (1, -1)
    if (picked == 0u && x + 1 < w_i && y > 0) {
        let n = cells[u32((y - 1) * w_i + (x + 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (-1, 1)
    if (picked == 0u && x > 0 && y + 1 < h_i) {
        let n = cells[u32((y + 1) * w_i + (x - 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }
    // (-1, -1)
    if (picked == 0u && x > 0 && y > 0) {
        let n = cells[u32((y - 1) * w_i + (x - 1))];
        let nel = cf_el(n);
        if (cf_has_flame(nel)) { picked = nel; }
        else if (nel == EL_WATER) {
            let nse = cf_solute_el(n);
            if (cf_has_flame(nse)) { picked = nse; }
        }
    }

    if (picked != 0u) {
        let hi = c.w & 0xFFFFFF00u;
        cells[i].w = hi | (picked & 0xFFu);
    }
}
"#;

const FLAME_TEST_EMISSION_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    phase: u32,
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> cells: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> has_flame: array<vec4<u32>, 24>;

const EL_EMPTY: u32 = 0u;
const EL_FIRE:  u32 = 5u;

fn fte_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn fte_temp(c: vec4<u32>) -> i32 {
    let raw = (c.y >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}
fn fte_burn(c: vec4<u32>) -> u32 { return (c.z >> 8u) & 0xFFu; }
fn fte_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn fte_updated(c: vec4<u32>) -> bool { return (fte_flag(c) & 1u) != 0u; }
fn fte_has_flame(el: u32) -> bool {
    let safe_id = min(el, 95u);
    return has_flame[safe_id / 4u][safe_id % 4u] != 0u;
}

// Element::Fire baseline cell. Matches Cell::new(Element::Fire) with
// life fixed at 60 (CPU randomizes 40..80; midpoint is fine), seed 0.
fn make_fire(solute_el: u32) -> vec4<u32> {
    let pack0 = 5u | (60u << 16u);
    let pack1 = 20u << 16u;
    let pack2 = 0u;
    let pack3 = solute_el & 0xFFu;
    return vec4<u32>(pack0, pack1, pack2, pack3);
}

fn is_emitter(c: vec4<u32>) -> bool {
    return fte_el(c) != EL_FIRE
        && fte_burn(c) == 0u
        && !fte_updated(c)
        && fte_temp(c) > 600
        && fte_has_flame(fte_el(c));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let off_x = u.phase & 1u;
    let off_y = (u.phase >> 1u) & 1u;
    let bx = gid.x * 2u + off_x;
    let by = gid.y * 2u + off_y;
    if (bx + 1u >= u.width || by + 1u >= u.height) { return; }
    // Frame parity gate ≈ CPU's 0.40 probability — alternating
    // emit/no-emit averages 50% emission rate per cell-frame.
    if ((u.frame & 1u) != 0u) { return; }

    let i00 = by * u.width + bx;
    let i10 = by * u.width + bx + 1u;
    let i01 = (by + 1u) * u.width + bx;
    let i11 = (by + 1u) * u.width + bx + 1u;

    var c00 = cells[i00];
    var c10 = cells[i10];
    var c01 = cells[i01];
    var c11 = cells[i11];

    if (is_emitter(c00)) {
        let src = fte_el(c00);
        if (fte_el(c10) == EL_EMPTY) { c10 = make_fire(src); }
        else if (fte_el(c01) == EL_EMPTY) { c01 = make_fire(src); }
        else if (fte_el(c11) == EL_EMPTY) { c11 = make_fire(src); }
    }
    if (is_emitter(c10)) {
        let src = fte_el(c10);
        if (fte_el(c00) == EL_EMPTY) { c00 = make_fire(src); }
        else if (fte_el(c11) == EL_EMPTY) { c11 = make_fire(src); }
        else if (fte_el(c01) == EL_EMPTY) { c01 = make_fire(src); }
    }
    if (is_emitter(c01)) {
        let src = fte_el(c01);
        if (fte_el(c11) == EL_EMPTY) { c11 = make_fire(src); }
        else if (fte_el(c00) == EL_EMPTY) { c00 = make_fire(src); }
        else if (fte_el(c10) == EL_EMPTY) { c10 = make_fire(src); }
    }
    if (is_emitter(c11)) {
        let src = fte_el(c11);
        if (fte_el(c01) == EL_EMPTY) { c01 = make_fire(src); }
        else if (fte_el(c10) == EL_EMPTY) { c10 = make_fire(src); }
        else if (fte_el(c00) == EL_EMPTY) { c00 = make_fire(src); }
    }

    cells[i00] = c00;
    cells[i10] = c10;
    cells[i01] = c01;
    cells[i11] = c11;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ChemSimpleUniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FlameTestUniforms {
    width: u32,
    height: u32,
    phase: u32,
    frame: u32,
}

/// GPU port of `World::clear_flags`. Trivial per-cell bit clear of
/// the FLAG_UPDATED flag.
struct ClearFlagsCtx {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ClearFlagsCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let uniforms = ChemSimpleUniforms { width: W as u32, height: H as u32, _pad0: 0, _pad1: 0 };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-clearflags-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-clearflags-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-clearflags-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-clearflags-shader"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_FLAGS_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-clearflags-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-clearflags-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        ClearFlagsCtx { pipeline, uniform_buf, bind_group }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-clearflags-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

/// GPU port of `World::color_fires`.
struct ColorFiresCtx {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    flame_lut_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ColorFiresCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let uniforms = ChemSimpleUniforms { width: W as u32, height: H as u32, _pad0: 0, _pad1: 0 };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-colorfires-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut lut: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0u32..96u32 {
            let p = crate::flame_color_flag_props(el_id as u8);
            lut[(el_id / 4) as usize][(el_id % 4) as usize] = p[0];
        }
        let flame_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-colorfires-flame-lut"),
            contents: bytemuck::cast_slice(&lut),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-colorfires-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-colorfires-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flame_lut_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-colorfires-shader"),
            source: wgpu::ShaderSource::Wgsl(COLOR_FIRES_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-colorfires-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-colorfires-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        ColorFiresCtx { pipeline, uniform_buf, flame_lut_buf, bind_group }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-colorfires-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

/// GPU port of `World::flame_test_emission`. Margolus 2×2 4-phase.
struct FlameTestEmissionCtx {
    pipeline: wgpu::ComputePipeline,
    phase_uniform_bufs: [wgpu::Buffer; 4],
    #[allow(dead_code)]
    flame_lut_buf: wgpu::Buffer,
    phase_bind_groups: [wgpu::BindGroup; 4],
}

impl FlameTestEmissionCtx {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
        let mk_uniform = |label: &str, phase: u32| {
            let u = FlameTestUniforms { width: W as u32, height: H as u32, phase, frame: 0 };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[u]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let phase_uniform_bufs: [wgpu::Buffer; 4] = [
            mk_uniform("alembic-flameemit-uniforms-p0", 0),
            mk_uniform("alembic-flameemit-uniforms-p1", 1),
            mk_uniform("alembic-flameemit-uniforms-p2", 2),
            mk_uniform("alembic-flameemit-uniforms-p3", 3),
        ];
        let mut lut: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0u32..96u32 {
            let p = crate::flame_color_flag_props(el_id as u8);
            lut[(el_id / 4) as usize][(el_id % 4) as usize] = p[0];
        }
        let flame_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-flameemit-flame-lut"),
            contents: bytemuck::cast_slice(&lut),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-flameemit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-flameemit-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: phase_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flame_lut_buf.as_entire_binding() },
            ],
        });
        let phase_bind_groups: [wgpu::BindGroup; 4] = [mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3)];
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-flameemit-shader"),
            source: wgpu::ShaderSource::Wgsl(FLAME_TEST_EMISSION_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-flameemit-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-flameemit-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        FlameTestEmissionCtx { pipeline, phase_uniform_bufs, flame_lut_buf, phase_bind_groups }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32) {
        let frame_arr = [frame];
        let frame_bytes: &[u8] = bytemuck::cast_slice(&frame_arr);
        for i in 0..4 {
            queue.write_buffer(&self.phase_uniform_bufs[i], 12, frame_bytes);
        }
        let blocks_x = (W as u32 + 1) / 2;
        let blocks_y = (H as u32 + 1) / 2;
        let wg_x = (blocks_x + 7) / 8;
        let wg_y = (blocks_y + 7) / 8;
        for i in 0..4 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-flameemit-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.phase_bind_groups[i], &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
    }
}

const RENDER_COMPUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cells: array<vec4<u32>>;
// Per-element base colors: one packed RGBA8 (u32) per element id,
// 96 elements packed 4 per vec4<u32>.
@group(0) @binding(2) var<uniform> color_lut: array<vec4<u32>, 24>;
@group(0) @binding(3) var sim_tex: texture_storage_2d<rgba8unorm, write>;
// Derived compound color mirror — vec4<f32> per slot, RGBA in [0,1].
@group(0) @binding(4) var<uniform> derived_color: array<vec4<f32>, 256>;

const FLAG_FROZEN: u32 = 0x02u;
const PHASE_MASK:  u32 = 0x0Cu;
const PHASE_NATIVE: u32 = 0u;
const PHASE_SOLID:  u32 = 1u;
const PHASE_LIQUID: u32 = 2u;
const PHASE_GAS:    u32 = 3u;

const EL_EMPTY:   u32 = 0u;
const EL_WATER:   u32 = 2u;
const EL_FIRE:    u32 = 5u;
const EL_DERIVED: u32 = 41u;

fn cell_el_render(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cell_derived_id_render(c: vec4<u32>) -> u32 { return (c.x >> 8u) & 0xFFu; }
fn cell_seed_render(c: vec4<u32>) -> u32 { return c.y & 0xFFu; }
fn cell_flag_render(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn cell_moisture_render(c: vec4<u32>) -> u32 { return c.z & 0xFFu; }
fn cell_temp_render(c: vec4<u32>) -> i32 {
    let raw = (c.y >> 16u) & 0xFFFFu;
    return i32(raw) - i32(select(0u, 65536u, raw >= 32768u));
}
fn cell_phase(c: vec4<u32>) -> u32 {
    return (cell_flag_render(c) & PHASE_MASK) >> 2u;
}
fn cell_frozen_render(c: vec4<u32>) -> bool {
    return (cell_flag_render(c) & FLAG_FROZEN) != 0u;
}

fn unpack_rgb(packed: u32) -> vec3<f32> {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

fn lookup_color(el_id: u32) -> vec3<f32> {
    return unpack_rgb(color_lut[el_id / 4u][el_id % 4u]);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) { return; }
    let i = y * u.width + x;
    let c = cells[i];
    let el = cell_el_render(c);

    var color = lookup_color(el);
    if (el == EL_DERIVED) {
        color = derived_color[cell_derived_id_render(c)].rgb;
    }

    if (el == EL_EMPTY) {
        textureStore(sim_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
        return;
    }

    // Seed-driven brightness variation (small per-cell noise).
    let seed_i = i32(cell_seed_render(c));
    let v = f32((seed_i - 128) / 16) / 255.0;
    color = clamp(color + vec3<f32>(v, v, v), vec3<f32>(0.0), vec3<f32>(1.0));

    // Moisture darkening — wet powders/solids look darker and blue-tinged.
    let moisture = cell_moisture_render(c);
    if (moisture > 20u && el != EL_WATER) {
        let wet = clamp((f32(moisture) - 20.0) / 235.0, 0.0, 1.0) * 0.55;
        color.r = color.r * (1.0 - wet);
        color.g = color.g * (1.0 - wet * 0.9);
        color.b = color.b * (1.0 - wet * 0.6) + (70.0 / 255.0) * wet;
    }

    // Thermal glow — hot cells shift through red→orange→yellow→white.
    let temp = cell_temp_render(c);
    if (temp > 250 && el != EL_FIRE) {
        let warm_heat = clamp(f32(temp - 250) / 1500.0, 0.0, 1.0);
        let warm_mix = warm_heat * 0.8;
        color = mix(color, vec3<f32>(1.0, 200.0/255.0, 80.0/255.0), warm_mix);
        if (temp > 1750) {
            let white_t = clamp(f32(temp - 1750) / 1250.0, 0.0, 1.0);
            let white_mix = white_t * 0.9;
            color = mix(color, vec3<f32>(1.0, 1.0, 1.0), white_mix);
        }
    }

    // Frozen (rigid-body) cells get a cool brighten so locked walls
    // are visually distinct.
    if (cell_frozen_render(c)) {
        color.r = min(color.r + 20.0/255.0, 1.0);
        color.g = min(color.g + 20.0/255.0, 1.0);
        color.b = min(color.b + 30.0/255.0, 1.0);
    }

    // Phase tint — forced (non-native) phases shift toward cold,
    // hot, or washed-out depending on which way the phase changed.
    let phase = cell_phase(c);
    if (phase == PHASE_SOLID) {
        color = vec3<f32>(color.r * 0.7, color.g * 0.7, color.b * 0.9 + 20.0/255.0);
    } else if (phase == PHASE_LIQUID) {
        color = vec3<f32>(color.r * 0.9 + 30.0/255.0, color.g * 0.7 + 15.0/255.0, color.b * 0.5);
    } else if (phase == PHASE_GAS) {
        color = vec3<f32>(color.r * 0.5 + 8.0/255.0, color.g * 0.5 + 8.0/255.0, color.b * 0.5 + 12.0/255.0);
    }

    textureStore(sim_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RenderUniforms {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU compute pipeline that fills `sim_texture` from `cells_buf`.
/// Replaces the per-frame CPU pixel-fill loop + `queue.write_texture`
/// upload — now CPU does no per-cell work for rendering.
struct RenderComputeCtx {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    color_lut_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl RenderComputeCtx {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cells_buf: &wgpu::Buffer,
        sim_view: &wgpu::TextureView,
        derived_color_buf: &wgpu::Buffer,
    ) -> Self {
        let _ = queue;
        let uniforms = RenderUniforms {
            width: W as u32,
            height: H as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-render-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Color LUT — packed RGBA8 per element, 96 elements in 24 vec4<u32>.
        let mut color_lut: Vec<[u32; 4]> = vec![[0u32; 4]; 24];
        for el_id in 0u32..96u32 {
            let rgba = crate::base_color_props(el_id as u8);
            let packed = (rgba[0] as u32)
                | ((rgba[1] as u32) << 8)
                | ((rgba[2] as u32) << 16)
                | ((rgba[3] as u32) << 24);
            color_lut[(el_id / 4) as usize][(el_id % 4) as usize] = packed;
        }
        let color_lut_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-render-color-lut"),
            contents: bytemuck::cast_slice(&color_lut),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-render-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-render-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: color_lut_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sim_view) },
                wgpu::BindGroupEntry { binding: 4, resource: derived_color_buf.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-render-shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_COMPUTE_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-render-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("alembic-render-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        RenderComputeCtx {
            pipeline,
            uniform_buf,
            color_lut_buf,
            bind_group,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (W as u32 + 7) / 8;
        let wg_y = (H as u32 + 7) / 8;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-render-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

/// Per-frame uniform data for the sim display shader. Carries the
/// sim's rendered rectangle within the window (for proper letterbox
/// + aspect preservation) plus cursor + brush info (in screen pixels
/// so the brush outline is a true circle regardless of window aspect).
/// 8 × f32 = 32 bytes; wgpu's uniform 16-byte alignment is satisfied.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct DisplayUniform {
    /// Sim rectangle in framebuffer pixel coords. Pixels outside this
    /// rect render as background; pixels inside sample the sim texture.
    sim_min_x: f32,
    sim_min_y: f32,
    sim_max_x: f32,
    sim_max_y: f32,
    /// Cursor position in framebuffer pixels.
    cursor_x: f32,
    cursor_y: f32,
    /// Brush radius in screen pixels (= sim cells × pixels-per-cell).
    brush_pixel_radius: f32,
    /// Visibility flag — 1.0 when cursor is over the sim rect.
    visible: f32,
}

const WINDOW_TITLE: &str = "Alembic (wgpu)";
const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 1024;

/// GPU state — created once after the window is up. Holds the wgpu
/// device + queue, the swapchain configuration, and the textured-quad
/// pipeline that displays our sim's color buffer. Owns the live
/// `World` so the CPU sim ticks each frame; the result is uploaded to
/// `sim_texture` and sampled by the fragment shader.
///
/// As the migration progresses, the cell-state will move into a wgpu
/// storage buffer and the per-frame upload disappears.
struct GpuState {
    /// wgpu surface backed by the winit window. Re-created on
    /// every resize, but the Arc<Window> handle stays stable.
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// CPU sim. Ticks every frame; we read its cells out, run them
    /// through `color_rgb()`, and upload the result for display.
    world: World,
    /// CPU scratch — packed RGBA8 pixels for one frame, W*H*4 bytes.
    /// GPU texture written by the render compute. Bound to the fragment
    /// shader's sampler, drawn as a fullscreen quad.
    sim_texture: wgpu::Texture,
    sim_bind_group: wgpu::BindGroup,
    sim_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer carrying cursor pos + brush radius to the
    /// fragment shader so it can render a brush outline overlay.
    display_uniform: wgpu::Buffer,
    /// GPU-resident cell state. Source of truth for the simulation
    /// runtime. CPU's `world.cells` is a mirror synced once per frame
    /// at the start of render() and pushed back before dispatch.
    cells_buf: wgpu::Buffer,
    /// GPU mirror of the CPU `DERIVED_COMPOUNDS` registry. 256 slots,
    /// each holding (kind_id, density, viscosity, molar_mass) +
    /// (r,g,b,1) in 2 vec4<f32>. Synced at the start of each frame
    /// because compounds are registered dynamically by CPU chemistry.
    derived_phys_buf: wgpu::Buffer,
    derived_color_buf: wgpu::Buffer,
    /// CPU staging vectors for the derived sync — reused per frame.
    derived_phys_staging: Vec<[f32; 4]>,
    derived_color_staging: Vec<[f32; 4]>,
    /// GPU compute pipeline for pressure diffusion. Replaces the CPU
    /// `World::pressure()` pass; lets us scale the grid without paying
    /// the linear CPU cost.
    pressure_compute: PressureComputeCtx,
    /// GPU compute pipeline for thermal diffusion (heat exchange +
    /// ambient blend). Replaces `World::thermal_diffuse()`.
    thermal_compute: ThermalComputeCtx,
    /// GPU compute pipeline for hydrostatic + thermal pressure tgt
    /// (`World::pressure_sources`). Largest single CPU pass; column
    /// scan is GPU-friendly with one thread per column.
    pressure_sources_compute: PressureSourcesCtx,
    /// GPU compute pipeline for motion (Margolus 2x2 block updates).
    /// Replaces the per-cell `update_cell` sweep — saves ~6.7ms/frame
    /// and is a foundation for future chemistry compute shaders that
    /// also live on the packed `CellGpu` buffer.
    motion_compute: MotionComputeCtx,
    /// GPU compute pipeline that fills `sim_texture` from `cells_buf`.
    /// Replaces the per-frame CPU pixel fill + texture upload.
    render_compute: RenderComputeCtx,
    /// GPU port of `World::clear_flags`. Trivial single-bit clear.
    clear_flags_compute: ClearFlagsCtx,
    /// GPU port of `World::color_fires`. Per-cell flame-color inheritance.
    color_fires_compute: ColorFiresCtx,
    /// GPU port of `World::flame_test_emission`. Margolus 2x2.
    flame_emit_compute: FlameTestEmissionCtx,
    /// Generic lifecycle pass — element-data-driven life decrement
    /// and decay-to-product (Fire → Empty after life=0, Steam →
    /// Water at boiling temp, etc.).
    lifecycle_compute: LifecycleCtx,
    /// GPU port of `World::tree_support_check`. Multi-iter flood-fill
    /// from anchored wood through connected wood. Marks unsupported
    /// wood with life=1; the wood-life-fall branch in vertical_fall
    /// then makes those cells fall.
    tree_support_compute: TreeSupportCtx,
    /// GPU port of the per-cell core of `World::thermal_post` —
    /// combustion ignition, burn-out, and phase changes.
    thermal_post_compute: ThermalPostCtx,
    /// GPU port of `World::dissolve` + `World::diffuse_solute`.
    /// Margolus 2x2 4-phase × 2 modes = 8 dispatches per frame.
    solute_compute: SoluteCtx,
    /// Fire emission from burning cells — visible flames above wood
    /// fires, gunpowder ignition columns, etc. Two parity sub-passes.
    fire_emit_compute: FireEmitCtx,
    /// Moisture chemistry: Water+Sand→Mud. Margolus 2x2 4-phase.
    water_sand_compute: WaterSandCtx,
    /// Glass etching: F + Glass → SiF (Derived) + O. Margolus 2x2.
    glass_etch_compute: GlassEtchCtx,
    frame_counter: u32,
    // Lightweight perf counter — prints fps + sim time once per second.
    prof_last_print: std::time::Instant,
    prof_frame_count: u32,
    prof_sim_us: u64,
    prof_compute_us: u64,
    prof_render_us: u64,
    /// Window must outlive the surface. Held as Arc so the surface's
    /// 'static lifetime contract is satisfied without unsafe.
    window: Arc<Window>,

    // ---- egui UI ----
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,

    // ---- Tool / panel state (mirrors the macroquad version) ----
    tool_mode: crate::ToolMode,
    build_mode: bool,
    /// Periodic-table modal overlay open. Tab toggles.
    pt_open: bool,
    /// Selected derived compound id when `selected == Element::Derived`.
    selected_did: u8,
    /// Wind vector set by the wind pad widget.
    wind: macroquad::math::Vec2,

    // ---- Input state ----
    /// Currently-selected element for the paint brush.
    selected: Element,
    /// Paint brush radius in cells.
    brush_radius: i32,
    /// Last known cursor position in window pixels (top-left origin).
    cursor_pos: Option<(f64, f64)>,
    /// True while left mouse is held — paint while held.
    paint_down: bool,
    /// True while right mouse is held — erase while held.
    erase_down: bool,
    /// Pause toggle (Space).
    paused: bool,
    /// Either Ctrl is currently held (gates wheel → zoom).
    ctrl_held: bool,
    /// Last cursor pos while middle-mouse drag is active. None when
    /// middle button is up — middle press latches the start, each
    /// cursor move applies the pan delta and re-anchors here.
    middle_drag_from: Option<(f64, f64)>,

    // ---- Camera state ----
    /// Set by the C key. Render reads it after motion readback and
    /// before the CPU step so the clear actually sticks (otherwise
    /// motion's full-cells readback overwrites the cleared cells).
    pending_clear: bool,
    /// World-space cell coordinate at the center of the view.
    /// Default: (W/2, H/2) → sim is centered. Pan moves this around.
    cam_center_x: f32,
    cam_center_y: f32,
    /// Pixels per cell. Default: whatever fits the sim into the window
    /// while preserving aspect (the "1× zoom" baseline). Larger values
    /// zoom in (fewer cells visible, each takes more pixels). Smaller
    /// values zoom out below baseline — the surrounding void shows.
    cam_scale: f32,
}

impl GpuState {
    /// Initialize wgpu against the given window. Panics on fatal
    /// driver / shader-compile failures — those are non-recoverable
    /// at this layer.
    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        // Backend selection: prefer Vulkan, then DX12, then Metal. We
        // explicitly EXCLUDE GL because the GL ES backend (mesa via
        // WSLg etc.) advertises zero compute workgroups. Compute is
        // load-bearing for this migration — without it the rest of
        // the port is moot.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN
                | wgpu::Backends::DX12
                | wgpu::Backends::METAL,
            ..Default::default()
        });
        let surface = match instance.create_surface(window.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("FATAL: wgpu create_surface failed: {e:?}");
                std::process::exit(1);
            }
        };
        // Diagnostic: enumerate ALL adapters so we know what's available
        // before requesting one. On WSL2 / virtualized setups it's common
        // to have multiple Vulkan ICDs, including software fallbacks like
        // lavapipe; we want to prefer real hardware (Discrete > Integrated)
        // and only fall back to Cpu / Other if nothing else exists.
        let all_adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(
                wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
            )
            .into_iter()
            .collect();
        eprintln!("[wgpu] {} adapter(s) available:", all_adapters.len());
        for a in &all_adapters {
            let info = a.get_info();
            eprintln!(
                "  • {} | backend: {:?} | type: {:?} | driver: {} {}",
                info.name, info.backend, info.device_type, info.driver, info.driver_info,
            );
        }
        // Pick best: prefer DiscreteGpu, then IntegratedGpu, then VirtualGpu;
        // explicitly avoid Cpu (software) unless it's the only option.
        let rank = |t: wgpu::DeviceType| -> i32 {
            match t {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Other => 3,
                wgpu::DeviceType::Cpu => 4,
            }
        };
        let mut sorted = all_adapters;
        sorted.sort_by_key(|a| rank(a.get_info().device_type));
        let adapter = match sorted.into_iter().next() {
            Some(a) => a,
            None => {
                eprintln!("FATAL: no compatible wgpu adapter found");
                std::process::exit(1);
            }
        };
        let info = adapter.get_info();
        eprintln!(
            "[wgpu] selected: {} | backend: {:?} | type: {:?}",
            info.name, info.backend, info.device_type,
        );
        let adapter_limits = adapter.limits();
        eprintln!(
            "[wgpu] limits: max_compute_workgroups_per_dimension = {}, max_storage_buffer_binding_size = {}",
            adapter_limits.max_compute_workgroups_per_dimension,
            adapter_limits.max_storage_buffer_binding_size,
        );
        // Use the adapter's actual limits rather than wgpu's default, which
        // assumes a desktop-class GPU. This lets us at least get a device on
        // limited backends so we can diagnose; we'll fail later with a clear
        // message if we genuinely need compute and don't have it.
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("alembic-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("FATAL: wgpu request_device failed: {e:?}");
                std::process::exit(1);
            }
        };
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // ---- Sim texture (W × H, Rgba8Unorm) ----
        // Created once at startup; updated each frame from `image_buffer`.
        let sim_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("alembic-sim-texture"),
            size: wgpu::Extent3d {
                width: W as u32,
                height: H as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // Sampled by the display pipeline AND written by the
            // render compute. STORAGE_BINDING lets the compute write
            // pixels directly without a CPU intermediate.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_DST
                 | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let sim_view = sim_texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Nearest-neighbor — cells are pixel-precise atoms. No bilinear blur.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("alembic-sim-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let display_init = DisplayUniform {
            sim_min_x: 0.0, sim_min_y: 0.0,
            sim_max_x: 1.0, sim_max_y: 1.0,
            cursor_x: -1.0, cursor_y: -1.0,
            brush_pixel_radius: 0.0, visible: 0.0,
        };
        let display_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-display-uniform"),
            contents: bytemuck::cast_slice(&[display_init]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sim_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-sim-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sim_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-sim-bg"),
            layout: &sim_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sim_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: display_uniform.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alembic-sim-shader"),
            source: wgpu::ShaderSource::Wgsl(SIM_DISPLAY_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("alembic-sim-pipeline-layout"),
            bind_group_layouts: &[&sim_bind_group_layout],
            push_constant_ranges: &[],
        });
        let sim_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("alembic-sim-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ---- World ----
        // Pre-paint a small showcase scene so a fresh window has something
        // to look at: a sand pile on the floor, a water column above-left,
        // and a stone block to anchor the eye. Diagnostic only — once we
        // have mouse painting wired up these go away.
        let mut world = World::new();
        // Pre-paint near the floor of the grid since the default
        // camera is floor-aligned. User sees the showcase scene
        // immediately; the rest of the grid is empty playspace
        // that opens up as they zoom out.
        let cx = W as i32 / 2;
        let floor_y = H as i32 - 30;
        world.paint(cx, floor_y, 12, Element::Sand, 0, false);
        world.paint(cx - 60, floor_y - 80, 6, Element::Water, 0, false);
        world.paint(cx + 50, floor_y - 5, 4, Element::Stone, 0, true);

        // GPU-resident cell state. Single 16 MB allocation, populated
        // once from initial world.cells, then mutated only by GPU
        // compute and small CPU paint/clear writes. Every compute
        // pipeline that needs cell fields binds this buffer.
        let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alembic-cells-buf"),
            size: (W * H * std::mem::size_of::<crate::Cell>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initial upload: world.cells → cells_buf via the zero-copy
        // bytes view. After this, world.cells stays in sync via the
        // single per-frame readback at the start of render(), and CPU
        // chemistry mutations are pushed back via a single per-frame
        // upload before the GPU dispatch.
        queue.write_buffer(&cells_buf, 0, crate::cells_as_bytes(&world.cells));

        // Derived compound registry mirror. 256 entries × 2 vec4<f32>.
        let derived_buf_bytes = (crate::DERIVED_GPU_CAPACITY * 16) as wgpu::BufferAddress;
        let derived_phys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alembic-derived-phys"),
            size: derived_buf_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let derived_color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alembic-derived-color"),
            size: derived_buf_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut derived_phys_staging = vec![[0.0f32; 4]; crate::DERIVED_GPU_CAPACITY];
        let mut derived_color_staging = vec![[0.0f32; 4]; crate::DERIVED_GPU_CAPACITY];
        crate::export_derived_to_gpu(&mut derived_phys_staging, &mut derived_color_staging);
        queue.write_buffer(&derived_phys_buf, 0, bytemuck::cast_slice(&derived_phys_staging));
        queue.write_buffer(&derived_color_buf, 0, bytemuck::cast_slice(&derived_color_staging));

        let pressure_compute = PressureComputeCtx::new(&device, &queue, &cells_buf);
        let thermal_compute = ThermalComputeCtx::new(&device, &queue, &cells_buf);
        let pressure_sources_compute = PressureSourcesCtx::new(&device, &queue, &cells_buf);
        let motion_compute = MotionComputeCtx::new(&device, &queue, &cells_buf, &derived_phys_buf);
        let render_compute = RenderComputeCtx::new(&device, &queue, &cells_buf, &sim_view, &derived_color_buf);
        let clear_flags_compute = ClearFlagsCtx::new(&device, &queue, &cells_buf);
        let color_fires_compute = ColorFiresCtx::new(&device, &queue, &cells_buf);
        let flame_emit_compute = FlameTestEmissionCtx::new(&device, &queue, &cells_buf);
        let lifecycle_compute = LifecycleCtx::new(&device, &queue, &cells_buf);
        let tree_support_compute = TreeSupportCtx::new(&device, &queue, &cells_buf);
        let thermal_post_compute = ThermalPostCtx::new(&device, &queue, &cells_buf);
        let solute_compute = SoluteCtx::new(&device, &queue, &cells_buf);
        let fire_emit_compute = FireEmitCtx::new(&device, &queue, &cells_buf);
        let water_sand_compute = WaterSandCtx::new(&device, &queue, &cells_buf);
        // Pre-register the SiF derived compound so the glass-etching
        // shader can write its derived_id directly. If registration
        // fails (registry full / Si or F not atom), shader gates on
        // `sif_derived_id == 0` would silently no-op.
        let sif_derived_id = crate::register_compound(crate::Element::Si, crate::Element::F)
            .unwrap_or(0) as u32;
        let glass_etch_compute = GlassEtchCtx::new(&device, &queue, &cells_buf, sif_derived_id);

        // egui setup. Renderer matches the swapchain format so we can
        // paint the UI directly into the same surface pass that draws
        // the sim.
        let egui_ctx = egui::Context::default();
        let egui_viewport_id = egui_ctx.viewport_id();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_viewport_id,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            None,  // depth format
            1,     // msaa samples
            false, // dithering
        );

        let mut state = GpuState {
            surface,
            surface_config,
            device,
            queue,
            world,
            sim_texture,
            sim_bind_group,
            sim_pipeline,
            display_uniform,
            cells_buf,
            derived_phys_buf,
            derived_color_buf,
            derived_phys_staging,
            derived_color_staging,
            pressure_compute,
            thermal_compute,
            pressure_sources_compute,
            motion_compute,
            render_compute,
            clear_flags_compute,
            color_fires_compute,
            flame_emit_compute,
            lifecycle_compute,
            tree_support_compute,
            thermal_post_compute,
            solute_compute,
            fire_emit_compute,
            water_sand_compute,
            glass_etch_compute,
            frame_counter: 0,
            prof_last_print: std::time::Instant::now(),
            prof_frame_count: 0,
            prof_sim_us: 0,
            prof_compute_us: 0,
            prof_render_us: 0,
            window,
            selected: Element::Sand,
            brush_radius: 4,
            cursor_pos: None,
            paint_down: false,
            erase_down: false,
            paused: false,
            ctrl_held: false,
            middle_drag_from: None,
            pending_clear: false,
            cam_center_x: W as f32 * 0.5,
            cam_center_y: H as f32 * 0.5,
            cam_scale: 1.0,
            egui_ctx,
            egui_state,
            egui_renderer,
            tool_mode: crate::ToolMode::Paint,
            build_mode: false,
            pt_open: false,
            selected_did: 0,
            wind: macroquad::math::Vec2::new(0.0, 0.0),
        };
        state.camera_reset();
        state
    }

    /// Pixels-per-cell that fills the window without letterbox bars
    /// (cover-fit). When the window aspect differs from the sim aspect,
    /// part of the sim is cropped off-screen — the user can pan to see
    /// it. This is the default "1× zoom" reference; zoom-out below
    /// this value reveals void around the sim grid.
    fn fit_scale(&self) -> f32 {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let by_w = win_w / W as f32;
        let by_h = win_h / H as f32;
        by_w.max(by_h)
    }

    /// Reset camera to the default view: bottom-aligned (floor visible
    /// immediately — falling-sand sims need that as the home pose),
    /// horizontally centered, zoomed in 2× past cover-fit so zooming
    /// out reveals the playspace above and to the sides. Bound to
    /// Backspace.
    fn camera_reset(&mut self) {
        self.cam_scale = self.fit_scale() * 2.0;
        let win_h = self.surface_config.height as f32;
        let visible_half_y = (win_h * 0.5) / self.cam_scale;
        self.cam_center_x = W as f32 * 0.5;
        // cam_center_y at H − half_y puts the bottom edge of the view
        // exactly at the grid floor.
        self.cam_center_y = (H as f32 - visible_half_y).max(H as f32 * 0.5);
        self.clamp_camera();
    }

    /// Cursor-anchored zoom. The cell currently under the cursor stays
    /// under the cursor across the zoom — the camera center shifts to
    /// preserve that invariant. Zoom-out is hard-capped at cover-fit
    /// (`fit_scale`): you can never see void around the grid; the
    /// minimum zoom is "as much grid as fits without bars."
    fn zoom_at(&mut self, screen_x: f32, screen_y: f32, factor: f32) {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let cell_x = self.cam_center_x + (screen_x - win_w * 0.5) / self.cam_scale;
        let cell_y = self.cam_center_y + (screen_y - win_h * 0.5) / self.cam_scale;
        let fit = self.fit_scale();
        // Min = cover-fit (no void, may crop one axis). Max = 16× zoom in.
        let new_scale = (self.cam_scale * factor).clamp(fit, fit * 16.0);
        self.cam_center_x = cell_x - (screen_x - win_w * 0.5) / new_scale;
        self.cam_center_y = cell_y - (screen_y - win_h * 0.5) / new_scale;
        self.cam_scale = new_scale;
        self.clamp_camera();
    }

    /// Pan the camera by a screen-pixel delta. Used by middle-mouse drag.
    fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.cam_center_x -= dx / self.cam_scale;
        self.cam_center_y -= dy / self.cam_scale;
        self.clamp_camera();
    }

    /// Keep the camera center inside the grid: the visible window edges
    /// can never extend past the grid boundary, so the user never sees
    /// void around the playable area. Run after any camera-mutating op
    /// (zoom, pan, reset).
    fn clamp_camera(&mut self) {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let half_x = (win_w * 0.5) / self.cam_scale;
        let half_y = (win_h * 0.5) / self.cam_scale;
        // If the visible half-extent exceeds the grid half-extent on
        // either axis, the grid is fully visible on that axis — pin
        // the camera center to the grid center for that axis.
        let w = W as f32;
        let h = H as f32;
        self.cam_center_x = if half_x >= w * 0.5 {
            w * 0.5
        } else {
            self.cam_center_x.clamp(half_x, w - half_x)
        };
        self.cam_center_y = if half_y >= h * 0.5 {
            h * 0.5
        } else {
            self.cam_center_y.clamp(half_y, h - half_y)
        };
    }

    /// Sim render rectangle in framebuffer pixels, derived from the
    /// camera state. May extend past the window when zoomed in (the
    /// shader clips); may be smaller than the window when zoomed out
    /// (void fills the gap). Returns (min_x, min_y, max_x, max_y) and
    /// the per-cell scale.
    fn sim_pixel_rect(&self) -> ((f32, f32, f32, f32), f32) {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let scale = self.cam_scale;
        let min_x = win_w * 0.5 - self.cam_center_x * scale;
        let min_y = win_h * 0.5 - self.cam_center_y * scale;
        let max_x = min_x + W as f32 * scale;
        let max_y = min_y + H as f32 * scale;
        ((min_x, min_y, max_x, max_y), scale)
    }

    /// Translate window-pixel cursor coords into integer grid coords,
    /// returning None when the cursor is outside the sim rectangle.
    /// Camera-aware: works correctly under any zoom/pan.
    fn cursor_to_grid(&self, px: f64, py: f64) -> Option<(i32, i32)> {
        let (rect, scale) = self.sim_pixel_rect();
        let pxf = px as f32;
        let pyf = py as f32;
        if pxf < rect.0 || pxf > rect.2 || pyf < rect.1 || pyf > rect.3 {
            return None;
        }
        let gx = ((pxf - rect.0) / scale).floor() as i32;
        let gy = ((pyf - rect.1) / scale).floor() as i32;
        Some((gx.clamp(0, W as i32 - 1), gy.clamp(0, H as i32 - 1)))
    }

    /// Apply a brush stroke at the current cursor location, if any.
    /// Skips when the cursor is outside the sim rect (over letterbox).
    fn apply_brush(&mut self) {
        let Some((px, py)) = self.cursor_pos else { return; };
        let Some((gx, gy)) = self.cursor_to_grid(px, py) else { return; };
        if self.paint_down {
            self.world.paint(gx, gy, self.brush_radius, self.selected, 0, false);
        }
        if self.erase_down {
            self.world.paint(gx, gy, self.brush_radius, Element::Empty, 0, false);
        }
    }

    /// Map a winit keycode to its element shortcut, if any. Mirrors
    /// the macroquad version's number-key bindings so muscle memory
    /// transfers.
    fn element_for_key(key: KeyCode) -> Option<Element> {
        match key {
            KeyCode::Digit1 => Some(Element::Sand),
            KeyCode::Digit2 => Some(Element::Water),
            KeyCode::Digit3 => Some(Element::Stone),
            KeyCode::Digit4 => Some(Element::Wood),
            KeyCode::Digit5 => Some(Element::CO2),
            KeyCode::Digit6 => Some(Element::Oil),
            KeyCode::Digit7 => Some(Element::Lava),
            KeyCode::Digit8 => Some(Element::Fire),
            KeyCode::Digit9 => Some(Element::Seed),
            KeyCode::Digit0 => Some(Element::Empty),
            _ => None,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Build the side panel UI in egui — faithful port of the macroquad
    /// `panel_button_rects` / `panel_ambient_rects` / `wind_pad_rect`
    /// layout in lib.rs. Same colors, same sections, same behavior.
    /// The periodic-table modal (Tab) is drawn separately on top.
    fn draw_ui(&mut self, ctx: &egui::Context) {
        // Panel chrome — colors mirror panel_bg() and draw_panel_button.
        let panel_bg = egui::Color32::from_rgb(18, 18, 24);
        let btn_normal = egui::Color32::from_rgb(30, 30, 38);
        let btn_hover = egui::Color32::from_rgb(42, 42, 54);
        let btn_selected = egui::Color32::from_rgb(62, 90, 140);
        let btn_border = egui::Color32::from_rgb(60, 60, 72);
        let text_btn = egui::Color32::from_rgb(220, 220, 230);
        let section_header = egui::Color32::from_rgb(130, 130, 150);
        let dim_label = egui::Color32::from_rgb(150, 150, 165);
        let value_color = egui::Color32::from_rgb(230, 230, 240);

        egui::SidePanel::right("alembic-side-panel")
            .resizable(false)
            .exact_width(240.0)
            .frame(
                egui::Frame::default()
                    .fill(panel_bg)
                    .inner_margin(egui::Margin {
                        left: 12, right: 12, top: 10, bottom: 10,
                    }),
            )
            .show(ctx, |ui| {
                // Override default egui button styling so our buttons
                // look like draw_panel_button: solid fill, 1px border,
                // 14pt centered label.
                Self::style_panel(ui, btn_normal, btn_hover, btn_border);

                // ---- TOOLS section ----
                ui.label(
                    egui::RichText::new("TOOLS").color(section_header).size(11.0),
                );
                ui.add_space(8.0);

                // Tool buttons (Paint / Heat / Vacuum / Pipet) at fixed 30px
                // height, full width minus margins.
                let tools: [(&str, crate::ToolMode); 4] = [
                    ("Paint", crate::ToolMode::Paint),
                    ("Heat", crate::ToolMode::Heat),
                    ("Vacuum", crate::ToolMode::Vacuum),
                    ("Pipet", crate::ToolMode::Pipet),
                ];
                for (label, mode) in tools {
                    if Self::tool_button(ui, label, self.tool_mode == mode,
                        btn_selected, btn_normal, btn_hover, btn_border, text_btn) {
                        self.tool_mode = mode;
                    }
                }

                // Prefab + (eventually) its dropdown.
                if Self::tool_button(
                    ui, "Prefab", self.tool_mode == crate::ToolMode::Prefab,
                    btn_selected, btn_normal, btn_hover, btn_border, text_btn,
                ) {
                    self.tool_mode = crate::ToolMode::Prefab;
                }

                // Wire + (eventually) its dropdown.
                if Self::tool_button(
                    ui, "Wire", self.tool_mode == crate::ToolMode::Wire,
                    btn_selected, btn_normal, btn_hover, btn_border, text_btn,
                ) {
                    self.tool_mode = crate::ToolMode::Wire;
                }

                // Extra gap before Build (matches macroquad +14px).
                ui.add_space(14.0);
                let build_label = if self.build_mode { "Build: ON" } else { "Build: OFF" };
                if Self::tool_button(
                    ui, build_label, self.build_mode,
                    btn_selected, btn_normal, btn_hover, btn_border, text_btn,
                ) {
                    self.build_mode = !self.build_mode;
                }

                // ---- Element readout ----
                ui.add_space(14.0);
                let sel_name = crate::ui_element_name(self.selected);
                let (sr, sg, sb) = self.selected.base_color();
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(16.0, 16.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(sr, sg, sb),
                    );
                    ui.label(
                        egui::RichText::new(format!("Element: {}", sel_name))
                            .color(value_color)
                            .size(13.0),
                    );
                });

                // ---- SIMULATION section ----
                ui.add_space(14.0);
                ui.label(
                    egui::RichText::new("SIMULATION").color(section_header).size(11.0),
                );
                ui.add_space(4.0);

                // Brush radius row — not in the macroquad panel directly
                // (it uses [/] keys + scroll), but exposing it here is
                // a quality-of-life addition matching the readout style.
                Self::ambient_row(
                    ui, "Brush",
                    &format!("{}", self.brush_radius),
                    dim_label, value_color,
                );

                let ambient_actual = 20 + self.world.ambient_offset;
                Self::ambient_row(
                    ui, "Temp",
                    &format!("{:+}°C", ambient_actual),
                    dim_label, value_color,
                );
                Self::ambient_row(
                    ui, "O₂",
                    &format!("{:.0}%", self.world.ambient_oxygen * 100.0),
                    dim_label, value_color,
                );
                Self::ambient_row(
                    ui, "Grav",
                    &format!("{:.1}×", self.world.gravity),
                    dim_label, value_color,
                );

                // ---- Wind pad ----
                ui.add_space(14.0);
                ui.label(
                    egui::RichText::new("Wind").color(dim_label).size(11.0),
                );
                ui.add_space(4.0);
                let pad_size = 84.0;
                let (wpad, wresp) = ui.allocate_exact_size(
                    egui::vec2(pad_size, pad_size),
                    egui::Sense::click_and_drag(),
                );
                let p = ui.painter();
                let bg_pad = if wresp.hovered() {
                    egui::Color32::from_rgb(34, 34, 46)
                } else {
                    egui::Color32::from_rgb(24, 24, 32)
                };
                p.rect_filled(wpad, 2.0, bg_pad);
                p.rect_stroke(wpad, 2.0, egui::Stroke::new(1.0, btn_border),
                    egui::StrokeKind::Inside);
                let center = wpad.center();
                let radius = pad_size * 0.5 - 2.0;
                p.circle_stroke(center, radius, egui::Stroke::new(1.0, btn_border));
                // Cardinal ticks
                for (dx, dy) in [(radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)] {
                    p.line_segment(
                        [
                            egui::pos2(center.x + dx * 0.85, center.y + dy * 0.85),
                            egui::pos2(center.x + dx, center.y + dy),
                        ],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 70, 84)),
                    );
                }
                // Drag/click sets wind vector. Distance from center, scaled
                // into [0, WIND_MAX].
                if wresp.clicked() || wresp.dragged() {
                    if let Some(pos) = wresp.interact_pointer_pos() {
                        let dx = pos.x - center.x;
                        let dy = pos.y - center.y;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist > 1.0 {
                            const WIND_MAX: f32 = 2.0;
                            let mag = (dist / radius).min(1.0) * WIND_MAX;
                            let n = mag / dist;
                            self.wind = macroquad::math::Vec2::new(dx * n, dy * n);
                        }
                    }
                }
                // Arrow from center → current wind.
                let mag = self.wind.length();
                if mag > 0.001 {
                    const WIND_MAX: f32 = 2.0;
                    let scale_ = (radius / WIND_MAX) * mag.min(WIND_MAX);
                    let dir = self.wind.normalize();
                    let tip = egui::pos2(
                        center.x + dir.x * scale_,
                        center.y + dir.y * scale_,
                    );
                    p.line_segment(
                        [center, tip],
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(230, 200, 120)),
                    );
                    p.circle_filled(tip, 3.0, egui::Color32::from_rgb(240, 210, 130));
                }
                p.circle_filled(center, 2.0, egui::Color32::from_rgb(140, 140, 160));

                // Numeric readout right of the pad.
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!("x {:+.2}", self.wind.x))
                        .color(value_color).size(12.0),
                );
                ui.label(
                    egui::RichText::new(format!("y {:+.2}", self.wind.y))
                        .color(value_color).size(12.0),
                );
                ui.label(
                    egui::RichText::new(format!("|v| {:.2}", mag))
                        .color(dim_label).size(12.0),
                );

                // Reset wind button.
                ui.add_space(8.0);
                if Self::tool_button(
                    ui, "Reset Wind", false,
                    btn_selected, btn_normal, btn_hover, btn_border, text_btn,
                ) {
                    self.wind = macroquad::math::Vec2::new(0.0, 0.0);
                }

                // ---- Footnote (open periodic table) ----
                ui.add_space(18.0);
                ui.separator();
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Press Tab — periodic table")
                        .color(dim_label).size(11.0),
                );
                ui.label(
                    egui::RichText::new("Space — pause   |   C — clear")
                        .color(dim_label).size(11.0),
                );
            });

        // Periodic-table modal overlay.
        if self.pt_open {
            self.draw_periodic_table(ctx);
        }
    }

    fn style_panel(
        ui: &mut egui::Ui,
        normal: egui::Color32,
        hover: egui::Color32,
        border: egui::Color32,
    ) {
        let v = &mut ui.style_mut().visuals.widgets;
        v.inactive.bg_fill = normal;
        v.inactive.weak_bg_fill = normal;
        v.inactive.bg_stroke = egui::Stroke::new(1.0, border);
        v.hovered.bg_fill = hover;
        v.hovered.weak_bg_fill = hover;
        v.hovered.bg_stroke = egui::Stroke::new(1.0, border);
        v.active.bg_fill = hover;
        v.active.weak_bg_fill = hover;
        v.active.bg_stroke = egui::Stroke::new(1.0, border);
    }

    /// Draws a fixed-width tool button matching draw_panel_button:
    /// 30px tall, full panel width, 1px border, label centered, three
    /// states (normal / hovered / selected).
    fn tool_button(
        ui: &mut egui::Ui,
        label: &str,
        selected: bool,
        sel_color: egui::Color32,
        normal: egui::Color32,
        hover: egui::Color32,
        border: egui::Color32,
        text: egui::Color32,
    ) -> bool {
        let bw = ui.available_width();
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(bw, 30.0),
            egui::Sense::click(),
        );
        let bg = if selected { sel_color }
                 else if resp.hovered() { hover }
                 else { normal };
        let p = ui.painter();
        p.rect_filled(rect, 2.0, bg);
        p.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, border),
            egui::StrokeKind::Inside);
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(14.0),
            text,
        );
        // Manual gap between buttons (6px in macroquad).
        ui.add_space(6.0);
        resp.clicked()
    }

    /// One ambient row (Temp / O₂ / Grav) — label left, value right,
    /// hover-glow background. Mirrors the macroquad ambient row.
    fn ambient_row(
        ui: &mut egui::Ui,
        label: &str,
        value: &str,
        dim: egui::Color32,
        val_color: egui::Color32,
    ) {
        let bw = ui.available_width();
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(bw, 28.0),
            egui::Sense::hover(),
        );
        let bg = if resp.hovered() {
            egui::Color32::from_rgb(38, 38, 48)
        } else {
            egui::Color32::from_rgb(24, 24, 32)
        };
        let p = ui.painter();
        p.rect_filled(rect, 2.0, bg);
        p.rect_stroke(rect, 2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 50, 62)),
            egui::StrokeKind::Inside);
        p.text(
            egui::pos2(rect.left() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(13.0),
            dim,
        );
        p.text(
            egui::pos2(rect.right() - 10.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            value,
            egui::FontId::proportional(14.0),
            val_color,
        );
        ui.add_space(5.0);
    }

    /// Periodic-table modal overlay. Mirrors `draw_periodic_table` —
    /// dim background, 18-col atom grid, compound row, derived row,
    /// detail panel below. Click an atom or compound to paint it.
    fn draw_periodic_table(&mut self, ctx: &egui::Context) {
        // Full-screen dimmer + click-to-close on the dim area.
        let bg = egui::Color32::from_rgba_unmultiplied(8, 8, 14, 220);
        egui::Area::new(egui::Id::new("alembic-pt-dim"))
            .fixed_pos(egui::pos2(0.0, 0.0))
            .order(egui::Order::Background)
            .show(ctx, |ui| {
                let screen = ctx.screen_rect();
                ui.painter().rect_filled(screen, 0.0, bg);
            });

        let title = egui::RichText::new("Periodic Table of Elements")
            .color(egui::Color32::WHITE).size(22.0);
        let close_hint = egui::RichText::new("Tab or Esc to close")
            .color(egui::Color32::from_rgb(180, 180, 180)).size(13.0);

        egui::Window::new("Periodic Table")
            .title_bar(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(18, 18, 26))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 76)))
                    .inner_margin(egui::Margin::same(20)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(title);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(close_hint);
                    });
                });
                ui.label(
                    egui::RichText::new("click an atom or compound to paint")
                        .color(egui::Color32::GRAY).size(11.0),
                );
                ui.add_space(8.0);

                let pt_tile = 38.0;
                let pt_gap = 3.0;

                // Atom grid — 18 groups × 9 rows (periods 1-7 + lanth + actin).
                let atoms = crate::ui_atoms();
                let mut by_pp: std::collections::HashMap<(u8, u8), (Element, &'static str, u8)> =
                    std::collections::HashMap::new();
                for (el, num, sym, group, period) in &atoms {
                    by_pp.insert((*period, *group), (*el, *sym, *num));
                }
                let max_period = atoms.iter().map(|a| a.4).max().unwrap_or(7);
                let mut picked = false;
                for period in 1..=max_period {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = pt_gap;
                        for group in 1..=18u8 {
                            if let Some(&(el, sym, num)) = by_pp.get(&(period, group)) {
                                if Self::pt_atom_tile(
                                    ui, el, sym, num, pt_tile, &mut self.selected,
                                ) {
                                    picked = true;
                                }
                            } else {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(pt_tile, pt_tile),
                                    egui::Sense::hover(),
                                );
                                let _ = rect;
                            }
                        }
                    });
                    ui.add_space(pt_gap);
                }

                ui.add_space(16.0);
                ui.label(
                    egui::RichText::new("Compounds").color(egui::Color32::LIGHT_GRAY)
                        .size(13.0),
                );
                ui.add_space(4.0);

                // Compound row — same tile size, contiguous.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = pt_gap;
                    for &el in crate::ui_compound_palette() {
                        if Self::pt_compound_tile(ui, el, pt_tile, &mut self.selected) {
                            picked = true;
                        }
                    }
                });

                // Derived row.
                let derived = crate::ui_derived_palette();
                if !derived.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Derived")
                            .color(egui::Color32::LIGHT_GRAY).size(13.0),
                    );
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = pt_gap;
                        for (did, formula, [r, g, b]) in &derived {
                            let highlight = self.selected == Element::Derived
                                && self.selected_did == *did;
                            let (rect, resp) = ui.allocate_exact_size(
                                egui::vec2(pt_tile, pt_tile),
                                egui::Sense::click(),
                            );
                            let p = ui.painter();
                            p.rect_filled(rect, 2.0, egui::Color32::from_rgb(*r, *g, *b));
                            p.rect_stroke(rect, 2.0,
                                egui::Stroke::new(1.0,
                                    egui::Color32::from_rgb(40, 40, 50)),
                                egui::StrokeKind::Inside);
                            if highlight {
                                p.rect_stroke(rect.expand(2.0), 3.0,
                                    egui::Stroke::new(3.0, egui::Color32::GREEN),
                                    egui::StrokeKind::Inside);
                            }
                            if resp.hovered() {
                                p.rect_stroke(rect.expand(1.0), 3.0,
                                    egui::Stroke::new(3.0, egui::Color32::YELLOW),
                                    egui::StrokeKind::Inside);
                            }
                            p.text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                formula,
                                egui::FontId::proportional(11.0),
                                egui::Color32::WHITE,
                            );
                            if resp.clicked() {
                                self.selected = Element::Derived;
                                self.selected_did = *did;
                                picked = true;
                            }
                        }
                    });
                }

                if picked {
                    self.pt_open = false;
                }

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Close (Esc / Tab)").clicked() {
                        self.pt_open = false;
                    }
                });
            });
    }

    fn pt_atom_tile(
        ui: &mut egui::Ui,
        el: Element,
        sym: &'static str,
        num: u8,
        size: f32,
        selected: &mut Element,
    ) -> bool {
        let (r, g, b) = el.base_color();
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(size, size),
            egui::Sense::click(),
        );
        let p = ui.painter();
        p.rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
        p.rect_stroke(rect, 2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 40, 50)),
            egui::StrokeKind::Inside);
        if *selected == el {
            p.rect_stroke(rect.expand(2.0), 3.0,
                egui::Stroke::new(3.0, egui::Color32::GREEN),
                egui::StrokeKind::Inside);
        }
        if resp.hovered() {
            p.rect_stroke(rect.expand(1.0), 3.0,
                egui::Stroke::new(3.0, egui::Color32::YELLOW),
                egui::StrokeKind::Inside);
        }
        // Atomic number (top-left, small).
        p.text(
            rect.left_top() + egui::vec2(3.0, 3.0),
            egui::Align2::LEFT_TOP,
            num.to_string(),
            egui::FontId::proportional(10.0),
            egui::Color32::BLACK,
        );
        // Symbol (centered, large).
        p.text(
            rect.center() + egui::vec2(0.0, 3.0),
            egui::Align2::CENTER_CENTER,
            sym,
            egui::FontId::proportional(18.0),
            egui::Color32::BLACK,
        );
        if resp.clicked() {
            *selected = el;
            return true;
        }
        false
    }

    fn pt_compound_tile(
        ui: &mut egui::Ui,
        el: Element,
        size: f32,
        selected: &mut Element,
    ) -> bool {
        let (r, g, b) = el.base_color();
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(size, size),
            egui::Sense::click(),
        );
        let p = ui.painter();
        if el == Element::Empty {
            p.rect_filled(rect, 2.0, egui::Color32::from_rgb(40, 40, 46));
            // Red X — eraser indicator.
            let red = egui::Color32::from_rgb(200, 70, 70);
            p.line_segment(
                [
                    rect.left_top() + egui::vec2(8.0, 8.0),
                    rect.right_bottom() - egui::vec2(8.0, 8.0),
                ],
                egui::Stroke::new(2.0, red),
            );
            p.line_segment(
                [
                    rect.right_top() + egui::vec2(-8.0, 8.0),
                    rect.left_bottom() + egui::vec2(8.0, -8.0),
                ],
                egui::Stroke::new(2.0, red),
            );
        } else {
            p.rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
        }
        p.rect_stroke(rect, 2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 40, 50)),
            egui::StrokeKind::Inside);
        if *selected == el {
            p.rect_stroke(rect.expand(2.0), 3.0,
                egui::Stroke::new(3.0, egui::Color32::GREEN),
                egui::StrokeKind::Inside);
        }
        if resp.hovered() {
            p.rect_stroke(rect.expand(1.0), 3.0,
                egui::Stroke::new(3.0, egui::Color32::YELLOW),
                egui::StrokeKind::Inside);
        }
        if resp.clicked() {
            *selected = el;
            return true;
        }
        false
    }

    fn render(&mut self) {
        // Single CPU↔GPU sync point per frame: motion's readback gives
        // us the post-everything state in world.cells. apply_brush and
        // CPU chemistry mutate world.cells. Then ONE upload pushes the
        // mutated cells back to cells_buf for the next GPU dispatch.
        let t_compute_start = std::time::Instant::now();
        if !self.paused {
            let _ = self.device.poll(wgpu::Maintain::Wait);
            self.motion_compute.read_back_prev_into(&mut self.world);
        }
        let t_compute_readback = t_compute_start.elapsed();

        // Pending C-key clear runs AFTER readback so it sticks.
        if self.pending_clear {
            for c in self.world.cells.iter_mut() {
                if !c.is_frozen() {
                    *c = crate::Cell::EMPTY;
                }
            }
            self.pending_clear = false;
        }
        self.apply_brush();

        let t_sim_start = std::time::Instant::now();
        if !self.paused {
            // GPU runs: pressure_sources, pressure, thermal_diffuse,
            // motion, clear_flags, color_fires, flame_test_emission.
            // CPU runs everything else (combustion in thermal_post,
            // chem_reactions, etc.) — those will follow in later phases.
            let gpu_chem = crate::GpuChem {
                clear_flags: true,
                color_fires: true,
                flame_test_emission: true,
                tree_support: true,
                thermal_post: true,
                dissolve: true,
                diffuse_solute: true,
                reactions: true,
                glass_etching: true,
            };
            self.world.step_skip_gpu_v2(macroquad::math::Vec2::new(0.0, 0.0), gpu_chem);
        }
        let t_sim = t_sim_start.elapsed();

        let t_dispatch_start = std::time::Instant::now();
        if !self.paused {
            let run_ps = self.frame_counter & 1 == 0;

            // Single zero-copy upload: world.cells (with this frame's
            // chemistry/paint changes) → cells_buf. Every GPU compute
            // pass below reads/writes cells_buf directly — no per-cell
            // CPU stage loops anywhere.
            self.queue.write_buffer(&self.cells_buf, 0, crate::cells_as_bytes(&self.world.cells));
            // Sync Derived registry: CPU chemistry passes can register
            // new compounds (FeCl, KCl, Al₂O₃, …); GPU motion + render
            // need their physics + color. 8KB upload per frame; cheap.
            crate::export_derived_to_gpu(
                &mut self.derived_phys_staging,
                &mut self.derived_color_staging,
            );
            self.queue.write_buffer(&self.derived_phys_buf, 0,
                bytemuck::cast_slice(&self.derived_phys_staging));
            self.queue.write_buffer(&self.derived_color_buf, 0,
                bytemuck::cast_slice(&self.derived_color_staging));
            let amb = self.world.ambient_offset;
            self.thermal_compute.update_frame(&self.queue, self.frame_counter, amb);
            if run_ps {
                self.pressure_sources_compute.update_frame(&self.queue, &self.world);
            }

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("alembic-combined-compute-encoder"),
            });
            // 1. clear_flags — zero out FLAG_UPDATED on every cell so
            //    motion/chemistry start from a clean slate.
            self.clear_flags_compute.encode(&mut encoder);
            // 1b. lifecycle — tick down life on ephemeral elements
            //     (Fire, Steam, …) so they age and decay this frame.
            self.lifecycle_compute.encode(&mut encoder);
            // 1c. tree_support — runs every 30 frames (matches CPU
            //     cadence). Multi-iter flood-fill from anchored
            //     wood; marks unrooted wood with life=1 so motion
            //     can drop it.
            if self.frame_counter % 30 == 0 {
                self.tree_support_compute.encode(&mut encoder);
            }
            // 2. PS: column scan + asymmetric blend. Writes new pressure
            //    straight into cells_buf, so the diffusion below sees
            //    post-PS values when it extracts.
            if run_ps {
                self.pressure_sources_compute.encode(&mut encoder);
            }
            // 3. Pressure: extract → 3 diffuse iters → writeback.
            self.pressure_compute.encode(&mut encoder);
            // 4. Thermal: extract → 1 diffuse iter → writeback.
            self.thermal_compute.encode(&mut encoder);
            // 4b. thermal_post — combustion ignition, burn-out, and
            //     phase changes (Water→Steam, Ice→Water, Wood→ash).
            self.thermal_post_compute.update_frame(&self.queue, self.frame_counter);
            self.thermal_post_compute.encode(&mut encoder);
            // 4c. dissolve + diffuse_solute — Margolus 2x2 4-phase
            //     for water-salt dissolution and inter-water solute
            //     equalization.
            self.solute_compute.encode(&mut encoder, &self.queue, self.frame_counter);
            // 4d. fire emit-above — burning cells spawn visible Fire
            //     in the empty cell above (1/10 per frame). Two
            //     parity sub-passes for race-safe multi-cell write.
            self.fire_emit_compute.encode(&mut encoder, &self.queue, self.frame_counter);
            // 4e. water+sand → mud moisture chemistry. Margolus 2x2
            //     4-phase. Replaces CPU `World::reactions`.
            self.water_sand_compute.encode(&mut encoder, &self.queue, self.frame_counter);
            // 4f. glass etching: F + Glass → SiF + O. Margolus 2x2
            //     4-phase. Replaces CPU `World::glass_etching`.
            self.glass_etch_compute.encode(&mut encoder, &self.queue, self.frame_counter);
            // 5. flame_test_emission — Margolus 4-phase, hot flame-
            //    coloring elements emit colored Fire into block-local
            //    empty cells.
            self.flame_emit_compute.encode(&mut encoder, &self.queue, self.frame_counter);
            // 6. color_fires — Fire cells inherit flame color from any
            //    flame-coloring neighbor.
            self.color_fires_compute.encode(&mut encoder);
            // 7. Motion: 5 passes (vfall, lspread-even/odd, dslide-even/odd).
            self.motion_compute.encode(&mut encoder, &self.queue, self.frame_counter, &self.cells_buf);
            self.queue.submit(std::iter::once(encoder.finish()));
            self.motion_compute.start_map();
            self.motion_compute.advance_frame();
            self.frame_counter = self.frame_counter.wrapping_add(1);
        }
        let t_compute = t_compute_readback + t_dispatch_start.elapsed();
        let t_render_start = std::time::Instant::now();

        // Compute the sim rectangle in framebuffer pixels, preserving
        // the W:H aspect ratio. Whatever space is left over becomes
        // letterbox / pillarbox bars on the sides.
        let (sim_rect, pixels_per_cell) = self.sim_pixel_rect();

        // Cursor in framebuffer pixels + brush radius in pixels so
        // the fragment shader can draw a true circle regardless of
        // how the sim is letterboxed inside the window.
        let display_data = match self.cursor_pos {
            Some((px, py)) => DisplayUniform {
                sim_min_x: sim_rect.0,
                sim_min_y: sim_rect.1,
                sim_max_x: sim_rect.2,
                sim_max_y: sim_rect.3,
                cursor_x: px as f32,
                cursor_y: py as f32,
                brush_pixel_radius: self.brush_radius as f32 * pixels_per_cell,
                visible: 1.0,
            },
            None => DisplayUniform {
                sim_min_x: sim_rect.0,
                sim_min_y: sim_rect.1,
                sim_max_x: sim_rect.2,
                sim_max_y: sim_rect.3,
                cursor_x: -1.0,
                cursor_y: -1.0,
                brush_pixel_radius: self.brush_radius as f32 * pixels_per_cell,
                visible: 0.0,
            },
        };
        self.queue.write_buffer(
            &self.display_uniform,
            0,
            bytemuck::cast_slice(&[display_data]),
        );

        // The render compute pass below dispatches as part of the
        // frame encoder — it reads cells_buf and writes sim_texture
        // directly, replacing the old CPU pixel-fill + texture upload.

        // Build the egui UI for this frame. Run BEFORE acquiring the
        // swapchain image so any UI state changes (selected element,
        // brush radius, ambient sliders, paused) take effect this frame.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_full_output = self.egui_ctx.clone().run(raw_input, |ctx| {
            self.draw_ui(ctx);
        });
        self.egui_state.handle_platform_output(&self.window, egui_full_output.platform_output);
        let egui_clipped = self
            .egui_ctx
            .tessellate(egui_full_output.shapes, egui_full_output.pixels_per_point);
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: egui_full_output.pixels_per_point,
        };

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(_) => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("alembic-frame-encoder"),
        });
        // First: render compute pass fills sim_texture from cells_buf.
        // Then: the display pipeline samples sim_texture to the swapchain.
        self.render_compute.encode(&mut encoder);
        // egui texture upkeep — must happen on the encoder before the
        // render pass begins.
        for (id, image_delta) in &egui_full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &egui_clipped,
            &screen_desc,
        );
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("alembic-sim-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02, g: 0.02, b: 0.06, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.sim_pipeline);
            rpass.set_bind_group(0, &self.sim_bind_group, &[]);
            rpass.draw(0..3, 0..1);
            // egui draws on top of the sim. forget_lifetime cast lets
            // the renderer accept our short-lived RenderPass handle —
            // egui-wgpu wants a static-borrowed lifetime here.
            let mut rpass_static = rpass.forget_lifetime();
            self.egui_renderer.render(&mut rpass_static, &egui_clipped, &screen_desc);
        }
        for id in &egui_full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        frame.present();
        let t_render = t_render_start.elapsed();

        self.prof_sim_us     += t_sim.as_micros() as u64;
        self.prof_compute_us += t_compute.as_micros() as u64;
        self.prof_render_us  += t_render.as_micros() as u64;
        self.prof_frame_count += 1;
        if self.prof_last_print.elapsed().as_secs_f32() >= 1.0 {
            let f = self.prof_frame_count.max(1) as f32;
            eprintln!(
                "[fps] {:>3} | sim {:>5.1}ms | compute {:>5.1}ms | render {:>5.1}ms",
                self.prof_frame_count,
                (self.prof_sim_us as f32 / f) / 1000.0,
                (self.prof_compute_us as f32 / f) / 1000.0,
                (self.prof_render_us as f32 / f) / 1000.0,
            );
            self.prof_frame_count = 0;
            self.prof_sim_us = 0;
            self.prof_compute_us = 0;
            self.prof_render_us = 0;
            self.prof_last_print = std::time::Instant::now();
        }
    }
}

/// Application-level state — held by winit's event loop driver. The
/// `GpuState` is created lazily on the first `resumed` event because
/// some platforms don't have a window/surface available until then.
pub struct App {
    state: Option<GpuState>,
}

impl App {
    pub fn new() -> Self {
        App { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_W, WINDOW_H))
            .with_resizable(true);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("FATAL: window creation failed: {e:?}");
                event_loop.exit();
                return;
            }
        };
        self.state = Some(GpuState::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return; };

        // Intercept app-level shortcut keys BEFORE egui sees them.
        // egui consumes Tab for its own focus traversal, which would
        // otherwise eat our Tab → periodic-table toggle.
        if let WindowEvent::KeyboardInput { event: ref ke, .. } = event {
            if ke.state == ElementState::Pressed {
                if let PhysicalKey::Code(code) = ke.physical_key {
                    match code {
                        KeyCode::Tab => {
                            state.pt_open = !state.pt_open;
                            return;
                        }
                        KeyCode::Escape => {
                            if state.pt_open {
                                state.pt_open = false;
                                return;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // egui consumes input events first so the side panel responds
        // to clicks/hovers/typing. If egui says it consumed pointer or
        // keyboard input, we suppress the matching sim handler below.
        let egui_resp = state.egui_state.on_window_event(&state.window, &event);
        let egui_wants_pointer = egui_resp.consumed
            && matches!(
                event,
                WindowEvent::MouseInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseWheel { .. }
            );
        let egui_wants_keyboard = egui_resp.consumed
            && matches!(event, WindowEvent::KeyboardInput { .. });
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                // Apply pan if middle-mouse drag is in progress.
                if let Some(prev) = state.middle_drag_from {
                    let dx = (new_pos.0 - prev.0) as f32;
                    let dy = (new_pos.1 - prev.1) as f32;
                    state.pan_pixels(dx, dy);
                    state.middle_drag_from = Some(new_pos);
                }
                state.cursor_pos = Some(new_pos);
            }
            WindowEvent::CursorLeft { .. } => {
                state.cursor_pos = None;
                state.middle_drag_from = None;
            }
            WindowEvent::MouseInput { state: mouse_state, button, .. } => {
                if egui_wants_pointer || state.egui_ctx.is_pointer_over_area() {
                    return;
                }
                let pressed = mouse_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => state.paint_down = pressed,
                    MouseButton::Right => state.erase_down = pressed,
                    MouseButton::Middle => {
                        if pressed {
                            state.middle_drag_from = state.cursor_pos;
                        } else {
                            state.middle_drag_from = None;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if egui_wants_pointer || state.egui_ctx.is_pointer_over_area() {
                    return;
                }
                // Plain wheel → brush radius. Ctrl+wheel → camera zoom
                // anchored at the cursor (cell under the cursor stays
                // under the cursor across the zoom).
                let raw_y: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 30.0,
                };
                if state.ctrl_held {
                    if let Some((px, py)) = state.cursor_pos {
                        let factor = if raw_y > 0.0 { 1.15 }
                                     else if raw_y < 0.0 { 1.0 / 1.15 }
                                     else { 1.0 };
                        state.zoom_at(px as f32, py as f32, factor);
                    }
                } else {
                    let dir = if raw_y > 0.0 { 1 }
                              else if raw_y < 0.0 { -1 }
                              else { 0 };
                    state.brush_radius = (state.brush_radius + dir).clamp(1, 30);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if egui_wants_keyboard { return; }
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    // Track Ctrl (both sides) for wheel-zoom gating.
                    if matches!(code, KeyCode::ControlLeft | KeyCode::ControlRight) {
                        state.ctrl_held = pressed;
                        return;
                    }
                    // Everything else fires on press only.
                    if !pressed { return; }
                    if let Some(el) = GpuState::element_for_key(code) {
                        state.selected = el;
                        return;
                    }
                    // Arrow keys / WASD pan the camera by ~12% of the
                    // visible window each press. Sensitive to zoom —
                    // smaller cells = bigger jumps, since you cover
                    // more world per keypress at lower zoom.
                    let pan_step_px = 0.12;
                    let win_w = state.surface_config.width as f32;
                    let win_h = state.surface_config.height as f32;
                    match code {
                        KeyCode::Space => state.paused = !state.paused,
                        KeyCode::Backspace => state.camera_reset(),
                        KeyCode::ArrowLeft | KeyCode::KeyA => {
                            state.pan_pixels(win_w * pan_step_px, 0.0);
                        }
                        KeyCode::ArrowRight | KeyCode::KeyD => {
                            state.pan_pixels(-win_w * pan_step_px, 0.0);
                        }
                        KeyCode::ArrowUp | KeyCode::KeyW => {
                            state.pan_pixels(0.0, win_h * pan_step_px);
                        }
                        KeyCode::ArrowDown | KeyCode::KeyS => {
                            state.pan_pixels(0.0, -win_h * pan_step_px);
                        }
                        KeyCode::KeyC => {
                            // Defer the actual clear until render() so it
                            // happens AFTER motion readback (which would
                            // otherwise resurrect the cleared cells).
                            state.pending_clear = true;
                        }
                        KeyCode::Tab => {
                            state.pt_open = !state.pt_open;
                        }
                        KeyCode::Escape => {
                            if state.pt_open {
                                state.pt_open = false;
                            } else {
                                event_loop.exit();
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

/// Sim display shader — letterboxes the sim texture inside the
/// window so it keeps its native W:H aspect ratio, with a thin
/// cursor outline drawn in screen-pixel space (true circle).
const SIM_DISPLAY_SHADER: &str = r#"
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

struct Display {
    sim_min_x: f32,
    sim_min_y: f32,
    sim_max_x: f32,
    sim_max_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    brush_pixel_radius: f32,
    visible: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(pos[vid], 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var sim_tex: texture_2d<f32>;
@group(0) @binding(1) var sim_samp: sampler;
@group(0) @binding(2) var<uniform> disp: Display;

const BG_COLOR: vec4<f32> = vec4<f32>(0.012, 0.012, 0.024, 1.0);

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = frag.x;
    let py = frag.y;
    var color: vec4<f32>;
    // Inside the sim rect → sample the texture using normalized
    // coordinates relative to the rect. Outside → background fill.
    if (px >= disp.sim_min_x && px <= disp.sim_max_x
        && py >= disp.sim_min_y && py <= disp.sim_max_y) {
        let u = (px - disp.sim_min_x) / (disp.sim_max_x - disp.sim_min_x);
        let v = (py - disp.sim_min_y) / (disp.sim_max_y - disp.sim_min_y);
        color = textureSample(sim_tex, sim_samp, vec2<f32>(u, v));
    } else {
        color = BG_COLOR;
    }
    // Cursor outline: thin ring at brush_pixel_radius around cursor.
    // Computed in framebuffer pixels so it's a true circle regardless
    // of the sim's letterbox shape. Thickness = ~1.5 pixels at any zoom.
    if (disp.visible > 0.5) {
        let dx = px - disp.cursor_x;
        let dy = py - disp.cursor_y;
        let dist = sqrt(dx * dx + dy * dy);
        let on_ring = abs(dist - disp.brush_pixel_radius) < 0.75;
        if (on_ring) {
            color = mix(color, vec4<f32>(1.0, 1.0, 1.0, 1.0), 0.7);
        }
    }
    return color;
}
"#;

/// Entry point used by `wgpu_main.rs`. Sets up the winit event loop
/// and runs until the window is closed.
pub fn run() {
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("FATAL: event loop init failed: {e:?}");
            std::process::exit(1);
        }
    };
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("FATAL: event loop terminated with error: {e:?}");
        std::process::exit(1);
    }
}
