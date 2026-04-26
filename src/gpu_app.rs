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
// vec4 per element: x = kind_id, y = density, z = falling, w = is_liquid
@group(0) @binding(2) var<uniform> motion_props: array<vec4<f32>, 96>;

const KIND_EMPTY: u32  = 0u;
const KIND_SOLID: u32  = 1u;
const KIND_GRAVEL: u32 = 2u;
const KIND_POWDER: u32 = 3u;
const KIND_LIQUID: u32 = 4u;
const KIND_GAS: u32    = 5u;
const KIND_FIRE: u32   = 6u;

const FLAG_FROZEN: u32 = 0x02u;

fn cell_idx(x: u32, y: u32) -> u32 { return y * u.width + x; }
// Cell layout (16 bytes, repr(C)):
//   x.byte0 = el, x.byte1 = derived_id, x.bytes2-3 = life
//   y.byte0 = seed, y.byte1 = flag, y.bytes2-3 = temp
//   z.byte0 = moisture, z.byte1 = burn, z.bytes2-3 = pressure
//   w.byte0 = solute_el, w.byte1 = solute_amt, w.byte2 = solute_derived_id
fn cell_el(c: vec4<u32>) -> u32 { return c.x & 0xFFu; }
fn cell_flag(c: vec4<u32>) -> u32 { return (c.y >> 8u) & 0xFFu; }
fn cell_kind(c: vec4<u32>) -> u32 { return u32(motion_props[cell_el(c)].x); }
fn cell_density(c: vec4<u32>) -> f32 { return motion_props[cell_el(c)].y; }
fn cell_frozen(c: vec4<u32>) -> bool { return (cell_flag(c) & FLAG_FROZEN) != 0u; }

// Pass 0 — vertical fall: one thread per column, bottom-up walk.
// When sand at row Y swaps with empty at Y+1, the next iteration
// (row Y-1) sees the now-empty Y and can fall into it. Cascading is
// automatic, so a column shifts down by one per frame without
// fragmenting (which is what Margolus block updates produced).
//
// Adjacent columns are independent — straight vertical fall doesn't
// cross columns — so per-column threads have no race conditions.
fn vertical_fall(x: u32) {
    var y = i32(u.height) - 2;
    loop {
        if (y < 0) { break; }
        let i_here = cell_idx(x, u32(y));
        let i_below = cell_idx(x, u32(y + 1));
        let c_here = cells[i_here];
        let c_below = cells[i_below];

        if (!cell_frozen(c_here) && !cell_frozen(c_below)) {
            let kh = cell_kind(c_here);
            let kb = cell_kind(c_below);
            let dh = cell_density(c_here);
            let db = cell_density(c_below);

            let h_falls = (kh == KIND_POWDER || kh == KIND_GRAVEL || kh == KIND_LIQUID);
            let b_open  = (kb == KIND_EMPTY || kb == KIND_GAS || kb == KIND_FIRE);

            if (h_falls && b_open && dh > db) {
                cells[i_here]  = c_below;
                cells[i_below] = c_here;
            }
        }
        y = y - 1;
    }
}

// Pass 2/3 — diagonal slide for powder/gravel/liquid. Each thread is
// one column, walks bottom-up. For cells where straight-down was
// blocked (sand-on-sand from vfall), try diagonally — this is what
// turns a vertical sand pillar into a triangular pile.
//
// Race-free via column parity: pass 2 only acts on even columns,
// pass 3 only acts on odd. Each pass writes to its own column AND
// the diagonal target column, but in pass 2 the diagonal target is
// ODD (even+1 or even-1), and odd columns aren't running — so no
// races. Same logic in pass 3.
//
// Direction alternates by frame parity to avoid one-sided drift.
fn diagonal_slide(x: u32, parity: u32) {
    if ((x & 1u) != parity) { return; }
    let dir: i32 = select(-1, 1, (u.frame & 1u) == 0u);
    let nx = i32(x) + dir;
    if (nx < 0 || nx >= i32(u.width)) { return; }

    var y = i32(u.height) - 2;
    loop {
        if (y < 0) { break; }
        let i_here = cell_idx(x, u32(y));
        let i_below = cell_idx(x, u32(y + 1));
        let i_diag  = cell_idx(u32(nx), u32(y + 1));
        let c_here  = cells[i_here];
        let c_below = cells[i_below];
        let c_diag  = cells[i_diag];

        if (!cell_frozen(c_here) && !cell_frozen(c_diag)) {
            let kh = cell_kind(c_here);
            let kb = cell_kind(c_below);
            let kd = cell_kind(c_diag);
            let h_slides = (kh == KIND_POWDER || kh == KIND_GRAVEL || kh == KIND_LIQUID);
            let b_blocks = !(kb == KIND_EMPTY || kb == KIND_GAS || kb == KIND_FIRE);
            let d_open   = (kd == KIND_EMPTY || kd == KIND_GAS || kd == KIND_FIRE);
            // Only the TOP cell of a pile column may slide. If there's
            // another sand-like cell above us, sliding here would leave
            // a gap that vfall couldn't fill until next frame — visible
            // as horizontal voids running through the pile. With this
            // check, internal cells stay put while the surface slides.
            var above_blocks = false;
            if (y > 0) {
                let i_above = cell_idx(x, u32(y - 1));
                let ka = cell_kind(cells[i_above]);
                above_blocks = (ka == KIND_POWDER || ka == KIND_GRAVEL || ka == KIND_LIQUID);
            }
            if (h_slides && b_blocks && d_open && !above_blocks) {
                cells[i_here] = c_diag;
                cells[i_diag] = c_here;
            }
        }
        y = y - 1;
    }
}

// Pass 1/2 — liquid horizontal spread. Each thread handles one row,
// swapping liquid with adjacent empty/gas if the cell below is solid
// (so liquids spread on top of a floor instead of refusing to fall).
//
// Race fix vs the previous single-pass version: thread y reads
// row y+1 for the support check while thread y+1 might be writing
// it. Split into even-row and odd-row sub-passes (parity 0 / 1)
// so adjacent rows never run simultaneously.
fn liquid_spread(y: u32, parity: u32) {
    if ((y & 1u) != parity) { return; }
    let lr = (u.frame & 1u) == 0u;
    let w_i = i32(u.width);
    var x: i32;
    var step: i32;
    var x_end: i32;
    if (lr) { x = 0; step = 1; x_end = w_i - 1; }
    else    { x = w_i - 1; step = -1; x_end = 0; }
    loop {
        if (lr) { if (x >= x_end) { break; } }
        else    { if (x <= x_end) { break; } }
        let nx = x + step;
        let i_a = cell_idx(u32(x),  y);
        let i_b = cell_idx(u32(nx), y);
        let c_a = cells[i_a];
        let c_b = cells[i_b];
        if (!cell_frozen(c_a) && !cell_frozen(c_b)) {
            let ka = cell_kind(c_a);
            let kb = cell_kind(c_b);
            // Only spread if cell BELOW is solid/liquid (i.e. liquid
            // is sitting on something — otherwise vertical fall handles it).
            // Cheap proxy: just check directly-below cell at this y+1.
            let i_below = cell_idx(u32(x), y + 1u);
            let c_below = cells[i_below];
            let kbel = cell_kind(c_below);
            let supported = (kbel != KIND_EMPTY && kbel != KIND_GAS && kbel != KIND_FIRE);
            if (supported && ka == KIND_LIQUID && (kb == KIND_EMPTY || kb == KIND_GAS)) {
                cells[i_a] = c_b;
                cells[i_b] = c_a;
            }
        }
        x = x + step;
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
    pass_uniform_bufs: [wgpu::Buffer; 5],
    #[allow(dead_code)]
    motion_props_buf: wgpu::Buffer,
    readback_bufs: [wgpu::Buffer; 2],
    pass_bind_groups: [wgpu::BindGroup; 5],
    write_idx: usize,
    has_data: [bool; 2],
}

impl MotionComputeCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, cells_buf: &wgpu::Buffer) -> Self {
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
        let pass_uniform_bufs: [wgpu::Buffer; 5] = [
            mk_pass_uniform(0, "alembic-motion-uniforms-vfall"),
            mk_pass_uniform(1, "alembic-motion-uniforms-lspread-even"),
            mk_pass_uniform(2, "alembic-motion-uniforms-lspread-odd"),
            mk_pass_uniform(3, "alembic-motion-uniforms-dslide-even"),
            mk_pass_uniform(4, "alembic-motion-uniforms-dslide-odd"),
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
            ],
        });
        let mk_bind = |i: usize| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-motion-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniform_bufs[i].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cells_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: motion_props_buf.as_entire_binding() },
            ],
        });
        let pass_bind_groups: [wgpu::BindGroup; 5] = [mk_bind(0), mk_bind(1), mk_bind(2), mk_bind(3), mk_bind(4)];
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

    /// Encode all 5 motion passes for one frame, then snapshot the
    /// shared `cells_buf` into the readback buffer for next-frame
    /// CPU sync.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, frame: u32, cells_buf: &wgpu::Buffer) {
        let cell_count = W * H;
        let frame_arr = [frame];
        let frame_bytes = bytemuck::cast_slice(&frame_arr);
        for i in 0..5 {
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
    image_buffer: Vec<u8>,
    /// GPU texture mirroring `image_buffer`. Bound to the fragment
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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

        let image_buffer = vec![0u8; W * H * 4];

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

        let pressure_compute = PressureComputeCtx::new(&device, &queue, &cells_buf);
        let thermal_compute = ThermalComputeCtx::new(&device, &queue, &cells_buf);
        let pressure_sources_compute = PressureSourcesCtx::new(&device, &queue, &cells_buf);
        let motion_compute = MotionComputeCtx::new(&device, &queue, &cells_buf);

        let mut state = GpuState {
            surface,
            surface_config,
            device,
            queue,
            world,
            image_buffer,
            sim_texture,
            sim_bind_group,
            sim_pipeline,
            display_uniform,
            cells_buf,
            pressure_compute,
            thermal_compute,
            pressure_sources_compute,
            motion_compute,
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
            self.world.step_skip_gpu_passes_and_motion(macroquad::math::Vec2::new(0.0, 0.0));
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
            let amb = self.world.ambient_offset;
            self.thermal_compute.update_frame(&self.queue, self.frame_counter, amb);
            if run_ps {
                self.pressure_sources_compute.update_frame(&self.queue, &self.world);
            }

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("alembic-combined-compute-encoder"),
            });
            // PS: column scan + asymmetric blend. Writes new pressure
            // straight into cells_buf, so the pressure diffusion below
            // sees post-PS values when it extracts.
            if run_ps {
                self.pressure_sources_compute.encode(&mut encoder);
            }
            // Pressure: extract → 3 diffuse iters → writeback.
            self.pressure_compute.encode(&mut encoder);
            // Thermal: extract → 1 diffuse iter → writeback.
            self.thermal_compute.encode(&mut encoder);
            // Motion: 4 passes (vfall, lspread, dslide-even, dslide-odd).
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

        // Fill the CPU pixel buffer from cell colors. Same path as the
        // legacy macroquad render loop, just writing to our local Vec
        // instead of a macroquad Image.
        for i in 0..(W * H) {
            let c = self.world.cells[i];
            if c.el == Element::Empty {
                let base = i * 4;
                self.image_buffer[base]     = 2;
                self.image_buffer[base + 1] = 2;
                self.image_buffer[base + 2] = 6;
                self.image_buffer[base + 3] = 255;
                continue;
            }
            let [r, g, b] = color_rgb(c);
            let base = i * 4;
            self.image_buffer[base]     = r;
            self.image_buffer[base + 1] = g;
            self.image_buffer[base + 2] = b;
            self.image_buffer[base + 3] = 255;
        }

        // Upload to GPU texture.
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.sim_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.image_buffer,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * W as u32),
                rows_per_image: Some(H as u32),
            },
            wgpu::Extent3d {
                width: W as u32,
                height: H as u32,
                depth_or_array_layers: 1,
            },
        );

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
                        KeyCode::Escape => event_loop.exit(),
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
