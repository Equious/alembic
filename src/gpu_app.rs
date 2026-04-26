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

use crate::{color_rgb, pressure_source_props, thermal_profile_vec4, Element, World, H, W};

const THERMAL_COMPUTE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    ambient_offset: i32,
    frame: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> temp_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> temp_out: array<i32>;
@group(0) @binding(3) var<storage, read> el: array<u32>;
// Per-element thermal profile, indexed by el[i].
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

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    if (x >= w_i || y >= h_i) { return; }

    let i = u32(y * w_i + x);
    let me_el = el[i];
    let me_props = profiles[me_el];
    let my_k = me_props.x;
    let me_t = f32(temp_in[i]);

    var delta: f32 = 0.0;
    var diff_neighbors: f32 = 0.0;
    var oob_neighbors: f32 = 0.0;
    // 4 neighbors (cardinals).
    if (x > 0) {
        let ni = u32(y * w_i + (x - 1));
        let n_el = el[ni];
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (x < w_i - 1) {
        let ni = u32(y * w_i + (x + 1));
        let n_el = el[ni];
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (y > 0) {
        let ni = u32((y - 1) * w_i + x);
        let n_el = el[ni];
        let n_k = profiles[n_el].x;
        let n_t = f32(temp_in[ni]);
        let k = min(my_k, n_k);
        delta += k * (n_t - me_t);
        if (n_el != me_el) { diff_neighbors += 1.0; }
    } else { oob_neighbors += 1.0; }
    if (y < h_i - 1) {
        let ni = u32((y + 1) * w_i + x);
        let n_el = el[ni];
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
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pressure_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> pressure_out: array<i32>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;

const DIFF_SCALE: i32 = 2048;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w_i = i32(u.width);
    let h_i = i32(u.height);
    if (x >= w_i || y >= h_i) { return; }
    let i = u32(y * w_i + x);
    let me_perm = i32(perm[i]);
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
        let n_perm = i32(perm[ni]);
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    } else {
        new_p += (-me_p) * min(me_perm, 255) / DIFF_SCALE;
    }
    // RIGHT
    if (x < w_i - 1) {
        let ni = u32(y * w_i + (x + 1));
        let n_p = pressure_in[ni];
        let n_perm = i32(perm[ni]);
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    } else {
        new_p += (-me_p) * min(me_perm, 255) / DIFF_SCALE;
    }
    // UP — vertical OOB sealed (skip neighbor entirely).
    if (y > 0) {
        let ni = u32((y - 1) * w_i + x);
        let n_p = pressure_in[ni];
        let n_perm = i32(perm[ni]);
        let mp = min(me_perm, n_perm);
        if (mp > 0) { new_p += (n_p - me_p) * mp / DIFF_SCALE; }
    }
    // DOWN
    if (y < h_i - 1) {
        let ni = u32((y + 1) * w_i + x);
        let n_p = pressure_in[ni];
        let n_perm = i32(perm[ni]);
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

/// GPU compute pipeline + storage buffers for pressure diffusion. Replaces
/// the CPU `World::pressure()` pass — same numerics (4-neighbor flux at
/// DIFF_SCALE=2048, ITERS=3 by default), runs on the GPU as a compute
/// shader on storage buffers. Sync readback into `cell.pressure` at the
/// end so downstream CPU passes see the diffused field.
struct PressureComputeCtx {
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    pressure_a: wgpu::Buffer,
    pressure_b: wgpu::Buffer,
    perm_buf: wgpu::Buffer,
    /// Double-buffered readback. Frame N writes to readback_bufs[write_idx]
    /// and queues a map_async; frame N+1 reads from the same buffer (now
    /// mapped after a single shared device.poll). Removes the per-frame
    /// GPU sync stall — by the time we want the data, it's already
    /// finished computing during the inter-frame interval.
    readback_bufs: [wgpu::Buffer; 2],
    bind_a_to_b: wgpu::BindGroup,
    bind_b_to_a: wgpu::BindGroup,
    pressure_staging: Vec<i32>,
    perm_staging: Vec<u32>,
    iters: u32,
    write_idx: usize,
    has_data: [bool; 2],
}

impl PressureComputeCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cell_count = W * H;
        let buf_bytes = (cell_count * 4) as wgpu::BufferAddress; // i32 / u32 = 4 bytes

        let uniforms = ComputeUniforms {
            width: W as u32,
            height: H as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-pressure-compute-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Force initial uniform upload.
        queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[uniforms]));

        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let pressure_a = make_storage("alembic-pressure-a");
        let pressure_b = make_storage("alembic-pressure-b");
        let perm_buf = make_storage("alembic-perm");
        let make_readback = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let readback_bufs = [
            make_readback("alembic-pressure-readback-0"),
            make_readback("alembic-pressure-readback-1"),
        ];

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-pressure-compute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bind-a-to-b"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: perm_buf.as_entire_binding() },
            ],
        });
        let bind_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-pressure-bind-b-to-a"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: perm_buf.as_entire_binding() },
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
            uniform_buf,
            pressure_a,
            pressure_b,
            perm_buf,
            readback_bufs,
            bind_a_to_b,
            bind_b_to_a,
            pressure_staging: vec![0i32; cell_count],
            perm_staging: vec![0u32; cell_count],
            iters: 3,
            write_idx: 0,
            has_data: [false; 2],
        }
    }

    fn stage_and_upload(&mut self, world: &World, queue: &wgpu::Queue) {
        let cell_count = W * H;
        for i in 0..cell_count {
            self.pressure_staging[i] = world.cells[i].pressure as i32;
            self.perm_staging[i] = world.cells[i].el.pressure_p().permeability as u32;
        }
        queue.write_buffer(&self.pressure_a, 0, bytemuck::cast_slice(&self.pressure_staging));
        queue.write_buffer(&self.perm_buf, 0, bytemuck::cast_slice(&self.perm_staging));
    }

    /// Upload only the permeability LUT — used when pressure_a is going
    /// to be filled by a GPU-side copy from the pressure_sources output
    /// instead of by a CPU upload.
    fn stage_perm_only(&mut self, world: &World, queue: &wgpu::Queue) {
        let cell_count = W * H;
        for i in 0..cell_count {
            self.perm_staging[i] = world.cells[i].el.pressure_p().permeability as u32;
        }
        queue.write_buffer(&self.perm_buf, 0, bytemuck::cast_slice(&self.perm_staging));
    }

    /// Borrow the pressure_a storage buffer so other ctxs (e.g.
    /// PressureSourcesCtx) can copy their output directly into the
    /// diffusion input slot — avoiding a CPU round-trip when chaining.
    fn pressure_a_buf(&self) -> &wgpu::Buffer { &self.pressure_a }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let cell_count = W * H;
        let wg_x = (W as u32 + 15) / 16;
        let wg_y = (H as u32 + 15) / 16;
        for iter in 0..self.iters {
            let bind = if iter % 2 == 0 { &self.bind_a_to_b } else { &self.bind_b_to_a };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-pressure-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, bind, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        let final_buf = if self.iters % 2 == 1 { &self.pressure_b } else { &self.pressure_a };
        encoder.copy_buffer_to_buffer(
            final_buf, 0, &self.readback_bufs[self.write_idx], 0,
            (cell_count * 4) as wgpu::BufferAddress,
        );
    }

    fn start_map(&mut self) {
        self.readback_bufs[self.write_idx]
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.has_data[self.write_idx] = true;
    }

    /// Read the OTHER buffer (filled last frame). Skips on the first
    /// frame when no prior data exists.
    fn read_back_prev_into(&mut self, world: &mut World) {
        let read_idx = 1 - self.write_idx;
        if !self.has_data[read_idx] { return; }
        let slice = self.readback_bufs[read_idx].slice(..);
        {
            let data = slice.get_mapped_range();
            let result: &[i32] = bytemuck::cast_slice(&data);
            for (i, &v) in result.iter().enumerate() {
                world.cells[i].pressure = v as i16;
            }
        }
        self.readback_bufs[read_idx].unmap();
        self.has_data[read_idx] = false;
    }

    fn advance_frame(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ThermalUniforms {
    width: u32,
    height: u32,
    ambient_offset: i32,
    frame: u32,
}

/// GPU compute pipeline for thermal diffusion (heat exchange + ambient
/// blend). Runs the inner per-cell math of `World::thermal_diffuse()`
/// in a fragment shader, leaving moisture/combustion/phase changes
/// (`thermal_post`) on the CPU. Saves the largest CPU-linear cost
/// after pressure was moved to GPU.
struct ThermalComputeCtx {
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    profiles_buf: wgpu::Buffer,
    temp_a: wgpu::Buffer,
    temp_b: wgpu::Buffer,
    el_buf: wgpu::Buffer,
    readback_bufs: [wgpu::Buffer; 2],
    bind_a_to_b: wgpu::BindGroup,
    bind_b_to_a: wgpu::BindGroup,
    temp_staging: Vec<i32>,
    el_staging: Vec<u32>,
    write_idx: usize,
    has_data: [bool; 2],
}

impl ThermalComputeCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cell_count = W * H;
        let buf_bytes = (cell_count * 4) as wgpu::BufferAddress;

        // Build the per-element thermal profile uniform (96 vec4s).
        let mut profile_data: Vec<[f32; 4]> = vec![[0.0, 20.0, 0.0, 1.0]; 96];
        for i in 0..96 {
            profile_data[i] = thermal_profile_vec4(i as u8);
        }
        let profiles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-thermal-profiles"),
            contents: bytemuck::cast_slice(&profile_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = ThermalUniforms {
            width: W as u32,
            height: H as u32,
            ambient_offset: 0,
            frame: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alembic-thermal-uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let _ = queue; // queue write happens per-frame in dispatch.

        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let temp_a = make_storage("alembic-temp-a");
        let temp_b = make_storage("alembic-temp-b");
        let el_buf = make_storage("alembic-el");
        let make_readback = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let readback_bufs = [
            make_readback("alembic-thermal-readback-0"),
            make_readback("alembic-thermal-readback-1"),
        ];

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
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bind-a-to-b"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: el_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: profiles_buf.as_entire_binding() },
            ],
        });
        let bind_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-thermal-bind-b-to-a"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: temp_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: el_buf.as_entire_binding() },
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
            uniform_buf,
            profiles_buf,
            temp_a,
            temp_b,
            el_buf,
            readback_bufs,
            bind_a_to_b,
            bind_b_to_a,
            temp_staging: vec![0i32; cell_count],
            el_staging: vec![0u32; cell_count],
            write_idx: 0,
            has_data: [false; 2],
        }
    }

    fn stage_and_upload(&mut self, world: &World, queue: &wgpu::Queue, frame: u32, ambient_offset: i16) {
        let cell_count = W * H;
        for i in 0..cell_count {
            self.temp_staging[i] = world.cells[i].temp as i32;
            self.el_staging[i] = world.cells[i].el as u32;
        }
        queue.write_buffer(&self.temp_a, 0, bytemuck::cast_slice(&self.temp_staging));
        queue.write_buffer(&self.el_buf, 0, bytemuck::cast_slice(&self.el_staging));
        let uniforms = ThermalUniforms {
            width: W as u32,
            height: H as u32,
            ambient_offset: ambient_offset as i32,
            frame,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let cell_count = W * H;
        let wg_x = (W as u32 + 15) / 16;
        let wg_y = (H as u32 + 15) / 16;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alembic-thermal-cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_a_to_b, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.temp_b, 0, &self.readback_bufs[self.write_idx], 0,
            (cell_count * 4) as wgpu::BufferAddress,
        );
        let _ = (&self.bind_b_to_a, &self.profiles_buf, &self.temp_a);
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
            let result: &[i32] = bytemuck::cast_slice(&data);
            for (i, &v) in result.iter().enumerate() {
                world.cells[i].temp = v.clamp(-273, 4000) as i16;
            }
        }
        self.readback_bufs[read_idx].unmap();
        self.has_data[read_idx] = false;
    }

    fn advance_frame(&mut self) {
        self.write_idx = 1 - self.write_idx;
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
@group(0) @binding(1) var<storage, read> el_in: array<u32>;
@group(0) @binding(2) var<storage, read> flag_in: array<u32>;
@group(0) @binding(3) var<storage, read> temp_in: array<i32>;
@group(0) @binding(4) var<storage, read> pressure_in: array<i32>;
@group(0) @binding(5) var<storage, read_write> pressure_out: array<i32>;
// vec4 per element: x = kind_id, y = weight, z = _, w = _
@group(0) @binding(6) var<uniform> profiles: array<vec4<f32>, 96>;

const FLAG_FROZEN: u32 = 0x02u;
const KIND_EMPTY: u32  = 0u;
const KIND_LIQUID: u32 = 4u;
const KIND_GAS: u32    = 5u;
const KIND_FIRE: u32   = 6u;

// One thread per column. Walks vertically computing the column-
// integrated hydrostatic pressure plus per-cell thermal target,
// then blends current pressure → target with the asymmetric rule
// (gas/fire only blend up; everything else blends both ways).
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
        let id = el_in[i];
        let prof = profiles[id];
        let kind_id = u32(prof.x);
        let weight = prof.y;

        let is_pressurizable = (kind_id == KIND_GAS || kind_id == KIND_FIRE);
        let is_wallable = (kind_id != KIND_EMPTY
            && kind_id != KIND_LIQUID
            && kind_id != KIND_GAS
            && kind_id != KIND_FIRE);
        let is_frozen = (flag_in[i] & FLAG_FROZEN) != 0u;

        // Pass 1: thermal target.
        var target: i32 = 0;
        if (is_pressurizable) {
            let t = (temp_in[i] - 20) * 5;
            target = clamp(t, -300, 4000);
        }

        // Pass 2: hydrostatic column integration.
        if (u.gravity_present != 0u && u.gy != 0) {
            if (is_frozen && is_wallable) {
                col_p = 0.0;
            } else {
                col_p = col_p + weight * u.gravity_mag;
                let p_c = i32(clamp(col_p, -4000.0, 4000.0));
                target = clamp(target + p_c, -4000, 4000);
            }
        }

        // Pass 3: asymmetric blend toward target.
        let current = pressure_in[i];
        let delta = target - current;
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
        pressure_out[i] = new_p;

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

/// GPU compute pipeline for the hydrostatic + thermal pressure target
/// pass (`World::pressure_sources`). Largest single CPU-bound pass at
/// 1200×900 (~9-11ms); has no race conditions (column scan + per-cell
/// blend), so it ports cleanly with one thread per column.
struct PressureSourcesCtx {
    pipeline: wgpu::ComputePipeline,
    uniform_buf: wgpu::Buffer,
    #[allow(dead_code)]
    profiles_buf: wgpu::Buffer,
    el_buf: wgpu::Buffer,
    flag_buf: wgpu::Buffer,
    temp_buf: wgpu::Buffer,
    pressure_in_buf: wgpu::Buffer,
    pressure_out_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    el_staging: Vec<u32>,
    flag_staging: Vec<u32>,
    temp_staging: Vec<i32>,
    pressure_staging: Vec<i32>,
}

impl PressureSourcesCtx {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cell_count = W * H;
        let buf_bytes = (cell_count * 4) as wgpu::BufferAddress;

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

        let make_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let el_buf = make_storage("alembic-ps-el");
        let flag_buf = make_storage("alembic-ps-flag");
        let temp_buf = make_storage("alembic-ps-temp");
        let pressure_in_buf = make_storage("alembic-ps-pressure-in");
        let pressure_out_buf = make_storage("alembic-ps-pressure-out");
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alembic-ps-bgl"),
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
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-ps-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: el_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flag_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: temp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pressure_in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: pressure_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: profiles_buf.as_entire_binding() },
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
            el_buf,
            flag_buf,
            temp_buf,
            pressure_in_buf,
            pressure_out_buf,
            bind_group,
            el_staging: vec![0u32; cell_count],
            flag_staging: vec![0u32; cell_count],
            temp_staging: vec![0i32; cell_count],
            pressure_staging: vec![0i32; cell_count],
        }
    }

    fn stage_and_upload(&mut self, world: &World, queue: &wgpu::Queue) {
        let cell_count = W * H;
        for i in 0..cell_count {
            let c = world.cells[i];
            self.el_staging[i] = c.el as u32;
            self.flag_staging[i] = c.flag as u32;
            self.temp_staging[i] = c.temp as i32;
            self.pressure_staging[i] = c.pressure as i32;
        }
        queue.write_buffer(&self.el_buf, 0, bytemuck::cast_slice(&self.el_staging));
        queue.write_buffer(&self.flag_buf, 0, bytemuck::cast_slice(&self.flag_staging));
        queue.write_buffer(&self.temp_buf, 0, bytemuck::cast_slice(&self.temp_staging));
        queue.write_buffer(&self.pressure_in_buf, 0, bytemuck::cast_slice(&self.pressure_staging));

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

    /// Dispatch only — no readback copy. The output buffer is consumed
    /// directly by the next compute stage (pressure diffusion) via a
    /// GPU-side copy, so we don't need to round-trip through CPU.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        // One thread per column; workgroup size 64.
        let wg_x = (W as u32 + 63) / 64;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("alembic-ps-cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, 1, 1);
    }

    /// Borrow the output buffer so the diffusion ctx can copy from it
    /// in the chained dispatch.
    fn pressure_out_buf(&self) -> &wgpu::Buffer { &self.pressure_out_buf }
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
    /// GPU compute pipeline for pressure diffusion. Replaces the CPU
    /// `World::pressure()` pass; lets us scale the grid without paying
    /// the linear CPU cost.
    pressure_compute: PressureComputeCtx,
    /// GPU compute pipeline for thermal diffusion (heat exchange +
    /// ambient blend). Replaces `World::thermal_diffuse()`.
    thermal_compute: ThermalComputeCtx,
    /// GPU compute pipeline for hydrostatic + thermal pressure target
    /// (`World::pressure_sources`). Largest single CPU pass; column
    /// scan is GPU-friendly with one thread per column.
    pressure_sources_compute: PressureSourcesCtx,
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

        let pressure_compute = PressureComputeCtx::new(&device, &queue);
        let thermal_compute = ThermalComputeCtx::new(&device, &queue);
        let pressure_sources_compute = PressureSourcesCtx::new(&device, &queue);

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
            pressure_compute,
            thermal_compute,
            pressure_sources_compute,
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
        // Apply paint/erase from any held mouse button before stepping
        // so the new cells participate in this tick.
        self.apply_brush();

        // Per-frame timing
        let t_sim_start = std::time::Instant::now();
        if !self.paused {
            self.world.step_skip_gpu_passes(macroquad::math::Vec2::new(0.0, 0.0));
        }
        let t_sim = t_sim_start.elapsed();
        let t_compute_start = std::time::Instant::now();
        if !self.paused {
            // Double-buffered compute. Sequence:
            //   1. poll() — drains last frame's GPU work + map_async
            //      callbacks (near-zero in steady state since the GPU
            //      had a full frame to finish during winit's vsync).
            //   2. read PREVIOUS frame's results from the buffer that
            //      was queued for map last frame.
            //   3. stage current frame's inputs (CPU → write_buffer).
            //   4. encode + submit current compute (writes to the
            //      OTHER buffer).
            //   5. queue map_async on that buffer for next frame.
            //   6. swap write_idx for both ctxs.
            let _ = self.device.poll(wgpu::Maintain::Wait);
            // Pressure flow: PS (every other frame) → GPU copy → DIFF.
            // Both effects are in the diffusion buffer's readback that
            // lands in world.cells[i].pressure. PS output never round-
            // trips through CPU — the chain is entirely GPU-side.
            let run_ps = self.frame_counter & 1 == 0;
            self.pressure_compute.read_back_prev_into(&mut self.world);
            self.thermal_compute.read_back_prev_into(&mut self.world);

            if run_ps {
                self.pressure_sources_compute.stage_and_upload(&self.world, &self.queue);
                // On PS frames, perm only — pressure_a will be filled
                // from PS output via GPU copy.
                self.pressure_compute.stage_perm_only(&self.world, &self.queue);
            } else {
                self.pressure_compute.stage_and_upload(&self.world, &self.queue);
            }
            let amb = self.world.ambient_offset;
            self.thermal_compute.stage_and_upload(&self.world, &self.queue, self.frame_counter, amb);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("alembic-combined-compute-encoder"),
            });
            if run_ps {
                self.pressure_sources_compute.encode(&mut encoder);
                // Chain: PS.pressure_out → DIFF.pressure_a (GPU copy).
                let bytes = (W * H * 4) as wgpu::BufferAddress;
                encoder.copy_buffer_to_buffer(
                    self.pressure_sources_compute.pressure_out_buf(), 0,
                    self.pressure_compute.pressure_a_buf(), 0,
                    bytes,
                );
            }
            self.pressure_compute.encode(&mut encoder);
            self.thermal_compute.encode(&mut encoder);
            self.queue.submit(std::iter::once(encoder.finish()));
            self.pressure_compute.start_map();
            self.thermal_compute.start_map();
            self.pressure_compute.advance_frame();
            self.thermal_compute.advance_frame();
            self.frame_counter = self.frame_counter.wrapping_add(1);
        }
        let t_compute = t_compute_start.elapsed();
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
                            for c in state.world.cells.iter_mut() {
                                if !c.is_frozen() {
                                    *c = crate::Cell::EMPTY;
                                }
                            }
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
