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

use crate::{color_rgb, Element, World, H, W};

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
        let cx = W as i32 / 2;
        let floor_y = H as i32 - 30;
        world.paint(cx, floor_y, 12, Element::Sand, 0, false);
        world.paint(cx - 60, floor_y - 80, 6, Element::Water, 0, false);
        world.paint(cx + 50, floor_y - 5, 4, Element::Stone, 0, true);

        let image_buffer = vec![0u8; W * H * 4];

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

    /// Reset camera to the default fit-to-window view (sim centered,
    /// scale = fit_scale). Bound to Backspace.
    fn camera_reset(&mut self) {
        self.cam_center_x = W as f32 * 0.5;
        self.cam_center_y = H as f32 * 0.5;
        self.cam_scale = self.fit_scale();
    }

    /// Cursor-anchored zoom. The cell currently under the cursor stays
    /// under the cursor across the zoom — the camera center shifts to
    /// preserve that invariant.
    fn zoom_at(&mut self, screen_x: f32, screen_y: f32, factor: f32) {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        // Cell under cursor at the *current* scale.
        let cell_x = self.cam_center_x + (screen_x - win_w * 0.5) / self.cam_scale;
        let cell_y = self.cam_center_y + (screen_y - win_h * 0.5) / self.cam_scale;
        let fit = self.fit_scale();
        // Bound to a reasonable range — too small loses the sim in a
        // sea of void; too large makes individual cells fill the screen.
        let new_scale = (self.cam_scale * factor).clamp(fit * 0.25, fit * 16.0);
        // After zoom, solve for cam_center such that cell_x maps to screen_x.
        self.cam_center_x = cell_x - (screen_x - win_w * 0.5) / new_scale;
        self.cam_center_y = cell_y - (screen_y - win_h * 0.5) / new_scale;
        self.cam_scale = new_scale;
    }

    /// Pan the camera by a screen-pixel delta. Used by middle-mouse drag.
    fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.cam_center_x -= dx / self.cam_scale;
        self.cam_center_y -= dy / self.cam_scale;
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

        // Tick the CPU sim unless paused.
        if !self.paused {
            self.world.step(macroquad::math::Vec2::new(0.0, 0.0));
        }

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
                    match code {
                        KeyCode::Space => state.paused = !state.paused,
                        KeyCode::Backspace => state.camera_reset(),
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
