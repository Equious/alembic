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
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::{color_rgb, Element, World, H, W};

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
    /// Window must outlive the surface. Held as Arc so the surface's
    /// 'static lifetime contract is satisfied without unsafe.
    window: Arc<Window>,
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
            ],
        });
        let sim_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alembic-sim-bg"),
            layout: &sim_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&sim_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
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

        GpuState {
            surface,
            surface_config,
            device,
            queue,
            world,
            image_buffer,
            sim_texture,
            sim_bind_group,
            sim_pipeline,
            window,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn render(&mut self) {
        // Tick the CPU sim. Wind is zero for now — input wiring lands
        // in a later phase.
        self.world.step(macroquad::math::Vec2::new(0.0, 0.0));

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
            _ => {}
        }
    }
}

/// Sim display shader — fullscreen triangle samples the W×H sim
/// texture. UVs map to [0,1]² with origin at the top-left to match
/// our cell indexing (`cells[y * W + x]`). Out-of-range pixels go
/// to a panel-grey background so the sim stays in its proper
/// aspect ratio when the window is wider/taller than W:H.
const SIM_DISPLAY_SHADER: &str = r#"
const SIM_W: f32 = 320.0;
const SIM_H: f32 = 315.0;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Standard fullscreen triangle trick.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    // UVs with origin top-left (y inverted vs clip space).
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

@group(0) @binding(0) var sim_tex: texture_2d<f32>;
@group(0) @binding(1) var sim_samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // For now: stretch the sim across the whole window. Letterbox
    // logic comes later when we wire up the side panel.
    return textureSample(sim_tex, sim_samp, in.uv);
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
