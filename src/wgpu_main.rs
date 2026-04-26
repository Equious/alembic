//! v0.3 wgpu binary entry — opens an alembic window backed by wgpu
//! instead of macroquad. Runs in parallel with the macroquad binary
//! (`alembic`) until the migration is complete.

fn main() {
    alembic::gpu_app::run();
}
