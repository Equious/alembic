use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Elements Sandbox".to_owned(),
        window_width: 1280,
        window_height: 1024,
        window_resizable: false,
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    alembic::run_game().await
}
