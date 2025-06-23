use vulkan_n_body_simulation::{
  config,
  app,
  parse_args,
};

use winit::event_loop::EventLoop;

use std::error::Error;


fn main() -> Result<(), impl Error> {
  let config_path = parse_args().unwrap_or("./examples/two_galaxies/config.toml".into());
  let config = config::Config::new(&config_path);
  let event_loop = EventLoop::new().unwrap();
  let mut app = app::App::new(&event_loop, config);

  event_loop.run_app(&mut app)
}

