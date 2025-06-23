use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
  pub constants: Constants,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Constants {
  pub point_size: f32,
  pub min_mass: f32,
  pub max_mass: f32,
}

impl Config {
  pub fn new(path: &str) -> Self {
    let config: Config = config::Config::builder()
      .add_source(config::File::new(path, config::FileFormat::Toml))
      .build()
      .expect("Could not find the config file")
      .try_deserialize()
      .expect("Could not parse the config file");

    config
  }
}

