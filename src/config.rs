use serde::{Serialize, Deserialize};

use crate::app::MyVertex;
use crate::galaxy;


pub trait Vertices {
  fn vertices(&self) -> Vec<MyVertex> where Self: Sized;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
  pub constants: Constants,
  pub vertexes: Vec<VertexSource>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Constants {
  pub point_size: f32,
  pub min_mass: f32,
  pub max_mass: f32,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum VertexSource {
  Point(Point),
  Galaxy(Galaxy),
}

impl Vertices for VertexSource {
  fn vertices(&self) -> Vec<MyVertex> where Self: Sized {
    match self {
      VertexSource::Point(p) => p.vertices(),
      VertexSource::Galaxy(g) => g.vertices(),
    }
  }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Point {
  pos: [f32; 2],
  vel: [f32; 2],
  mass: f32,
}

impl Vertices for Point {
  fn vertices(&self) -> Vec<MyVertex> {
    let vert = MyVertex::new(
      self.pos,
      self.vel,
      self.mass,
    );

    vec![vert]
  }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Galaxy {
  particles: u64,
  arms: u8,
  twist: f32,
  max_radius: f32,
  safety: f32,
  center_pos: [f32; 2],
  center_vel: [f32; 2],
  center_mass: f32,
}

impl Vertices for Galaxy {
  fn vertices(&self) -> Vec<MyVertex> {
    galaxy::generate(
      self.particles as usize,
      self.arms.into(),
      self.twist,
      self.max_radius,
      self.safety,
      self.center_pos,
      self.center_vel,
      self.center_mass,
    )
  }
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

