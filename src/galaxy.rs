use rand_distr::{Distribution, Normal, Uniform};

use std::f32::consts::TAU;

use crate::app::MyVertex;


const G: f32 = 6.67408E-11;


pub fn generate(
  amount: usize,
  arms: u32,
  twist: f32,
  max_radius: f32,
  safety: f32,
  center_pos: [f32; 2],
  center_vel: [f32; 2],
  center_mass: f32,
) -> Vec<MyVertex> {
  let mut rng = rand::rng();
  let mut particles = Vec::with_capacity(amount);

  let core_count = amount / 5;
  let norm = Normal::new(0.0, max_radius * 0.2).unwrap();
  for _ in 0..core_count {
    let r = norm.sample(&mut rng).abs().min(max_radius);
    let theta = Uniform::new(0.0, TAU).unwrap().sample(&mut rng);

    let x = center_pos[0] + r * theta.cos();
    let y = center_pos[1] + r * theta.sin();

    let speed = ((G * center_mass * (r as f32)) / (r * r + safety)).sqrt();
    let vx = -center_vel[0] + speed * theta.sin();
    let vy = -center_vel[1] - speed * theta.cos();
    let mass = Uniform::new(1.8E5, 2.2E5).unwrap().sample(&mut rng);

    particles.push(MyVertex::new([x, y], [vx, vy], mass));
  }

  let arm_count = amount - core_count;
  let radius_dist = Normal::new(0.0, max_radius * 0.5).unwrap();
  for _ in 0..arm_count {
    let arm_idx = Uniform::new(0, arms).unwrap().sample(&mut rng) as f32;
    let base_angle = arm_idx / (arms as f32) * TAU;
    let r = radius_dist.sample(&mut rng).abs().min(max_radius);
    let theta = base_angle + twist * r + Uniform::new(-0.1, 0.1).unwrap().sample(&mut rng);

    let x = center_pos[0] + r * theta.cos();
    let y = center_pos[1] + r * theta.sin();

    let speed = ((G * center_mass * (r as f32)) / (r * r + safety)).sqrt();
    let vx = -center_vel[0] + speed * theta.sin();
    let vy = -center_vel[1] - speed * theta.cos();
    let mass = Uniform::new(0.9E5, 1.2E5).unwrap().sample(&mut rng);

    particles.push(MyVertex::new([x, y], [vx, vy], mass));
  }

  particles
}

