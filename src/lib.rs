pub mod app;
pub mod config;
mod galaxy;


use std::env;


pub fn parse_args() -> Option<String> {
  let args: Vec<String> = env::args().collect();
  let program_name = &args[0];
  let mut i = 1;
  let mut config_path = None;

  while i < args.len() {
    match args[i].as_str() {
      "-h" | "--help" => {
        print_help(program_name);
        std::process::exit(0);
      }
      "-c" | "--config" => {
        if i + 1 < args.len() {
          config_path = Some(args[i + 1].clone());
          i += 1;
        } else {
          eprintln!("Error: Missing value after {}", args[i]);
          std::process::exit(1);
        }
      }
      _ => {
        eprintln!("Unknown argument: {}", args[i]);
        std::process::exit(1);
      }
    }
    i += 1;
  }

  config_path
}

fn print_help(program_name: &str) {
  println!("Usage: {} [OPTIONS]", program_name);
  println!();
  println!("Options:");
  println!("  -c, --config <FILE>    Path to config file");
  println!("  -h, --help             Print this help message");
}

