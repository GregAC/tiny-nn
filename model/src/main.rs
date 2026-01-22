mod cli;
mod config;
mod executor;
mod fsm;
mod input_source;
mod ops;
mod test_data;
mod tnn_types;
mod utils;

use clap::Parser;
use std::path::Path;

use cli::{Cli, Commands};
use config::load_config;
use executor::{execute_operation, ExecutionMode};
use fsm::TnnSimulator;
use input_source::HexFileInput;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { config, output_dir } => {
            run_operations(&config, &output_dir, ExecutionMode::Generate);
        }
        Commands::Simulate { input, output } => {
            run_simulation(&input, &output);
        }
    }
}

fn run_simulation(input_path: &Path, output_path: &Path) {
    println!("Running FSM simulation");
    println!("  Input:  {:?}", input_path);
    println!("  Output: {:?}", output_path);

    // Load input
    let mut input = HexFileInput::from_file(input_path)
        .expect(&format!("Failed to read input file: {:?}", input_path));

    // Run simulation
    let mut simulator = TnnSimulator::new();
    simulator.run(&mut input);

    // Write output
    simulator
        .write_output(output_path)
        .expect(&format!("Failed to write output file: {:?}", output_path));

    println!("Done! Wrote {} output bytes", simulator.output().len());
}

fn run_operations(config_path: &Path, output_dir: &Path, mode: ExecutionMode) {
    let config = load_config(config_path);

    println!(
        "Running {} operation(s) in {:?} mode",
        config.operation.len(),
        match mode {
            ExecutionMode::Generate => "generate",
            ExecutionMode::Simulate => "simulate",
        }
    );
    println!("Output directory: {:?}", output_dir);
    println!();

    // Ensure output directory exists
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)
            .expect(&format!("Failed to create output directory: {:?}", output_dir));
    }

    for op in &config.operation {
        execute_operation(op, output_dir, mode);
        println!();
    }

    println!("Done!");
}
