mod cli;
mod config;
mod executor;
mod fsm;
mod input_source;
mod network;
mod ops;
mod test_data;
mod tnn_types;
mod utils;

use clap::Parser;
use std::path::Path;

use cli::{Cli, Commands};
use config::load_config;
use executor::execute_operation;
use fsm::TnnSimulator;
use input_source::HexFileInput;
use network::TnnNetworkServer;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { config, output_dir } => {
            run_generate(&config, &output_dir);
        }
        Commands::Simulate { input, output } => {
            run_simulation(&input, &output);
        }
        Commands::Serve { port, host } => {
            run_server(&host, port);
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

fn run_generate(config_path: &Path, output_dir: &Path) {
    let config = load_config(config_path);

    println!(
        "Generating test vectors for {} operation(s)",
        config.operation.len()
    );
    println!("Output directory: {:?}", output_dir);
    println!();

    // Ensure output directory exists
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)
            .expect(&format!("Failed to create output directory: {:?}", output_dir));
    }

    for op in &config.operation {
        execute_operation(op, output_dir);
        println!();
    }

    println!("Done!");
}

fn run_server(host: &str, port: u16) {
    let addr = format!("{}:{}", host, port);
    println!("Starting TNN simulation server on {}", addr);

    let server = TnnNetworkServer::bind(&addr)
        .expect(&format!("Failed to bind to {}", addr));

    println!("Server listening on {}", server.local_addr().unwrap());

    // Accept and run connections in a loop
    loop {
        if let Err(e) = server.accept_and_run() {
            eprintln!("Connection error: {}", e);
        }
        println!("Connection closed, waiting for next connection...");
    }
}
