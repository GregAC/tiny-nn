use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tnn-model")]
#[command(about = "Tiny-NN Model CLI for test vector generation and simulation")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate test vectors (input .hex and expected output _expected.hex files)
    Generate {
        /// Path to TOML configuration file
        config: PathBuf,

        /// Output directory for generated test vectors
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,
    },

    /// Run cycle-accurate FSM simulation on binary input
    Simulate {
        /// Path to input hex file (16-bit words, one per line)
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output hex file (8-bit bytes, one per line)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Start TCP server for network-based simulation
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "9876")]
        port: u16,

        /// Host address to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
}
