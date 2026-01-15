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

    /// Simulate operations and output results as hex (without timing information)
    Simulate {
        /// Path to TOML configuration file
        config: PathBuf,

        /// Output directory for result files
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,
    },
}
