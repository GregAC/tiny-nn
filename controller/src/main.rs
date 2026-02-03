use std::path::PathBuf;

use clap::{Parser, Subcommand};

use controller::{
    load_hex_file, load_model, plan_execution, read_input_json, write_fp16_vec,
    write_output_json, CnnRunner, LayerPlan, TnnNetworkClient,
};

#[derive(Parser)]
#[command(name = "tnn-controller")]
#[command(about = "Controller for TNN neural network accelerator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate test vectors (hex files) from a CNN model
    Generate {
        /// Path to the CNN model TOML file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to input data hex file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for generated hex files
        #[arg(short, long, default_value = ".")]
        output_dir: PathBuf,
    },

    /// Run model live through TNN server
    Run {
        /// Path to the CNN model TOML file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to input JSON file
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output JSON file
        #[arg(short, long)]
        output: PathBuf,

        /// TNN server address (host:port)
        #[arg(long, default_value = "localhost:9876")]
        host: String,

        /// Include intermediate layer outputs
        #[arg(long)]
        intermediate: bool,
    },

    /// Show execution plan for a model
    Plan {
        /// Path to the CNN model TOML file
        #[arg(short, long)]
        model: PathBuf,
    },

    /// Validate a CNN model TOML file
    Validate {
        /// Path to the CNN model TOML file
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            input,
            output_dir,
        } => {
            println!("Loading model from: {}", model.display());
            let cnn_model = load_model(&model)?;

            println!("Model: {}", cnn_model.metadata.name);
            if let Some(desc) = &cnn_model.metadata.description {
                println!("Description: {}", desc);
            }
            println!(
                "Input shape: {:?}",
                cnn_model.metadata.input_shape
            );
            println!("Output size: {}", cnn_model.metadata.output_size);
            println!("Layers: {}", cnn_model.layers.len());

            println!("\nLoading input from: {}", input.display());
            let input_data = load_hex_file(&input)?;
            println!("Input values: {}", input_data.len());

            let default_path = PathBuf::from(".");
            let base_path = model.parent().unwrap_or(&default_path);
            let runner = CnnRunner::new(cnn_model, base_path);

            println!("\nGenerating commands...");
            let commands = runner.generate_commands(&input_data)?;

            std::fs::create_dir_all(&output_dir)?;
            let output_path = output_dir.join("commands.hex");
            commands.write_to_file(&output_path)?;
            println!(
                "Wrote {} commands to: {}",
                commands.len(),
                output_path.display()
            );

            // Also write input as FP16 hex
            let input_fp16_path = output_dir.join("input_fp16.hex");
            write_fp16_vec(&input_data, &input_fp16_path)?;
            println!("Wrote input to: {}", input_fp16_path.display());

            Ok(())
        }

        Commands::Run {
            model,
            input,
            output,
            host,
            intermediate,
        } => {
            println!("Loading model from: {}", model.display());
            let cnn_model = load_model(&model)?;

            println!("Model: {}", cnn_model.metadata.name);
            println!(
                "Input shape: {:?}",
                cnn_model.metadata.input_shape
            );
            println!("Output size: {}", cnn_model.metadata.output_size);

            println!("\nLoading input from: {}", input.display());
            let input_data = read_input_json(&input)?;
            println!("Input values: {}", input_data.len());

            let default_path = PathBuf::from(".");
            let base_path = model.parent().unwrap_or(&default_path);
            let runner = CnnRunner::new(cnn_model, base_path);

            println!("\nConnecting to TNN server at {}...", host);
            let mut interface = TnnNetworkClient::connect(&host)?;

            println!("Running model...");
            let results = runner.run(&input_data, &mut interface, intermediate)?;

            println!("Got {} output values", results.final_output.len());
            if intermediate {
                println!(
                    "Captured {} layer outputs",
                    results.layer_outputs.len()
                );
            }

            let layer_outputs = if intermediate {
                Some(&results.layer_outputs)
            } else {
                None
            };

            write_output_json(
                &output,
                runner.model_name(),
                &results.final_output,
                layer_outputs,
            )?;
            println!("Wrote output to: {}", output.display());

            Ok(())
        }

        Commands::Plan { model } => {
            println!("Loading model from: {}", model.display());
            let cnn_model = load_model(&model)?;

            println!("\nModel: {}", cnn_model.metadata.name);
            println!(
                "Input: {:?}",
                cnn_model.metadata.input_shape
            );
            println!("Output: {} classes\n", cnn_model.metadata.output_size);

            let plans = plan_execution(&cnn_model);

            println!("Execution Plan:");
            println!("{:-<60}", "");

            for (i, plan) in plans.iter().enumerate() {
                match plan {
                    LayerPlan::Conv2d {
                        name,
                        in_shape,
                        out_shape,
                        num_convolve_ops,
                        num_accumulate_ops,
                    } => {
                        println!(
                            "{:2}. Conv2d '{}': {:?} -> {:?}",
                            i + 1,
                            name,
                            in_shape,
                            out_shape
                        );
                        println!(
                            "    {} convolve + {} accumulate ops",
                            num_convolve_ops, num_accumulate_ops
                        );
                    }

                    LayerPlan::Linear {
                        name,
                        in_size,
                        out_size,
                        num_mul_acc_ops,
                    } => {
                        println!(
                            "{:2}. Linear '{}': {} -> {}",
                            i + 1,
                            name,
                            in_size,
                            out_size
                        );
                        println!("    {} mul_acc ops", num_mul_acc_ops);
                    }

                    LayerPlan::MaxPool2d {
                        name,
                        in_shape,
                        out_shape,
                    } => {
                        println!(
                            "{:2}. MaxPool2d '{}': {:?} -> {:?}",
                            i + 1,
                            name,
                            in_shape,
                            out_shape
                        );
                        println!("    1 max_pool op");
                    }

                    LayerPlan::AvgPool2d {
                        name,
                        in_shape,
                        out_shape,
                    } => {
                        println!(
                            "{:2}. AvgPool2d '{}': {:?} -> {:?}",
                            i + 1,
                            name,
                            in_shape,
                            out_shape
                        );
                        println!("    1 fixed_mul_acc op");
                    }

                    LayerPlan::Flatten {
                        name,
                        in_shape,
                        out_size,
                    } => {
                        println!(
                            "{:2}. Flatten '{}': {:?} -> {} elements",
                            i + 1,
                            name,
                            in_shape,
                            out_size
                        );
                        println!("    (no TNN operation)");
                    }
                }
            }

            Ok(())
        }

        Commands::Validate { model } => {
            println!("Validating model: {}", model.display());

            let cnn_model = load_model(&model)?;

            println!("Model parsed successfully!");
            println!("  Name: {}", cnn_model.metadata.name);
            println!(
                "  Input shape: {:?}",
                cnn_model.metadata.input_shape
            );
            println!("  Output size: {}", cnn_model.metadata.output_size);
            println!("  Layers: {}", cnn_model.layers.len());

            // Check layer constraints
            let mut errors = Vec::new();

            for layer in &cnn_model.layers {
                match layer {
                    controller::cnn::Layer::Conv2d(conv) => {
                        if conv.stride != 1 {
                            errors.push(format!(
                                "Conv2d '{}': stride must be 1, got {}",
                                conv.name, conv.stride
                            ));
                        }
                    }
                    _ => {}
                }
            }

            if errors.is_empty() {
                println!("\nAll constraints satisfied!");
            } else {
                println!("\nConstraint violations:");
                for err in &errors {
                    println!("  - {}", err);
                }
                std::process::exit(1);
            }

            Ok(())
        }
    }
}
