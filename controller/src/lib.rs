//! # TNN Controller
//!
//! A library for translating CNN (Convolutional Neural Network) architectures
//! into TNN (Tiny Neural Network) accelerator operations.
//!
//! ## Overview
//!
//! The TNN controller bridges high-level neural network descriptions with the
//! hardware-level operations of the TNN accelerator. It provides:
//!
//! - TOML-based model parsing
//! - Automatic tensor shape tracking
//! - Layer-to-operation translation
//! - Command stream generation
//!
//! ## Quick Start
//!
//! ```no_run
//! use controller::{load_model, CnnRunner, TinyNNFP16};
//!
//! // Load a CNN model from TOML
//! let model = load_model("model.toml").unwrap();
//!
//! // Create a runner with the model
//! let runner = CnnRunner::new(model, ".");
//!
//! // Generate commands for an input
//! let input = vec![TinyNNFP16::from_f32(1.0); 64];
//! let commands = runner.generate_commands(&input).unwrap();
//!
//! // Write commands to a hex file
//! commands.write_to_file("commands.hex").unwrap();
//! ```
//!
//! ## Module Structure
//!
//! - [`cnn`] - CNN model parsing and representation
//! - [`comm`] - Communication interfaces for TNN
//! - [`translation`] - Layer-to-operation translation
//! - [`tnn_ops`] - TNN operation encoding
//! - [`fp16`] - TNN 16-bit floating point type
//! - [`runner`] - Execution orchestration
//! - [`error`] - Error types
//!
//! ## Supported Layers
//!
//! | Layer | TNN Operations |
//! |-------|----------------|
//! | Conv2d | convolve + accumulate |
//! | Linear | mul_acc |
//! | AvgPool2d | fixed_mul_acc |
//! | MaxPool2d | max_pool |
//! | Flatten | (shape tracking only) |

pub mod cnn;
pub mod comm;
pub mod error;
pub mod fp16;
pub mod runner;
pub mod tnn_ops;
pub mod translation;

pub use cnn::{load_hex_file, load_model, CnnModel, TensorShape};
pub use comm::{write_fp16_vec, CommandBuffer, TnnInterface};
pub use error::ControllerError;
pub use fp16::TinyNNFP16;
pub use runner::{plan_execution, CnnRunner, LayerPlan};
