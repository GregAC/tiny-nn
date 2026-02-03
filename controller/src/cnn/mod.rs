//! CNN model parsing and representation.
//!
//! This module provides types and functions for loading CNN architectures
//! from TOML files and tracking tensor shapes through the network.
//!
//! ## Schema
//!
//! CNN models are described in TOML format with two main sections:
//!
//! 1. **Metadata**: Model name, input shape, output size
//! 2. **Layers**: Sequential list of layer definitions
//!
//! See [`schema`] for the complete type definitions.
//!
//! ## Example
//!
//! ```no_run
//! use controller::cnn::{load_model, TensorShape};
//!
//! let model = load_model("model.toml").unwrap();
//! println!("Model: {}", model.metadata.name);
//!
//! let mut shape = TensorShape::from_chw(model.metadata.input_shape);
//! for layer in &model.layers {
//!     println!("Layer: {}", layer.name());
//! }
//! ```

pub mod parser;
pub mod schema;
pub mod tensor;

pub use parser::{load_hex_file, load_model, load_weights};
pub use schema::{CnnModel, DataSource, Layer};
pub use tensor::TensorShape;
