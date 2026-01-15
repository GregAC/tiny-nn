use rand::Rng;
use serde::Deserialize;
use std::fs;
use std::path::Path;

use crate::test_data;
use crate::tnn_types::{TinyNNFP16, TNNFP16ExpWidth, TNNFP16MantWidth};
use ndarray::Array2;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub operation: Vec<Operation>,
}

#[derive(Debug, Deserialize)]
pub struct Operation {
    #[serde(rename = "type")]
    pub op_type: String,
    pub name: String,
    pub output_dir: Option<String>,

    // Operation-specific fields (all optional, validated at runtime)
    pub group_size: Option<usize>,
    pub relu: Option<bool>,
    pub values_per_pool: Option<usize>,

    // Data sources
    pub image: Option<DataSource>,
    pub params: Option<DataSource>,
    pub values: Option<DataSource>,
    pub bias: Option<DataSource>,
    pub param: Option<DataSource>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DataSource {
    pub source: String, // "inline", "file", "builtin", "random"

    // Source-specific fields (all optional)
    pub name: Option<String>,       // for builtin
    pub value: Option<f32>,         // for inline scalar
    pub values: Option<Vec<f32>>,   // for inline vector
    pub path: Option<String>,       // for file
    pub count: Option<usize>,       // for random
}

impl DataSource {
    /// Resolve a scalar value from the data source
    pub fn resolve_scalar(&self) -> TinyNNFP16 {
        match self.source.as_str() {
            "inline" => {
                if let Some(v) = self.value {
                    TinyNNFP16::from_f32(v)
                } else if let Some(values) = &self.values {
                    if values.len() == 1 {
                        TinyNNFP16::from_f32(values[0])
                    } else {
                        panic!("Expected single value for scalar, got {} values", values.len());
                    }
                } else {
                    panic!("Inline source requires 'value' or single-element 'values'");
                }
            }
            "random" => generate_random_fp16(),
            _ => panic!(
                "Cannot resolve scalar from source type '{}'. Use 'inline' or 'random'.",
                self.source
            ),
        }
    }

    /// Resolve a vector of values from the data source
    pub fn resolve_vector(&self) -> Vec<TinyNNFP16> {
        match self.source.as_str() {
            "inline" => {
                if let Some(values) = &self.values {
                    values.iter().map(|v| TinyNNFP16::from_f32(*v)).collect()
                } else if let Some(v) = self.value {
                    vec![TinyNNFP16::from_f32(v)]
                } else {
                    panic!("Inline source requires 'value' or 'values'");
                }
            }
            "file" => {
                let path = self.path.as_ref().expect("File source requires 'path'");
                load_vector_from_file(path)
            }
            "builtin" => {
                let name = self.name.as_ref().expect("Builtin source requires 'name'");
                // Builtin vector sources - flatten 2D arrays if needed
                match name.as_str() {
                    "mnist_image" => test_data::get_mnist_image().iter().cloned().collect(),
                    "incrementing_test_image" => {
                        test_data::get_incrementing_test_image().iter().cloned().collect()
                    }
                    _ => panic!("Unknown builtin vector source: '{}'", name),
                }
            }
            "random" => {
                let count = self.count.expect("Random source requires 'count'");
                generate_random_vector(count)
            }
            _ => panic!("Unknown source type: '{}'", self.source),
        }
    }

    /// Resolve a 2D array from the data source
    pub fn resolve_array2(&self) -> Array2<TinyNNFP16> {
        match self.source.as_str() {
            "builtin" => {
                let name = self.name.as_ref().expect("Builtin source requires 'name'");
                match name.as_str() {
                    "mnist_image" => test_data::get_mnist_image(),
                    "mnist_convolve_0_params" => test_data::get_mnist_convolve_0_params(),
                    "mnist_convolve_1_params" => test_data::get_mnist_convolve_1_params(),
                    "incrementing_test_image" => test_data::get_incrementing_test_image(),
                    "half_const_convolve_params" => test_data::get_half_const_convolve_params(),
                    _ => panic!("Unknown builtin array source: '{}'", name),
                }
            }
            "inline" => {
                // For inline 2D arrays, expect a flat list of values and reshape
                // For convolution params, we expect 4x2 = 8 values
                let values = self
                    .values
                    .as_ref()
                    .expect("Inline array source requires 'values'");
                if values.len() == 8 {
                    // Assume 4x2 convolution params
                    let fp_values: Vec<TinyNNFP16> =
                        values.iter().map(|v| TinyNNFP16::from_f32(*v)).collect();
                    Array2::from_shape_vec((4, 2), fp_values)
                        .expect("Failed to reshape inline values to 4x2 array")
                } else {
                    panic!(
                        "Inline array source with {} values not supported. Expected 8 for 4x2 params.",
                        values.len()
                    );
                }
            }
            "file" => {
                let path = self.path.as_ref().expect("File source requires 'path'");
                load_array2_from_file(path)
            }
            _ => panic!(
                "Cannot resolve 2D array from source type '{}'. Use 'builtin', 'inline', or 'file'.",
                self.source
            ),
        }
    }
}

/// Load a vector of TinyNNFP16 values from a hex file
fn load_vector_from_file(path: &str) -> Vec<TinyNNFP16> {
    let content = fs::read_to_string(path).expect(&format!("Failed to read file: {}", path));
    content
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(|line| {
            let trimmed = line.trim();
            // Try parsing as hex first (4 hex digits = 16 bits)
            if trimmed.len() == 4 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
                let raw = u16::from_str_radix(trimmed, 16)
                    .expect(&format!("Invalid hex value: {}", trimmed));
                TinyNNFP16::from_u16(raw)
            } else {
                // Try parsing as float
                let f: f32 = trimmed
                    .parse()
                    .expect(&format!("Invalid float value: {}", trimmed));
                TinyNNFP16::from_f32(f)
            }
        })
        .collect()
}

/// Load a 2D array from a file (assumes 4x2 convolution params format)
fn load_array2_from_file(path: &str) -> Array2<TinyNNFP16> {
    let values = load_vector_from_file(path);
    if values.len() == 8 {
        Array2::from_shape_vec((4, 2), values).expect("Failed to reshape file values to 4x2 array")
    } else {
        panic!(
            "File array source with {} values not supported. Expected 8 for 4x2 params.",
            values.len()
        );
    }
}

/// Generate a single random TinyNNFP16 value
fn generate_random_fp16() -> TinyNNFP16 {
    let mut rng = rand::thread_rng();
    let high_exp =
        rng.gen_range((1 << (TNNFP16ExpWidth - 1)) - 10..(1 << (TNNFP16ExpWidth - 1)) + 10);
    TinyNNFP16::new(
        rng.gen(),
        rng.gen_range(high_exp - 5..high_exp),
        rng.gen_range(0..(1 << TNNFP16MantWidth)),
    )
}

/// Generate a random vector of TinyNNFP16 values
fn generate_random_vector(count: usize) -> Vec<TinyNNFP16> {
    (0..count).map(|_| generate_random_fp16()).collect()
}

/// Load and parse a TOML config file
pub fn load_config(path: &Path) -> Config {
    let content =
        fs::read_to_string(path).expect(&format!("Failed to read config file: {:?}", path));
    toml::from_str(&content).expect(&format!("Failed to parse config file: {:?}", path))
}
