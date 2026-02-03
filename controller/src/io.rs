//! JSON input/output for TNN controller.
//!
//! Provides human-readable JSON format for input data and results.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;

/// Input data in JSON format.
///
/// # Example
///
/// ```json
/// {
///   "values": [0.1, 0.2, 0.3, ...]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonInput {
    /// Input values as floating point numbers.
    pub values: Vec<f64>,
}

/// Output data in JSON format.
///
/// # Example
///
/// ```json
/// {
///   "model": "SimpleCNN",
///   "final_output": [0.123, -0.456, ...],
///   "layers": {
///     "pool1": [0.5, 0.3, ...],
///     "conv1": [0.1, 0.2, ...]
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonOutput {
    /// Model name.
    pub model: String,
    /// Final output values as floating point numbers.
    pub final_output: Vec<f64>,
    /// Intermediate layer outputs (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<HashMap<String, Vec<f64>>>,
}

/// Read input from a JSON file.
pub fn read_input_json(path: &Path) -> Result<Vec<TinyNNFP16>, ControllerError> {
    let file = File::open(path).map_err(|e| {
        ControllerError::IoError(format!("Failed to open input file {:?}: {}", path, e))
    })?;
    let reader = BufReader::new(file);
    let input: JsonInput = serde_json::from_reader(reader).map_err(|e| {
        ControllerError::ParseError(format!("Failed to parse JSON input: {}", e))
    })?;

    Ok(input
        .values
        .iter()
        .map(|&v| TinyNNFP16::from_f32(v as f32))
        .collect())
}

/// Write output to a JSON file.
pub fn write_output_json(
    path: &Path,
    model_name: &str,
    final_output: &[TinyNNFP16],
    layer_outputs: Option<&HashMap<String, Vec<TinyNNFP16>>>,
) -> Result<(), ControllerError> {
    let output = JsonOutput {
        model: model_name.to_string(),
        final_output: final_output.iter().map(|v| v.to_f32() as f64).collect(),
        layers: layer_outputs.map(|layers| {
            layers
                .iter()
                .map(|(name, values)| {
                    (
                        name.clone(),
                        values.iter().map(|v| v.to_f32() as f64).collect(),
                    )
                })
                .collect()
        }),
    };

    let file = File::create(path).map_err(|e| {
        ControllerError::IoError(format!("Failed to create output file {:?}: {}", path, e))
    })?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &output).map_err(|e| {
        ControllerError::IoError(format!("Failed to write JSON output: {}", e))
    })?;

    Ok(())
}

/// Convert FP16 values to f64 for JSON serialization.
pub fn fp16_to_f64_vec(values: &[TinyNNFP16]) -> Vec<f64> {
    values.iter().map(|v| v.to_f32() as f64).collect()
}

/// Convert f64 values from JSON to FP16.
pub fn f64_to_fp16_vec(values: &[f64]) -> Vec<TinyNNFP16> {
    values.iter().map(|&v| TinyNNFP16::from_f32(v as f32)).collect()
}
