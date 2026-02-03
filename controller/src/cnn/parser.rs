//! Model and data file parsing utilities.
//!
//! This module provides functions for loading CNN models from TOML files
//! and weight data from hex files.

use std::fs;
use std::path::Path;

use crate::cnn::schema::{CnnModel, DataSource};
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;

/// Load a CNN model from a TOML file.
///
/// # Arguments
///
/// * `path` - Path to the TOML model file
///
/// # Returns
///
/// The parsed [`CnnModel`] or an error if parsing fails.
///
/// # Example
///
/// ```no_run
/// use controller::cnn::load_model;
///
/// let model = load_model("model.toml").unwrap();
/// println!("Loaded model: {}", model.metadata.name);
/// ```
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<CnnModel, ControllerError> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        ControllerError::IoError(format!(
            "Failed to read model file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;

    toml::from_str(&content).map_err(|e| {
        ControllerError::ParseError(format!("Failed to parse TOML: {}", e))
    })
}

/// Load weight or bias values from a data source.
///
/// Handles both inline values (embedded in TOML) and file-based values
/// (stored in separate hex files).
///
/// # Arguments
///
/// * `source` - The data source specification
/// * `base_path` - Base path for resolving relative file paths
///
/// # Returns
///
/// Vector of [`TinyNNFP16`] values.
pub fn load_weights<P: AsRef<Path>>(
    source: &DataSource,
    base_path: P,
) -> Result<Vec<TinyNNFP16>, ControllerError> {
    match source {
        DataSource::Inline { values } => {
            Ok(values.iter().map(|&v| TinyNNFP16::from_f32(v)).collect())
        }
        DataSource::File { path } => {
            let full_path = base_path.as_ref().join(path);
            load_hex_file(&full_path)
        }
    }
}

/// Load FP16 values from a hex file.
///
/// The file format is one 4-character hex value per line, representing
/// a 16-bit TNN floating point number.
///
/// # Arguments
///
/// * `path` - Path to the hex file
///
/// # Returns
///
/// Vector of [`TinyNNFP16`] values.
///
/// # Example
///
/// ```no_run
/// use controller::cnn::load_hex_file;
///
/// let values = load_hex_file("weights.hex").unwrap();
/// println!("Loaded {} values", values.len());
/// ```
///
/// # File Format
///
/// ```text
/// 3f80
/// bf00
/// 3e80
/// ```
pub fn load_hex_file<P: AsRef<Path>>(path: P) -> Result<Vec<TinyNNFP16>, ControllerError> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| {
        ControllerError::IoError(format!(
            "Failed to read hex file {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let mut values = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw = u16::from_str_radix(line, 16).map_err(|e| {
            ControllerError::ParseError(format!(
                "Invalid hex value '{}' at line {}: {}",
                line,
                line_num + 1,
                e
            ))
        })?;
        values.push(TinyNNFP16::from_u16(raw));
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inline_weights() {
        let source = DataSource::Inline {
            values: vec![1.0, 2.0, 3.0],
        };
        let weights = load_weights(&source, ".").unwrap();
        assert_eq!(weights.len(), 3);
    }
}
