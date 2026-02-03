//! Error types for the controller.
//!
//! This module defines [`ControllerError`], the main error type used
//! throughout the controller library.

use thiserror::Error;

/// Error type for controller operations.
///
/// Covers errors from file I/O, parsing, translation, and communication.
#[derive(Error, Debug)]
pub enum ControllerError {
    /// File I/O error (reading/writing model or data files).
    #[error("IO error: {0}")]
    IoError(String),

    /// TOML or hex file parsing error.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Error during layer-to-operation translation.
    #[error("Translation error: {0}")]
    TranslationError(String),

    /// Communication error with TNN hardware/simulation.
    #[error("Communication error: {0}")]
    CommError(String),

    /// Tensor shape doesn't match layer requirements.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Layer requires weights/bias that weren't provided.
    #[error("Missing weights for layer: {0}")]
    MissingWeights(String),
}
