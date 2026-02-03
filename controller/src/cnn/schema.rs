//! TOML schema types for CNN model definitions.
//!
//! This module defines the Rust types that correspond to the CNN TOML schema.
//! Models are deserialized directly from TOML using serde.
//!
//! ## Schema Structure
//!
//! ```toml
//! [metadata]
//! name = "ModelName"
//! input_shape = [C, H, W]
//! output_size = N
//!
//! [[layer]]
//! type = "conv2d"
//! name = "conv1"
//! # ... layer-specific parameters
//! ```

use serde::Deserialize;

/// A complete CNN model with metadata and layers.
#[derive(Debug, Clone, Deserialize)]
pub struct CnnModel {
    /// Model metadata (name, input/output dimensions).
    pub metadata: Metadata,
    /// Sequential list of layers.
    #[serde(rename = "layer")]
    pub layers: Vec<Layer>,
}

/// Model metadata describing input/output dimensions.
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    /// Model identifier.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Input dimensions as [channels, height, width].
    pub input_shape: [usize; 3],
    /// Number of output classes/features.
    pub output_size: usize,
}

/// A layer in the CNN, tagged by type.
///
/// Each variant corresponds to a different layer type in the TOML schema.
/// The `type` field in TOML determines which variant is used.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Layer {
    /// 2D convolution layer.
    Conv2d(Conv2dLayer),
    /// Fully connected (dense) layer.
    Linear(LinearLayer),
    /// Average pooling layer.
    AvgPool2d(AvgPool2dLayer),
    /// Max pooling layer.
    MaxPool2d(MaxPool2dLayer),
    /// Flatten layer (reshapes to 1D).
    Flatten(FlattenLayer),
}

impl Layer {
    /// Get the layer's name.
    pub fn name(&self) -> &str {
        match self {
            Layer::Conv2d(l) => &l.name,
            Layer::Linear(l) => &l.name,
            Layer::AvgPool2d(l) => &l.name,
            Layer::MaxPool2d(l) => &l.name,
            Layer::Flatten(l) => &l.name,
        }
    }
}

/// 2D convolution layer parameters.
///
/// Performs 2D convolution over an input image with learnable kernels.
///
/// ## TNN Mapping
///
/// Maps to `convolve` + `accumulate` operations. Large kernels are
/// decomposed into 4x2 tiles.
///
/// ## Constraints
///
/// - `stride` must be 1 (strided convolutions not supported)
/// - Only square kernels supported
#[derive(Debug, Clone, Deserialize)]
pub struct Conv2dLayer {
    /// Layer identifier.
    pub name: String,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (filters).
    pub out_channels: usize,
    /// Size of the square convolution kernel.
    pub kernel_size: usize,
    /// Convolution stride (must be 1).
    pub stride: usize,
    /// Whether to apply ReLU activation after convolution.
    #[serde(default)]
    pub relu: bool,
    /// Kernel weights, shape: `[out_channels, in_channels, kernel_size, kernel_size]`.
    pub weights: Option<DataSource>,
    /// Bias values, shape: \[out_channels\].
    pub bias: Option<DataSource>,
}

/// Fully connected (linear) layer parameters.
///
/// Performs a linear transformation: `y = Wx + b`
///
/// ## TNN Mapping
///
/// Maps to `mul_acc` operations, one per output neuron.
#[derive(Debug, Clone, Deserialize)]
pub struct LinearLayer {
    /// Layer identifier.
    pub name: String,
    /// Size of input vector.
    pub in_features: usize,
    /// Size of output vector.
    pub out_features: usize,
    /// Whether to apply ReLU activation after transformation.
    #[serde(default)]
    pub relu: bool,
    /// Weight matrix, shape: `[out_features, in_features]` (row-major).
    pub weights: Option<DataSource>,
    /// Bias values, shape: \[out_features\].
    pub bias: Option<DataSource>,
}

/// Average pooling layer parameters.
///
/// Computes the average value within each pooling window.
///
/// ## TNN Mapping
///
/// Maps to `fixed_mul_acc` with parameter = 1/(kernel_size²).
#[derive(Debug, Clone, Deserialize)]
pub struct AvgPool2dLayer {
    /// Layer identifier.
    pub name: String,
    /// Size of the square pooling window.
    pub kernel_size: usize,
    /// Stride of the pooling window.
    pub stride: usize,
}

/// Max pooling layer parameters.
///
/// Returns the maximum value within each pooling window.
///
/// ## TNN Mapping
///
/// Maps directly to `max_pool` operation.
#[derive(Debug, Clone, Deserialize)]
pub struct MaxPool2dLayer {
    /// Layer identifier.
    pub name: String,
    /// Size of the square pooling window.
    pub kernel_size: usize,
    /// Stride of the pooling window.
    pub stride: usize,
}

/// Flatten layer parameters.
///
/// Reshapes a multi-dimensional tensor into a 1D vector.
/// This is a logical operation; no TNN operation is required.
#[derive(Debug, Clone, Deserialize)]
pub struct FlattenLayer {
    /// Layer identifier.
    pub name: String,
}

/// Source for weight/bias data.
///
/// Weights can be embedded inline in the TOML or stored in external hex files.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum DataSource {
    /// Values embedded directly in the TOML as floats.
    Inline {
        /// Float values (will be converted to TNN FP16).
        values: Vec<f32>,
    },
    /// Values stored in an external hex file.
    File {
        /// Path to the hex file (relative to model TOML).
        path: String,
    },
}
