//! Pooling layer translation.
//!
//! Translates pooling layers (avg_pool2d, max_pool2d) to TNN operations.

use crate::cnn::schema::{AvgPool2dLayer, MaxPool2dLayer};
use crate::cnn::TensorShape;
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;
use crate::tnn_ops::{fixed_mul_acc_stream, max_pool_stream};

/// Translates a max_pool2d layer to TNN max_pool operations.
/// Extracts non-overlapping windows from input and finds maximum in each.
///
/// Input layout: `[channels, height, width]` flattened in row-major order per channel
/// Output: pooled values for each channel
pub fn translate_max_pool2d(
    layer: &MaxPool2dLayer,
    input_shape: &TensorShape,
    input: &[TinyNNFP16],
) -> Result<Vec<u16>, ControllerError> {
    let (channels, height, width) = input_shape
        .as_image()
        .ok_or_else(|| ControllerError::ShapeMismatch("MaxPool2d expects image input".into()))?;

    let pool_size = layer.kernel_size * layer.kernel_size;
    let out_height = (height - layer.kernel_size) / layer.stride + 1;
    let out_width = (width - layer.kernel_size) / layer.stride + 1;

    // Extract pooling windows and feed to max_pool
    let mut pool_values = Vec::new();

    for c in 0..channels {
        let channel_offset = c * height * width;

        for out_y in 0..out_height {
            for out_x in 0..out_width {
                let start_y = out_y * layer.stride;
                let start_x = out_x * layer.stride;

                // Extract kernel_size x kernel_size window
                for ky in 0..layer.kernel_size {
                    for kx in 0..layer.kernel_size {
                        let y = start_y + ky;
                        let x = start_x + kx;
                        let idx = channel_offset + y * width + x;
                        pool_values.push(input[idx]);
                    }
                }
            }
        }
    }

    Ok(max_pool_stream(&pool_values, pool_size))
}

/// Translates an avg_pool2d layer to TNN fixed_mul_acc operations.
/// Computes average by multiplying each element by 1/pool_size and summing.
///
/// Input layout: `[channels, height, width]` flattened in row-major order per channel
/// Output: pooled values for each channel
pub fn translate_avg_pool2d(
    layer: &AvgPool2dLayer,
    input_shape: &TensorShape,
    input: &[TinyNNFP16],
) -> Result<Vec<u16>, ControllerError> {
    let (channels, height, width) = input_shape
        .as_image()
        .ok_or_else(|| ControllerError::ShapeMismatch("AvgPool2d expects image input".into()))?;

    let pool_size = layer.kernel_size * layer.kernel_size;
    let out_height = (height - layer.kernel_size) / layer.stride + 1;
    let out_width = (width - layer.kernel_size) / layer.stride + 1;

    // Parameter is 1/pool_size for averaging
    let param = TinyNNFP16::from_f32(1.0 / pool_size as f32);

    // Extract pooling windows and feed to fixed_mul_acc
    let mut pool_values = Vec::new();

    for c in 0..channels {
        let channel_offset = c * height * width;

        for out_y in 0..out_height {
            for out_x in 0..out_width {
                let start_y = out_y * layer.stride;
                let start_x = out_x * layer.stride;

                // Extract kernel_size x kernel_size window
                for ky in 0..layer.kernel_size {
                    for kx in 0..layer.kernel_size {
                        let y = start_y + ky;
                        let x = start_x + kx;
                        let idx = channel_offset + y * width + x;
                        pool_values.push(input[idx]);
                    }
                }
            }
        }
    }

    Ok(fixed_mul_acc_stream(&pool_values, pool_size, param))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_max_pool2d() {
        let layer = MaxPool2dLayer {
            name: "pool1".to_string(),
            kernel_size: 2,
            stride: 2,
        };

        let input_shape = TensorShape::Image {
            channels: 1,
            height: 4,
            width: 4,
        };

        // 4x4 input with values 0-15
        let input: Vec<TinyNNFP16> = (0..16)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let stream = translate_max_pool2d(&layer, &input_shape, &input).unwrap();

        // Should have: cmd + 16 values (4 pools * 4 values each) + NaN
        assert!(stream.len() > 0);
        assert_eq!(stream[0] & 0xF000, 0x5000); // Max pool opcode
    }

    #[test]
    fn test_translate_avg_pool2d() {
        let layer = AvgPool2dLayer {
            name: "pool1".to_string(),
            kernel_size: 2,
            stride: 2,
        };

        let input_shape = TensorShape::Image {
            channels: 1,
            height: 4,
            width: 4,
        };

        let input: Vec<TinyNNFP16> = (0..16)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let stream = translate_avg_pool2d(&layer, &input_shape, &input).unwrap();

        // Should have: cmd + param + 16 values + NaN
        assert!(stream.len() > 0);
        assert_eq!(stream[0] & 0xF000, 0x4000); // Fixed mul acc opcode
    }
}
