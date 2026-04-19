//! Conv2d layer translation.
//!
//! Translates 2D convolution layers to TNN convolve and accumulate operations.
//!
//! ## Kernel Decomposition
//!
//! TNN's convolve operation uses a fixed 4×2 kernel. Larger kernels are
//! decomposed into multiple 4×2 tiles:
//!
//! ```text
//! 5×5 kernel decomposition:
//! +-------+-------+
//! | 4×2   | 1×2   |  (tiles 0, 1)
//! +-------+-------+
//! | 4×2   | 1×2   |  (tiles 2, 3)
//! +-------+-------+
//! | 4×1   | 1×1   |  (tiles 4, 5) - padded with zeros
//! +-------+-------+
//! ```
//!
//! Each tile produces partial convolution results that are summed by the
//! accumulate operation.

use crate::cnn::schema::Conv2dLayer;
use crate::cnn::TensorShape;
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;
use crate::tnn_ops::{accumulate_stream, convolve_stream, CONV_HEIGHT, CONV_WIDTH};

/// A 4×2 tile extracted from a larger kernel.
/// Coordinates are relative to the full kernel's top-left.
#[derive(Debug)]
pub struct KernelTile {
    /// X offset in the full kernel (0, 4, 8, ...)
    pub x_offset: usize,
    /// Y offset in the full kernel (0, 2, 4, ...)
    pub y_offset: usize,
    /// The 8 kernel values in column-major order
    pub params: [TinyNNFP16; CONV_WIDTH * CONV_HEIGHT],
}

/// Decomposes a square kernel into 4x2 tiles for TNN convolution.
/// Returns tiles in order: iterate y_offset (outer), then x_offset (inner).
///
/// Kernel layout: row-major `[kernel_size, kernel_size]`
/// Each tile: column-major [4, 2] for TNN
pub fn decompose_kernel(
    kernel: &[TinyNNFP16],
    kernel_size: usize,
) -> Vec<KernelTile> {
    let num_tiles_x = (kernel_size + CONV_WIDTH - 1) / CONV_WIDTH;
    let num_tiles_y = (kernel_size + CONV_HEIGHT - 1) / CONV_HEIGHT;

    let mut tiles = Vec::with_capacity(num_tiles_x * num_tiles_y);

    for tile_y in 0..num_tiles_y {
        for tile_x in 0..num_tiles_x {
            let y_offset = tile_y * CONV_HEIGHT;
            let x_offset = tile_x * CONV_WIDTH;

            let mut params = [TinyNNFP16::zero(); CONV_WIDTH * CONV_HEIGHT];

            // Fill in column-major order: p_x_y where x is column, y is row
            for x in 0..CONV_WIDTH {
                for y in 0..CONV_HEIGHT {
                    let kernel_x = x_offset + x;
                    let kernel_y = y_offset + y;

                    let value = if kernel_x < kernel_size && kernel_y < kernel_size {
                        // Kernel is stored row-major: kernel[kernel_y * kernel_size + kernel_x]
                        kernel[kernel_y * kernel_size + kernel_x]
                    } else {
                        // Padding with zeros for tiles that extend beyond kernel
                        TinyNNFP16::zero()
                    };

                    // Column-major index: x * CONV_HEIGHT + y
                    params[x * CONV_HEIGHT + y] = value;
                }
            }

            tiles.push(KernelTile {
                x_offset,
                y_offset,
                params,
            });
        }
    }

    tiles
}

/// Extracts a row of image data for convolution in column-major 2-row chunks.
/// This matches TNN's expected data format.
///
/// For a given y position (top of 2-row chunk), extracts values in order:
/// (x=0,y), (x=0,y+1), (x=1,y), (x=1,y+1), ...
fn extract_conv_row(
    image: &[TinyNNFP16],
    width: usize,
    height: usize,
    start_x: usize,
    end_x: usize,
    y: usize,
) -> Vec<TinyNNFP16> {
    let mut values = Vec::with_capacity((end_x - start_x) * CONV_HEIGHT);

    for x in start_x..end_x {
        for inner_y in 0..CONV_HEIGHT {
            let actual_y = y + inner_y;
            if actual_y < height && x < width {
                values.push(image[actual_y * width + x]);
            } else {
                values.push(TinyNNFP16::zero());
            }
        }
    }

    values
}

/// Generates the convolution input stream for a single channel image.
/// The stream covers the entire image for all rows that will produce output.
fn conv_input_stream_for_channel(
    image: &[TinyNNFP16],
    width: usize,
    height: usize,
    kernel_size: usize,
    tile_y_offset: usize,
) -> Vec<TinyNNFP16> {
    let conv_result_height = height - (kernel_size - 1);
    // When kernel_size is not a multiple of CONV_WIDTH, the last valid output pixel
    // requires image columns beyond `width`. Pad each row with zeros so the sliding
    // window can reach all out_width output positions.
    let extra_cols = (CONV_WIDTH - kernel_size % CONV_WIDTH) % CONV_WIDTH;
    let padded_width = width + extra_cols;
    let mut stream = Vec::new();

    for out_y in 0..conv_result_height {
        // The actual y in the image for this output row, accounting for tile offset
        let y = out_y + tile_y_offset;
        let row = extract_conv_row(image, width, height, 0, padded_width, y);
        stream.extend(row);
    }

    stream
}

/// Result of translating a conv2d layer
pub struct Conv2dTranslation {
    /// Command streams for convolve operations (one per tile per input channel per output channel)
    pub convolve_streams: Vec<Vec<u16>>,
    /// Command streams for accumulate operations (one per output channel)
    pub accumulate_streams: Vec<Vec<u16>>,
}

/// Translates a conv2d layer to TNN operations.
///
/// For each output channel:
///   - For each input channel:
///     - For each kernel tile: run convolve operation
///   - Run accumulate to sum all partial results + bias + optional ReLU
///
/// Input layout: `[channels, height, width]` flattened, row-major within each channel
/// Weights layout: `[out_channels, in_channels, kernel_h, kernel_w]` flattened
/// Bias layout: `[out_channels]`
pub fn translate_conv2d(
    layer: &Conv2dLayer,
    input_shape: &TensorShape,
    input: &[TinyNNFP16],
    weights: &[TinyNNFP16],
    bias: &[TinyNNFP16],
) -> Result<Conv2dTranslation, ControllerError> {
    let (in_channels, height, width) = input_shape
        .as_image()
        .ok_or_else(|| ControllerError::ShapeMismatch("Conv2d expects image input".into()))?;

    if in_channels != layer.in_channels {
        return Err(ControllerError::ShapeMismatch(format!(
            "Conv2d {} expects {} input channels, got {}",
            layer.name, layer.in_channels, in_channels
        )));
    }

    let kernel_size = layer.kernel_size;
    let out_channels = layer.out_channels;
    let kernel_elements = kernel_size * kernel_size;

    // Verify weight dimensions
    let expected_weights = out_channels * in_channels * kernel_elements;
    if weights.len() != expected_weights {
        return Err(ControllerError::ShapeMismatch(format!(
            "Conv2d {} expects {} weights, got {}",
            layer.name, expected_weights, weights.len()
        )));
    }

    if bias.len() != out_channels {
        return Err(ControllerError::ShapeMismatch(format!(
            "Conv2d {} expects {} bias values, got {}",
            layer.name, out_channels, bias.len()
        )));
    }

    let out_height = height - kernel_size + 1;
    let out_width = width - kernel_size + 1;
    let _out_pixels = out_height * out_width;

    // Number of tiles needed to cover the kernel
    let tiles_per_kernel = {
        let num_tiles_x = (kernel_size + CONV_WIDTH - 1) / CONV_WIDTH;
        let num_tiles_y = (kernel_size + CONV_HEIGHT - 1) / CONV_HEIGHT;
        num_tiles_x * num_tiles_y
    };

    // Total partial results per output pixel = in_channels * tiles_per_kernel
    let partials_per_pixel = in_channels * tiles_per_kernel;

    let mut convolve_streams = Vec::new();
    let mut accumulate_streams = Vec::new();

    for out_ch in 0..out_channels {
        // Note: In a full implementation, we would collect partial results here
        // to feed into the accumulate operation. For command generation, we just
        // track that the operations will produce out_pixels outputs.

        for in_ch in 0..in_channels {
            // Extract kernel for this (out_ch, in_ch) pair
            let kernel_offset = (out_ch * in_channels + in_ch) * kernel_elements;
            let kernel = &weights[kernel_offset..kernel_offset + kernel_elements];

            // Decompose kernel into tiles
            let tiles = decompose_kernel(kernel, kernel_size);

            // Extract input channel
            let channel_offset = in_ch * height * width;
            let channel_data = &input[channel_offset..channel_offset + height * width];

            for tile in &tiles {
                // Generate input stream for this tile
                // The tile's y_offset determines which rows of the image we read
                let conv_data = conv_input_stream_for_channel(
                    channel_data,
                    width,
                    height,
                    kernel_size,
                    tile.y_offset,
                );

                // Generate convolve command
                let stream = convolve_stream(&tile.params, &conv_data);
                convolve_streams.push(stream);

                // Note: In actual execution, we would collect the outputs here.
                // For command generation, we just record the operations.
                // The accumulate operation will expect partials_per_pixel values
                // per output pixel, arranged for proper grouping.
            }
        }

        // Generate accumulate command for this output channel
        // In practice, the convolve outputs would be fed here.
        // For command generation, we create a placeholder accumulate that expects
        // the right number of partial results per output.
        //
        // The accumulate operation groups values and adds bias.
        // Each output pixel needs partials_per_pixel values summed together.
        //
        // Note: This is a simplification - the actual data flow would need the
        // runtime to collect convolve outputs and feed them to accumulate.
        // For now, we generate the accumulate command structure.

        // Create a placeholder stream - in practice the partials would come from convolve outputs
        let accum_stream = accumulate_stream(
            &[], // Placeholder - actual values come from convolve outputs
            partials_per_pixel,
            bias[out_ch],
            layer.relu,
        );
        accumulate_streams.push(accum_stream);
    }

    Ok(Conv2dTranslation {
        convolve_streams,
        accumulate_streams,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_kernel_4x4() {
        // 4x4 kernel fits exactly in one 4x2 tile horizontally, two vertically
        let kernel: Vec<TinyNNFP16> = (0..16)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let tiles = decompose_kernel(&kernel, 4);

        assert_eq!(tiles.len(), 2); // 1 tile wide, 2 tiles tall

        // First tile covers rows 0-1
        assert_eq!(tiles[0].x_offset, 0);
        assert_eq!(tiles[0].y_offset, 0);

        // Second tile covers rows 2-3
        assert_eq!(tiles[1].x_offset, 0);
        assert_eq!(tiles[1].y_offset, 2);
    }

    #[test]
    fn test_decompose_kernel_3x3() {
        // 3x3 kernel requires padding
        let kernel: Vec<TinyNNFP16> = (0..9)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let tiles = decompose_kernel(&kernel, 3);

        // 3x3 kernel: ceil(3/4)=1 tile wide, ceil(3/2)=2 tiles tall
        assert_eq!(tiles.len(), 2);

        // Check that out-of-bounds positions are zero
        // First tile (y=0,1): positions (3,0) and (3,1) should be zero
        let first_tile = &tiles[0];
        // Column 3 (index 3) in column-major: positions 6 and 7
        assert!(first_tile.params[6].is_zero());
        assert!(first_tile.params[7].is_zero());
    }

    #[test]
    fn test_decompose_kernel_5x5() {
        // 5x5 kernel: ceil(5/4)=2 tiles wide, ceil(5/2)=3 tiles tall
        let kernel: Vec<TinyNNFP16> = (0..25)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let tiles = decompose_kernel(&kernel, 5);

        assert_eq!(tiles.len(), 6); // 2 * 3

        // Check tile order: y-major
        assert_eq!((tiles[0].x_offset, tiles[0].y_offset), (0, 0));
        assert_eq!((tiles[1].x_offset, tiles[1].y_offset), (4, 0));
        assert_eq!((tiles[2].x_offset, tiles[2].y_offset), (0, 2));
        assert_eq!((tiles[3].x_offset, tiles[3].y_offset), (4, 2));
        assert_eq!((tiles[4].x_offset, tiles[4].y_offset), (0, 4));
        assert_eq!((tiles[5].x_offset, tiles[5].y_offset), (4, 4));
    }

    #[test]
    fn test_translate_conv2d_simple() {
        let layer = Conv2dLayer {
            name: "conv1".to_string(),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 4,
            stride: 1,
            relu: false,
            weights: None,
            bias: None,
        };

        let input_shape = TensorShape::Image {
            channels: 1,
            height: 6,
            width: 6,
        };

        let input: Vec<TinyNNFP16> = (0..36)
            .map(|i| TinyNNFP16::from_f32(i as f32))
            .collect();

        let weights: Vec<TinyNNFP16> = (0..16)
            .map(|i| TinyNNFP16::from_f32(i as f32 * 0.1))
            .collect();

        let bias = vec![TinyNNFP16::from_f32(0.5)];

        let result = translate_conv2d(&layer, &input_shape, &input, &weights, &bias).unwrap();

        // 4x4 kernel = 2 tiles (1 wide, 2 tall)
        // 1 input channel, 1 output channel
        // Should have 2 convolve streams
        assert_eq!(result.convolve_streams.len(), 2);

        // 1 accumulate stream (one per output channel)
        assert_eq!(result.accumulate_streams.len(), 1);
    }
}
