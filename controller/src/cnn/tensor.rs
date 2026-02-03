//! Tensor shape tracking through neural network layers.
//!
//! This module provides the [`TensorShape`] type for tracking how tensor
//! dimensions change as data flows through network layers.

/// Represents the shape of a tensor as it flows through the network.
///
/// Tensors can be either multi-dimensional images (for conv/pool layers)
/// or 1D vectors (after flatten/for linear layers).
///
/// # Example
///
/// ```
/// use controller::cnn::TensorShape;
///
/// // Start with 1-channel 28x28 image
/// let shape = TensorShape::from_chw([1, 28, 28]);
///
/// // After 4x4 convolution with 8 output channels
/// let shape = shape.after_conv2d(8, 4);
/// // Now: Image { channels: 8, height: 25, width: 25 }
///
/// // After 2x2 max pooling with stride 2
/// let shape = shape.after_pool(2, 2);
/// // Now: Image { channels: 8, height: 12, width: 12 }
///
/// // After flatten
/// let shape = shape.after_flatten();
/// // Now: Vector { size: 1152 }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum TensorShape {
    /// Multi-dimensional image tensor.
    Image {
        /// Number of channels (e.g., 1 for grayscale, 3 for RGB).
        channels: usize,
        /// Image height in pixels.
        height: usize,
        /// Image width in pixels.
        width: usize,
    },
    /// Flattened 1D vector.
    Vector {
        /// Number of elements.
        size: usize,
    },
}

impl TensorShape {
    /// Create a new image shape from [C, H, W] array.
    ///
    /// # Arguments
    ///
    /// * `shape` - Array of [channels, height, width]
    pub fn from_chw(shape: [usize; 3]) -> Self {
        TensorShape::Image {
            channels: shape[0],
            height: shape[1],
            width: shape[2],
        }
    }

    /// Compute the shape after a Conv2d operation.
    ///
    /// For valid (no padding) convolution with stride 1:
    /// - Output height = input_height - kernel_size + 1
    /// - Output width = input_width - kernel_size + 1
    /// - Output channels = out_channels
    ///
    /// # Arguments
    ///
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the square convolution kernel
    ///
    /// # Panics
    ///
    /// Panics if called on a Vector shape.
    pub fn after_conv2d(&self, out_channels: usize, kernel_size: usize) -> Self {
        match self {
            TensorShape::Image { height, width, .. } => TensorShape::Image {
                channels: out_channels,
                height: height - kernel_size + 1,
                width: width - kernel_size + 1,
            },
            TensorShape::Vector { .. } => {
                panic!("Cannot apply conv2d to a vector tensor")
            }
        }
    }

    /// Compute the shape after a pooling operation.
    ///
    /// For pooling with given kernel and stride:
    /// - Output height = (input_height - kernel_size) / stride + 1
    /// - Output width = (input_width - kernel_size) / stride + 1
    /// - Channels remain unchanged
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the square pooling window
    /// * `stride` - Stride of the pooling operation
    ///
    /// # Panics
    ///
    /// Panics if called on a Vector shape.
    pub fn after_pool(&self, kernel_size: usize, stride: usize) -> Self {
        match self {
            TensorShape::Image {
                channels,
                height,
                width,
            } => TensorShape::Image {
                channels: *channels,
                height: (height - kernel_size) / stride + 1,
                width: (width - kernel_size) / stride + 1,
            },
            TensorShape::Vector { .. } => {
                panic!("Cannot apply pooling to a vector tensor")
            }
        }
    }

    /// Flatten the tensor to a 1D vector.
    ///
    /// For an image, the total size is channels × height × width.
    /// For a vector, returns self unchanged.
    pub fn after_flatten(&self) -> Self {
        match self {
            TensorShape::Image {
                channels,
                height,
                width,
            } => TensorShape::Vector {
                size: channels * height * width,
            },
            TensorShape::Vector { .. } => self.clone(),
        }
    }

    /// Get total number of elements in the tensor.
    pub fn size(&self) -> usize {
        match self {
            TensorShape::Image {
                channels,
                height,
                width,
            } => channels * height * width,
            TensorShape::Vector { size } => *size,
        }
    }

    /// Get image dimensions if this is an image shape.
    ///
    /// # Returns
    ///
    /// `Some((channels, height, width))` for Image, `None` for Vector.
    pub fn as_image(&self) -> Option<(usize, usize, usize)> {
        match self {
            TensorShape::Image {
                channels,
                height,
                width,
            } => Some((*channels, *height, *width)),
            TensorShape::Vector { .. } => None,
        }
    }

    /// Get vector size if this is a vector shape.
    ///
    /// # Returns
    ///
    /// `Some(size)` for Vector, `None` for Image.
    pub fn as_vector(&self) -> Option<usize> {
        match self {
            TensorShape::Vector { size } => Some(*size),
            TensorShape::Image { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_shape() {
        let input = TensorShape::from_chw([1, 28, 28]);
        let output = input.after_conv2d(8, 4);
        assert_eq!(
            output,
            TensorShape::Image {
                channels: 8,
                height: 25,
                width: 25
            }
        );
    }

    #[test]
    fn test_pool_shape() {
        let input = TensorShape::from_chw([8, 25, 25]);
        let output = input.after_pool(2, 2);
        assert_eq!(
            output,
            TensorShape::Image {
                channels: 8,
                height: 12,
                width: 12
            }
        );
    }

    #[test]
    fn test_flatten() {
        let input = TensorShape::from_chw([8, 5, 6]);
        let output = input.after_flatten();
        assert_eq!(output, TensorShape::Vector { size: 240 });
    }
}
