//! Layer-to-operation translation.
//!
//! This module translates high-level CNN layer descriptions into
//! low-level TNN operation sequences.
//!
//! ## Translation Summary
//!
//! | Layer | TNN Operation(s) | Module |
//! |-------|------------------|--------|
//! | Conv2d | convolve + accumulate | [`conv2d`] |
//! | Linear | mul_acc | [`linear`] |
//! | AvgPool2d | fixed_mul_acc | [`pooling`] |
//! | MaxPool2d | max_pool | [`pooling`] |
//! | Flatten | (none) | - |
//!
//! ## Convolution Decomposition
//!
//! TNN's convolve operation uses a fixed 4×2 kernel. Larger kernels are
//! decomposed into tiles:
//!
//! - A 4×4 kernel → 2 tiles (1×2)
//! - A 5×5 kernel → 6 tiles (2×3)
//! - A 3×3 kernel → 2 tiles (1×2, with padding)

pub mod conv2d;
pub mod linear;
pub mod pooling;

pub use conv2d::{decompose_kernel, translate_conv2d, Conv2dTranslation, KernelTile};
pub use linear::translate_linear;
pub use pooling::{translate_avg_pool2d, translate_max_pool2d};
