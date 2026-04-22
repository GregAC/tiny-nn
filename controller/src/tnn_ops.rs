//! TNN operation encoding and command stream building.
//!
//! This module defines the TNN hardware operation opcodes and provides
//! functions for building command streams that can be sent to TNN.
//!
//! ## TNN Operations
//!
//! | Operation | Opcode | Description |
//! |-----------|--------|-------------|
//! | Convolve | 0x1xxx | 4x2 kernel convolution |
//! | Accumulate | 0x2xxx | Sum groups with bias and optional ReLU |
//! | Mul-Acc | 0x3xxx | Multiply-accumulate pairs with bias |
//! | Fixed Mul-Acc | 0x4xxx | Multiply all by fixed param and sum |
//! | Max Pool | 0x5xxx | Find maximum in groups |
//!
//! ## Command Format
//!
//! All operations are terminated by a NaN value (0xFFFF).

use crate::fp16::TinyNNFP16;

/// TNN convolution kernel width (fixed by hardware).
pub const CONV_WIDTH: usize = 4;

/// TNN convolution kernel height (fixed by hardware).
pub const CONV_HEIGHT: usize = 2;

/// Opcode for convolve operation.
pub const CMD_CONVOLVE: u16 = 0x1000;

/// Opcode for accumulate operation.
pub const CMD_ACCUMULATE: u16 = 0x2000;

/// Opcode for multiply-accumulate operation.
pub const CMD_MUL_ACC: u16 = 0x3000;

/// Opcode for fixed multiply-accumulate operation.
pub const CMD_FIXED_MUL_ACC: u16 = 0x4000;

/// Opcode for max pool operation.
pub const CMD_MAX_POOL: u16 = 0x5000;

/// NaN terminator value (signals end of operation data).
pub const NAN_TERMINATOR: u16 = 0xFFFF;

/// Cycles from command word to first output for convolve.
pub const CONV_FIRST_OUTPUT_DELAY: usize = 22;

/// Extra cycles between convolve output rows.
pub const CONV_ROW_OUTPUT_DELAY: usize = 6;

/// Cycles from command word to first output for accumulate.
pub const ACCUM_FIRST_OUTPUT_DELAY: usize = 5;

/// Cycles from command word to first output for fixed_mul_acc.
pub const FIXED_MUL_ACC_FIRST_OUTPUT_DELAY: usize = 6;

/// Cycles from command word to first output for max_pool.
pub const MAX_POOL_FIRST_OUTPUT_DELAY: usize = 1;

/// Cycles from command word to output for mul_acc.
pub const MUL_ACC_OUTPUT_DELAY: usize = 6;

/// Zero words to send after convolve NaN so FSM returns to idle (5 drain + 1 idle).
pub const CONVOLVE_IDLE_DRAIN: usize = 6;
/// Zero words to send after accumulate NaN so FSM returns to idle (3 drain + 1 idle).
pub const ACCUMULATE_IDLE_DRAIN: usize = 4;
/// Zero words to send after mul_acc NaN so FSM returns to idle (4 drain + 1 idle).
pub const MUL_ACC_IDLE_DRAIN: usize = 5;
/// Zero words to send after fixed_mul_acc NaN so FSM returns to idle (4 drain + 1 idle).
pub const FIXED_MUL_ACC_IDLE_DRAIN: usize = 5;
/// Zero words to send after max_pool NaN so FSM returns to idle (1 drain + 1 idle).
pub const MAX_POOL_IDLE_DRAIN: usize = 2;

/// Build a convolve command stream.
///
/// The convolve operation performs a 4x2 kernel convolution over streaming data.
///
/// # Arguments
///
/// * `params` - 8 kernel parameters in column-major order:
///   `[p_0_0, p_0_1, p_1_0, p_1_1, p_2_0, p_2_1, p_3_0, p_3_1]`
/// * `data` - Input data values in column-major 2-row chunks
///
/// # Panics
///
/// Panics if `params.len() != 8`.
///
/// # Returns
///
/// Command stream: `[CMD, params..., data..., NaN]`
pub fn convolve_stream(params: &[TinyNNFP16], data: &[TinyNNFP16]) -> Vec<u16> {
    assert_eq!(
        params.len(),
        CONV_WIDTH * CONV_HEIGHT,
        "Convolve requires 8 parameters"
    );

    let mut output = vec![CMD_CONVOLVE];
    output.extend(params.iter().map(|x| x.as_u16()));
    output.extend(data.iter().map(|x| x.as_u16()));
    output.push(NAN_TERMINATOR);
    output
}

/// Build an accumulate command stream.
///
/// The accumulate operation sums groups of values, adds a bias, and optionally
/// applies ReLU activation.
///
/// # Arguments
///
/// * `values` - Values to accumulate
/// * `count` - Number of values per group (1-256)
/// * `bias` - Constant added to each sum
/// * `relu` - Whether to apply ReLU to output
///
/// # Panics
///
/// Panics if `count` is not in range 1-256.
///
/// # Returns
///
/// Command stream: `[CMD, bias, values..., NaN]`
///
/// The command word encodes: `opcode | (relu << 8) | (count - 1)`
pub fn accumulate_stream(
    values: &[TinyNNFP16],
    count: usize,
    bias: TinyNNFP16,
    relu: bool,
) -> Vec<u16> {
    assert!(count >= 1 && count <= 256, "Count must be 1-256");

    let relu_flag: u16 = if relu { 0x100 } else { 0x0 };
    let cmd = CMD_ACCUMULATE | relu_flag | ((count - 1) as u16 & 0xFF);

    let mut output = vec![cmd];
    output.push(bias.as_u16());
    output.extend(values.iter().map(|x| x.as_u16()));
    output.push(NAN_TERMINATOR);
    output
}

/// Build a multiply-accumulate command stream.
///
/// Computes: `relu(sum(v[i] * p[i] for all pairs) + bias)`
///
/// # Arguments
///
/// * `values` - Interleaved (value, weight) pairs
/// * `bias` - Constant added to the sum
/// * `relu` - Whether to apply ReLU to output
///
/// # Returns
///
/// Command stream: `[CMD, bias, pairs..., NaN]`
pub fn mul_acc_stream(values: &[TinyNNFP16], bias: TinyNNFP16, relu: bool) -> Vec<u16> {
    let relu_flag: u16 = if relu { 0x100 } else { 0x0 };
    let cmd = CMD_MUL_ACC | relu_flag;

    let mut output = vec![cmd];
    output.push(bias.as_u16());
    output.extend(values.iter().map(|x| x.as_u16()));
    output.push(NAN_TERMINATOR);
    output
}

/// Build a fixed multiply-accumulate command stream.
///
/// Computes: `sum(v[i] * param)` for each group of `count` values.
/// Used for average pooling where `param = 1/pool_size`.
///
/// # Arguments
///
/// * `values` - Values to process
/// * `count` - Number of values per group (1-256)
/// * `param` - Fixed multiplier for all values
///
/// # Panics
///
/// Panics if `count` is not in range 1-256.
///
/// # Returns
///
/// Command stream: `[CMD, param, values..., NaN]`
pub fn fixed_mul_acc_stream(values: &[TinyNNFP16], count: usize, param: TinyNNFP16) -> Vec<u16> {
    assert!(count >= 1 && count <= 256, "Count must be 1-256");

    let cmd = CMD_FIXED_MUL_ACC | ((count - 1) as u16 & 0xFF);

    let mut output = vec![cmd];
    output.push(param.as_u16());
    output.extend(values.iter().map(|x| x.as_u16()));
    output.push(NAN_TERMINATOR);
    output
}

/// Build a max pool command stream.
///
/// Finds the maximum value in each group of `count` values.
///
/// # Arguments
///
/// * `values` - Values to process
/// * `count` - Number of values per pool (1-256)
///
/// # Panics
///
/// Panics if `count` is not in range 1-256.
///
/// # Returns
///
/// Command stream: `[CMD, values..., NaN]`
pub fn max_pool_stream(values: &[TinyNNFP16], count: usize) -> Vec<u16> {
    assert!(count >= 1 && count <= 256, "Count must be 1-256");

    let cmd = CMD_MAX_POOL | ((count - 1) as u16 & 0xFF);

    let mut output = vec![cmd];
    output.extend(values.iter().map(|x| x.as_u16()));
    output.push(NAN_TERMINATOR);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_stream() {
        let params: Vec<TinyNNFP16> = (0..8).map(|_| TinyNNFP16::zero()).collect();
        let data = vec![TinyNNFP16::from_f32(1.0), TinyNNFP16::from_f32(2.0)];
        let stream = convolve_stream(&params, &data);

        assert_eq!(stream[0], CMD_CONVOLVE);
        assert_eq!(*stream.last().unwrap(), NAN_TERMINATOR);
        assert_eq!(stream.len(), 1 + 8 + 2 + 1);
    }

    #[test]
    fn test_accumulate_stream() {
        let values = vec![TinyNNFP16::from_f32(1.0); 8];
        let bias = TinyNNFP16::from_f32(0.5);
        let stream = accumulate_stream(&values, 4, bias, true);

        assert_eq!(stream[0], CMD_ACCUMULATE | 0x100 | 3);
        assert_eq!(*stream.last().unwrap(), NAN_TERMINATOR);
    }

    #[test]
    fn test_mul_acc_stream() {
        let values = vec![TinyNNFP16::from_f32(1.0); 4];
        let bias = TinyNNFP16::zero();
        let stream = mul_acc_stream(&values, bias, false);

        assert_eq!(stream[0], CMD_MUL_ACC);
        assert_eq!(*stream.last().unwrap(), NAN_TERMINATOR);
    }

    #[test]
    fn test_max_pool_stream() {
        let values = vec![TinyNNFP16::from_f32(1.0); 4];
        let stream = max_pool_stream(&values, 4);

        assert_eq!(stream[0], CMD_MAX_POOL | 3);
        assert_eq!(*stream.last().unwrap(), NAN_TERMINATOR);
    }
}
