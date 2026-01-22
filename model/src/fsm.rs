use ndarray::Array2;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::input_source::InputSource;
use crate::ops::{do_accumulate, do_convolve, do_fixed_mul_acc, do_max_pool, ConvHeight, ConvWidth};
use crate::tnn_types::{TinyNNFP16, TinyNNFP16StdNaN, TinyNNFP16Zero};

// Command opcodes (upper 4 bits of command word)
const CMD_CONVOLVE: u16 = 0x1;
const CMD_ACCUMULATE: u16 = 0x2;
const CMD_MUL_ACC: u16 = 0x3;
const CMD_FIXED_MUL_ACC: u16 = 0x4;
const CMD_MAX_POOL: u16 = 0x5;
const CMD_TEST: u16 = 0xF;

// Latency constants (matching utils.rs)
const CONV_FIRST_OUTPUT_DELAY: usize = 22; // 14 + 8
const CONV_ROW_OUTPUT_DELAY: usize = 6;
const ACCUM_FIRST_OUTPUT_DELAY: usize = 5;
const FIXED_MUL_ACC_FIRST_OUTPUT_DELAY: usize = 6;
const MAX_POOL_FIRST_OUTPUT_DELAY: usize = 1;
const MUL_ACC_OUTPUT_DELAY: usize = 6;

/// The TNN simulator that models the FSM behavior.
pub struct TnnSimulator {
    output: Vec<u8>,
}

impl TnnSimulator {
    pub fn new() -> Self {
        TnnSimulator { output: Vec::new() }
    }

    /// Run the simulation, consuming input and producing output.
    pub fn run(&mut self, input: &mut dyn InputSource) {
        while let Some(cmd_word) = input.next_word() {
            self.decode_and_execute(cmd_word, input);
        }
    }

    /// Get the output bytes produced by the simulation.
    pub fn output(&self) -> &[u8] {
        &self.output
    }

    /// Write output to a hex file (one byte per line as 2 hex digits).
    pub fn write_output(&self, path: &Path) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        for byte in &self.output {
            writeln!(file, "{:02x}", byte)?;
        }
        file.flush()?;
        Ok(())
    }

    fn decode_and_execute(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        let opcode = (cmd_word >> 12) & 0xF;

        match opcode {
            CMD_CONVOLVE => self.handle_convolve(input),
            CMD_ACCUMULATE => self.handle_accumulate(cmd_word, input),
            CMD_MUL_ACC => self.handle_mul_acc(cmd_word, input),
            CMD_FIXED_MUL_ACC => self.handle_fixed_mul_acc(cmd_word, input),
            CMD_MAX_POOL => self.handle_max_pool(cmd_word, input),
            CMD_TEST => self.handle_test(cmd_word, input),
            _ => {
                // Unknown command - ignore
            }
        }
    }

    /// Read n words from input.
    fn read_n_words(&self, input: &mut dyn InputSource, n: usize) -> Vec<u16> {
        let mut words = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(word) = input.next_word() {
                words.push(word);
            } else {
                break;
            }
        }
        words
    }

    /// Read words until we see the standard NaN (0xFFFF).
    fn read_until_nan(&self, input: &mut dyn InputSource) -> Vec<u16> {
        let mut words = Vec::new();
        while let Some(word) = input.next_word() {
            if word == TinyNNFP16StdNaN.as_u16() {
                break;
            }
            words.push(word);
        }
        words
    }

    /// Output FP16 values as bytes (low byte first, then high byte).
    fn output_fp16_values(&mut self, values: &[TinyNNFP16]) {
        for v in values {
            let raw = v.as_u16();
            self.output.push((raw & 0xFF) as u8);
            self.output.push((raw >> 8) as u8);
        }
    }

    /// Output NaN bytes for the latency delay (0xFF for each cycle).
    fn output_latency_padding(&mut self, cycles: usize) {
        for _ in 0..cycles {
            self.output.push(0xFF);
        }
    }

    fn handle_convolve(&mut self, input: &mut dyn InputSource) {
        // Read 8 parameter words
        let param_words = self.read_n_words(input, 8);
        if param_words.len() < 8 {
            return; // Not enough parameters
        }

        // Convert to Array2<TinyNNFP16> (4x2 in column-major order)
        let params: Vec<TinyNNFP16> = param_words
            .iter()
            .map(|w| TinyNNFP16::from_u16(*w))
            .collect();
        let params_array =
            Array2::from_shape_vec((ConvWidth, ConvHeight), params).expect("Failed to create params array");

        // Read image data until NaN
        let image_words = self.read_until_nan(input);
        let image_values: Vec<TinyNNFP16> =
            image_words.iter().map(|w| TinyNNFP16::from_u16(*w)).collect();

        // Compute convolution result
        let results = do_convolve(&image_values, &params_array);

        // For convolve, output latency padding first, then results
        // The latency structure is: initial delay, then results with row gaps
        // But since we're doing batch processing, we just output:
        // - Initial latency padding
        // - All results (with inter-row delays for proper timing)

        // Calculate how many rows of output
        // Each row produces (image_width - 3) results
        // The convolve result is computed row by row in do_convolve
        // For now, output initial padding + all results
        // TODO: Add row delays if needed for exact timing match

        self.output_latency_padding(CONV_FIRST_OUTPUT_DELAY);
        self.output_fp16_values(&results);
    }

    fn handle_accumulate(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        // Parse command word: [8] = RELU, [7:0] = count-1
        let relu = (cmd_word & 0x100) != 0;
        let count = ((cmd_word & 0xFF) as usize) + 1; // count field is count-1

        // Read bias
        let bias_word = match input.next_word() {
            Some(w) => w,
            None => return,
        };
        let bias = TinyNNFP16::from_u16(bias_word);

        // Read values until NaN
        let value_words = self.read_until_nan(input);
        let values: Vec<TinyNNFP16> = value_words
            .iter()
            .map(|w| TinyNNFP16::from_u16(*w))
            .collect();

        // Compute accumulate result
        let results = do_accumulate(&values, count, bias, relu);

        // Output with proper latency structure
        // Initial delay, then for each result: gap of (count - 2) followed by 2 bytes
        self.output_latency_padding(ACCUM_FIRST_OUTPUT_DELAY + 2);

        for v in results.iter() {
            // Gap before each result
            self.output_latency_padding(count - 2);
            let raw = v.as_u16();
            self.output.push((raw & 0xFF) as u8);
            self.output.push((raw >> 8) as u8);
        }
    }

    fn handle_mul_acc(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        // Parse command word: [8] = RELU
        let relu = (cmd_word & 0x100) != 0;

        // Read bias
        let bias_word = match input.next_word() {
            Some(w) => w,
            None => return,
        };
        let bias = TinyNNFP16::from_u16(bias_word);

        // Read value/param pairs until NaN
        let pair_words = self.read_until_nan(input);
        if pair_words.len() % 2 != 0 {
            // Invalid - must be even number of values
            return;
        }

        // Compute mul_acc: sum(v_i * p_i) + bias, with optional RELU
        // Note: do_mul_acc in ops.rs is incomplete, so we implement correctly here
        let mut result = TinyNNFP16Zero;
        for chunk in pair_words.chunks(2) {
            let v = TinyNNFP16::from_u16(chunk[0]);
            let p = TinyNNFP16::from_u16(chunk[1]);
            result = result + (v * p);
        }
        result = result + bias;

        if relu && result.sgn() {
            result = TinyNNFP16Zero;
        }

        // Output: delay of (MulAccOutputDelay + num_values * 2) + result
        let num_values = pair_words.len();
        self.output_latency_padding(MUL_ACC_OUTPUT_DELAY + num_values);

        let raw = result.as_u16();
        self.output.push((raw & 0xFF) as u8);
        self.output.push((raw >> 8) as u8);
    }

    fn handle_fixed_mul_acc(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        // Parse command word: [7:0] = count-1
        let count = ((cmd_word & 0xFF) as usize) + 1;

        // Read fixed parameter
        let param_word = match input.next_word() {
            Some(w) => w,
            None => return,
        };
        let param = TinyNNFP16::from_u16(param_word);

        // Read values until NaN
        let value_words = self.read_until_nan(input);
        let values: Vec<TinyNNFP16> = value_words
            .iter()
            .map(|w| TinyNNFP16::from_u16(*w))
            .collect();

        // Compute fixed_mul_acc result
        let results = do_fixed_mul_acc(&values, count, param);

        // Output with proper latency structure
        // Initial delay, then for each result: gap of (count - 2) followed by 2 bytes
        self.output_latency_padding(FIXED_MUL_ACC_FIRST_OUTPUT_DELAY + 2);

        for v in results.iter() {
            // Gap before each result
            self.output_latency_padding(count - 2);
            let raw = v.as_u16();
            self.output.push((raw & 0xFF) as u8);
            self.output.push((raw >> 8) as u8);
        }
    }

    fn handle_max_pool(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        // Parse command word: [7:0] = count-1
        let count = ((cmd_word & 0xFF) as usize) + 1;

        // Read values until NaN
        let value_words = self.read_until_nan(input);
        let values: Vec<TinyNNFP16> = value_words
            .iter()
            .map(|w| TinyNNFP16::from_u16(*w))
            .collect();

        // Compute max_pool result
        let results = do_max_pool(&values, count);

        // Output with proper latency structure
        // Initial delay, then for each result: gap of (count - 2) followed by 2 bytes
        self.output_latency_padding(MAX_POOL_FIRST_OUTPUT_DELAY + 2);

        for v in results.iter() {
            // Gap before each result
            self.output_latency_padding(count - 2);
            let raw = v.as_u16();
            self.output.push((raw & 0xFF) as u8);
            self.output.push((raw >> 8) as u8);
        }
    }

    fn handle_test(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        let subtype = (cmd_word >> 8) & 0xF;

        match subtype {
            0xF => {
                // ASCII test - output "T-NN" continuously while input[15:8] == 0xFF
                self.output.extend_from_slice(&[0x54, 0x2D, 0x4E, 0x4E]);
                while let Some(word) = input.next_word() {
                    if (word >> 8) != 0xFF {
                        break;
                    }
                    self.output.extend_from_slice(&[0x54, 0x2D, 0x4E, 0x4E]);
                }
            }
            0x0 => {
                // Pulse test - output 0xAA, 0x55 continuously while input[15:8] == 0xF0
                self.output.extend_from_slice(&[0xAA, 0x55]);
                while let Some(word) = input.next_word() {
                    if (word >> 8) != 0xF0 {
                        break;
                    }
                    self.output.extend_from_slice(&[0xAA, 0x55]);
                }
            }
            0x1 => {
                // Count test - output count from provided value
                let mut count = (cmd_word & 0xFF) as u8;
                loop {
                    self.output.push(count);
                    if count == 0 {
                        break;
                    }
                    count = count.wrapping_sub(1);
                }
            }
            _ => {
                // Unknown test subtype - ignore
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_source::HexFileInput;

    #[test]
    fn test_count_test() {
        let mut sim = TnnSimulator::new();
        // Command: 0xF1XX where XX is starting count (e.g., 0xF103 for count from 3)
        let mut input = HexFileInput::from_vec(vec![0xF103]);
        sim.run(&mut input);

        assert_eq!(sim.output(), &[3, 2, 1, 0]);
    }

    #[test]
    fn test_pulse_test() {
        let mut sim = TnnSimulator::new();
        // Command: 0xF0XX, then continue while [15:8] == 0xF0
        let mut input = HexFileInput::from_vec(vec![0xF000, 0xF000, 0x0000]);
        sim.run(&mut input);

        assert_eq!(sim.output(), &[0xAA, 0x55, 0xAA, 0x55]);
    }

    #[test]
    fn test_ascii_test() {
        let mut sim = TnnSimulator::new();
        // Command: 0xFFXX, then continue while [15:8] == 0xFF
        let mut input = HexFileInput::from_vec(vec![0xFF00, 0xFF00, 0x0000]);
        sim.run(&mut input);

        // "T-NN" = 0x54, 0x2D, 0x4E, 0x4E
        assert_eq!(sim.output(), &[0x54, 0x2D, 0x4E, 0x4E, 0x54, 0x2D, 0x4E, 0x4E]);
    }
}
