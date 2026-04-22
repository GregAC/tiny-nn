use std::fs;
use std::io::BufWriter;
use std::path::Path;

use crate::tnn_types::TinyNNFP16StdNaN;
use crate::utils::write_output_stream;

const CONV_FIRST_OUTPUT_DELAY: usize = 22;
const ACCUM_FIRST_OUTPUT_DELAY_TOTAL: usize = 7; // AccumFirstOutputDelay(5) + 2
const FIXED_MUL_ACC_FIRST_OUTPUT_DELAY_TOTAL: usize = 8; // FixedMulAccFirstOutputDelay(6) + 2
const MAX_POOL_FIRST_OUTPUT_DELAY_TOTAL: usize = 3; // MaxPoolFirstOutputDelay(1) + 2
const MUL_ACC_OUTPUT_DELAY: usize = 6;

const OP_CONVOLVE: u16 = 0x1;
const OP_ACCUMULATE: u16 = 0x2;
const OP_MUL_ACC: u16 = 0x3;
const OP_FIXED_MUL_ACC: u16 = 0x4;
const OP_MAX_POOL: u16 = 0x5;
const OP_TEST: u16 = 0xF;

pub fn annotate_received(sent_path: &Path, received_path: &Path, output_path: &Path) {
    let sent_words = read_hex_words(sent_path);
    let received_bytes = read_hex_bytes(received_path);

    let validity_map = build_output_validity_map(&sent_words);

    if validity_map.len() != received_bytes.len() {
        eprintln!(
            "Warning: validity map length ({}) != received bytes length ({})",
            validity_map.len(),
            received_bytes.len()
        );
    }

    let annotated: Vec<Option<u8>> = received_bytes
        .iter()
        .enumerate()
        .map(|(i, &byte)| {
            if i < validity_map.len() && validity_map[i] {
                Some(byte)
            } else {
                None
            }
        })
        .collect();

    let file = fs::File::create(output_path)
        .unwrap_or_else(|_| panic!("Failed to create output file: {:?}", output_path));
    let mut writer = BufWriter::new(file);
    write_output_stream(&annotated, &mut writer);

    let valid_count = annotated.iter().filter(|b| b.is_some()).count();
    println!(
        "Wrote {} bytes ({} valid, {} marked as latency)",
        annotated.len(),
        valid_count,
        annotated.len() - valid_count
    );
}

fn read_hex_words(path: &Path) -> Vec<u16> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read sent hex file: {:?}", path));
    content
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(|line| {
            let trimmed = line.trim();
            u16::from_str_radix(trimmed, 16)
                .unwrap_or_else(|_| panic!("Invalid hex word: '{}'", trimmed))
        })
        .collect()
}

fn read_hex_bytes(path: &Path) -> Vec<u8> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read received hex file: {:?}", path));
    content
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(|line| {
            let trimmed = line.trim();
            u8::from_str_radix(trimmed, 16)
                .unwrap_or_else(|_| panic!("Invalid hex byte: '{}'", trimmed))
        })
        .collect()
}

fn build_output_validity_map(sent_words: &[u16]) -> Vec<bool> {
    let nan = TinyNNFP16StdNaN.as_u16();
    let mut validity = Vec::new();
    let mut i = 0;

    while i < sent_words.len() {
        let cmd_word = sent_words[i];
        i += 1;
        let opcode = (cmd_word >> 12) & 0xF;

        match opcode {
            OP_CONVOLVE => {
                i = (i + 8).min(sent_words.len()); // skip 8 param words
                let n = count_until_nan(sent_words, &mut i, nan);
                // total_results = out_h * image_width - (ConvWidth-1) = N/2 - 3
                validity.extend(std::iter::repeat(false).take(CONV_FIRST_OUTPUT_DELAY));
                if n >= 6 {
                    let num_results = n / 2 - 3;
                    validity.extend(std::iter::repeat(true).take(num_results * 2));
                }
            }
            OP_ACCUMULATE => {
                let count = ((cmd_word & 0xFF) as usize) + 1;
                i = (i + 1).min(sent_words.len()); // skip bias
                let n = count_until_nan(sent_words, &mut i, nan);
                let num_results = n / count;
                validity.extend(std::iter::repeat(false).take(ACCUM_FIRST_OUTPUT_DELAY_TOTAL));
                for _ in 0..num_results {
                    validity.extend(std::iter::repeat(false).take(count.saturating_sub(2)));
                    validity.extend(std::iter::repeat(true).take(2));
                }
            }
            OP_MUL_ACC => {
                i = (i + 1).min(sent_words.len()); // skip bias
                let n = count_until_nan(sent_words, &mut i, nan);
                validity.extend(std::iter::repeat(false).take(MUL_ACC_OUTPUT_DELAY + n));
                validity.extend(std::iter::repeat(true).take(2));
            }
            OP_FIXED_MUL_ACC => {
                let count = ((cmd_word & 0xFF) as usize) + 1;
                i = (i + 1).min(sent_words.len()); // skip param
                let n = count_until_nan(sent_words, &mut i, nan);
                let num_results = n / count;
                validity.extend(
                    std::iter::repeat(false).take(FIXED_MUL_ACC_FIRST_OUTPUT_DELAY_TOTAL),
                );
                for _ in 0..num_results {
                    validity.extend(std::iter::repeat(false).take(count.saturating_sub(2)));
                    validity.extend(std::iter::repeat(true).take(2));
                }
            }
            OP_MAX_POOL => {
                let count = ((cmd_word & 0xFF) as usize) + 1;
                let n = count_until_nan(sent_words, &mut i, nan);
                let num_results = n / count;
                validity.extend(std::iter::repeat(false).take(MAX_POOL_FIRST_OUTPUT_DELAY_TOTAL));
                for _ in 0..num_results {
                    validity.extend(std::iter::repeat(false).take(count.saturating_sub(2)));
                    validity.extend(std::iter::repeat(true).take(2));
                }
            }
            OP_TEST => {
                let subtype = (cmd_word >> 8) & 0xF;
                match subtype {
                    0xF => {
                        // ASCII test: 4 valid bytes per word while [15:8] == 0xFF
                        validity.extend(std::iter::repeat(true).take(4));
                        while i < sent_words.len() {
                            let word = sent_words[i];
                            i += 1;
                            if (word >> 8) != 0xFF {
                                break;
                            }
                            validity.extend(std::iter::repeat(true).take(4));
                        }
                    }
                    0x0 => {
                        // Pulse test: 2 valid bytes per word while [15:8] == 0xF0
                        validity.extend(std::iter::repeat(true).take(2));
                        while i < sent_words.len() {
                            let word = sent_words[i];
                            i += 1;
                            if (word >> 8) != 0xF0 {
                                break;
                            }
                            validity.extend(std::iter::repeat(true).take(2));
                        }
                    }
                    0x1 => {
                        // Count test: count down from `count` to 0 inclusive
                        let count = (cmd_word & 0xFF) as usize;
                        validity.extend(std::iter::repeat(true).take(count + 1));
                    }
                    _ => {}
                }
            }
            _ => {
                // Drain/idle words (e.g. 0x0000 padding between operations) produce
                // one 0xFF output byte which is not valid data.
                validity.push(false);
            }
        }
    }

    validity
}

fn count_until_nan(words: &[u16], i: &mut usize, nan: u16) -> usize {
    let mut count = 0;
    while *i < words.len() {
        let word = words[*i];
        *i += 1;
        if word == nan {
            break;
        }
        count += 1;
    }
    count
}
