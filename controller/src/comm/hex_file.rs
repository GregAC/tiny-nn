//! Hex file output for TNN command streams.
//!
//! This module provides utilities for writing TNN commands and data to
//! hex files that can be used for simulation testing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;

/// Writes TNN command streams to hex files.
///
/// Each 16-bit word is written as a 4-character hexadecimal value on its own line.
///
/// # Example
///
/// ```no_run
/// use controller::comm::HexFileWriter;
///
/// let mut writer = HexFileWriter::new("commands.hex").unwrap();
/// writer.write_word(0x1000).unwrap();  // Convolve opcode
/// writer.write_word(0xFFFF).unwrap();  // NaN terminator
/// writer.flush().unwrap();
/// ```
pub struct HexFileWriter {
    writer: BufWriter<File>,
}

impl HexFileWriter {
    /// Create a new hex file writer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, ControllerError> {
        let file = File::create(path.as_ref()).map_err(|e| {
            ControllerError::IoError(format!(
                "Failed to create hex file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Ok(HexFileWriter {
            writer: BufWriter::new(file),
        })
    }

    /// Write a single 16-bit word as hex.
    pub fn write_word(&mut self, word: u16) -> Result<(), ControllerError> {
        writeln!(self.writer, "{:04x}", word).map_err(|e| {
            ControllerError::IoError(format!("Failed to write hex: {}", e))
        })
    }

    /// Write a stream of 16-bit words.
    pub fn write_stream(&mut self, stream: &[u16]) -> Result<(), ControllerError> {
        for &word in stream {
            self.write_word(word)?;
        }
        Ok(())
    }

    /// Write a FP16 value.
    pub fn write_fp16(&mut self, value: TinyNNFP16) -> Result<(), ControllerError> {
        self.write_word(value.as_u16())
    }

    /// Flush any buffered data to disk.
    pub fn flush(&mut self) -> Result<(), ControllerError> {
        self.writer.flush().map_err(|e| {
            ControllerError::IoError(format!("Failed to flush: {}", e))
        })
    }
}

/// Collects commands in memory before writing to file.
///
/// Useful when building command streams incrementally before saving.
///
/// # Example
///
/// ```
/// use controller::comm::CommandBuffer;
///
/// let mut buf = CommandBuffer::new();
/// buf.push(0x1000);  // Convolve opcode
/// buf.extend(&[0x3F80, 0x0000]);  // Some data
/// buf.push(0xFFFF);  // NaN terminator
///
/// assert_eq!(buf.len(), 4);
/// ```
#[derive(Default)]
pub struct CommandBuffer {
    commands: Vec<u16>,
}

impl CommandBuffer {
    /// Create a new empty command buffer.
    pub fn new() -> Self {
        CommandBuffer { commands: Vec::new() }
    }

    /// Add a single command word.
    pub fn push(&mut self, word: u16) {
        self.commands.push(word);
    }

    /// Add multiple command words.
    pub fn extend(&mut self, stream: &[u16]) {
        self.commands.extend_from_slice(stream);
    }

    /// Clear all commands.
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Get commands as a slice.
    pub fn as_slice(&self) -> &[u16] {
        &self.commands
    }

    /// Get number of commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Write all commands to a hex file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ControllerError> {
        let mut writer = HexFileWriter::new(path)?;
        writer.write_stream(&self.commands)?;
        writer.flush()
    }
}

/// Write a vector of FP16 values to a hex file.
///
/// # Arguments
///
/// * `values` - FP16 values to write
/// * `path` - Path to the output file
pub fn write_fp16_vec<P: AsRef<Path>>(
    values: &[TinyNNFP16],
    path: P,
) -> Result<(), ControllerError> {
    let mut writer = HexFileWriter::new(path)?;
    for &v in values {
        writer.write_fp16(v)?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_buffer() {
        let mut buf = CommandBuffer::new();
        buf.push(0x1000);
        buf.extend(&[0x0001, 0x0002]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.as_slice(), &[0x1000, 0x0001, 0x0002]);
    }
}
