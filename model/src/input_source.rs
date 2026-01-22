use std::fs;
use std::path::Path;

/// Trait for providing 16-bit input words to the TNN simulation.
/// This abstraction allows for different input sources:
/// - HexFileInput: Read from .hex file (for testing/verification)
/// - Future: SocketInput for real-time communication with a controller
pub trait InputSource {
    /// Get the next 16-bit input word.
    /// Returns None when input is exhausted.
    fn next_word(&mut self) -> Option<u16>;

    /// Check if more input is available without consuming it.
    fn has_more(&self) -> bool;
}

/// Read 16-bit words from a .hex file (one hex value per line).
/// Supports 4-digit hex values (16-bit words).
pub struct HexFileInput {
    values: Vec<u16>,
    index: usize,
}

impl HexFileInput {
    /// Load input from a hex file.
    /// Each line should contain a 4-digit hex value (16-bit word).
    /// Lines starting with '#' or empty lines are ignored.
    pub fn from_file(path: &Path) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let values = content
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
            .map(|line| {
                let trimmed = line.trim();
                u16::from_str_radix(trimmed, 16)
                    .expect(&format!("Invalid hex value on line: '{}'", trimmed))
            })
            .collect();

        Ok(HexFileInput { values, index: 0 })
    }

    /// Create from an existing vector (useful for testing).
    pub fn from_vec(values: Vec<u16>) -> Self {
        HexFileInput { values, index: 0 }
    }

    /// Reset to the beginning of the input.
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl InputSource for HexFileInput {
    fn next_word(&mut self) -> Option<u16> {
        if self.index < self.values.len() {
            let value = self.values[self.index];
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn has_more(&self) -> bool {
        self.index < self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_file_input_from_vec() {
        let mut input = HexFileInput::from_vec(vec![0x1234, 0x5678, 0xFFFF]);

        assert!(input.has_more());
        assert_eq!(input.next_word(), Some(0x1234));
        assert_eq!(input.next_word(), Some(0x5678));
        assert_eq!(input.next_word(), Some(0xFFFF));
        assert_eq!(input.next_word(), None);
        assert!(!input.has_more());
    }

    #[test]
    fn test_reset() {
        let mut input = HexFileInput::from_vec(vec![0x1234, 0x5678]);

        assert_eq!(input.next_word(), Some(0x1234));
        assert_eq!(input.next_word(), Some(0x5678));
        assert_eq!(input.next_word(), None);

        input.reset();

        assert_eq!(input.next_word(), Some(0x1234));
        assert_eq!(input.next_word(), Some(0x5678));
    }
}
