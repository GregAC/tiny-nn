//! Recording wrapper for TNN interfaces.
//!
//! Provides [`RecordingInterface`] which wraps any [`TnnInterface`] and
//! captures all sent/received data for analysis or replay.

use crate::comm::interface::TnnInterface;
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;

/// Wraps a TnnInterface and records all communication.
///
/// Useful for debugging, generating test vectors, or replaying operations.
///
/// # Example
///
/// ```ignore
/// use controller::comm::{RecordingInterface, NullInterface};
///
/// let mut recorder = RecordingInterface::new(NullInterface);
/// recorder.send_word(0x1000).unwrap();
/// recorder.send_word(0xFFFF).unwrap();
///
/// assert_eq!(recorder.sent_words(), &[0x1000, 0xFFFF]);
/// ```
pub struct RecordingInterface<T: TnnInterface> {
    inner: T,
    sent_words: Vec<u16>,
    received_bytes: Vec<u8>,
}

impl<T: TnnInterface> RecordingInterface<T> {
    /// Create a new recording interface wrapping an inner interface.
    pub fn new(inner: T) -> Self {
        RecordingInterface {
            inner,
            sent_words: Vec::new(),
            received_bytes: Vec::new(),
        }
    }

    /// Get all words sent to the interface.
    pub fn sent_words(&self) -> &[u16] {
        &self.sent_words
    }

    /// Get all bytes received from the interface.
    pub fn received_bytes(&self) -> &[u8] {
        &self.received_bytes
    }

    /// Clear the recording buffers.
    pub fn clear(&mut self) {
        self.sent_words.clear();
        self.received_bytes.clear();
    }

    /// Unwrap and return the inner interface.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: TnnInterface> TnnInterface for RecordingInterface<T> {
    fn send_word(&mut self, word: u16) -> Result<(), ControllerError> {
        self.sent_words.push(word);
        self.inner.send_word(word)
    }

    fn recv_byte(&mut self) -> Result<u8, ControllerError> {
        let byte = self.inner.recv_byte()?;
        self.received_bytes.push(byte);
        Ok(byte)
    }

    fn recv_fp16(&mut self) -> Result<TinyNNFP16, ControllerError> {
        let low = self.recv_byte()?;
        let high = self.recv_byte()?;
        let raw = (high as u16) << 8 | (low as u16);
        Ok(TinyNNFP16::from_u16(raw))
    }
}

/// A null interface that doesn't communicate.
///
/// Useful for testing or when only command generation is needed.
/// All sends succeed; all receives return 0.
#[derive(Default)]
pub struct NullInterface;

impl TnnInterface for NullInterface {
    fn send_word(&mut self, _word: u16) -> Result<(), ControllerError> {
        Ok(())
    }

    fn recv_byte(&mut self) -> Result<u8, ControllerError> {
        Ok(0)
    }
}
