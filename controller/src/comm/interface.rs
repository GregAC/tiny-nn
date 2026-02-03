//! TNN communication interface trait.
//!
//! Defines the [`TnnInterface`] trait for sending commands to and receiving
//! data from TNN hardware or simulation.

use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;

/// Trait for communicating with TNN hardware or simulation.
///
/// Implementations can target different backends:
/// - File-based (for generating test vectors)
/// - TCP sockets (for real-time communication)
/// - Direct hardware interface
///
/// # Example
///
/// ```ignore
/// use controller::comm::TnnInterface;
///
/// fn run_operation<T: TnnInterface>(iface: &mut T, cmd: &[u16]) -> Result<(), ControllerError> {
///     iface.send_stream(cmd)?;
///     let result = iface.recv_fp16()?;
///     println!("Result: {:?}", result);
///     Ok(())
/// }
/// ```
pub trait TnnInterface {
    /// Send a 16-bit word to TNN.
    fn send_word(&mut self, word: u16) -> Result<(), ControllerError>;

    /// Receive a single byte from TNN.
    fn recv_byte(&mut self) -> Result<u8, ControllerError>;

    /// Receive a FP16 value (low byte first, then high byte).
    ///
    /// TNN outputs FP16 values as two bytes, low byte first.
    fn recv_fp16(&mut self) -> Result<TinyNNFP16, ControllerError> {
        let low = self.recv_byte()?;
        let high = self.recv_byte()?;
        let raw = (high as u16) << 8 | (low as u16);
        Ok(TinyNNFP16::from_u16(raw))
    }

    /// Send a complete command stream.
    ///
    /// Convenience method that sends each word in the stream.
    fn send_stream(&mut self, stream: &[u16]) -> Result<(), ControllerError> {
        for &word in stream {
            self.send_word(word)?;
        }
        Ok(())
    }

    /// Receive multiple FP16 values.
    ///
    /// Convenience method that receives `count` FP16 values.
    fn recv_fp16_vec(&mut self, count: usize) -> Result<Vec<TinyNNFP16>, ControllerError> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.recv_fp16()?);
        }
        Ok(result)
    }
}
