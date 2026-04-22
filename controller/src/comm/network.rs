//! TCP client for TNN communication.
//!
//! Implements [`TnnInterface`] over a TCP connection for real-time
//! communication with the TNN model server.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;
use super::TnnInterface;

/// Latency configuration for receiving TNN operation results.
///
/// TNN operations have deterministic latency - a fixed number of cycles
/// before the first output, and potentially fixed gaps between subsequent
/// outputs. This struct captures those timing parameters.
#[derive(Debug, Clone, Copy)]
pub struct OutputLatency {
    /// Number of padding bytes before the first FP16 result.
    pub initial_padding: usize,
    /// Number of padding bytes between each subsequent FP16 result.
    pub inter_result_padding: usize,
}

/// Default port for TNN network communication.
pub const DEFAULT_PORT: u16 = 9876;

/// TCP client that implements TnnInterface for network communication.
pub struct TnnNetworkClient {
    stream: TcpStream,
}

impl TnnNetworkClient {
    /// Connect to a TNN server at the specified address.
    ///
    /// # Arguments
    ///
    /// * `addr` - Address in the form "host:port"
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = TnnNetworkClient::connect("localhost:9876")?;
    /// ```
    pub fn connect(addr: &str) -> Result<Self, ControllerError> {
        let stream = TcpStream::connect(addr).map_err(|e| {
            ControllerError::CommError(format!("Failed to connect to {}: {}", addr, e))
        })?;

        // Set a read timeout to avoid hanging forever
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| {
                ControllerError::CommError(format!("Failed to set read timeout: {}", e))
            })?;

        Ok(TnnNetworkClient { stream })
    }

    /// Connect with a specific timeout.
    pub fn connect_with_timeout(addr: &str, timeout: Duration) -> Result<Self, ControllerError> {
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| {
                ControllerError::CommError(format!("Invalid address '{}': {}", addr, e))
            })?,
            timeout,
        )
        .map_err(|e| {
            ControllerError::CommError(format!("Failed to connect to {}: {}", addr, e))
        })?;

        stream.set_read_timeout(Some(timeout)).map_err(|e| {
            ControllerError::CommError(format!("Failed to set read timeout: {}", e))
        })?;

        Ok(TnnNetworkClient { stream })
    }

    /// Close the connection gracefully.
    pub fn close(self) -> Result<(), ControllerError> {
        self.stream.shutdown(std::net::Shutdown::Both).map_err(|e| {
            ControllerError::CommError(format!("Failed to close connection: {}", e))
        })
    }
}

impl TnnInterface for TnnNetworkClient {
    fn send_word(&mut self, word: u16) -> Result<(), ControllerError> {
        // Send as big-endian (high byte first) to match server expectation
        let bytes = [(word >> 8) as u8, (word & 0xFF) as u8];
        self.stream.write_all(&bytes).map_err(|e| {
            ControllerError::CommError(format!("Failed to send word: {}", e))
        })?;
        self.stream.flush().map_err(|e| {
            ControllerError::CommError(format!("Failed to flush: {}", e))
        })
    }

    fn recv_byte(&mut self) -> Result<u8, ControllerError> {
        let mut buf = [0u8; 1];
        self.stream.read_exact(&mut buf).map_err(|e| {
            ControllerError::CommError(format!("Failed to receive byte: {}", e))
        })?;
        Ok(buf[0])
    }
}

/// Drain the TNN FSM back to idle after an operation completes.
///
/// Sends `drain_words` zero words (consumed by the ExecEnd drain states and
/// one idle cycle), then reads back the corresponding 0xFF output bytes.
/// This must be called after receiving all output for an operation and before
/// sending the next operation's command word.
pub fn drain_to_idle(
    iface: &mut impl TnnInterface,
    drain_words: usize,
) -> Result<(), ControllerError> {
    for _ in 0..drain_words {
        iface.send_word(0)?;
    }
    skip_padding_bytes(iface, drain_words)?;
    Ok(())
}

/// Skip a known number of padding bytes.
fn skip_padding_bytes(
    iface: &mut impl TnnInterface,
    count: usize,
) -> Result<(), ControllerError> {
    for _ in 0..count {
        let _ = iface.recv_byte()?;
    }
    Ok(())
}

/// Read a single FP16 value (low byte first, then high byte).
fn recv_fp16(iface: &mut impl TnnInterface) -> Result<TinyNNFP16, ControllerError> {
    let low = iface.recv_byte()?;
    let high = iface.recv_byte()?;
    let raw = ((high as u16) << 8) | (low as u16);
    Ok(TinyNNFP16::from_u16(raw))
}

/// Receive FP16 values with known latency padding.
///
/// This function skips the deterministic padding bytes output by TNN operations
/// based on known latency counts, rather than trying to detect padding by byte
/// value (which fails when valid FP16 values contain 0xFF).
///
/// The TNN output pattern is:
/// - Initial padding bytes
/// - For each result: inter-result padding bytes, then low byte, then high byte
///
/// # Arguments
///
/// * `iface` - TNN interface to receive from
/// * `count` - Number of FP16 values to receive
/// * `latency` - Latency configuration specifying padding byte counts
pub fn recv_fp16_values_with_latency(
    iface: &mut impl TnnInterface,
    count: usize,
    latency: OutputLatency,
) -> Result<Vec<TinyNNFP16>, ControllerError> {
    let mut results = Vec::with_capacity(count);

    // Skip initial padding before first result
    skip_padding_bytes(iface, latency.initial_padding)?;

    for _ in 0..count {
        // Skip inter-result padding (this comes BEFORE each result's data bytes)
        skip_padding_bytes(iface, latency.inter_result_padding)?;
        results.push(recv_fp16(iface)?);
    }

    Ok(results)
}
