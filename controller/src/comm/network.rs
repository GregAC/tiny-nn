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

/// Helper to skip latency padding bytes (0xFF) in the output stream.
pub fn skip_latency_padding(
    iface: &mut impl TnnInterface,
) -> Result<TinyNNFP16, ControllerError> {
    loop {
        let low = iface.recv_byte()?;
        if low != 0xFF {
            // This is the low byte of an FP16 value
            let high = iface.recv_byte()?;
            let raw = ((high as u16) << 8) | (low as u16);
            return Ok(TinyNNFP16::from_u16(raw));
        }
        // Otherwise it's a latency padding byte, continue
    }
}

/// Receive an expected number of FP16 values, skipping latency padding.
pub fn recv_fp16_values_with_padding(
    iface: &mut impl TnnInterface,
    count: usize,
) -> Result<Vec<TinyNNFP16>, ControllerError> {
    let mut results = Vec::with_capacity(count);
    for _ in 0..count {
        results.push(skip_latency_padding(iface)?);
    }
    Ok(results)
}
