//! Communication interfaces for TNN.
//!
//! This module provides traits and implementations for communicating with
//! TNN hardware or simulation. The primary interface is file-based for
//! generating test vectors.
//!
//! ## Components
//!
//! - [`TnnInterface`] - Trait for TNN communication
//! - [`HexFileWriter`] - Write commands to hex files
//! - [`CommandBuffer`] - Collect commands in memory
//! - [`RecordingInterface`] - Capture/replay wrapper
//! - [`TnnNetworkClient`] - TCP client for network communication

pub mod hex_file;
pub mod interface;
pub mod network;
pub mod recorder;

pub use hex_file::{write_bytes_hex_file, write_fp16_vec, CommandBuffer, HexFileWriter};
pub use interface::TnnInterface;
pub use network::{drain_to_idle, recv_fp16_values_with_latency, OutputLatency, TnnNetworkClient};
pub use recorder::{NullInterface, RecordingInterface};
