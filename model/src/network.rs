//! TCP server for TNN simulation.
//!
//! Provides network-based communication with the TNN simulator,
//! allowing external controllers to drive operations in real-time.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::input_source::InputSource;

/// Default port for TNN network server.
pub const DEFAULT_PORT: u16 = 9876;

/// An InputSource that reads 16-bit words from a TCP stream.
///
/// Words are received as big-endian byte pairs (high byte first).
pub struct TcpInputSource {
    stream: TcpStream,
}

impl TcpInputSource {
    pub fn new(stream: TcpStream) -> Self {
        TcpInputSource { stream }
    }
}

impl InputSource for TcpInputSource {
    fn next_word(&mut self) -> Option<u16> {
        let mut buf = [0u8; 2];
        match self.stream.read_exact(&mut buf) {
            Ok(_) => {
                // Big-endian: high byte first
                let word = ((buf[0] as u16) << 8) | (buf[1] as u16);
                Some(word)
            }
            Err(_) => None,
        }
    }

    fn has_more(&self) -> bool {
        // For TCP, we can't peek without consuming, so always return true
        // The actual end is detected when next_word returns None
        true
    }
}

/// Callback type for streaming output bytes.
pub type OutputCallback = Box<dyn FnMut(u8) + Send>;

/// TNN Network Server that accepts connections and runs simulation.
pub struct TnnNetworkServer {
    listener: TcpListener,
}

impl TnnNetworkServer {
    /// Bind to the specified address.
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(TnnNetworkServer { listener })
    }

    /// Accept a connection and run the simulation loop.
    ///
    /// Reads 16-bit words from the client, processes them through the simulator,
    /// and sends output bytes back immediately.
    pub fn accept_and_run(&self) -> std::io::Result<()> {
        println!("Waiting for connection...");
        let (stream, addr) = self.listener.accept()?;
        println!("Accepted connection from: {}", addr);

        // Clone stream for writing (we need both read and write)
        let write_stream = stream.try_clone()?;

        self.run_with_streams(stream, write_stream)
    }

    /// Run simulation with the given read and write streams.
    fn run_with_streams(&self, read_stream: TcpStream, mut write_stream: TcpStream) -> std::io::Result<()> {
        let mut input = TcpInputSource::new(read_stream);

        // Create simulator with output callback that writes to the stream
        let mut simulator = StreamingTnnSimulator::new(move |byte| {
            let _ = write_stream.write_all(&[byte]);
            let _ = write_stream.flush();
        });

        // Run simulation - this will block until input is exhausted
        simulator.run(&mut input);

        println!("Simulation complete");
        Ok(())
    }

    /// Get the local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }
}

/// A streaming version of TnnSimulator that outputs bytes immediately via callback.
///
/// This is based on the regular TnnSimulator but modified to support
/// streaming output for network operation.
pub struct StreamingTnnSimulator<F>
where
    F: FnMut(u8),
{
    output_callback: F,
}

impl<F> StreamingTnnSimulator<F>
where
    F: FnMut(u8),
{
    pub fn new(output_callback: F) -> Self {
        StreamingTnnSimulator { output_callback }
    }

    /// Run the simulation, consuming input and streaming output.
    pub fn run(&mut self, input: &mut dyn InputSource) {
        while let Some(cmd_word) = input.next_word() {
            self.decode_and_execute(cmd_word, input);
        }
    }

    fn decode_and_execute(&mut self, cmd_word: u16, input: &mut dyn InputSource) {
        use crate::tnn_types::{TinyNNFP16, TinyNNFP16Zero};
        use crate::ops::{do_accumulate, do_convolve, do_fixed_mul_acc, do_max_pool, ConvHeight, ConvWidth};
        use ndarray::Array2;

        const CMD_CONVOLVE: u16 = 0x1;
        const CMD_ACCUMULATE: u16 = 0x2;
        const CMD_MUL_ACC: u16 = 0x3;
        const CMD_FIXED_MUL_ACC: u16 = 0x4;
        const CMD_MAX_POOL: u16 = 0x5;
        const CMD_TEST: u16 = 0xF;

        const CONV_FIRST_OUTPUT_DELAY: usize = 22;
        const ACCUM_FIRST_OUTPUT_DELAY: usize = 5;
        const FIXED_MUL_ACC_FIRST_OUTPUT_DELAY: usize = 6;
        const MAX_POOL_FIRST_OUTPUT_DELAY: usize = 1;
        const MUL_ACC_OUTPUT_DELAY: usize = 6;

        let opcode = (cmd_word >> 12) & 0xF;

        match opcode {
            CMD_CONVOLVE => {
                // Read 8 parameter words
                let param_words = self.read_n_words(input, 8);
                if param_words.len() < 8 {
                    return;
                }

                let params: Vec<TinyNNFP16> = param_words
                    .iter()
                    .map(|w| TinyNNFP16::from_u16(*w))
                    .collect();
                let params_array =
                    Array2::from_shape_vec((ConvWidth, ConvHeight), params).expect("Failed to create params array");

                let image_words = self.read_until_nan(input);
                let image_values: Vec<TinyNNFP16> =
                    image_words.iter().map(|w| TinyNNFP16::from_u16(*w)).collect();

                let results = do_convolve(&image_values, &params_array);

                self.output_latency_padding(CONV_FIRST_OUTPUT_DELAY);
                self.output_fp16_values(&results);
            }

            CMD_ACCUMULATE => {
                let relu = (cmd_word & 0x100) != 0;
                let count = ((cmd_word & 0xFF) as usize) + 1;

                let bias_word = match input.next_word() {
                    Some(w) => w,
                    None => return,
                };
                let bias = TinyNNFP16::from_u16(bias_word);

                let value_words = self.read_until_nan(input);
                let values: Vec<TinyNNFP16> = value_words
                    .iter()
                    .map(|w| TinyNNFP16::from_u16(*w))
                    .collect();

                let results = do_accumulate(&values, count, bias, relu);

                self.output_latency_padding(ACCUM_FIRST_OUTPUT_DELAY + 2);

                for v in results.iter() {
                    self.output_latency_padding(count - 2);
                    let raw = v.as_u16();
                    (self.output_callback)((raw & 0xFF) as u8);
                    (self.output_callback)((raw >> 8) as u8);
                }
            }

            CMD_MUL_ACC => {
                let relu = (cmd_word & 0x100) != 0;

                let bias_word = match input.next_word() {
                    Some(w) => w,
                    None => return,
                };
                let bias = TinyNNFP16::from_u16(bias_word);

                let pair_words = self.read_until_nan(input);
                if pair_words.len() % 2 != 0 {
                    return;
                }

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

                let num_values = pair_words.len();
                self.output_latency_padding(MUL_ACC_OUTPUT_DELAY + num_values);

                let raw = result.as_u16();
                (self.output_callback)((raw & 0xFF) as u8);
                (self.output_callback)((raw >> 8) as u8);
            }

            CMD_FIXED_MUL_ACC => {
                let count = ((cmd_word & 0xFF) as usize) + 1;

                let param_word = match input.next_word() {
                    Some(w) => w,
                    None => return,
                };
                let param = TinyNNFP16::from_u16(param_word);

                let value_words = self.read_until_nan(input);
                let values: Vec<TinyNNFP16> = value_words
                    .iter()
                    .map(|w| TinyNNFP16::from_u16(*w))
                    .collect();

                let results = do_fixed_mul_acc(&values, count, param);

                self.output_latency_padding(FIXED_MUL_ACC_FIRST_OUTPUT_DELAY + 2);

                for v in results.iter() {
                    self.output_latency_padding(count - 2);
                    let raw = v.as_u16();
                    (self.output_callback)((raw & 0xFF) as u8);
                    (self.output_callback)((raw >> 8) as u8);
                }
            }

            CMD_MAX_POOL => {
                let count = ((cmd_word & 0xFF) as usize) + 1;

                let value_words = self.read_until_nan(input);
                let values: Vec<TinyNNFP16> = value_words
                    .iter()
                    .map(|w| TinyNNFP16::from_u16(*w))
                    .collect();

                let results = do_max_pool(&values, count);

                self.output_latency_padding(MAX_POOL_FIRST_OUTPUT_DELAY + 2);

                for v in results.iter() {
                    self.output_latency_padding(count - 2);
                    let raw = v.as_u16();
                    (self.output_callback)((raw & 0xFF) as u8);
                    (self.output_callback)((raw >> 8) as u8);
                }
            }

            CMD_TEST => {
                let subtype = (cmd_word >> 8) & 0xF;

                match subtype {
                    0xF => {
                        // ASCII test - output "T-NN"
                        for &b in &[0x54u8, 0x2D, 0x4E, 0x4E] {
                            (self.output_callback)(b);
                        }
                        while let Some(word) = input.next_word() {
                            if (word >> 8) != 0xFF {
                                break;
                            }
                            for &b in &[0x54u8, 0x2D, 0x4E, 0x4E] {
                                (self.output_callback)(b);
                            }
                        }
                    }
                    0x0 => {
                        // Pulse test
                        for &b in &[0xAAu8, 0x55] {
                            (self.output_callback)(b);
                        }
                        while let Some(word) = input.next_word() {
                            if (word >> 8) != 0xF0 {
                                break;
                            }
                            for &b in &[0xAAu8, 0x55] {
                                (self.output_callback)(b);
                            }
                        }
                    }
                    0x1 => {
                        // Count test
                        let mut count = (cmd_word & 0xFF) as u8;
                        loop {
                            (self.output_callback)(count);
                            if count == 0 {
                                break;
                            }
                            count = count.wrapping_sub(1);
                        }
                    }
                    _ => {}
                }
            }

            _ => {
                (self.output_callback)(0xFF);
            }
        }
    }

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

    fn read_until_nan(&self, input: &mut dyn InputSource) -> Vec<u16> {
        use crate::tnn_types::TinyNNFP16StdNaN;
        let mut words = Vec::new();
        while let Some(word) = input.next_word() {
            if word == TinyNNFP16StdNaN.as_u16() {
                break;
            }
            words.push(word);
        }
        words
    }

    fn output_fp16_values(&mut self, values: &[crate::tnn_types::TinyNNFP16]) {
        for v in values {
            let raw = v.as_u16();
            (self.output_callback)((raw & 0xFF) as u8);
            (self.output_callback)((raw >> 8) as u8);
        }
    }

    fn output_latency_padding(&mut self, cycles: usize) {
        for _ in 0..cycles {
            (self.output_callback)(0xFF);
        }
    }
}
