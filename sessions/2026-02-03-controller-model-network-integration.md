# Session: Controller-Model Network Integration

**Date:** 2026-02-03
**Duration:** ~45 minutes
**Model:** Claude Opus 4.5

## Objective

Enable running a complete CNN through TNN by adding network communication between the controller and model, implementing live execution, and creating validation tools.

## Changes Made

### Phase 1: Model TCP Server

**New file: `model/src/network.rs`**
- `TcpInputSource`: InputSource implementation that reads 16-bit words from TCP stream
- `TnnNetworkServer`: TCP server that accepts connections and runs simulation
- `StreamingTnnSimulator`: Modified simulator that outputs bytes immediately via callback (for real-time streaming over network)

**Modified: `model/src/cli.rs`**
- Added `Serve` subcommand with `--port` (default 9876) and `--host` (default 127.0.0.1) options

**Modified: `model/src/main.rs`**
- Added `network` module
- Wired up `run_server()` function for the serve command

### Phase 2: Controller TCP Client

**New file: `controller/src/comm/network.rs`**
- `TnnNetworkClient`: Implements `TnnInterface` over TCP connection
- `skip_latency_padding()`: Helper to skip 0xFF padding bytes in output stream
- `recv_fp16_values_with_padding()`: Receive expected number of FP16 values, skipping padding

**Modified: `controller/src/comm/mod.rs`**
- Added network module and exports

### Phase 3: Live Execution in Runner

**Modified: `controller/src/runner.rs`**
- Added `LayerOutputs` struct with `final_output` and `layer_outputs` HashMap
- Added `run<T: TnnInterface>()` method that:
  - Executes model layer by layer through live TNN interface
  - Collects real outputs from TNN and feeds to subsequent layers
  - Supports optional intermediate layer output capture
- Added `model_name()` accessor method

### Phase 4: JSON I/O

**New file: `controller/src/io.rs`**
- `JsonInput` / `JsonOutput`: Serde structs for JSON format
- `read_input_json()`: Load input values from JSON file
- `write_output_json()`: Write results to JSON file with optional intermediate layers

**Modified: `controller/Cargo.toml`**
- Added `serde_json = "1.0"` dependency

**Modified: `controller/src/main.rs`**
- Added `Run` subcommand with `--model`, `--input`, `--output`, `--host`, `--intermediate` options

**Modified: `controller/src/lib.rs`**
- Added io module and exports for new types

### Phase 5: PyTorch Comparison & Testing

**New file: `util/pytorch_compare.py`**
- Builds PyTorch model from TOML config
- Loads weights from inline values or hex files
- Compares TNN output with PyTorch reference
- Reports mismatches with configurable tolerance

**New file: `controller/examples/simple_cnn.py`**
- Reference PyTorch implementation of SimpleCNN model
- Can load weights from TOML and run inference

**New file: `controller/examples/test_input_8x8.json`**
- Test input data (8x8 gradient pattern)

**New file: `util/test_simple_cnn.sh`**
- End-to-end test script that:
  1. Starts model server
  2. Runs controller
  3. Compares with PyTorch (if available)
  4. Reports pass/fail

## Files Changed

| File | Status | Lines |
|------|--------|-------|
| `model/src/network.rs` | New | ~280 |
| `model/src/cli.rs` | Modified | +12 |
| `model/src/main.rs` | Modified | +20 |
| `controller/src/comm/network.rs` | New | ~120 |
| `controller/src/comm/mod.rs` | Modified | +5 |
| `controller/src/runner.rs` | Modified | +170 |
| `controller/src/io.rs` | New | ~100 |
| `controller/src/main.rs` | Modified | +50 |
| `controller/src/lib.rs` | Modified | +5 |
| `controller/Cargo.toml` | Modified | +1 |
| `util/pytorch_compare.py` | New | ~200 |
| `controller/examples/simple_cnn.py` | New | ~130 |
| `controller/examples/test_input_8x8.json` | New | ~12 |
| `util/test_simple_cnn.sh` | New | ~90 |

## Usage

**Start model server:**
```bash
cargo run -p model -- serve --port 9876
```

**Run controller with live TNN:**
```bash
cargo run -p controller -- run \
  --model controller/examples/simple_cnn.toml \
  --input controller/examples/test_input_8x8.json \
  --output /tmp/tnn_output.json \
  --host localhost:9876 \
  --intermediate
```

**Compare with PyTorch:**
```bash
python util/pytorch_compare.py \
  controller/examples/simple_cnn.toml \
  controller/examples/test_input_8x8.json \
  /tmp/tnn_output.json \
  --tolerance 0.05
```

**Run full test:**
```bash
./util/test_simple_cnn.sh
```

## Architecture

```
┌─────────────────┐     TCP (9876)      ┌─────────────────┐
│   Controller    │◄──────────────────►│   Model Server   │
│                 │                     │                 │
│  - Load model   │  send: 16-bit words │  - TnnSimulator │
│  - JSON I/O     │  recv: 8-bit bytes  │  - Streaming    │
│  - Live exec    │                     │    output       │
└─────────────────┘                     └─────────────────┘
        │
        ▼
┌─────────────────┐
│  PyTorch        │
│  Comparison     │
└─────────────────┘
```

## Notes

- The streaming simulator duplicates some logic from `fsm.rs` to support immediate output via callback rather than accumulating in a Vec
- Network protocol uses big-endian for 16-bit words (high byte first)
- TNN outputs FP16 as two bytes (low byte first) with 0xFF padding for latency
- Conv2d live execution collects partial results from all convolve ops before running accumulate

## Future Work

- Consider refactoring `fsm.rs` and `StreamingTnnSimulator` to share code
- Add timeout handling for network operations
- Support multiple concurrent connections
- Add progress reporting for long-running models
