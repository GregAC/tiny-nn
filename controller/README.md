# TNN Controller

A Rust controller that translates CNN (Convolutional Neural Network) architecture
descriptions into low-level TNN (Tiny Neural Network) accelerator operations.

## Overview

The TNN controller bridges the gap between high-level neural network descriptions
and the hardware-level operations of the TNN accelerator. It takes CNN models
described in TOML format and generates sequences of TNN commands that can be
executed on the hardware or in simulation.

## Features

- **TOML Model Parsing**: Load CNN architectures from human-readable TOML files
- **Layer Translation**: Convert high-level layers to TNN operations:
  - `conv2d` → `convolve` + `accumulate`
  - `linear` → `mul_acc`
  - `avg_pool2d` → `fixed_mul_acc`
  - `max_pool2d` → `max_pool`
  - `flatten` → (shape tracking only)
- **Shape Tracking**: Automatically compute tensor dimensions through the network
- **Hex File Generation**: Output command streams compatible with TNN simulation

## Installation

```bash
cd controller
cargo build --release
```

## Usage

### Validate a Model

Check that a CNN model TOML file is valid and meets TNN constraints:

```bash
cargo run -- validate --model path/to/model.toml
```

### Show Execution Plan

Display the execution plan showing how each layer maps to TNN operations:

```bash
cargo run -- plan --model path/to/model.toml
```

Example output:
```
Model: SimpleCNN
Input: [1, 8, 8]
Output: 2 classes

Execution Plan:
------------------------------------------------------------
 1. AvgPool2d 'pool1': Image { channels: 1, height: 8, width: 8 } -> Image { channels: 1, height: 4, width: 4 }
    1 fixed_mul_acc op
 2. Conv2d 'conv1': Image { channels: 1, height: 4, width: 4 } -> Image { channels: 2, height: 1, width: 1 }
    4 convolve + 2 accumulate ops
 3. Flatten 'flatten1': Image { channels: 2, height: 1, width: 1 } -> 2 elements
    (no TNN operation)
 4. Linear 'fc1': 2 -> 2
    2 mul_acc ops
```

### Generate Commands

Generate TNN command streams from a model and input data:

```bash
cargo run -- generate \
    --model path/to/model.toml \
    --input path/to/input.hex \
    --output-dir ./output
```

This produces:
- `commands.hex`: TNN command stream (16-bit words, one per line)
- `input_fp16.hex`: Input data converted to TNN FP16 format

## Model Format

Models are described in TOML format. See [docs/cnn_schema.md](../docs/cnn_schema.md)
for the complete schema specification.

Example model:
```toml
[metadata]
name = "SimpleCNN"
description = "Example CNN"
input_shape = [1, 28, 28]  # [channels, height, width]
output_size = 10

[[layer]]
type = "conv2d"
name = "conv1"
in_channels = 1
out_channels = 8
kernel_size = 4
stride = 1
relu = true

[layer.weights]
source = "inline"
values = [0.1, 0.2, ...]  # 8 * 1 * 4 * 4 = 128 values

[layer.bias]
source = "inline"
values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

[[layer]]
type = "flatten"
name = "flatten1"

[[layer]]
type = "linear"
name = "fc1"
in_features = 200
out_features = 10

[layer.weights]
source = "file"
path = "fc1_weights.hex"

[layer.bias]
source = "file"
path = "fc1_bias.hex"
```

## Hex File Format

Input and weight files use a simple hex format with one 16-bit value per line:

```
3f80
bf00
3e80
```

Each line is a 4-character hexadecimal value representing a TNN FP16 number.

## Architecture

### Modules

- **`cnn`**: CNN model parsing and representation
  - `schema`: TOML schema types (CnnModel, Layer variants)
  - `parser`: File loading utilities
  - `tensor`: TensorShape tracking through layers

- **`translation`**: Layer-to-operation translation
  - `conv2d`: Convolution decomposition into 4x2 tiles
  - `linear`: Linear layer to mul_acc
  - `pooling`: Pooling operations

- **`comm`**: Communication interfaces
  - `interface`: TnnInterface trait
  - `hex_file`: File-based output
  - `recorder`: Recording wrapper for capture/replay

- **`tnn_ops`**: TNN operation encoding (opcodes, command building)
- **`fp16`**: TNN 16-bit floating point type
- **`runner`**: CnnRunner orchestration

### TNN Operations

| Layer Type | TNN Operation(s) | Notes |
|------------|------------------|-------|
| `conv2d` | convolve, accumulate | Multiple convolve for large kernels |
| `linear` | mul_acc | One per output neuron |
| `avg_pool2d` | fixed_mul_acc | Parameter = 1/(kernel_size²) |
| `max_pool2d` | max_pool | Direct mapping |
| `flatten` | (none) | Shape tracking only |

### Convolution Decomposition

TNN's convolve operation uses a fixed 4x2 kernel. Larger kernels are decomposed:

- A 4x4 kernel → 2 tiles (1 wide × 2 tall)
- A 5x5 kernel → 6 tiles (2 wide × 3 tall)
- A 3x3 kernel → 2 tiles (with zero padding)

Each tile's results are accumulated with bias and optional ReLU.

## Constraints

Models must satisfy these constraints:
- Conv2d stride must be 1
- Only ReLU activation is supported
- Square kernels and pooling windows only
- No padding (valid convolutions)
- Sequential layer execution

## Testing

Run the test suite:

```bash
cargo test
```

## Library Usage

The controller can also be used as a library:

```rust
use controller::{load_model, CnnRunner, TensorShape};

// Load a model
let model = load_model("model.toml")?;

// Create a runner
let runner = CnnRunner::new(model, ".");

// Generate commands for an input
let input: Vec<TinyNNFP16> = /* load input */;
let commands = runner.generate_commands(&input)?;

// Write to file
commands.write_to_file("commands.hex")?;
```

## Related Documentation

- [CNN Schema](../docs/cnn_schema.md) - TOML schema for CNN layers
- [TNN Operations](../docs/operations.md) - Low-level operation encoding
- [Architecture](../docs/architecture.md) - TNN hardware architecture
