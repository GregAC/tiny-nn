# CNN Architecture Extractor for TNN

This tool extracts CNN architecture from PyTorch modules and outputs TOML files
describing the architecture and weights for use with TNN (Tiny Neural Network
accelerator).

## Installation

Requires Python 3.8+ with PyTorch installed:

```bash
pip install torch
```

## Usage

### Command Line

```bash
# Basic usage - output to stdout (uses randomly initialized weights)
python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28

# Load pre-trained weights from checkpoint
python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 --checkpoint weights.pt -o model.toml

# Save to file
python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 -o model.toml

# Use file-based weights (hex files)
python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 --checkpoint weights.pt --weights-format file -o model.toml

# Validate only (no output)
python -m util.cnn_extractor --validate-only model.py MiniCNN --input-shape 1 28 28
```

### Loading Pre-trained Weights

The `--checkpoint` option accepts checkpoint files saved with `torch.save()`. Supported formats:

```python
# Format 1: Save state_dict (recommended)
torch.save(model.state_dict(), 'weights.pt')

# Format 2: Save full model
torch.save(model, 'model.pt')

# Format 3: Training checkpoint with state_dict key
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')
```

All three formats are automatically detected and handled.

### Python API

```python
import torch.nn as nn
from util.cnn_extractor import CNNExtractor

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 12 * 12, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Extract architecture
model = MiniCNN()
extractor = CNNExtractor(model, input_shape=(1, 28, 28))

# Get TOML string
toml_content = extractor.to_toml(
    name="MiniCNN",
    description="Example CNN for MNIST"
)
print(toml_content)

# Or save directly to file
extractor.save_toml("model.toml", name="MiniCNN")
```

## Supported Layers

| PyTorch Layer | TOML `type` | TNN Operation(s) | Notes |
|---------------|-------------|------------------|-------|
| `nn.Conv2d` | `conv2d` | convolve + accumulate | stride must be 1 |
| `nn.Linear` | `linear` | mul_acc | - |
| `nn.AvgPool2d` | `avg_pool2d` | fixed_mul_acc | - |
| `nn.MaxPool2d` | `max_pool2d` | max_pool | - |
| `nn.Flatten` | `flatten` | (reshape only) | - |
| `F.relu` / `nn.ReLU` | (relu flag) | - | Attached to preceding layer |

## Constraints

The extractor requires models to follow these constraints:

1. **All layers must be defined as class attributes**
   - The model must use `nn.Module` subclasses as attributes
   - Inline layer creation in `forward()` is not supported

2. **`forward()` must use only supported operations**
   - See supported layers table above
   - Unsupported layers will raise `UnsupportedLayerError`

3. **No branching or conditional logic in `forward()`**
   - No `if` statements based on input
   - No loops with dynamic bounds
   - FX tracing requires a single execution path

4. **Conv2d stride must be 1**
   - TNN does not support strided convolutions
   - Use pooling layers for downsampling

5. **Only ReLU activation is supported**
   - No sigmoid, tanh, softmax, etc.
   - ReLU is attached as a flag on the preceding layer

6. **No grouped convolutions**
   - `groups` parameter must be 1

7. **No dilation in convolutions**
   - `dilation` parameter must be 1

## Output Format

The TOML output describes layers at a high level. The controller (not this tool)
is responsible for decomposing layers into low-level TNN operations.

### Example Output

```toml
[metadata]
name = "MiniCNN"
description = "Example CNN for MNIST"
input_shape = [1, 28, 28]
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
values = [0.1, 0.2, ...]

[layer.bias]
source = "inline"
values = [0.0, 0.0, ...]

[[layer]]
type = "max_pool2d"
name = "pool1"
kernel_size = 2
stride = 2

[[layer]]
type = "flatten"
name = "flatten1"

[[layer]]
type = "linear"
name = "fc1"
in_features = 1152
out_features = 10

[layer.weights]
source = "inline"
values = [...]

[layer.bias]
source = "inline"
values = [...]
```

## Weight Formats

### Inline (default)

Weights are stored directly in the TOML file as float arrays:

```toml
[layer.weights]
source = "inline"
values = [0.1, 0.2, 0.3, ...]
```

### File

Weights are stored in separate hex files (TNN FP16 format):

```toml
[layer.weights]
source = "file"
path = "conv1_weights.hex"
```

Hex files contain one 4-character hex value per line representing TNN's custom
16-bit floating point format (1 sign + 8 exponent + 7 mantissa bits).

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `UnsupportedLayerError` | Layer type not supported | Remove or replace the layer |
| `UnsupportedActivationError` | Non-ReLU activation | Use only `F.relu()` or `nn.ReLU` |
| `UnsupportedConfigError` | Invalid layer configuration | Check stride, groups, dilation |
| `ControlFlowError` | Control flow in `forward()` | Remove conditionals and loops |

## FP16 Format

TNN uses a custom 16-bit floating point format:

- 1 sign bit (bit 15)
- 8 exponent bits (bits 14-7)
- 7 mantissa bits (bits 6-0)
- Exponent bias: 127

This is similar to BF16 but with a 7-bit mantissa instead of 7-bit.
