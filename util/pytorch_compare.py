#!/usr/bin/env python3
"""
PyTorch comparison script for TNN output validation.

This script:
1. Loads a CNN architecture from a TOML file
2. Constructs an equivalent PyTorch model
3. Loads weights from the TOML/hex files
4. Runs input through the PyTorch model
5. Compares output with TNN results

Usage:
    python pytorch_compare.py model.toml input.json tnn_output.json [--max-ulps 4]
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_toml(path):
    """Load TOML file."""
    try:
        import toml
        return toml.load(path)
    except ImportError:
        print("Error: 'toml' package required. Install with: pip install toml")
        sys.exit(1)


def load_weights_from_source(source, base_path):
    """Load weights from a DataSource definition."""
    source_type = source.get('source')

    if source_type == 'inline':
        return np.array(source['values'], dtype=np.float32)

    elif source_type == 'file':
        # Load from hex file
        hex_path = base_path / source['path']
        values = []
        with open(hex_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse as FP16 hex value
                    raw = int(line, 16)
                    # Convert TNN FP16 to float (simplified - assumes similar to BF16)
                    values.append(tnn_fp16_to_float(raw))
        return np.array(values, dtype=np.float32)

    else:
        raise ValueError(f"Unknown source type: {source_type}")


def tnn_fp16_to_float(raw):
    """Convert TNN FP16 (1-8-7 format) to float."""
    sign = (raw >> 15) & 1
    exp = (raw >> 7) & 0xFF
    mant = raw & 0x7F

    if exp == 0:
        # Denormal or zero
        if mant == 0:
            return -0.0 if sign else 0.0
        # Denormal
        value = (mant / 128.0) * (2 ** -126)
    elif exp == 0xFF:
        # Inf or NaN
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        # Normal
        value = (1 + mant / 128.0) * (2 ** (exp - 127))

    return -value if sign else value


def build_pytorch_model(config, base_path):
    """Build a PyTorch model from TOML config."""
    layers = []
    input_shape = config['metadata']['input_shape']
    current_channels = input_shape[0]
    current_size = input_shape[1]  # Assuming square

    for layer_config in config['layer']:
        layer_type = layer_config['type']

        if layer_type == 'avg_pool2d':
            kernel_size = layer_config['kernel_size']
            stride = layer_config['stride']
            layers.append(nn.AvgPool2d(kernel_size, stride))
            current_size = (current_size - kernel_size) // stride + 1

        elif layer_type == 'max_pool2d':
            kernel_size = layer_config['kernel_size']
            stride = layer_config['stride']
            layers.append(nn.MaxPool2d(kernel_size, stride))
            current_size = (current_size - kernel_size) // stride + 1

        elif layer_type == 'conv2d':
            in_ch = layer_config['in_channels']
            out_ch = layer_config['out_channels']
            kernel_size = layer_config['kernel_size']
            relu = layer_config.get('relu', False)

            conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=True)

            # Load weights
            if 'weights' in layer_config:
                weights = load_weights_from_source(layer_config['weights'], base_path)
                weights = weights.reshape(out_ch, in_ch, kernel_size, kernel_size)
                conv.weight.data = torch.tensor(weights)

            if 'bias' in layer_config:
                bias = load_weights_from_source(layer_config['bias'], base_path)
                conv.bias.data = torch.tensor(bias)
            else:
                conv.bias.data.zero_()

            layers.append(conv)
            if relu:
                layers.append(nn.ReLU())

            current_channels = out_ch
            current_size = current_size - kernel_size + 1

        elif layer_type == 'flatten':
            layers.append(nn.Flatten())

        elif layer_type == 'linear':
            in_features = layer_config['in_features']
            out_features = layer_config['out_features']
            relu = layer_config.get('relu', False)

            linear = nn.Linear(in_features, out_features, bias=True)

            # Load weights
            if 'weights' in layer_config:
                weights = load_weights_from_source(layer_config['weights'], base_path)
                weights = weights.reshape(out_features, in_features)
                linear.weight.data = torch.tensor(weights)

            if 'bias' in layer_config:
                bias = load_weights_from_source(layer_config['bias'], base_path)
                linear.bias.data = torch.tensor(bias)
            else:
                linear.bias.data.zero_()

            layers.append(linear)
            if relu:
                layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def run_pytorch(model, input_data, input_shape):
    """Run PyTorch model on input data."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_data, dtype=torch.float32)
        x = x.view(1, *input_shape)  # Add batch dimension
        output = model(x)
        return output.numpy().flatten()


def float_to_bf16_bits(f):
    """Get BF16 bit pattern by truncating the low 16 bits of float32."""
    if math.isnan(f):
        return 0x7FC0
    bits32 = struct.unpack('>I', struct.pack('>f', float(f)))[0]
    return (bits32 >> 16) & 0xFFFF


def bf16_ulp_distance(a, b):
    """ULP distance between two values measured in BF16 (1-8-7) space."""
    if math.isnan(a) or math.isnan(b):
        return float('inf')
    if math.isinf(a) and math.isinf(b):
        return 0 if a == b else float('inf')
    if math.isinf(a) or math.isinf(b):
        return float('inf')

    def to_ordered(bits):
        # Map sign-magnitude BF16 to a linearly ordered integer.
        # Negative values (bit 15 set) map to <= 0; positive to >= 0.
        # -0 and +0 both map to 0.
        if bits >= 0x8000:
            return 0x8000 - bits
        return bits

    return abs(to_ordered(float_to_bf16_bits(a)) - to_ordered(float_to_bf16_bits(b)))


def compare_outputs(pytorch_out, tnn_out, max_ulps):
    """Compare PyTorch and TNN outputs using BF16 ULP distance."""
    if len(pytorch_out) != len(tnn_out):
        print(f"ERROR: Output size mismatch! PyTorch: {len(pytorch_out)}, TNN: {len(tnn_out)}")
        return False

    max_ulp_seen = 0
    mismatches = []

    for i, (pt_val, tnn_val) in enumerate(zip(pytorch_out, tnn_out)):
        ulps = bf16_ulp_distance(float(pt_val), float(tnn_val))
        if ulps > max_ulp_seen:
            max_ulp_seen = ulps

        if ulps > max_ulps:
            mismatches.append((i, pt_val, tnn_val, ulps))

    print(f"Output size: {len(pytorch_out)}")
    print(f"Max ULP difference (BF16): {max_ulp_seen}")
    print(f"Allowed max ULPs: {max_ulps}")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)} values exceed {max_ulps} ULPs):")
        for i, pt_val, tnn_val, ulps in mismatches:
            print(f"  [{i}]: PyTorch={pt_val:.6f}, TNN={tnn_val:.6f}, ULPs={ulps}")
        return False

    print("\nAll values within tolerance!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch model output with TNN output'
    )
    parser.add_argument('model', help='Path to CNN model TOML file')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('tnn_output', help='Path to TNN output JSON file')
    parser.add_argument(
        '--max-ulps', type=int, default=4,
        help='Maximum allowed BF16 ULP distance per output value (default: 4)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output values'
    )
    args = parser.parse_args()

    # Load model config
    model_path = Path(args.model)
    config = load_toml(model_path)
    base_path = model_path.parent

    print(f"Model: {config['metadata']['name']}")
    print(f"Input shape: {config['metadata']['input_shape']}")
    print(f"Output size: {config['metadata']['output_size']}")
    print()

    # Build PyTorch model
    model = build_pytorch_model(config, base_path)
    print(f"Built PyTorch model with {len(list(model.modules())) - 1} layers")

    # Load input
    with open(args.input) as f:
        input_json = json.load(f)
    input_data = input_json['values']
    print(f"Loaded {len(input_data)} input values")

    # Load TNN output
    with open(args.tnn_output) as f:
        tnn_json = json.load(f)
    tnn_output = tnn_json['final_output']
    print(f"Loaded {len(tnn_output)} TNN output values")
    print()

    # Run PyTorch
    input_shape = config['metadata']['input_shape']
    pytorch_output = run_pytorch(model, input_data, input_shape)

    if args.verbose:
        print("PyTorch output:", pytorch_output)
        print("TNN output:", tnn_output)
        print()

    # Compare
    success = compare_outputs(pytorch_output, tnn_output, args.max_ulps)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
