#!/usr/bin/env python3
"""
Reference PyTorch implementation of the SimpleCNN model.

This matches the architecture defined in simple_cnn.toml:
- Input: 1 channel, 8x8 image
- AvgPool2d: 2x2, stride 2 -> 1x4x4
- Conv2d: 1->2 channels, 4x4 kernel, ReLU -> 2x1x1
- Flatten -> 2
- Linear: 2->2, no activation -> 2
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 2, 4, bias=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


def load_weights_from_toml(model, toml_path):
    """Load weights from a TOML model file into PyTorch model."""
    import toml

    with open(toml_path) as f:
        config = toml.load(f)

    for layer_config in config['layer']:
        layer_type = layer_config['type']
        layer_name = layer_config['name']

        if layer_type == 'conv2d':
            # Find corresponding PyTorch layer
            if layer_name == 'conv1':
                pt_layer = model.conv1
            else:
                continue

            if 'weights' in layer_config and layer_config['weights']['source'] == 'inline':
                weights = layer_config['weights']['values']
                out_ch = layer_config['out_channels']
                in_ch = layer_config['in_channels']
                k_size = layer_config['kernel_size']

                # Reshape weights: [out_ch, in_ch, k, k]
                w = np.array(weights).reshape(out_ch, in_ch, k_size, k_size)
                pt_layer.weight.data = torch.tensor(w, dtype=torch.float32)

            if 'bias' in layer_config and layer_config['bias']['source'] == 'inline':
                bias = layer_config['bias']['values']
                pt_layer.bias.data = torch.tensor(bias, dtype=torch.float32)

        elif layer_type == 'linear':
            if layer_name == 'fc1':
                pt_layer = model.fc1
            else:
                continue

            if 'weights' in layer_config and layer_config['weights']['source'] == 'inline':
                weights = layer_config['weights']['values']
                out_f = layer_config['out_features']
                in_f = layer_config['in_features']

                # Reshape weights: [out_features, in_features]
                w = np.array(weights).reshape(out_f, in_f)
                pt_layer.weight.data = torch.tensor(w, dtype=torch.float32)

            if 'bias' in layer_config and layer_config['bias']['source'] == 'inline':
                bias = layer_config['bias']['values']
                pt_layer.bias.data = torch.tensor(bias, dtype=torch.float32)


def run_inference(model, input_data):
    """Run inference and return output."""
    model.eval()
    with torch.no_grad():
        # Input shape: [batch, channels, height, width]
        x = torch.tensor(input_data, dtype=torch.float32).view(1, 1, 8, 8)
        output = model(x)
        return output.numpy().flatten()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run SimpleCNN inference')
    parser.add_argument('--model', default='simple_cnn.toml', help='Path to model TOML')
    parser.add_argument('--input', help='Path to input JSON file')
    parser.add_argument('--output', help='Path to output JSON file')
    args = parser.parse_args()

    # Create model and load weights
    model = SimpleCNN()
    load_weights_from_toml(model, args.model)

    # Load input if provided
    if args.input:
        import json
        with open(args.input) as f:
            input_json = json.load(f)
        input_data = input_json['values']
    else:
        # Default: all ones
        input_data = [1.0] * 64

    # Run inference
    output = run_inference(model, input_data)
    print(f"Output: {output}")

    # Save output if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'model': 'SimpleCNN',
                'final_output': output.tolist()
            }, f, indent=2)


if __name__ == '__main__':
    main()
