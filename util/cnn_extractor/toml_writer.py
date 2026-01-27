"""TOML generation for CNN architecture."""

import os
from typing import Dict, List, Optional, Any

from .layer_handlers import LayerInfo
from .fp16_converter import write_hex_file


def generate_toml(
    name: str,
    description: str,
    input_shape: List[int],
    output_size: int,
    layers: List[LayerInfo],
    weights_format: str = "inline",
    output_dir: Optional[str] = None,
) -> str:
    """Generate TOML string for CNN architecture.

    Args:
        name: Model name
        description: Model description
        input_shape: Input shape [channels, height, width]
        output_size: Number of output classes
        layers: List of extracted layer information
        weights_format: "inline" or "file"
        output_dir: Directory for weight files (required if weights_format="file")

    Returns:
        TOML string
    """
    lines = []

    # Metadata section
    lines.append("[metadata]")
    lines.append(f'name = "{name}"')
    lines.append(f'description = "{description}"')
    lines.append(f"input_shape = {input_shape}")
    lines.append(f"output_size = {output_size}")
    lines.append("")

    # Layer sections
    for layer in layers:
        lines.append("[[layer]]")
        lines.append(f'type = "{layer.layer_type}"')
        lines.append(f'name = "{layer.name}"')

        # Write layer-specific parameters
        for key, value in layer.params.items():
            lines.append(f"{key} = {_format_value(value)}")

        # Write relu flag if set
        if layer.relu:
            lines.append("relu = true")

        # Write weights
        if layer.weights is not None:
            lines.append("")
            if weights_format == "inline":
                lines.append("[layer.weights]")
                lines.append('source = "inline"')
                lines.append(f"values = {_format_float_list(layer.weights)}")
            else:
                # Write to file
                weights_path = _write_weights_file(
                    layer.weights, layer.name, "weights", output_dir
                )
                lines.append("[layer.weights]")
                lines.append('source = "file"')
                lines.append(f'path = "{weights_path}"')

        # Write bias
        if layer.bias is not None:
            lines.append("")
            if weights_format == "inline":
                lines.append("[layer.bias]")
                lines.append('source = "inline"')
                lines.append(f"values = {_format_float_list(layer.bias)}")
            else:
                # Write to file
                bias_path = _write_weights_file(
                    layer.bias, layer.name, "bias", output_dir
                )
                lines.append("[layer.bias]")
                lines.append('source = "file"')
                lines.append(f'path = "{bias_path}"')

        lines.append("")

    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a value for TOML output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (list, tuple)):
        return f"[{', '.join(str(v) for v in value)}]"
    else:
        return str(value)


def _format_float_list(values: List[float], max_per_line: int = 8) -> str:
    """Format a list of floats for TOML output.

    For short lists, outputs on a single line.
    For longer lists, outputs with line breaks.
    """
    if len(values) <= max_per_line:
        formatted = ", ".join(f"{v}" for v in values)
        return f"[{formatted}]"

    # Multi-line format for long lists
    lines = ["["]
    for i in range(0, len(values), max_per_line):
        chunk = values[i:i + max_per_line]
        formatted = ", ".join(f"{v}" for v in chunk)
        if i + max_per_line < len(values):
            lines.append(f"    {formatted},")
        else:
            lines.append(f"    {formatted}")
    lines.append("]")
    return "\n".join(lines)


def _write_weights_file(
    values: List[float],
    layer_name: str,
    weight_type: str,
    output_dir: str
) -> str:
    """Write weights to a hex file.

    Args:
        values: Float values to write
        layer_name: Layer name for filename
        weight_type: "weights" or "bias"
        output_dir: Output directory

    Returns:
        Relative path to the weights file
    """
    if output_dir is None:
        raise ValueError("output_dir required when weights_format='file'")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{layer_name}_{weight_type}.hex"
    filepath = os.path.join(output_dir, filename)
    write_hex_file(values, filepath)
    return filename


def write_toml_file(content: str, path: str) -> None:
    """Write TOML content to a file.

    Args:
        content: TOML string
        path: Output file path
    """
    with open(path, 'w') as f:
        f.write(content)
