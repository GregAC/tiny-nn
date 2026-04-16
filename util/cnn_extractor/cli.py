"""Command-line interface for CNN extractor."""

import argparse
import importlib.util
import sys
import os
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

from .extractor import CNNExtractor
from .validation import ExtractionError


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load weights from a checkpoint file into a model.

    Supports both state_dict files and full model files saved with torch.save().

    Args:
        model: The model instance to load weights into
        checkpoint_path: Path to the checkpoint file (.pt or .pth)

    Returns:
        The model with loaded weights

    Raises:
        RuntimeError: If the checkpoint cannot be loaded
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Could be a state_dict or a training checkpoint with 'model_state_dict' key
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's a raw state_dict
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, nn.Module):
        # Full model was saved - copy its state
        model.load_state_dict(checkpoint.state_dict())
    else:
        raise RuntimeError(
            f"Unknown checkpoint format: {type(checkpoint)}. "
            "Expected state_dict or nn.Module."
        )

    return model


def load_model_class(module_path: str, class_name: str) -> Type[nn.Module]:
    """Load a model class from a Python file.

    Args:
        module_path: Path to the Python file containing the model
        class_name: Name of the model class

    Returns:
        The model class

    Raises:
        ImportError: If the module or class cannot be loaded
    """
    # Load the module from file
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)

    # Get the class from the module
    if not hasattr(module, class_name):
        raise ImportError(f"Class {class_name} not found in {module_path}")

    return getattr(module, class_name)


def parse_input_shape(shape_args: list) -> Tuple[int, int, int]:
    """Parse input shape from command line arguments.

    Args:
        shape_args: List of 3 integers [channels, height, width]

    Returns:
        Tuple of (channels, height, width)
    """
    if len(shape_args) != 3:
        raise ValueError(
            f"Input shape must have 3 values (channels, height, width), "
            f"got {len(shape_args)}"
        )
    return tuple(int(x) for x in shape_args)


def main(args=None):
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Extract CNN architecture from PyTorch modules to TOML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with random weights (for testing architecture)
  python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 -o model.toml

  # Extract with pre-trained weights from checkpoint
  python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 --checkpoint weights.pt -o model.toml

  # Use file-based weight storage
  python -m util.cnn_extractor model.py MiniCNN --input-shape 1 28 28 --checkpoint weights.pt --weights-format file -o model.toml

  # Validate model structure only
  python -m util.cnn_extractor --validate-only model.py MiniCNN --input-shape 1 28 28
        """
    )

    parser.add_argument(
        "module_path",
        help="Path to Python file containing the model class"
    )
    parser.add_argument(
        "class_name",
        help="Name of the model class to extract"
    )
    parser.add_argument(
        "--input-shape", "-s",
        nargs=3,
        type=int,
        required=True,
        metavar=("C", "H", "W"),
        help="Input shape as channels height width (e.g., 1 28 28)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        help="Path to checkpoint file (.pt/.pth) with pre-trained weights"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output TOML file path (default: stdout)"
    )
    parser.add_argument(
        "--name", "-n",
        help="Model name (default: class name)"
    )
    parser.add_argument(
        "--description", "-d",
        default="",
        help="Model description"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        help="Number of output classes (auto-detected if not provided)"
    )
    parser.add_argument(
        "--weights-format",
        choices=["inline", "file"],
        default="inline",
        help="Weight storage format (default: inline)"
    )
    parser.add_argument(
        "--weights-dir",
        help="Directory for weight files (default: same as output file)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the model, don't generate output"
    )

    parsed_args = parser.parse_args(args)

    #try:
    # Load the model class
    model_class = load_model_class(parsed_args.module_path, parsed_args.class_name)

    # Instantiate the model
    model = model_class()

    # Load checkpoint if provided
    if parsed_args.checkpoint:
        model = load_checkpoint(model, parsed_args.checkpoint)
        model.eval()  # Set to evaluation mode

    # Parse input shape
    input_shape = parse_input_shape(parsed_args.input_shape)

    # Create extractor
    extractor = CNNExtractor(model, input_shape)

    if parsed_args.validate_only:
        # Validate only
        warnings = extractor.validate_only()
        print(f"Validation successful: {parsed_args.class_name}")
        for warning in warnings:
            print(f"Warning: {warning}")
        return 0

    # Generate TOML
    name = parsed_args.name or parsed_args.class_name

    if parsed_args.output:
        # Save to file
        extractor.save_toml(
            path=parsed_args.output,
            name=name,
            description=parsed_args.description,
            output_size=parsed_args.output_size,
            weights_format=parsed_args.weights_format,
            output_dir=parsed_args.weights_dir,
        )
        print(f"Saved to {parsed_args.output}")
    else:
        # Print to stdout
        toml_content = extractor.to_toml(
            name=name,
            description=parsed_args.description,
            output_size=parsed_args.output_size,
            weights_format=parsed_args.weights_format,
            output_dir=parsed_args.weights_dir,
        )
        print(toml_content)

    return 0

    #except ExtractionError as e:
    #    print(f"Error: {e}", file=sys.stderr)
    #    return 1
    #except ImportError as e:
    #    print(f"Import error: {e}", file=sys.stderr)
    #    return 1
    #except Exception as e:
    #    print(f"Unexpected error: {e}", file=sys.stderr)
    #    return 1


if __name__ == "__main__":
    sys.exit(main())
