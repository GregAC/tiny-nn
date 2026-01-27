"""Per-layer-type extraction logic."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .validation import (
    UnsupportedConfigError,
    validate_conv2d,
    validate_pool2d,
)


@dataclass
class LayerInfo:
    """Extracted layer information."""

    layer_type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    weights: Optional[List[float]] = None
    bias: Optional[List[float]] = None
    relu: bool = False


def extract_conv2d(module, name: str) -> LayerInfo:
    """Extract Conv2d layer information.

    Args:
        module: PyTorch Conv2d module
        name: Layer name

    Returns:
        LayerInfo with extracted parameters
    """
    import torch.nn as nn

    # Validate configuration
    validate_conv2d(module, name)

    # Extract kernel size
    kernel_size = module.kernel_size
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise UnsupportedConfigError(
                name, f"non-square kernel not supported: {kernel_size}"
            )
        kernel_size = kernel_size[0]

    info = LayerInfo(
        layer_type="conv2d",
        name=name,
        params={
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": kernel_size,
            "stride": 1,  # Always 1 (validated above)
        }
    )

    # Extract weights: [out_channels, in_channels, kH, kW] -> flattened
    if module.weight is not None:
        info.weights = module.weight.detach().cpu().numpy().flatten().tolist()

    # Extract bias
    if module.bias is not None:
        info.bias = module.bias.detach().cpu().numpy().flatten().tolist()

    return info


def extract_linear(module, name: str) -> LayerInfo:
    """Extract Linear layer information.

    Args:
        module: PyTorch Linear module
        name: Layer name

    Returns:
        LayerInfo with extracted parameters
    """
    info = LayerInfo(
        layer_type="linear",
        name=name,
        params={
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
    )

    # Extract weights: [out_features, in_features] -> flattened
    if module.weight is not None:
        info.weights = module.weight.detach().cpu().numpy().flatten().tolist()

    # Extract bias
    if module.bias is not None:
        info.bias = module.bias.detach().cpu().numpy().flatten().tolist()

    return info


def extract_avgpool2d(module, name: str) -> LayerInfo:
    """Extract AvgPool2d layer information.

    Args:
        module: PyTorch AvgPool2d module
        name: Layer name

    Returns:
        LayerInfo with extracted parameters
    """
    validate_pool2d(module, name)

    # Extract kernel size
    kernel_size = module.kernel_size
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise UnsupportedConfigError(
                name, f"non-square kernel not supported: {kernel_size}"
            )
        kernel_size = kernel_size[0]

    # Extract stride
    stride = module.stride
    if isinstance(stride, tuple):
        if stride[0] != stride[1]:
            raise UnsupportedConfigError(
                name, f"non-square stride not supported: {stride}"
            )
        stride = stride[0]

    return LayerInfo(
        layer_type="avg_pool2d",
        name=name,
        params={
            "kernel_size": kernel_size,
            "stride": stride,
        }
    )


def extract_maxpool2d(module, name: str) -> LayerInfo:
    """Extract MaxPool2d layer information.

    Args:
        module: PyTorch MaxPool2d module
        name: Layer name

    Returns:
        LayerInfo with extracted parameters
    """
    validate_pool2d(module, name)

    # Extract kernel size
    kernel_size = module.kernel_size
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise UnsupportedConfigError(
                name, f"non-square kernel not supported: {kernel_size}"
            )
        kernel_size = kernel_size[0]

    # Extract stride
    stride = module.stride
    if isinstance(stride, tuple):
        if stride[0] != stride[1]:
            raise UnsupportedConfigError(
                name, f"non-square stride not supported: {stride}"
            )
        stride = stride[0]

    return LayerInfo(
        layer_type="max_pool2d",
        name=name,
        params={
            "kernel_size": kernel_size,
            "stride": stride,
        }
    )


def extract_flatten(module, name: str) -> LayerInfo:
    """Extract Flatten layer information.

    Args:
        module: PyTorch Flatten module
        name: Layer name

    Returns:
        LayerInfo with extracted parameters (no params for flatten)
    """
    return LayerInfo(
        layer_type="flatten",
        name=name,
    )


# Map PyTorch module types to extraction functions
LAYER_EXTRACTORS = {
    "Conv2d": extract_conv2d,
    "Linear": extract_linear,
    "AvgPool2d": extract_avgpool2d,
    "MaxPool2d": extract_maxpool2d,
    "Flatten": extract_flatten,
}


# Layers that don't need explicit handling (handled implicitly or as flags)
IMPLICIT_LAYERS = {"ReLU"}


def get_extractor(module_type: str):
    """Get the extraction function for a module type.

    Args:
        module_type: PyTorch module type name (e.g., "Conv2d")

    Returns:
        Extraction function or None if implicit/unsupported
    """
    return LAYER_EXTRACTORS.get(module_type)


def is_implicit_layer(module_type: str) -> bool:
    """Check if a layer type is handled implicitly.

    Args:
        module_type: PyTorch module type name

    Returns:
        True if the layer is handled implicitly
    """
    return module_type in IMPLICIT_LAYERS
