"""Error classes and validation for CNN extractor."""


class ExtractionError(Exception):
    """Base exception for CNN extraction errors."""
    pass


class UnsupportedLayerError(ExtractionError):
    """Raised when an unsupported layer type is encountered."""

    def __init__(self, layer_type: str, layer_name: str = ""):
        self.layer_type = layer_type
        self.layer_name = layer_name
        msg = f"Unsupported layer type: {layer_type}"
        if layer_name:
            msg += f" (layer: {layer_name})"
        super().__init__(msg)


class UnsupportedActivationError(ExtractionError):
    """Raised when an unsupported activation function is encountered."""

    def __init__(self, activation: str):
        self.activation = activation
        super().__init__(
            f"Unsupported activation: {activation}. Only ReLU is supported."
        )


class UnsupportedConfigError(ExtractionError):
    """Raised when a layer has unsupported configuration."""

    def __init__(self, layer_name: str, config_issue: str):
        self.layer_name = layer_name
        self.config_issue = config_issue
        super().__init__(f"Unsupported configuration in {layer_name}: {config_issue}")


class ValidationError(ExtractionError):
    """Raised when validation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class ControlFlowError(ExtractionError):
    """Raised when control flow is detected in forward()."""

    def __init__(self, message: str = "Control flow (if/loops) detected in forward()"):
        super().__init__(message)


def validate_conv2d(module, name: str) -> None:
    """Validate Conv2d layer configuration.

    Args:
        module: PyTorch Conv2d module
        name: Layer name for error messages

    Raises:
        UnsupportedConfigError: If configuration is not supported
    """
    import torch.nn as nn

    if not isinstance(module, nn.Conv2d):
        raise ValidationError(f"Expected Conv2d, got {type(module).__name__}")

    # Check stride
    stride = module.stride
    if isinstance(stride, tuple):
        if stride != (1, 1):
            raise UnsupportedConfigError(
                name, f"stride must be 1, got {stride}"
            )
    elif stride != 1:
        raise UnsupportedConfigError(name, f"stride must be 1, got {stride}")

    # Check groups (no grouped convolutions)
    if module.groups != 1:
        raise UnsupportedConfigError(
            name, f"grouped convolutions not supported (groups={module.groups})"
        )

    # Check dilation
    dilation = module.dilation
    if isinstance(dilation, tuple):
        if dilation != (1, 1):
            raise UnsupportedConfigError(
                name, f"dilation not supported, got {dilation}"
            )
    elif dilation != 1:
        raise UnsupportedConfigError(name, f"dilation not supported, got {dilation}")


def validate_pool2d(module, name: str) -> None:
    """Validate pooling layer configuration.

    Args:
        module: PyTorch pooling module
        name: Layer name for error messages

    Raises:
        UnsupportedConfigError: If configuration is not supported
    """
    import torch.nn as nn

    if not isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        raise ValidationError(
            f"Expected AvgPool2d or MaxPool2d, got {type(module).__name__}"
        )
