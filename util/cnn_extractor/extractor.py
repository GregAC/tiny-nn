"""Main CNNExtractor class with FX tracing."""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.fx as fx

from .validation import (
    ExtractionError,
    UnsupportedLayerError,
    UnsupportedActivationError,
    ControlFlowError,
)
from .layer_handlers import (
    LayerInfo,
    get_extractor,
    is_implicit_layer,
)
from .toml_writer import generate_toml, write_toml_file


class CNNExtractor:
    """Extract CNN architecture from PyTorch modules.

    Uses torch.fx symbolic tracing to determine layer order and detect
    ReLU activations attached to layers.
    """

    def __init__(self, model: nn.Module, input_shape: Tuple[int, int, int]):
        """Initialize the extractor.

        Args:
            model: PyTorch model to extract
            input_shape: Input shape as (channels, height, width)
        """
        self.model = model
        self.input_shape = list(input_shape)
        self._layers: List[LayerInfo] = []
        self._traced: Optional[fx.GraphModule] = None

    def extract(self) -> List[LayerInfo]:
        """Extract layer information from the model.

        Returns:
            List of LayerInfo objects in execution order

        Raises:
            ExtractionError: If extraction fails
        """
        # Trace the model to get execution order
        try:
            self._traced = fx.symbolic_trace(self.model)
        except Exception as e:
            raise ControlFlowError(
                f"Failed to trace model. This may be due to control flow "
                f"in forward(): {e}"
            )

        # Build a map of module names to modules
        module_map: Dict[str, nn.Module] = dict(self.model.named_modules())

        # Track which layers have ReLU after them
        relu_after: Dict[str, bool] = {}

        # First pass: find ReLU calls and what they follow
        prev_node = None
        for node in self._traced.graph.nodes:
            if self._is_relu_call(node):
                if prev_node is not None and prev_node.op == "call_module":
                    relu_after[prev_node.target] = True
            prev_node = node

        # Second pass: extract layers in order
        self._layers = []
        for node in self._traced.graph.nodes:
            if node.op == "call_module":
                module_name = node.target
                module = module_map.get(module_name)

                if module is None:
                    raise ExtractionError(f"Module not found: {module_name}")

                layer_info = self._extract_module(module, module_name)
                if layer_info is not None:
                    # Attach ReLU flag if ReLU follows this layer
                    if relu_after.get(module_name, False):
                        layer_info.relu = True
                    self._layers.append(layer_info)

            elif node.op == "call_function":
                # Handle functional calls like F.relu, torch.flatten
                layer_info = self._extract_function(node)
                if layer_info is not None:
                    self._layers.append(layer_info)

            elif node.op == "call_method":
                # Handle method calls like x.flatten()
                layer_info = self._extract_method(node)
                if layer_info is not None:
                    self._layers.append(layer_info)

        return self._layers

    def _is_relu_call(self, node: fx.Node) -> bool:
        """Check if a node is a ReLU call."""
        if node.op == "call_function":
            func = node.target
            func_name = getattr(func, "__name__", str(func))
            return func_name == "relu"
        elif node.op == "call_module":
            module = dict(self.model.named_modules()).get(node.target)
            return isinstance(module, nn.ReLU)
        return False

    def _extract_module(self, module: nn.Module, name: str) -> Optional[LayerInfo]:
        """Extract information from a module.

        Args:
            module: PyTorch module
            name: Module name

        Returns:
            LayerInfo or None if layer should be skipped
        """
        module_type = type(module).__name__

        # Skip ReLU modules (handled as flags)
        if is_implicit_layer(module_type):
            return None

        # Get extractor for this module type
        extractor = get_extractor(module_type)
        if extractor is None:
            raise UnsupportedLayerError(module_type, name)

        return extractor(module, name)

    def _extract_function(self, node: fx.Node) -> Optional[LayerInfo]:
        """Extract information from a function call.

        Args:
            node: FX graph node

        Returns:
            LayerInfo or None if function should be skipped
        """
        func = node.target
        func_name = getattr(func, "__name__", str(func))

        # Handle torch.flatten or F.relu
        if func_name == "flatten":
            return LayerInfo(
                layer_type="flatten",
                name=f"flatten_{node.name}",
            )
        elif func_name == "relu":
            # ReLU is handled by attaching to previous layer
            # Find the previous layer and attach relu flag
            # This is handled in the main extract loop
            return None
        elif func_name in ("sigmoid", "tanh", "softmax", "gelu", "leaky_relu"):
            raise UnsupportedActivationError(func_name)

        # Unknown function - may be okay (e.g., view, reshape for internal use)
        return None

    def _extract_method(self, node: fx.Node) -> Optional[LayerInfo]:
        """Extract information from a method call.

        Args:
            node: FX graph node

        Returns:
            LayerInfo or None if method should be skipped

        Raises:
            UnsupportedActivationError: If an unsupported activation is found
        """
        method_name = node.target

        if method_name == "flatten":
            return LayerInfo(
                layer_type="flatten",
                name=f"flatten_{node.name}",
            )
        elif method_name in ("view", "reshape", "contiguous"):
            # These are internal tensor operations, skip
            return None
        elif method_name in ("sigmoid", "tanh", "softmax", "gelu", "leaky_relu",
                            "softmax_", "sigmoid_", "tanh_"):
            raise UnsupportedActivationError(method_name)

        return None

    def to_toml(
        self,
        name: str,
        description: str = "",
        output_size: Optional[int] = None,
        weights_format: str = "inline",
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate TOML representation of the model.

        Args:
            name: Model name
            description: Model description
            output_size: Number of output classes (auto-detected if not provided)
            weights_format: "inline" or "file"
            output_dir: Directory for weight files (required if weights_format="file")

        Returns:
            TOML string
        """
        if not self._layers:
            self.extract()

        # Auto-detect output size from last linear layer
        if output_size is None:
            for layer in reversed(self._layers):
                if layer.layer_type == "linear":
                    output_size = layer.params.get("out_features", 10)
                    break
            else:
                output_size = 10  # Default

        return generate_toml(
            name=name,
            description=description,
            input_shape=self.input_shape,
            output_size=output_size,
            layers=self._layers,
            weights_format=weights_format,
            output_dir=output_dir,
        )

    def save_toml(
        self,
        path: str,
        name: str,
        description: str = "",
        output_size: Optional[int] = None,
        weights_format: str = "inline",
        output_dir: Optional[str] = None,
    ) -> None:
        """Save TOML representation to a file.

        Args:
            path: Output file path
            name: Model name
            description: Model description
            output_size: Number of output classes
            weights_format: "inline" or "file"
            output_dir: Directory for weight files
        """
        # Use directory of output file for weights if not specified
        if weights_format == "file" and output_dir is None:
            import os
            output_dir = os.path.dirname(path) or "."

        toml_content = self.to_toml(
            name=name,
            description=description,
            output_size=output_size,
            weights_format=weights_format,
            output_dir=output_dir,
        )
        write_toml_file(toml_content, path)

    def validate_only(self) -> List[str]:
        """Validate the model without generating output.

        Returns:
            List of warning messages (empty if fully valid)

        Raises:
            ExtractionError: If validation fails
        """
        warnings = []

        # Try extraction (will raise on errors)
        self.extract()

        # Check for potential issues
        has_conv = any(l.layer_type == "conv2d" for l in self._layers)
        has_linear = any(l.layer_type == "linear" for l in self._layers)
        has_flatten = any(l.layer_type == "flatten" for l in self._layers)

        if has_conv and has_linear and not has_flatten:
            warnings.append(
                "Model has Conv2d and Linear layers but no Flatten. "
                "Make sure the tensor is properly reshaped."
            )

        return warnings
