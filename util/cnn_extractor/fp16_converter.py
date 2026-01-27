"""Float to TNN FP16 conversion.

TNN uses a custom 16-bit floating point format:
- 1 sign bit (bit 15)
- 8 exponent bits (bits 14-7)
- 7 mantissa bits (bits 6-0)
- Exponent bias: 127

This matches the format in model/src/tnn_types.rs.
"""

import math
from typing import List, Union

# TNN FP16 format constants (matching model/src/tnn_types.rs)
TNNFP16_MANT_WIDTH = 7
TNNFP16_EXP_WIDTH = 8
TNNFP16_MANT_MASK = (1 << TNNFP16_MANT_WIDTH) - 1  # 0x7F
TNNFP16_EXP_MASK = ((1 << TNNFP16_EXP_WIDTH) - 1) << TNNFP16_MANT_WIDTH  # 0x7F80
TNNFP16_SGN_MASK = 1 << (TNNFP16_MANT_WIDTH + TNNFP16_EXP_WIDTH)  # 0x8000
TNNFP16_BIAS = (1 << (TNNFP16_EXP_WIDTH - 1)) - 1  # 127


def float_to_tnn_fp16(value: float) -> int:
    """Convert a float32 value to TNN's custom FP16 format.

    Args:
        value: Float value to convert

    Returns:
        16-bit unsigned integer in TNN FP16 format
    """
    if value == 0.0:
        return 0

    # Handle sign
    sgn = value < 0
    unsigned_fp = abs(value)

    # Handle infinity
    if math.isinf(unsigned_fp):
        exp = (1 << TNNFP16_EXP_WIDTH) - 1
        if sgn:
            return TNNFP16_SGN_MASK | (exp << TNNFP16_MANT_WIDTH)
        return exp << TNNFP16_MANT_WIDTH

    # Handle NaN
    if math.isnan(value):
        # Return standard NaN encoding
        return TNNFP16_SGN_MASK | TNNFP16_EXP_MASK | TNNFP16_MANT_MASK

    # Calculate exponent and mantissa
    exp = math.floor(math.log2(unsigned_fp))
    mant = (unsigned_fp / (2 ** exp)) * (2 ** TNNFP16_MANT_WIDTH)

    # Adjust exponent with bias
    biased_exp = int(exp) + TNNFP16_BIAS

    # Handle underflow (too small)
    if biased_exp <= 0:
        return 0

    # Handle overflow (too large)
    if biased_exp >= (1 << TNNFP16_EXP_WIDTH) - 1:
        exp_val = (1 << TNNFP16_EXP_WIDTH) - 1
        if sgn:
            return TNNFP16_SGN_MASK | (exp_val << TNNFP16_MANT_WIDTH)
        return exp_val << TNNFP16_MANT_WIDTH

    # Build the FP16 value
    result = 0
    if sgn:
        result |= TNNFP16_SGN_MASK
    result |= (biased_exp << TNNFP16_MANT_WIDTH) & TNNFP16_EXP_MASK
    result |= int(mant) & TNNFP16_MANT_MASK

    return result


def tnn_fp16_to_float(raw: int) -> float:
    """Convert TNN FP16 format back to float.

    Args:
        raw: 16-bit unsigned integer in TNN FP16 format

    Returns:
        Float value
    """
    sgn = (raw & TNNFP16_SGN_MASK) != 0
    exp = (raw & TNNFP16_EXP_MASK) >> TNNFP16_MANT_WIDTH
    mant = raw & TNNFP16_MANT_MASK

    # Zero
    if exp == 0 and mant == 0 and not sgn:
        return 0.0

    # Infinity
    if exp == (1 << TNNFP16_EXP_WIDTH) - 1 and mant == 0:
        return float('-inf') if sgn else float('inf')

    # NaN
    if exp == (1 << TNNFP16_EXP_WIDTH) - 1 and mant != 0:
        return float('nan')

    # Normal number
    mant_with_msb = mant + (1 << TNNFP16_MANT_WIDTH)
    mant_float = mant_with_msb / (1 << TNNFP16_MANT_WIDTH)
    result = mant_float * (2 ** (exp - TNNFP16_BIAS))

    return -result if sgn else result


def float_to_hex(value: float) -> str:
    """Convert a float to TNN FP16 hex string.

    Args:
        value: Float value to convert

    Returns:
        4-character hex string (e.g., "3f80")
    """
    return f"{float_to_tnn_fp16(value):04x}"


def floats_to_hex_list(values: List[float]) -> List[str]:
    """Convert a list of floats to TNN FP16 hex strings.

    Args:
        values: List of float values

    Returns:
        List of 4-character hex strings
    """
    return [float_to_hex(v) for v in values]


def write_hex_file(values: List[float], path: str) -> None:
    """Write float values as TNN FP16 hex to a file.

    Args:
        values: List of float values
        path: Output file path
    """
    with open(path, 'w') as f:
        for v in values:
            f.write(f"{float_to_hex(v)}\n")
