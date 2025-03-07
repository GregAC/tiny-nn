# Overview

Tiny-NN consists of a number of floating point multiply and add units in a fixed
arrangement that is controlled by a state machine. The state machine has
multiple operating modes that allow it to use the computation units to perform a
number of useful operations, e.g. a multiply-accumulate operation that
multiplies pairs of numbers, sums them all, adds a bias followed by an optional
RELU, this can be used to compute the result of a single neuron in a layer of a
neural network. It contains storage for a small number of parameters and a
portion of an image to allow efficient convolution operations.

Pipelining is employed to ensure maximum utilization of the floating point
units.

The floating point format used is a custom 16-bit format, referred to as fp16 in
this documentation.

Inputs are fed in 16 bits per clock with results output 8 bits per clock.
Results are typically fp16 numbers so take two clocks to output. Generally there
are multiple inputs per output so the two clock output time isn't a performance
issue.

# Operations

Available operations are described in [operations.md](./operations.md)

# FP16 Format

The FP16 format uses an 8-bit exponent, 7-bit mantissa and 1-bit sign. The
exponent bias is 128 (i.e. exponent of 128 == 2^0). The bit-level layout is:

| 15   | 14 - 7   | 6 - 0    |
|------|----------|----------|
| Sign | Exponent | Mantissa |

It does not support denormalized numbers. Special encodings are:

* Zero - All 0s
* Positive infinity - Sign: 0, Exponent: 8'hff, Mantissa: 0
* Negative infinity - Sign: 1, Exponent:8'hff, Mantissa: 0
* NaN
  - Any number with Exponent: 8'h0 other than Zero (i.e. where all other
    fields are 0)
  - Any number with Exponent: 8'hff other than the infinities (i.e. where
    Mantissa == 0)
  - The standard NaN is all 1s
