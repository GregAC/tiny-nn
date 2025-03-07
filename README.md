Overview
--------

Tiny-NN is a toy neural network accelerator written in system verilog. It's
specifically designed for use with tiny tapeout (https://tinytapeout.com/) and
for use with CNNs (convolutional neural networks) though can do generic
multiply-accumulate operations to implement linear neurons.

The matching tiny tapeout repository can be found at
https://github.com/GregAC/tt10-tiny-nn. sv2v is used to transform the system
verilog into verilog for using with the OpenLANE flow used by tiny tapeout.

See the documentation under `doc/` for further information.

Repository Structure
--------------------

This repository contains the RTL (in `rtl/`) and DV (in `dv/`) for Tiny-NN and a
model (in `model/`) written in Rust.  The model isn't fully developed and relies
on altering the `main` function to do what is required rather than providing a
usable CLI application.

A makefile is used to build the various testbenches and the verilog for tiny
tapeout. It's contained in the root directory and has these targets

- `top_dv` - Top-level testbench
- `top_dv_sv2v` - Same testbench as above but using the sv2v built verilog version
  of the RTL
- `fp_add_dv` - Block-level testbench for the floating point adder
- `fp_mul_dv` - Block-level testbench for the floating point multiplier
- `sv2v` - Builds verilog version of the RTL with sv2v, outputs to `sv2v_out/`

The rust model is built with cargo

Dependencies
------------

- [Verilator](https://www.veripool.org/verilator/) - Used to build the
  testbench. v5.032 is known to work, other versions may work as well, though it
  must be a v5 as it relies on v5 features.
- [GTKWave](https://gtkwave.sourceforge.net/) - For viewing the .fst waves
  output from the testbench
- [sv2v](https://github.com/zachjs/sv2v) - For building a verilog version of the
  system verilog source
- [rust](https://www.rust-lang.org/) - To build the model
