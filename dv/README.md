# Overview

There are three separate test benches

 - fp_add_tb.sv - Block level test for the floating point adder
 - fp_mul_tb.sv - Block level test for the floating point multiplier
 - tiny_nn_top_tb.sv - Top level test for the full design

The block level tests both read tests from the associated
`fp_add_tests.hex`/`fp_mul_tests.hex` files in this directory, these are not
included in the repository due to their size (they need regenerating using the
model). The top-level testbench uses plusargs to define which test it reads and
output produced.

The test vectors and expected outputs have all been generated using the rust
model.

# Using top-level testbench

From the repository root build it with make

```
make top_dv
```

The simulator binary has two plusargs '+test_file=' specifies the input and
'+out=' specifies where the output goes. The testbench just applies the 16-bit
hex numbers from the input file as successive inputs and writes outputs in the
output file. The testbench terminates once it runs out of inputs and Tiny-NN has
gone into the idle state (with a timeout if this takes two long). A number of
pre-generate test vectors and their expected outputs are found in
`dv/test_vectors`. The `dv/out_compare.py` script compares simulation output to
epxected output E.g. to run a multiply accumulate test, once the testbench is
built:

```
./obj_dir/Vtiny_nn_top_tb +test_data=./dv/test_vectors/test_simple_mul_acc.hex  +out=out.hex
./dv/out_compare.py -s ./out.hex -e ./dv/test_vectors/test_simple_mul_acc_expected.hex
```

The './dv/run_tests.py' will run all of the tests and do comparisons for you. It
must be run from the repository root.
