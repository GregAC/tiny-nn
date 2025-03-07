# Overview

With the FSM in idle state an operation begins when a valid command word is seen
on the 16-bit input. Data following the command word is operation dependent.
The termination condition for each operation (which will send the FSM back to
idle) is different, though a common one is to give a standard NaN (all 1st) as
the input.

There is no valid signal for the output (all available 8 bits are required for
the data). Each operation has different latentices that tell you which output
cycles relative to the input are valid output bytes. The low byte of each fp16
number is output first.

# Operations

## Convolve

| 15 - 12 | 11 - 0  |
|---------|---------|
| 4'b0001 | ignored |

Compute a 4x2 convolution kernel over an 2D array. The array is streamed in and
held in an internal buffer so once that buffer is filled every 2 new pixels can
compute a new covolution. All data is given in column major order. After the
command word 8 parameters are provided which are stored in internal memory. So
the first 8 words input are (where p_x_y is the parameter for location x, y in
the kernel):

 - p_0_0
 - p_0_1
 - p_1_0
 - p_1_1
 - p_2_0
 - p_2_1
 - p_3_0
 - p_3_1

Following the parameters array data is provided. The array data is
unbounded and the operation is terminated by providing a standard NaN as input.

Array data is streamed into an internal 4x2 input buffer, once it has been
filled convolutions begin to be computed. Each 2 values provided shifts 2 values
out of the input buffer and computes another convolvution. E.g. if the the
following 10 values are provided (where v_x_y is the value for location x, y in
the array):

 - v_0_0
 - v_0_1
 - v_1_0
 - v_1_1
 - v_2_0
 - v_2_1
 - v_3_0
 - v_3_1
 - v_4_0
 - v_4_1

The following would be computed
 - p_0_0 * v_0_0 + p_0_1 * v_0_1 + p_1_0 * v_1_0 + p_1_1 * v_1_1 + p_2_0 * v_2_0 + p_2_1 * v_2_1 + p_3_0 * v_3_0 + p_3_1 * v_3_1
 - p_1_0 * v_1_0 + p_1_1 * v_1_1 + p_2_0 * v_2_0 + p_2_1 * v_2_1 + p_3_0 * v_3_0 + p_3_1 * v_3_1 + p_4_0 * v_4_0 + p_4_1 * v_4_1

The first output is seen 20 cycles after the command word is input. Output is
continuous from that point onward (2 cycles to output each fp16 result and a new
result for every two values once the input buffer is full).

## Accumulate

| 15 - 12 | 11 - 9  | 8    | 7 - 0 |
|---------|---------|------|-------|
| 4'b0010 | ignored | RELU | count |

Computes sums of groups of numbers with a constant bias applied to each sum and
an optional RELU. The 'count' field of the command word specifies how big each
group of numbers is. When 'RELU' is set to 1 RELU is applied to the output.

Following the command word the bias is input. After the bias each input is a
number that will get summed. For example with count == 4 if the following is
input after the command word:

 - b
 - v_0
 - v_1
 - v_2
 - v_3
 - v_4
 - v_5
 - v_6
 - v_7

The following would be computed:

 - RELU(v_0 + v_1 + v_2 + v_3 + B)
 - RELU(v_4 + v_5 + v_6 + v_7 + B)

The first output is seen (3 + count) cycles after command word is input. The gap
between outputs is (count - 2) cycles (-2 to account for the 2 cycles each output
takes).

The operation is terminated with a standard NaN

## Multiply-Accumulate

| 15 - 12 | 11 - 9  | 8    | 7 - 0   |
|---------|---------|------|---------|
| 4'b0011 | ignored | RELU | ignored |

Computes the sum of pairs of numbers that are muliplied together. With a bias
added at the end and an optional RELU aplied.

The first input is the bias, the following values are the pairs of numbers. A
standard NaN terminates the inputs and will cause the output value to be
computed. E.g. if the following values are provided:

 - b
 - v_0
 - p_0
 - v_1
 - p_1
 - v_2
 - p_2
 - v_3
 - p_3
 - NaN

The following would be computed:
 - relu(v_0 * p_0 + v_1 * p_1+ v_2 * p_2 + v_3 * p_3 + b)

The output is seen 4 + num_values cycles after the command word is supplied
where num_values is the total number of values (excluding the bias) that are
provided, this is 8 in the example above.

## Tests

A number of test operations are provided to check basic input/output behaviour.


### ASCII Test

| 15 - 12 | 11 - 8  | 7 - 0   |
|---------|---------|---------|
| 4'b1111 | 4'b1111 | ignored |

Outputs the following bytes ('T-NN' is ASCII) continously whilst bits [15:8] of
the input remain 8'hff:

 - 8'h54
 - 8'h2d
 - 8'h4e
 - 8'h4e

### Pulse Test

| 15 - 12 | 11 - 8  | 7 - 0   |
|---------|---------|---------|
| 4'b1111 | 4'b0000 | ignored |

Outputs the following bytes continously whilst bits [15:8] of the input remain
8'f0:

 - 8'b1010_1010
 - 8'b0101_0101

### Count test

| 15 - 12 | 11 - 8  | 7 - 0 |
|---------|---------|-------|
| 4'b1111 | 4'b0001 | count |

Outputs a count starting from the the provided `count`. Input is ignored whilst
the count is being output.
