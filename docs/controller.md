# Overview

The various TNN operations can be used to implement a complete convolution
neural network (CNN) with a suitable controller. They key to this is
understanding how the various layers and operations used in a CNN map into the
TNN operations.

We're considering the following operations that could be part of a CNN
implementation (referencing pytorch where appropriate):

- 2D Convolve (torch.nn.Conv2d)
- Linear layer/fully connected network (torch.nn.Linear)
- Average Pool 2D (torch.nn.AvgPool2d)
- Max Pool 2D (torch.nn.MaxPool2d)
- RELU activation function (torch.nn.relu)

We have a way to take a CNN defined in pytorch and execute it against the TNN
RTL and Model. This is the controller program (written in rust) that takes a
CNN architecture description and weights (supplied as TOML). It sends input
to TNN via a network interface and receive output back via the same interface.
Either the RTL simulation or the model could be on the other side of this
interface, the controller will not know nor need to know.

The TOML architecture description is documented in `docs/cnn_schema.md`

# Operation Mapping

Here's how we map each to TNN operations

## 2D Convolve

Our TNN operation provides a convolve operation on a fixed 2D size (4x2). We can map
an arbitrary sized convolve to it by decomposing it into 4x2 chunks. Where we
don't need all of the 4x2 in a given chunk just set the unneeded weights to 0
(e.g. for a 3x2 chunk weights 3,0 and 3,1 would be 0). The Accumulate operation
can be used to add partial results together and apply any final bias.

At first glance the convolve operation only appears suitable for a single kernel
height row of the image and we need to start a new convolve operation for each
new row but this is not the case. If we just continue inputting data going
straight into the next row we receive convolution results that blend the old row
and the new row together but we can just ignore them. Eventually we'll have
flushed out the old row from the buffer and we'll be computing with the new row
data.

Concretely after we get the last valid convolution result for the next 6 inputs
the matching outputs will be junk (a mix of the previous and current rows convolved
together) after 8 inputs the buffer now has the first 4x2 pixels of the current
row and we get a valid convolution output.

Note we only need to support stride==1 from pytorch's torch.nn.Conv2d

## Linear Layer

This maps directly to the multiply-accumulate operation. Simply supply the input
values and weights pairwise (with the bias first) and get a result.

## Average Pool 2D

This maps to the fixed multiply-accumulate operation. The fixed multiplier is
just the divisor needed for the average. E.g. if we have a 2x2 avg pool our
divisor is 4, so our multiplier is 0.25. The fixed multiply-accumulate operation
will calculate i_0_0 * 0.25 + i_0_1 * 0.25 + i_1_0 * 0.25 + i_1_1 * 0.25 where
i_x_y is a 2x2 chunk of the matrix we are computing an average pool on.

## Max Pool 2D

This is directly implemented

## RELU

RELU is not available as a standalone operation, rather than is an optional
inclusion on other operations.

# Controller

We want to implement the controller that can take a TOML architecture
description and execute it against a TNN implementation.

The controller is written in rust, it will send data to TNN via a network
interface and receive data back on the same interface.

It can capture everything sent to TNN and received back from it.
This can be used to replay the commands from the controller against the RTL or
model and to compare their output to the output seen when running the
controller. This allows us to run tests without always requiring the
controller.

The controller must be able to work with arbitrary TOML architecture
descriptions but we also want a mode that acts as a full end to end digit
recognition setup, taking in an MNIST type image with a 28x28 greyscale image of
a single digit and producing probabilities for each digit.
