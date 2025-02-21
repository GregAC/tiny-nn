const ConvWidth: usize = 4;
const ConvHeight: usize = 2;

use super::tnn_types::{TinyNNFP16, TinyNNFP16Zero};
use ndarray::{s, Array2, ArrayView, Ix2};

fn sum_values(values: ArrayView<'_, TinyNNFP16, Ix2>) -> TinyNNFP16 {
    if values.shape()[1] == 1 {
        if values.shape()[0] == 1 {
            values[[0, 0]]
        } else if values.shape()[0] == 2 {
            values[[0, 0]] + values[[1, 0]]
        } else {
            sum_values(values.slice(s![..values.shape()[0] / 2, ..]))
                + sum_values(values.slice(s![values.shape()[0] / 2.., ..]))
        }
    } else {
        sum_values(values.slice(s![.., ..values.shape()[1] / 2]))
            + sum_values(values.slice(s![.., values.shape()[1] / 2..]))
    }
}

pub fn do_convolve(in_values: &Vec<TinyNNFP16>, params: &Array2<TinyNNFP16>) -> Vec<TinyNNFP16> {
    let mut active_values =
        Array2::<TinyNNFP16>::from_elem((ConvWidth, ConvHeight), TinyNNFP16Zero);
    let mut out_values: Vec<TinyNNFP16> = Vec::new();

    if in_values.len() < (ConvWidth * ConvHeight) {
        return out_values;
    }

    for x in 0..(ConvWidth - 1) {
        for y in 0..ConvHeight {
            active_values[[x + 1, y]] = in_values[x * ConvHeight + y];
        }
    }

    for n in (((ConvWidth - 1) * ConvHeight)..in_values.len()).step_by(ConvHeight) {
        if n + ConvHeight > in_values.len() {
            break;
        }

        let shift_slice = active_values.slice(s![1.., ..]).to_owned();

        active_values
            .slice_mut(s![..(ConvWidth - 1), ..])
            .assign(&shift_slice);

        for y in 0..ConvHeight {
            active_values[[ConvWidth - 1, y]] = in_values[n + y];
        }

        let multiply = &active_values * params;

        out_values.push(sum_values(multiply.slice(s![.., ..])));
    }

    return out_values;
}

pub fn do_accumulate(
    in_values: &Vec<TinyNNFP16>,
    values_per_accum: usize,
    bias: TinyNNFP16,
    relu: bool,
) -> Vec<TinyNNFP16> {
    let mut out_values: Vec<TinyNNFP16> = Vec::new();
    let mut cur_accum = bias;
    let mut cur_value_idx = 0;

    for v in in_values {
        cur_accum = cur_accum + *v;
        cur_value_idx += 1;

        if cur_value_idx == values_per_accum {
            if relu && cur_accum.sgn() {
                cur_accum = TinyNNFP16Zero;
            }

            out_values.push(cur_accum);

            cur_accum = bias;
            cur_value_idx = 0;
        }
    }

    return out_values;
}

pub fn input_conv_stream_from_image(
    image: &Vec<TinyNNFP16>,
    start_x: usize,
    end_x: usize,
    y: usize,
    height: usize,
) -> Vec<TinyNNFP16> {
    let mut out_values: Vec<TinyNNFP16> = Vec::new();

    for x in start_x..end_x {
        for inner_y in 0..ConvHeight {
            out_values.push(image[x * height + (y + inner_y)]);
        }
    }

    return out_values;
}
