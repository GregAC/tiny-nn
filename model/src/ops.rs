pub const ConvWidth: usize = 4;
pub const ConvHeight: usize = 2;

use super::tnn_types::{TinyNNFP16, TinyNNFP16Zero, TinyNNFP16NegInf};
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
    let mut cur_accum = TinyNNFP16Zero;
    let mut cur_value_idx = 0;

    for v in in_values {
        cur_accum = cur_accum + *v;
        cur_value_idx += 1;

        if cur_value_idx == values_per_accum {
            cur_accum = cur_accum + bias;

            if relu && cur_accum.sgn() {
                cur_accum = TinyNNFP16Zero;
            }

            out_values.push(cur_accum);

            cur_accum = TinyNNFP16Zero;
            cur_value_idx = 0;
        }
    }

    return out_values;
}

pub fn do_mul_acc(
    in_values: &Vec<TinyNNFP16>,
    bias: TinyNNFP16,
    relu: bool
) -> TinyNNFP16 {
    let mut result = TinyNNFP16Zero;
    let mut i = 0;

    for vp in in_values[..].chunks(2) {
        println!("val {}: {:x} {:x}", i, vp[0].exp(), vp[0].mant());
        println!("param {}: {:x} {:x}", i, vp[1].exp(), vp[1].mant());
        result = result + (vp[0] * vp[1]);
        println!("result {}: {:x} {:x}", i, result.exp(), result.mant());

        i = i + 1;
    }

    result = result + bias;

    if (result.sgn() && relu) {
        TinyNNFP16Zero
    } else {
        result
    }
}

pub fn do_fixed_mul_acc(
    in_values: &Vec<TinyNNFP16>,
    value_per_mul_acc: usize,
    param: TinyNNFP16,
) -> Vec<TinyNNFP16> {
    let mut out_values: Vec<TinyNNFP16> = Vec::new();
    let mut cur_mul_acc = TinyNNFP16Zero;
    let mut cur_value_idx = 0;

    for v in in_values {
        let mul_result = *v * param;
        cur_mul_acc = cur_mul_acc + mul_result;

        println!("v {}: {:x} {:x}", cur_value_idx, v.exp(), v.mant());
        println!("mul_result {}: {:x} {:x}", cur_value_idx, mul_result.exp(), mul_result.mant());
        println!("cur_mul_acc {}: {:x} {:x}", cur_value_idx, cur_mul_acc.exp(), cur_mul_acc.mant());

        cur_value_idx += 1;

        if cur_value_idx == value_per_mul_acc {
            out_values.push(cur_mul_acc);
            cur_mul_acc = TinyNNFP16Zero;
            cur_value_idx = 0;
        }
    }

    out_values
}

pub fn do_max_pool (
    in_values: &Vec<TinyNNFP16>,
    values_per_pool: usize
) -> Vec<TinyNNFP16> {
    let mut out_values: Vec<TinyNNFP16> = Vec::new();
    let mut cur_max = TinyNNFP16NegInf;
    let mut cur_value_idx = 0;

    for v in in_values {
        if *v > cur_max {
            cur_max = *v;
        }

        println!("V {} is {}", cur_value_idx, v.to_f32());

        cur_value_idx += 1;

        if cur_value_idx == values_per_pool {
            println!("Max was {}", cur_max.to_f32());
            out_values.push(cur_max);
            cur_max = TinyNNFP16NegInf;
            cur_value_idx = 0;
        }
    }

    out_values
}
