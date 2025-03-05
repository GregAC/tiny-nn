use rand::Rng;
use super::ops::{do_convolve, ConvHeight, ConvWidth};
use super::tnn_types::{TinyNNFP16, TinyNNFP16Zero, TinyNNFP16StdNaN, TNNFP16MantWidth, TNNFP16ExpWidth};
use ndarray::{s, Array1, Array2};
use std::io;

// 8 cycles for parameters, then 12 cycles for first valid convolve result
const ConvFirstOutputDelay: usize = 12 + 8;
const ConvRowOutputDelay: usize = 6;
const AccumFirstOutputDelay: usize = 3;
const MulAccOutputDelay: usize = 4;

const TNNCmdOpConvolve: u16 = 1 << 12;
const TNNCmdOpAccumulate: u16 = 2 << 12;
const TNNCmdOpMulAcc: u16 = 3 << 12;

pub fn conv_stream_from_image_row(
    image: &Array2<TinyNNFP16>,
    start_x: usize,
    end_x: usize,
    y: usize,
) -> Vec<TinyNNFP16> {
    let mut out_values: Vec<TinyNNFP16> = Vec::new();

    for x in start_x..end_x {
        for inner_y in 0..ConvHeight {
            out_values.push(image[[x, (y + inner_y)]]);
        }
    }

    return out_values;
}

pub fn input_conv_stream_from_image(image: &Array2<TinyNNFP16>) -> Vec<TinyNNFP16> {
    let mut out: Vec<TinyNNFP16> = Vec::new();
    let height = image.shape()[1];

    let conv_result_height = image.shape()[1] - (ConvHeight - 1);

    for y in 0..conv_result_height {
        let mut conv_stream = conv_stream_from_image_row(image, 0, image.shape()[0], y);
        out.append(&mut conv_stream);
    }

    return out;
}

pub fn conv_image(image: &Array2<TinyNNFP16>, params: &Array2<TinyNNFP16>, y_offset_start: usize, y_offset_end: usize) -> Array2<TinyNNFP16> {
    let conv_result_width = image.shape()[0] - (ConvWidth - 1);
    let conv_result_height = image.shape()[1] - (ConvHeight - 1) - y_offset_end - y_offset_start;

    let mut result =
        Array2::<TinyNNFP16>::from_elem((conv_result_width, conv_result_height), TinyNNFP16Zero);

    for y in 0..conv_result_height {
        let conv_stream = conv_stream_from_image_row(image, 0, image.shape()[0], y + y_offset_start);
        let accum_result = Array1::from_vec(do_convolve(&conv_stream, params));

        result.slice_mut(s![.., y]).assign(&accum_result);
    }

    return result;
}

pub fn write_fp_vec_to_file(v: &Vec<TinyNNFP16>, writer: &mut dyn io::Write) {
    for n in v {
        write!(writer, "{:04x}\n", n.as_u16());
    }
}

pub fn output_stream_from_conv_image(conv_image: &Array2<TinyNNFP16>) -> Vec<Option<u8>> {
    let mut output: Vec<Option<u8>> = vec![None; ConvFirstOutputDelay];

    for y in 0..conv_image.shape()[1] {
        output.extend(
            conv_image
                .slice(s![.., y])
                .iter()
                .flat_map(|x| {
                    let x_i = x.as_u16();
                    [(x_i & 0xff) as u8, (x_i >> 8) as u8]
                })
                .map(|x| Some(x)),
        );

        if y != conv_image.shape()[1] - 1 {
            output.extend(vec![None; ConvRowOutputDelay]);
        }
    }

    return output;
}

pub fn write_output_stream(output_stream: &Vec<Option<u8>>, writer: &mut dyn io::Write) {
    for x in output_stream {
        if let Some(b) = x {
            write!(writer, "{:02x}\n", b);
        } else {
            write!(writer, "X\n");
        }
    }
}

pub fn full_conv_stream(conv_image_stream: &Vec<TinyNNFP16>, params: &Array2<TinyNNFP16>) -> Vec<u16> {
    let mut output: Vec<u16> = vec![TNNCmdOpConvolve];

    output.extend(params.flatten().to_vec().iter().map(|x| x.as_u16()));
    output.extend(conv_image_stream.iter().map(|x| x.as_u16()));
    output.push(TinyNNFP16StdNaN.as_u16());

    return output;
}

pub fn full_accum_stream(accum_values: &Vec<TinyNNFP16>, values_per_accum: usize, bias: TinyNNFP16, relu: bool) -> Vec<u16> {
    let relu_flag: u16 = if relu {0x100} else {0x0};
    let mut output: Vec<u16> = vec![TNNCmdOpAccumulate | relu_flag | (((values_per_accum - 1) & 0xff) as u16)];

    output.push(bias.as_u16());
    output.extend(accum_values.iter().map(|x| x.as_u16()));
    output.push(TinyNNFP16StdNaN.as_u16());

    return output;
}

pub fn full_mul_acc_stream(mul_acc_values: &Vec<TinyNNFP16>, bias: TinyNNFP16, relu: bool) -> Vec<u16> {
    let relu_flag: u16 = if relu {0x100} else {0x0};
    let mut output: Vec<u16> = vec![TNNCmdOpMulAcc | relu_flag];

    output.push(bias.as_u16());
    output.extend(mul_acc_values.iter().map(|x| x.as_u16()));
    output.push(TinyNNFP16StdNaN.as_u16());

    return output;
}

pub fn output_stream_from_accum_result(accum_result: &Vec<TinyNNFP16>, values_per_accum: usize) -> Vec<Option<u8>> {
    let mut output: Vec<Option<u8>> = vec![None; AccumFirstOutputDelay + 2];

    for v in accum_result {
        output.extend(vec![None; values_per_accum - 2]);
        let v_16 = v.as_u16();
        output.push(Some((v_16 & 0xff) as u8));
        output.push(Some((v_16 >> 8) as u8));
    }

    return output;
}

pub fn output_stream_from_mul_acc_result(num_params: usize, result: TinyNNFP16) -> Vec<Option<u8>> {
    let mut output: Vec<Option<u8>> = vec![None; MulAccOutputDelay + num_params * 2];
    let result_16 = result.as_u16();
    output.push(Some((result_16 & 0xff) as u8));
    output.push(Some((result_16 >> 8) as u8));

    return output;
}

pub fn gen_rand_mul_acc(num_params: usize) -> (Vec<TinyNNFP16>, TinyNNFP16) {
    let mut output: Vec<TinyNNFP16> = Vec::new();
    let mut rng = rand::thread_rng();
    let high_exp = rng.gen_range((1 << (TNNFP16ExpWidth - 1)) - 10..(1 << (TNNFP16ExpWidth - 1)) + 10);
    let low_exp = high_exp - 5;

    for _ in 0..num_params {
        output.push(TinyNNFP16::new(rng.gen(), rng.gen_range(low_exp..high_exp), rng.gen_range(0..(1 << TNNFP16MantWidth))));
        output.push(TinyNNFP16::new(rng.gen(), rng.gen_range(low_exp..high_exp), rng.gen_range(0..(1 << TNNFP16MantWidth))));
    }

    return (output, TinyNNFP16::new(rng.gen(), rng.gen_range(high_exp-2..high_exp), rng.gen_range(0..(1 << TNNFP16MantWidth))));
}
