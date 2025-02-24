use super::ops::{do_convolve, ConvHeight, ConvWidth};
use super::tnn_types::{TinyNNFP16, TinyNNFP16Zero};
use ndarray::{s, Array1, Array2};
use std::io;

// 8 cycles for parameters, then 12 cycles for first valid convolve result
const ConvFirstOutputDelay: usize = 12 + 8;
const ConvRowOutputDelay: usize = 6;

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

pub fn conv_image(image: &Array2<TinyNNFP16>, params: &Array2<TinyNNFP16>) -> Array2<TinyNNFP16> {
    let conv_result_width = image.shape()[0] - (ConvWidth - 1);
    let conv_result_height = image.shape()[1] - (ConvHeight - 1);

    let mut result =
        Array2::<TinyNNFP16>::from_elem((conv_result_width, conv_result_height), TinyNNFP16Zero);

    for y in 0..conv_result_height {
        let conv_stream = conv_stream_from_image_row(image, 0, image.shape()[0], y);
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
