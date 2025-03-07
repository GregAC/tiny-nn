mod ops;
mod test_data;
mod tnn_types;
mod utils;

use itertools;
use ndarray::{array, s, Array, Array2};
use rand::Rng;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use tnn_types::{TinyNNFP16, TinyNNFP16Zero};
use tqdm::tqdm;

fn special_mul_check(
    a: tnn_types::TinyNNFP16,
    b: tnn_types::TinyNNFP16,
    c: tnn_types::TinyNNFP16,
) -> bool {
    (a.is_inf() && b.is_zero() && c.is_zero()) || (a.is_zero() && b.is_inf() && c.is_zero())
}

fn create_add_mul_tests() {
    let mut rng = rand::thread_rng();
    let mut failures = 0;

    let mut fp_add_tests_file = match File::create("fp_add_tests.hex") {
        Err(why) => panic!("couldn't open fp_add_tests.hex: {}", why),
        Ok(file) => BufWriter::new(file)
    };

    let mut fp_mul_tests_file = match File::create("fp_mul_tests.hex") {
        Err(why) => panic!("couldn't open fp_mul_tests.hex: {}", why),
        Ok(file) => BufWriter::new(file)
    };

    for i in tqdm(0..10_000_000) {
        let (num1, num2) = tnn_types::gen_rnd_fp_operands(tnn_types::gen_rnd_exp_kind());

        let mul_result = num1 * num2;
        let fp32_mul_result  = num1.to_f32() * num2.to_f32();
        if !special_mul_check(num1, num2, mul_result) && !mul_result.f32_cmp(fp32_mul_result) {
            failures += 1;

            println!("FAIL!");
            println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
            println!("Mul: {:e}, fp32_mul: {:e} fp32 denorm: {}", mul_result.to_f32(), fp32_mul_result, fp32_mul_result.is_subnormal());
            println!("num1 s:{:}, e:{}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
            println!("num2 s:{:}, e:{}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
            println!("mul_result s:{:}, e:{}, m:{:x}", mul_result.sgn(), mul_result.exp(), mul_result.mant());
        }

        write!(&mut fp_mul_tests_file, "{:04x}\n", num1.as_u16());
        write!(&mut fp_mul_tests_file, "{:04x}\n", num2.as_u16());
        write!(&mut fp_mul_tests_file, "{:04x}\n", mul_result.as_u16());

        let add_result = num1 + num2;
        let fp32_add_result  = (num1.to_f32() + num2.to_f32()) as f32;
        if !add_result.f32_cmp(fp32_add_result) {
            failures += 1;

            println!("FAIL!");
            println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
            println!("Add: {:e}, fp32_add: {:e} fp32 denorm: {}", add_result.to_f32(), fp32_add_result, fp32_add_result.is_subnormal());
            println!("num1 s:{:}, e:{:x}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
            println!("num2 s:{:}, e:{:x}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
            println!("add_result s:{:}, e:{:x}, m:{:x}", add_result.sgn(), add_result.exp(), add_result.mant());
        }

        write!(&mut fp_add_tests_file, "{:04x}\n", num1.as_u16());
        write!(&mut fp_add_tests_file, "{:04x}\n", num2.as_u16());
        write!(&mut fp_add_tests_file, "{:04x}\n", add_result.as_u16());
    }

    fp_add_tests_file.flush();
    fp_mul_tests_file.flush();

    if failures == 0 {
        println!("PASS!");
    } else {
        println!("{} failures", failures);
    }
}

fn gen_convolve() {
    let test_image = test_data::get_incrementing_test_image();
    let test_params = test_data::get_half_const_convolve_params();

    let conv_result = utils::conv_image(&test_image, &test_params, 0, 0);

    for y in 0..conv_result.shape()[1] {
        for x in 0..conv_result.shape()[0] {
            println!(
                "{:04x} {:02x} {:02x} ",
                conv_result[[x, y]].as_u16(),
                conv_result[[x, y]].mant(),
                conv_result[[x, y]].exp()
            );
        }
        println!("");
    }


    let mut test_file = match File::create("test_vec.hex") {
        Err(why) => panic!("couldn't open test_vec.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };


    let test_input_stream = utils::full_conv_stream(&utils::input_conv_stream_from_image(&test_image), &test_params);

    for x in test_input_stream {
        write!(test_file, "{:04x}\n", x);
    }

    let expected_output = utils::output_stream_from_conv_image(&conv_result);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}

fn gen_accumulate() {
    //let test_nums = vec![
    //    TinyNNFP16::from_f32(0.25),
    //    TinyNNFP16::from_f32(0.5),
    //    TinyNNFP16::from_f32(0.75),
    //    TinyNNFP16::from_f32(1.0),

    //    TinyNNFP16::from_f32(1.25),
    //    TinyNNFP16::from_f32(1.5),
    //    TinyNNFP16::from_f32(1.75),
    //    TinyNNFP16::from_f32(2.0),

    //    TinyNNFP16::from_f32(2.25),
    //    TinyNNFP16::from_f32(2.5),
    //    TinyNNFP16::from_f32(2.75),
    //    TinyNNFP16::from_f32(3.0),

    //    TinyNNFP16::from_f32(3.25),
    //    TinyNNFP16::from_f32(3.5),
    //    TinyNNFP16::from_f32(3.75),
    //    TinyNNFP16::from_f32(4.0),
    //];

    let test_nums = vec![
        TinyNNFP16::from_f32(1.0),
        TinyNNFP16::from_f32(2.0),

        TinyNNFP16::from_f32(3.0),
        TinyNNFP16::from_f32(4.0),

        TinyNNFP16::from_f32(5.0),
        TinyNNFP16::from_f32(6.0),

        TinyNNFP16::from_f32(7.0),
        TinyNNFP16::from_f32(8.0),
    ];

    let test_accumulate = ops::do_accumulate(&test_nums, 2, TinyNNFP16::from_f32(-3.5), true);

    for v in &test_accumulate {
        println!("{} {:04x} {:02x} {:02x}", v.to_f32(), v.as_u16(), v.mant(), v.exp());
    }

    let mut test_file = match File::create("test_vec.hex") {
        Err(why) => panic!("couldn't open test_vec.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };


    let test_input_stream = utils::full_accum_stream(&test_nums, 2, TinyNNFP16::from_f32(-3.5), true);

    for x in test_input_stream {
        write!(test_file, "{:04x}\n", x);
    }

    let expected_output = utils::output_stream_from_accum_result(&test_accumulate, 2);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}

fn gen_accumulate2() {
    let test_image = test_data::get_mnist_image();
    let test_params1 = test_data::get_mnist_convolve_0_params();
    let test_params2 = test_data::get_mnist_convolve_1_params();

    let conv_result1 = utils::conv_image(&test_image, &test_params1, 0, 2);
    let conv_result2 = utils::conv_image(&test_image, &test_params2, 2, 0);

    let mut full_interleave: Vec<TinyNNFP16> = Vec::new();
    let full_accum: Vec<TinyNNFP16> = Vec::new();

    let convolve_bias = TinyNNFP16::from_f32(-0.5370);

    for y in 0..11 {
        let interleaved = itertools::interleave(conv_result1.slice(s![.., y]).into_iter(),
            conv_result2.slice(s![.., y]).into_iter()).cloned().collect::<Vec<_>>();

        full_interleave.extend(interleaved);
    }

    let test_accumulate = ops::do_accumulate(&full_interleave, 2, convolve_bias, true);

    for v in &test_accumulate {
        println!("{} {:04x} {:02x} {:02x}", v.to_f32(), v.as_u16(), v.mant(), v.exp());
    }

    let mut test_file = match File::create("test_vec.hex") {
        Err(why) => panic!("couldn't open test_vec.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    let test_input_stream = utils::full_accum_stream(&full_interleave, 2, convolve_bias, true);

    for x in test_input_stream {
        write!(test_file, "{:04x}\n", x);
    }

    let expected_output = utils::output_stream_from_accum_result(&test_accumulate, 2);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}

fn gen_simple_mul_acc() {
    let test_vals = vec![
        TinyNNFP16::from_f32(1.0), TinyNNFP16::from_f32(2.0),
        TinyNNFP16::from_f32(3.0), TinyNNFP16::from_f32(4.0),
        TinyNNFP16::from_f32(5.0), TinyNNFP16::from_f32(6.0),
        TinyNNFP16::from_f32(7.0), TinyNNFP16::from_f32(8.0),
    ];

    let mul_acc_result = ops::do_mul_acc(&test_vals, TinyNNFP16::from_f32(0.5), true);

    println!("{} {:04x} {:02x} {:02x}", mul_acc_result.to_f32(), mul_acc_result.as_u16(), mul_acc_result.mant(), mul_acc_result.exp());

    let mut test_file = match File::create("test_vec.hex") {
        Err(why) => panic!("couldn't open test_vec.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    let test_input_stream = utils::full_mul_acc_stream(&test_vals, TinyNNFP16::from_f32(0.5), true);

    for x in test_input_stream {
        write!(test_file, "{:04x}\n", x);
    }

    let expected_output = utils::output_stream_from_mul_acc_result(4, mul_acc_result);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}

fn gen_rnd_mul_acc(num_params: usize, relu: bool) {
    let (test_vals, bias) = utils::gen_rand_mul_acc(num_params);
    let mul_acc_result = ops::do_mul_acc(&test_vals, bias, relu);

    println!("{} {:04x} {:02x} {:02x}", mul_acc_result.to_f32(), mul_acc_result.as_u16(), mul_acc_result.mant(), mul_acc_result.exp());

    let mut test_file = match File::create("test_vec.hex") {
        Err(why) => panic!("couldn't open test_vec.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    let test_input_stream = utils::full_mul_acc_stream(&test_vals, bias, relu);

    for x in test_input_stream {
        write!(test_file, "{:04x}\n", x);
    }

    let expected_output = utils::output_stream_from_mul_acc_result(num_params, mul_acc_result);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}

fn main() {
    gen_rnd_mul_acc(7, true);
}
