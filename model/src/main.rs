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

fn main() {
    let test_image = test_data::get_mnist_image();
    let test_params = test_data::get_mnist_convolve_0_params();

    let conv_result = utils::conv_image(&test_image, &test_params);

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

    utils::write_fp_vec_to_file(&test_params.flatten().to_vec(), &mut test_file);
    utils::write_fp_vec_to_file(
        &utils::input_conv_stream_from_image(&test_image),
        &mut test_file,
    );

    let expected_output = utils::output_stream_from_conv_image(&conv_result);

    let mut expected_output_file = match File::create("expected_out.hex") {
        Err(why) => panic!("couldn't open expected_out.hex: {}", why),
        Ok(file) => BufWriter::new(file),
    };

    utils::write_output_stream(&expected_output, &mut expected_output_file);
}
