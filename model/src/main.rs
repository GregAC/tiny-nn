mod ops;
mod tnn_types;
use ndarray::{array, Array, Array2, s};
use rand::Rng;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use tnn_types::{TinyNNFP16, TinyNNFP16Zero};
use tqdm::tqdm;
use itertools;

fn special_mul_check(
    a: tnn_types::TinyNNFP16,
    b: tnn_types::TinyNNFP16,
    c: tnn_types::TinyNNFP16,
) -> bool {
    (a.is_inf() && b.is_zero() && c.is_zero()) || (a.is_zero() && b.is_inf() && c.is_zero())
}

fn main() {
    //let test_fp16 = tnn_types::TinyNNFP16::new(false, 128, 0x40);
    //let test_fp16_2 = tnn_types::TinyNNFP16::new(true, 127, 0x00);
    //let result = test_fp16 + test_fp16_2;

    //let mut rng = rand::thread_rng();
    //let mut failures = 0;
    ////let num1 = tnn_types::TinyNNFP16::new(false, 143, 0b0101010);
    ////let num2 = tnn_types::TinyNNFP16::new(false, 142, 0b0001111);

    ////let add_result = num1 + num2;
    ////let fp32_add_result  = num1.to_f32() + num2.to_f32();

    ////if !add_result.f32_cmp(fp32_add_result) {
    ////    println!("FAIL!");
    ////} else {
    ////    println!("PASS!");
    ////}

    ////println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
    ////println!("Add: {:e}, fp32_add: {:e}", add_result.to_f32(), fp32_add_result);
    ////println!("num1 s:{:}, e:{:x}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
    ////println!("num2 s:{:}, e:{:x}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
    ////println!("add_result s:{:}, e:{:x}, m:{:x}", add_result.sgn(), add_result.exp(), add_result.mant());

    //let mut fp_add_tests_file = match File::create("fp_add_tests.hex") {
    //    Err(why) => panic!("couldn't open fp_add_tests.hex: {}", why),
    //    Ok(file) => BufWriter::new(file)
    //};

    //let mut fp_mul_tests_file = match File::create("fp_mul_tests.hex") {
    //    Err(why) => panic!("couldn't open fp_mul_tests.hex: {}", why),
    //    Ok(file) => BufWriter::new(file)
    //};

    //for i in tqdm(0..10_000_000) {
    //    let (num1, num2) = tnn_types::gen_rnd_fp_operands(tnn_types::gen_rnd_exp_kind());

    //    let mul_result = num1 * num2;
    //    let fp32_mul_result  = num1.to_f32() * num2.to_f32();
    //    if !special_mul_check(num1, num2, mul_result) && !mul_result.f32_cmp(fp32_mul_result) {
    //        failures += 1;

    //        println!("FAIL!");
    //        println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
    //        println!("Mul: {:e}, fp32_mul: {:e} fp32 denorm: {}", mul_result.to_f32(), fp32_mul_result, fp32_mul_result.is_subnormal());
    //        println!("num1 s:{:}, e:{}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
    //        println!("num2 s:{:}, e:{}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
    //        println!("mul_result s:{:}, e:{}, m:{:x}", mul_result.sgn(), mul_result.exp(), mul_result.mant());
    //    }

    //    write!(&mut fp_mul_tests_file, "{:04x}\n", num1.as_u16());
    //    write!(&mut fp_mul_tests_file, "{:04x}\n", num2.as_u16());
    //    write!(&mut fp_mul_tests_file, "{:04x}\n", mul_result.as_u16());

    //    let add_result = num1 + num2;
    //    let fp32_add_result  = (num1.to_f32() + num2.to_f32()) as f32;
    //    if !add_result.f32_cmp(fp32_add_result) {
    //        failures += 1;

    //        println!("FAIL!");
    //        println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
    //        println!("Add: {:e}, fp32_add: {:e} fp32 denorm: {}", add_result.to_f32(), fp32_add_result, fp32_add_result.is_subnormal());
    //        println!("num1 s:{:}, e:{:x}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
    //        println!("num2 s:{:}, e:{:x}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
    //        println!("add_result s:{:}, e:{:x}, m:{:x}", add_result.sgn(), add_result.exp(), add_result.mant());
    //    }

    //    write!(&mut fp_add_tests_file, "{:04x}\n", num1.as_u16());
    //    write!(&mut fp_add_tests_file, "{:04x}\n", num2.as_u16());
    //    write!(&mut fp_add_tests_file, "{:04x}\n", add_result.as_u16());
    //}

    //fp_add_tests_file.flush();
    //fp_mul_tests_file.flush();

    //if failures == 0 {
    //    println!("PASS!");
    //} else {
    //    println!("{} failures", failures);
    //}

    let test_params1 = array![
        [
          TinyNNFP16::from_f32(-0.008700000122189522),
          TinyNNFP16::from_f32(0.00139999995008111),
        ],
        [
          TinyNNFP16::from_f32(-0.09920000284910202),
          TinyNNFP16::from_f32(-0.3628000020980835),
        ],
        [
          TinyNNFP16::from_f32(0.028200000524520874),
          TinyNNFP16::from_f32(0.049300000071525574),
        ],
        [
          TinyNNFP16::from_f32(-0.0763000026345253),
          TinyNNFP16::from_f32(0.43709999322891235),
        ]
    ];

    let test_params2 = array![
        [
          TinyNNFP16::from_f32(-0.462799996137619),
          TinyNNFP16::from_f32(0.7610999941825867),
        ],
        [
          TinyNNFP16::from_f32(0.27239999175071716),
          TinyNNFP16::from_f32(0.6888999938964844),
        ],
        [
          TinyNNFP16::from_f32(0.8213000297546387),
          TinyNNFP16::from_f32(0.09920000284910202),
        ],
        [
          TinyNNFP16::from_f32(0.07729999721050262),
          TinyNNFP16::from_f32(-0.4505999982357025),
        ]
    ];

    //let test_image_in = vec![
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.06270000338554382),
    //    TinyNNFP16::from_f32(0.5098000168800354),
    //    TinyNNFP16::from_f32(0.6607999801635742),
    //    TinyNNFP16::from_f32(0.6626999974250793),
    //    TinyNNFP16::from_f32(0.6607999801635742),
    //    TinyNNFP16::from_f32(0.24709999561309814),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.24609999358654022),
    //    TinyNNFP16::from_f32(0.9157000184059143),
    //    TinyNNFP16::from_f32(0.7911999821662903),
    //    TinyNNFP16::from_f32(0.46860000491142273),
    //    TinyNNFP16::from_f32(0.3626999855041504),
    //    TinyNNFP16::from_f32(0.8813999891281128),
    //    TinyNNFP16::from_f32(0.8666999936103821),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.009800000116229057),
    //    TinyNNFP16::from_f32(0.4431000053882599),
    //    TinyNNFP16::from_f32(0.9569000005722046),
    //    TinyNNFP16::from_f32(0.3666999936103821),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.5108000040054321),
    //    TinyNNFP16::from_f32(0.9901999831199646),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.3314000070095062),
    //    TinyNNFP16::from_f32(0.9901999831199646),
    //    TinyNNFP16::from_f32(0.7088000178337097),
    //    TinyNNFP16::from_f32(0.04610000178217888),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.7519999742507935),
    //    TinyNNFP16::from_f32(0.6499999761581421),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.3303999900817871),
    //    TinyNNFP16::from_f32(0.9656999707221985),
    //    TinyNNFP16::from_f32(0.7930999994277954),
    //    TinyNNFP16::from_f32(0.08529999852180481),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.11180000007152557),
    //    TinyNNFP16::from_f32(0.907800018787384),
    //    TinyNNFP16::from_f32(0.14219999313354492),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.8980000019073486),
    //    TinyNNFP16::from_f32(0.9265000224113464),
    //    TinyNNFP16::from_f32(0.9656999707221985),
    //    TinyNNFP16::from_f32(0.13920000195503235),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.006899999920278788),
    //    TinyNNFP16::from_f32(0.6901999711990356),
    //    TinyNNFP16::from_f32(0.5098000168800354),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.6843000054359436),
    //    TinyNNFP16::from_f32(0.8048999905586243),
    //    TinyNNFP16::from_f32(0.40290001034736633),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.5146999955177307),
    //    TinyNNFP16::from_f32(0.5755000114440918),
    //    TinyNNFP16::from_f32(0.05490000173449516),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.4293999969959259),
    //    TinyNNFP16::from_f32(0.8264999985694885),
    //    TinyNNFP16::from_f32(0.9814000129699707),
    //    TinyNNFP16::from_f32(0.9901999831199646),
    //    TinyNNFP16::from_f32(0.9930999875068665),
    //    TinyNNFP16::from_f32(0.7206000089645386),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.21080000698566437),
    //    TinyNNFP16::from_f32(0.3824000060558319),
    //    TinyNNFP16::from_f32(0.33730000257492065),
    //    TinyNNFP16::from_f32(0.011800000444054604),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //    TinyNNFP16::from_f32(0.0),
    //];

    //let convolve_bias = TinyNNFP16::from_f32(-0.5370);


    //let mut first_convolve = Array2::<TinyNNFP16>::from_elem((11, 11), TinyNNFP16Zero);

    //for y in 0..11 {
    //    let image_stream = ops::input_conv_stream_from_image(&test_image_in, 0, 14, y, 14);
    //    let conv_result = ops::do_convolve(&image_stream, &test_params1);

    //    first_convolve.slice_mut(s![..,y]).assign(&Array::from_vec(conv_result));
    //}

    //let mut second_convolve = Array2::<TinyNNFP16>::from_elem((11, 11), TinyNNFP16Zero);

    //for y in 0..11 {
    //    let image_stream = ops::input_conv_stream_from_image(&test_image_in, 0, 14, y + 2, 14);
    //    let conv_result = ops::do_convolve(&image_stream, &test_params2);

    //    second_convolve.slice_mut(s![..,y]).assign(&Array::from_vec(conv_result));
    //}

    //for y in 0..11 {
    //    println!("[");

    //    let interleaved = itertools::interleave(first_convolve.slice(s![.., y]).into_iter(),
    //        second_convolve.slice(s![.., y]).into_iter()).cloned().collect::<Vec<_>>();

    //    let final_convolve = ops::do_accumulate(&interleaved, 2, convolve_bias, true);

    //    for r in final_convolve {
    //        println!("  {},", r.to_f32());
    //    }
    //    println!("],");
    //}

    //let image_stream = ops::input_conv_stream_from_image(&test_image_in, 0, 4, 1, 8);
    //for f in &image_stream {
    //    println!("Pix is {}", f.to_f32());
    //}

    //let conv_result = ops::do_convolve(&image_stream, test_params);
    //for r in conv_result {
    //    println!("Result {}", r.to_f32());
    //}
    //
    
    let test_params1 = array![
        [
          TinyNNFP16::from_f32(0.5),
          TinyNNFP16::from_f32(0.5),
        ],
        [
          TinyNNFP16::from_f32(0.5),
          TinyNNFP16::from_f32(0.5),
        ],
        [
          TinyNNFP16::from_f32(0.5),
          TinyNNFP16::from_f32(0.5),
        ],
        [
          TinyNNFP16::from_f32(0.5),
          TinyNNFP16::from_f32(0.5),
        ]
    ];
    let test_image_in = vec![
        TinyNNFP16::from_f32(0.000000),
        TinyNNFP16::from_f32(8.000000),
        TinyNNFP16::from_f32(16.000000),
        TinyNNFP16::from_f32(24.000000),
        TinyNNFP16::from_f32(32.000000),
        TinyNNFP16::from_f32(40.000000),
        TinyNNFP16::from_f32(48.000000),
        TinyNNFP16::from_f32(56.000000),
        TinyNNFP16::from_f32(1.000000),
        TinyNNFP16::from_f32(9.000000),
        TinyNNFP16::from_f32(17.000000),
        TinyNNFP16::from_f32(25.000000),
        TinyNNFP16::from_f32(33.000000),
        TinyNNFP16::from_f32(41.000000),
        TinyNNFP16::from_f32(49.000000),
        TinyNNFP16::from_f32(57.000000),
        TinyNNFP16::from_f32(2.000000),
        TinyNNFP16::from_f32(10.000000),
        TinyNNFP16::from_f32(18.000000),
        TinyNNFP16::from_f32(26.000000),
        TinyNNFP16::from_f32(34.000000),
        TinyNNFP16::from_f32(42.000000),
        TinyNNFP16::from_f32(50.000000),
        TinyNNFP16::from_f32(58.000000),
        TinyNNFP16::from_f32(3.000000),
        TinyNNFP16::from_f32(11.000000),
        TinyNNFP16::from_f32(19.000000),
        TinyNNFP16::from_f32(27.000000),
        TinyNNFP16::from_f32(35.000000),
        TinyNNFP16::from_f32(43.000000),
        TinyNNFP16::from_f32(51.000000),
        TinyNNFP16::from_f32(59.000000),
        TinyNNFP16::from_f32(4.000000),
        TinyNNFP16::from_f32(12.000000),
        TinyNNFP16::from_f32(20.000000),
        TinyNNFP16::from_f32(28.000000),
        TinyNNFP16::from_f32(36.000000),
        TinyNNFP16::from_f32(44.000000),
        TinyNNFP16::from_f32(52.000000),
        TinyNNFP16::from_f32(60.000000),
        TinyNNFP16::from_f32(5.000000),
        TinyNNFP16::from_f32(13.000000),
        TinyNNFP16::from_f32(21.000000),
        TinyNNFP16::from_f32(29.000000),
        TinyNNFP16::from_f32(37.000000),
        TinyNNFP16::from_f32(45.000000),
        TinyNNFP16::from_f32(53.000000),
        TinyNNFP16::from_f32(61.000000),
        TinyNNFP16::from_f32(6.000000),
        TinyNNFP16::from_f32(14.000000),
        TinyNNFP16::from_f32(22.000000),
        TinyNNFP16::from_f32(30.000000),
        TinyNNFP16::from_f32(38.000000),
        TinyNNFP16::from_f32(46.000000),
        TinyNNFP16::from_f32(54.000000),
        TinyNNFP16::from_f32(62.000000),
        TinyNNFP16::from_f32(7.000000),
        TinyNNFP16::from_f32(15.000000),
        TinyNNFP16::from_f32(23.000000),
        TinyNNFP16::from_f32(31.000000),
        TinyNNFP16::from_f32(39.000000),
        TinyNNFP16::from_f32(47.000000),
        TinyNNFP16::from_f32(55.000000),
        TinyNNFP16::from_f32(63.000000)
    ];


    let mut test_image_file = match File::create("test_image.hex") {
        Err(why) => panic!("couldn't open test_image.hex: {}", why),
        Ok(file) => BufWriter::new(file)
    };

    let image_stream = ops::input_conv_stream_from_image(&test_image_in, 0, 8, 0, 8);
    let conv_result = ops::do_convolve(&image_stream, &test_params1);

    for n in image_stream {
        write!(&mut test_image_file, "{:04x}\n", n.as_u16());
    }

    for n in conv_result {
        println!("{} {:x} {} {} {}", n.to_f32(), n.as_u16(), n.exp(), n.mant(), n.sgn());
    }

    let foo = TinyNNFP16::from_f32(0.5);
    println!("{:x}", foo.as_u16());
}

//num1 s:false, e:82, m:21
//num2 s:false, e:7b, m:38
//add_result s:false, e:82, m:21
