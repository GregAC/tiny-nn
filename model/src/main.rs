mod tnn_types;
use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::io::BufWriter;
use tqdm::tqdm;

fn main() {
    //let test_fp16 = tnn_types::TinyNNFP16::new(false, 128, 0x40);
    //let test_fp16_2 = tnn_types::TinyNNFP16::new(true, 127, 0x00);
    //let result = test_fp16 + test_fp16_2;

    let mut rng = rand::thread_rng();
    let mut failures = 0;
    //let num1 = tnn_types::TinyNNFP16::new(false, 143, 0b0101010);
    //let num2 = tnn_types::TinyNNFP16::new(false, 142, 0b0001111);

    //let add_result = num1 + num2;
    //let fp32_add_result  = num1.to_f32() + num2.to_f32();

    //if !add_result.f32_cmp(fp32_add_result) {
    //    println!("FAIL!");
    //} else {
    //    println!("PASS!");
    //}

    //println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
    //println!("Add: {:e}, fp32_add: {:e}", add_result.to_f32(), fp32_add_result);
    //println!("num1 s:{:}, e:{:x}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
    //println!("num2 s:{:}, e:{:x}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
    //println!("add_result s:{:}, e:{:x}, m:{:x}", add_result.sgn(), add_result.exp(), add_result.mant());

    let mut fp_add_tests_file = match File::create("fp_add_tests.hex") {
        Err(why) => panic!("couldn't open fp_add_tests.hex: {}", why),
        Ok(file) => BufWriter::new(file)
    };

    let mut fp_mul_tests_file = match File::create("fp_mul_tests.hex") {
        Err(why) => panic!("couldn't open fp_mul_tests.hex: {}", why),
        Ok(file) => BufWriter::new(file)
    };

    for i in tqdm(0..10_000_000) {
        let num1 = tnn_types::TinyNNFP16::new(rng.gen(), rng.gen_range(10..250),rng.gen_range(0..128));
        let num2 = tnn_types::TinyNNFP16::new(rng.gen(), rng.gen_range(10..250),rng.gen_range(0..128));

        let mul_result = num1 * num2;
        let fp32_mul_result  = num1.to_f32() * num2.to_f32();
        if !mul_result.f32_cmp(fp32_mul_result) {
            failures += 1;

            println!("FAIL!");
            println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
            println!("Mul: {:e}, fp32_mul: {:e}", mul_result.to_f32(), fp32_mul_result);
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
            println!("Add: {:e}, fp32_add: {:e}", add_result.to_f32(), fp32_add_result);
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

//num1 s:false, e:82, m:21
//num2 s:false, e:7b, m:38
//add_result s:false, e:82, m:21

