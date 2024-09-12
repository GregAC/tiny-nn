mod tnn_types;
use rand::Rng;
use std::fs::File;
use std::io::Write;

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
        Ok(file) => file
    };

    let mut fp_mul_tests_file = match File::create("fp_mul_tests.hex") {
        Err(why) => panic!("couldn't open fp_mul_tests.hex: {}", why),
        Ok(file) => file
    };

    for i in 0..1000 {
        let num1 = tnn_types::TinyNNFP16::new(rng.gen(), rng.gen_range(100..160),rng.gen_range(0..128));
        let num2 = tnn_types::TinyNNFP16::new(rng.gen(), rng.gen_range(100..160),rng.gen_range(0..128));
        //let num1 = tnn_types::TinyNNFP16::new(false, 0x82, 0x21);
        //let num2 = tnn_types::TinyNNFP16::new(false, 0x7b, 0x38);

        let mul_result = num1 * num2;
        let fp32_mul_result  = num1.to_f32() * num2.to_f32();
        if !mul_result.f32_cmp(fp32_mul_result) {
            failures += 1;

            println!("FAIL!");
            println!("Num1: {:e}, Num2: {:e}", num1.to_f32(), num2.to_f32());
            println!("Mul: {:e}, fp32_mul: {:e}", mul_result.to_f32(), fp32_mul_result);
            println!("num1 s:{:}, e:{:x}, m:{:x}", num1.sgn(), num1.exp(), num1.mant());
            println!("num2 s:{:}, e:{:x}, m:{:x}", num2.sgn(), num2.exp(), num2.mant());
            println!("mul_result s:{:}, e:{:x}, m:{:x}", mul_result.sgn(), mul_result.exp(), mul_result.mant());
        }

        write!(&mut fp_mul_tests_file, "{:04x}\n", num1.as_u16());
        write!(&mut fp_mul_tests_file, "{:04x}\n", num2.as_u16());
        write!(&mut fp_mul_tests_file, "{:04x}\n", mul_result.as_u16());

        let add_result = num1 + num2;
        let fp32_add_result  = num1.to_f32() + num2.to_f32();
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

        if ((i % 10_000) == 0) {
            println!("{}", i);
        }
    }



    if failures == 0 {
        println!("PASS!");
    }
}

//num1 s:false, e:82, m:21
//num2 s:false, e:7b, m:38
//add_result s:false, e:82, m:21

