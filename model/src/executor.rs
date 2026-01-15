use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::config::Operation;
use crate::ops;
use crate::tnn_types::TinyNNFP16;
use crate::utils;

#[derive(Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    Generate,  // Full test vectors with timing
    Simulate,  // Just compute results, no timing
}

pub fn execute_operation(op: &Operation, base_output_dir: &Path, mode: ExecutionMode) {
    let output_dir = op
        .output_dir
        .as_ref()
        .map(Path::new)
        .unwrap_or(base_output_dir);

    match op.op_type.as_str() {
        "convolve" => execute_convolve(op, output_dir, mode),
        "accumulate" => execute_accumulate(op, output_dir, mode),
        "mul_acc" => execute_mul_acc(op, output_dir, mode),
        "fixed_mul_acc" => execute_fixed_mul_acc(op, output_dir, mode),
        "max_pool" => execute_max_pool(op, output_dir, mode),
        _ => panic!("Unknown operation type: '{}'", op.op_type),
    }
}

fn execute_convolve(op: &Operation, output_dir: &Path, mode: ExecutionMode) {
    let image = op
        .image
        .as_ref()
        .expect("Convolve operation requires 'image' data source")
        .resolve_array2();
    let params = op
        .params
        .as_ref()
        .expect("Convolve operation requires 'params' data source")
        .resolve_array2();

    // Compute convolution result
    let conv_result = utils::conv_image(&image, &params, 0, 0);

    match mode {
        ExecutionMode::Generate => {
            // Generate input stream
            let input_stream = utils::full_conv_stream(
                &utils::input_conv_stream_from_image(&image),
                &params,
            );

            // Generate expected output stream
            let expected_output = utils::output_stream_from_conv_image(&conv_result);

            // Write files
            write_input_stream(output_dir, &op.name, &input_stream);
            write_output_stream(output_dir, &op.name, &expected_output);
        }
        ExecutionMode::Simulate => {
            // Write just the result values
            let result_values: Vec<TinyNNFP16> = conv_result.iter().cloned().collect();
            write_result_file(output_dir, &op.name, &result_values);
        }
    }
}

fn execute_accumulate(op: &Operation, output_dir: &Path, mode: ExecutionMode) {
    let values = op
        .values
        .as_ref()
        .expect("Accumulate operation requires 'values' data source")
        .resolve_vector();
    let group_size = op
        .group_size
        .expect("Accumulate operation requires 'group_size'");
    assert!(group_size > 0, "group_size must be greater than 0");
    let bias = op
        .bias
        .as_ref()
        .expect("Accumulate operation requires 'bias' data source")
        .resolve_scalar();
    let relu = op.relu.unwrap_or(false);

    // Compute accumulate result
    let accum_result = ops::do_accumulate(&values, group_size, bias, relu);

    match mode {
        ExecutionMode::Generate => {
            // Generate input stream
            let input_stream = utils::full_accum_stream(&values, group_size, bias, relu);

            // Generate expected output stream
            let expected_output = utils::output_stream_from_accum_result(&accum_result, group_size);

            // Write files
            write_input_stream(output_dir, &op.name, &input_stream);
            write_output_stream(output_dir, &op.name, &expected_output);
        }
        ExecutionMode::Simulate => {
            write_result_file(output_dir, &op.name, &accum_result);
        }
    }
}

fn execute_mul_acc(op: &Operation, output_dir: &Path, mode: ExecutionMode) {
    let values = op
        .values
        .as_ref()
        .expect("MulAcc operation requires 'values' data source (interleaved val,param pairs)")
        .resolve_vector();
    assert!(
        values.len() % 2 == 0,
        "MulAcc requires an even number of values (interleaved val,param pairs), got {}",
        values.len()
    );
    let bias = op
        .bias
        .as_ref()
        .expect("MulAcc operation requires 'bias' data source")
        .resolve_scalar();
    let relu = op.relu.unwrap_or(false);

    // Compute mul_acc result
    let mul_acc_result = ops::do_mul_acc(&values, bias, relu);

    let num_params = values.len() / 2;

    match mode {
        ExecutionMode::Generate => {
            // Generate input stream
            let input_stream = utils::full_mul_acc_stream(&values, bias, relu);

            // Generate expected output stream
            let expected_output = utils::output_stream_from_mul_acc_result(num_params, mul_acc_result);

            // Write files
            write_input_stream(output_dir, &op.name, &input_stream);
            write_output_stream(output_dir, &op.name, &expected_output);
        }
        ExecutionMode::Simulate => {
            write_result_file(output_dir, &op.name, &vec![mul_acc_result]);
        }
    }
}

fn execute_fixed_mul_acc(op: &Operation, output_dir: &Path, mode: ExecutionMode) {
    let values = op
        .values
        .as_ref()
        .expect("FixedMulAcc operation requires 'values' data source")
        .resolve_vector();
    let group_size = op
        .group_size
        .expect("FixedMulAcc operation requires 'group_size'");
    assert!(group_size > 0, "group_size must be greater than 0");
    let param = op
        .param
        .as_ref()
        .expect("FixedMulAcc operation requires 'param' data source")
        .resolve_scalar();

    // Compute fixed_mul_acc result
    let fixed_mul_acc_result = ops::do_fixed_mul_acc(&values, group_size, param);

    match mode {
        ExecutionMode::Generate => {
            // Generate input stream
            let input_stream = utils::full_fixed_mul_acc_stream(&values, group_size, param);

            // Generate expected output stream
            let expected_output =
                utils::output_stream_from_fixed_mul_acc_result(&fixed_mul_acc_result, group_size);

            // Write files
            write_input_stream(output_dir, &op.name, &input_stream);
            write_output_stream(output_dir, &op.name, &expected_output);
        }
        ExecutionMode::Simulate => {
            write_result_file(output_dir, &op.name, &fixed_mul_acc_result);
        }
    }
}

fn execute_max_pool(op: &Operation, output_dir: &Path, mode: ExecutionMode) {
    let values = op
        .values
        .as_ref()
        .expect("MaxPool operation requires 'values' data source")
        .resolve_vector();
    let pool_size = op
        .values_per_pool
        .expect("MaxPool operation requires 'values_per_pool'");
    assert!(pool_size > 0, "values_per_pool must be greater than 0");

    // Compute max_pool result
    let max_pool_result = ops::do_max_pool(&values, pool_size);

    match mode {
        ExecutionMode::Generate => {
            // Generate input stream
            let input_stream = utils::full_max_pool_stream(&values, pool_size);

            // Generate expected output stream
            let expected_output =
                utils::output_stream_from_max_pool_result(&max_pool_result, pool_size);

            // Write files
            write_input_stream(output_dir, &op.name, &input_stream);
            write_output_stream(output_dir, &op.name, &expected_output);
        }
        ExecutionMode::Simulate => {
            write_result_file(output_dir, &op.name, &max_pool_result);
        }
    }
}

// Helper functions for writing output files

fn write_input_stream(output_dir: &Path, name: &str, stream: &Vec<u16>) {
    let path = output_dir.join(format!("{}.hex", name));
    let mut file = BufWriter::new(
        File::create(&path).expect(&format!("Failed to create file: {:?}", path)),
    );

    for x in stream {
        writeln!(file, "{:04x}", x).expect("Failed to write to file");
    }

    file.flush().expect("Failed to flush file");
    println!("Wrote: {:?}", path);
}

fn write_output_stream(output_dir: &Path, name: &str, stream: &Vec<Option<u8>>) {
    let path = output_dir.join(format!("{}_expected.hex", name));
    let mut file = BufWriter::new(
        File::create(&path).expect(&format!("Failed to create file: {:?}", path)),
    );

    utils::write_output_stream(stream, &mut file);

    file.flush().expect("Failed to flush file");
    println!("Wrote: {:?}", path);
}

fn write_result_file(output_dir: &Path, name: &str, values: &Vec<TinyNNFP16>) {
    let path = output_dir.join(format!("{}_result.hex", name));
    let mut file = BufWriter::new(
        File::create(&path).expect(&format!("Failed to create file: {:?}", path)),
    );

    for v in values {
        writeln!(file, "{:04x}", v.as_u16()).expect("Failed to write to file");
    }

    file.flush().expect("Failed to flush file");
    println!("Wrote: {:?}", path);
}
