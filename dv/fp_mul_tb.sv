module fp_mul_tb;
  import tiny_nn_pkg::*;

  fp_t op_a, op_b, result;

  fp_op_tester u_tester (
    .op_a_o  (op_a),
    .op_b_o  (op_b),
    .result_i(result)
  );

  fp_mul u_dut (
    .op_a_i  (op_a),
    .op_b_i  (op_b),
    .result_o(result)
  );

  initial begin
    int num_tests = 0;
    int failures = 0;

    $dumpfile("sim.fst");
    $dumpvars;

    u_tester.read_tests_from_file("./fp_mul_tests.hex");

    u_tester.run_tests(num_tests, failures);

    if (failures == 0) begin
      $display("PASS! All tests succeeded %d run", num_tests);
    end else begin
      $display("FAIL! Saw %d failures out of %d tests", failures, num_tests);
    end

    $finish();
  end
endmodule
