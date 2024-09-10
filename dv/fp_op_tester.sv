module fp_op_tester import tiny_nn_pkg::*; (
  output fp_t op_a_o,
  output fp_t op_b_o,
  input  fp_t result_i
);
  typedef struct packed {
    fp_t op_a;
    fp_t op_b;
    fp_t expected_result;
  } fp_op_test_t;

  fp_op_test_t tests[$];

  function automatic void add_test(fp_t op_a, fp_t op_b, fp_t expected_result);
    fp_op_test_t new_test = '{op_a: op_a, op_b: op_b, expected_result: expected_result};
    tests.push_back(new_test);
  endfunction

  task automatic run_tests(output int num_tests, output int failures);
    foreach (tests[i]) begin
      op_a_o = tests[i].op_a;
      op_b_o = tests[i].op_b;

      #2ns;

      if (tests[i].expected_result != result_i) begin
        $display("Mismatch at %t", $time());
        $display("op_a.sgn: %b, op_a.exp: %d, op_a.mant: %b", op_a_o.sgn, op_a_o.exp, op_a_o.mant);
        $display("op_b.sgn: %b, op_b.exp: %d, op_b.mant: %b", op_b_o.sgn, op_b_o.exp, op_b_o.mant);

        $display("expected.sgn: %b, expected.exp: %d, expected.mant: %b",
          tests[i].expected_result.sgn, tests[i].expected_result.exp,
          tests[i].expected_result.mant);
        $display("result.sgn: %b, result.exp: %d, result.mant: %b", result_i.sgn, result_i.exp,
          result_i.mant);

        failures += 1;
      end

      num_tests += 1;
    end
  endtask

  task automatic read_tests_from_file(string filename);
    integer test_file;

    test_file = $fopen(filename, "r");

    if (test_file == 0) begin
      $display("Error opening %s", filename);
      return;
    end

    while (!$feof(test_file)) begin
      bit [15:0] a, b, expected;
      fp_op_test_t new_test;

      $fscanf(test_file, "%x\n", a);
      $fscanf(test_file, "%x\n", b);
      $fscanf(test_file, "%x\n", expected);

      new_test = '{op_a: a, op_b: b, expected_result: expected};

      tests.push_back(new_test);
    end
  endtask
endmodule
