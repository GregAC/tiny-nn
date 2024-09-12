module fp_add_tb;
  import tiny_nn_pkg::*;

  fp_t op_a, op_b, result;

  fp_op_tester u_tester (
    .op_a_o  (op_a),
    .op_b_o  (op_b),
    .result_i(result)
  );

  fp_add u_dut (
    .op_a_i  (op_a),
    .op_b_i  (op_b),
    .result_o(result)
  );

  initial begin
    int num_tests = 0;
    int failures = 0;

    $dumpfile("sim.fst");
    $dumpvars;

    u_tester.add_test('{sgn: 1'b1, exp: 8'd103, mant: 7'b1001011},
      '{sgn: 1'b1, exp: 8'd105, mant: 7'b0111101},
      '{sgn: 1'b1, exp: 8'd105, mant: 7'b1101111});

    u_tester.add_test('{sgn: 1'b0, exp: 8'd127, mant: 7'd0},
      '{sgn: 1'b0, exp: 8'd127, mant: 7'd0},
      '{sgn: 1'b0, exp: 8'd128, mant: 7'd0});

    u_tester.add_test('{sgn: 1'b0, exp: 8'd127, mant: 7'b100_0000},
      '{sgn: 1'b0, exp: 8'd125, mant: 7'b000_0000},
      '{sgn: 1'b0, exp: 8'd127, mant: 7'b110_0000});

    u_tester.add_test('{sgn: 1'b0, exp: 8'd127, mant: 7'd0},
      '{sgn: 1'b0, exp: 8'd126, mant: 7'd0},
      '{sgn: 1'b0, exp: 8'd127, mant: 7'b100_0000});

    u_tester.add_test('{sgn: 1'b0, exp: 8'd127, mant: 7'd0},
      '{sgn: 1'b1, exp: 8'd126, mant: 7'd0},
      '{sgn: 1'b0, exp: 8'd126, mant: 7'd0});

    u_tester.add_test('{sgn: 1'b0, exp: 8'd126, mant: 7'd0},
      '{sgn: 1'b1, exp: 8'd127, mant: 7'd0},
      '{sgn: 1'b1, exp: 8'd126, mant: 7'd0});

    u_tester.read_tests_from_file("./fp_add_tests.hex");

    //u_tester.add_test('{sgn: 1'b1, exp: 8'd143, mant: 7'b010_1010},
    //  '{sgn: 1'b1, exp: 8'd142, mant: 7'b000_1111},
    //  '{sgn: 1'b1, exp: 8'd143, mant: 7'b111_0001});

    u_tester.run_tests(num_tests, failures);

    if (failures == 0) begin
      $display("PASS! All tests succeeded %d run", num_tests);
    end else begin
      $display("FAIL! Saw %d failures out of %d tests", failures, num_tests);
    end

    $finish();
  end
endmodule
