module fp_cmp import tiny_nn_pkg::*; (
  input  fp_t op_a_i,
  input  fp_t op_b_i,

  output logic op_a_greater_o,
  output logic invalid_nan_o
);

  always_comb begin
    op_a_greater_o = 1'b0;
    invalid_nan_o = 1'b0;

    if (is_nan(op_a_i) || is_nan(op_b_i)) begin
      invalid_nan_o = 1'b1;
    end else if (op_a_i == op_b_i) begin
      op_a_greater_o = 1'b0;
    end else if (is_inf(op_a_i)) begin
      op_a_greater_o = op_a_i.sgn ? 1'b0 : 1'b1;
    end else if (is_inf(op_b_i)) begin
      op_a_greater_o = op_b_i.sgn ? 1'b1 : 1'b0;
    end else if (op_a_i == FPZero) begin
      op_a_greater_o = op_b_i.sgn ? 1'b1 : 1'b0;
    end else if (op_b_i == FPZero) begin
      op_a_greater_o = op_a_i.sgn ? 1'b0 : 1'b1;
    end else if (op_a_i.sgn != op_b_i.sgn) begin
      op_a_greater_o = op_a_i.sgn ? 1'b0 : 1'b1;
    end else if(op_a_i.exp > op_b_i.exp) begin
      op_a_greater_o = op_a_i.sgn ? 1'b0 : 1'b1;
    end else if(op_a_i.exp < op_b_i.exp) begin
      op_a_greater_o = op_a_i.sgn ? 1'b1 : 1'b0;
    end else if(op_a_i.mant > op_b_i.mant) begin
      op_a_greater_o = op_a_i.sgn ? 1'b0 : 1'b1;
    end
  end
endmodule
