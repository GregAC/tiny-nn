module fp_mul import tiny_nn_pkg::*; (
  input  fp_t op_a_i,
  input  fp_t op_b_i,
  output fp_t result_o
);
  logic [FPMantWidth:0] op_a_full_mant, op_b_full_mant;

  logic [FPMantWidth*2+1:0] mant_mul;
  logic [FPExpWidth:0]      exp_add;

  logic [FPMantWidth-1:0] result_mant;
  logic [FPExpWidth-1:0]  result_exp;
  logic                   result_sgn;

  logic shift_output_mant;

  assign op_a_full_mant = {1'b1, op_a_i.mant};
  assign op_b_full_mant = {1'b1, op_b_i.mant};

  assign mant_mul = op_a_full_mant * op_b_full_mant;
  assign exp_add = op_a_i.exp + op_b_i.exp - {2'b01, {(FPExpWidth - 1){1'b0}}};

  assign shift_output_mant = mant_mul[FPMantWidth*2+1];

  assign result_mant = shift_output_mant ? mant_mul[FPMantWidth*2:FPMantWidth+1] :
                                           mant_mul[FPMantWidth*2-1:FPMantWidth];

  assign result_exp = shift_output_mant ? FPExpWidth'(exp_add + 1'b1) : exp_add[FPExpWidth-1:0];

  assign result_sgn = op_a_i.sgn ^ op_b_i.sgn;

  assign result_o = '{
    sgn: result_sgn,
    exp: result_exp,
    mant: result_mant
  };
endmodule
