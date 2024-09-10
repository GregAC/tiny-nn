module fp_mul #(
  parameter int unsigned ExpWidth = 8,
  parameter int unsigned MantWidth = 7
)  (
  input  logic [ExpWidth + MantWidth:0] op_a_i,
  input  logic [ExpWidth + MantWidth:0] op_b_i,
  output logic [ExpWidth + MantWidth:0] result_o
);
  logic [MantWidth:0]  op_a_mant, op_b_mant;
  logic [ExpWidth-1:0] op_a_exp, op_b_exp;

  logic [MantWidth*2+1:0] mant_mul;
  logic [ExpWidth:0]      exp_add;

  logic [MantWidth-1:0] result_mant;
  logic [ExpWidth-1:0] result_exp;

  logic shift_output_mant;

  assign op_a_mant = {1'b1, op_a_i[MantWidth-1:0]};
  assign op_b_mant = {1'b1, op_b_i[MantWidth-1:0]};

  assign op_a_exp = op_a_i[MantWidth + ExpWidth - 1:MantWidth];
  assign op_b_exp = op_b_i[MantWidth + ExpWidth - 1:MantWidth];

  assign mant_mul = op_a_mant * op_b_mant;
  assign exp_add = op_a_exp + op_b_exp - {2'b0, {(ExpWidth - 1){1'b1}}};

  assign shift_output_mant = mant_mul[MantWidth*2+1];

  assign result_mant = shift_output_mant ? mant_mul[MantWidth*2:MantWidth+1] :
                                           mant_mul[MantWidth*2-1:MantWidth];

  assign result_exp = shift_output_mant ? ExpWidth'(exp_add + 1'b1) : exp_add[ExpWidth-1:0];

  assign result_o = {1'b0, result_exp, result_mant};
endmodule
