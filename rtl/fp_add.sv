module fp_add import tiny_nn_pkg::*; (
  input  fp_t op_a_i,
  input  fp_t op_b_i,
  output fp_t result_o
);
  fp_t op_x, op_y;

  // Bit width for working with full mantissas, multiple extra bits are required
  // - The top, implicit 1, bit
  // - Bit for sum overflow
  // - Sign bit
  // - Bit for extra precision when the exponent decreases
  localparam FPMantWidthExt = 1 + 1 + 1 + FPMantWidth + 1;

  logic [FPMantWidthExt-1:0]      op_x_full_mant, op_y_full_mant;
  logic [FPMantWidthExt-1:0]      op_y_full_mant_signed;
  logic [FPMantWidthExt-1:0]      op_x_full_mant_signed;
  logic signed [FPMantWidthExt-1:0]  foo;
  logic [FPMantWidthExt-1:0]      op_y_full_mant_signed_shifted;
  logic [FPMantWidth-1:0]         op_y_dropped_bits;
  logic [FPExpWidth-1:0]          full_mant_shift;
  logic [$clog2(FPMantWidth)-1:0] mant_shift;

  logic [FPMantWidthExt-1:0] mant_add_signed;
  logic [FPMantWidthExt-1:0] mant_add;
  logic                      mant_add_sign;
  logic [FPMantWidth:0]      mant_add_norm_all [FPMantWidth+3];
  logic [FPMantWidth:0]      mant_add_norm;
  logic [FPMantWidth-1:0]    mant_final;

  logic [$clog2(FPMantWidth)-1:0] exp_change;

  logic [FPMantWidth-1:0] result_mant;
  logic [FPExpWidth-1:0]  result_exp;

  always_comb begin
    if (op_a_i.exp >= op_b_i.exp) begin
      op_x = op_a_i;
      op_y = op_b_i;
    end else begin
      op_x = op_b_i;
      op_y = op_a_i;
    end
  end

  assign op_x_full_mant = {3'b001, op_x.mant, 1'b0};
  assign op_y_full_mant = {3'b001, op_y.mant, 1'b0};

  assign full_mant_shift = op_x.exp - op_y.exp;
  assign mant_shift = full_mant_shift[$clog2(FPMantWidth)-1:0];

  assign op_x_full_mant_signed = op_x.sgn ? ~(op_x_full_mant - 1'b1) : op_x_full_mant;
  assign op_y_full_mant_signed = op_y.sgn ? ~(op_y_full_mant - 1'b1) : op_y_full_mant;

  always_comb begin
    op_y_dropped_bits = '0;
    for (int i = 0;i < FPMantWidth; ++i) begin
      if (mant_shift >= ($clog2(FPMantWidth))'(i + 1)) begin
        op_y_dropped_bits[i] = op_y_full_mant_signed[i+1];
      end
    end
  end

  assign foo = $signed(op_y_full_mant_signed) >>> mant_shift;

  assign op_y_full_mant_signed_shifted =
    full_mant_shift <= FPExpWidth'(FPMantWidth) ?
      $unsigned(foo)                                          :
      {FPMantWidthExt{1'b0}};

  assign mant_add_signed = op_x_full_mant_signed + $unsigned(op_y_full_mant_signed_shifted);
  assign mant_add_sign = mant_add_signed[FPMantWidthExt-1];

  always_comb begin
    if (mant_add_sign) begin
      if (|op_y_dropped_bits && full_mant_shift <= FPExpWidth'(FPMantWidth)) begin
        mant_add = ~mant_add_signed;
      end else begin
        mant_add = ~mant_add_signed + 1'b1;
      end
    end else begin
      mant_add = mant_add_signed;
    end
  end

  for (genvar i_norm = 0; i_norm < FPMantWidth + 3;i_norm++) begin : g_norm
    if (i_norm == FPMantWidth + 2) begin
      assign mant_add_norm_all[i_norm] = mant_add[i_norm:2];
    end else if (i_norm == FPMantWidth + 1) begin
      assign mant_add_norm_all[i_norm] = mant_add[i_norm:1];
    end else begin
      localparam int unsigned BottomFillWidth = FPMantWidth-i_norm;

      assign mant_add_norm_all[i_norm] = {mant_add[i_norm:0], {BottomFillWidth{1'b0}}};
    end
  end

  always_comb begin
    exp_change    = '0;
    mant_add_norm = '0;

    for(int i = 0;i < FPMantWidth + 3;i++) begin

      if (mant_add[i]) begin
        mant_add_norm = mant_add_norm_all[i];

        if (i <= FPMantWidth) begin
          exp_change = $clog2(FPMantWidth)'(FPMantWidth - i);
        end
      end
    end
  end

  assign result_exp = mant_add[FPMantWidth + 2] ? op_x.exp + 1'b1 :
                      mant_add[FPMantWidth + 1] ? op_x.exp :
                                                  op_x.exp - 1'b1 - FPExpWidth'(exp_change);

  assign result_mant = mant_add_norm[FPMantWidth-1:0];

  assign result_o = mant_add_signed == '0 ? FPZero :
                                            '{
                                               sgn:  mant_add_sign,
                                               exp:  result_exp,
                                               mant: result_mant
                                             };
endmodule
