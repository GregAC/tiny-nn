module tiny_nn_core import tiny_nn_pkg::*; #(
  parameter int unsigned ValArrayWidth = 4,
  parameter int unsigned ValArrayHeight = 2
) (
  input clk_i,
  input rst_ni,

  input fp_t                       val_i,
  input logic [ValArrayHeight-1:0] val_shift_i,

  input fp_t                                         param_i,
  input logic [ValArrayHeight * ValArrayWidth - 1:0] param_write_i,

  input logic mul_row_sel_i,
  input logic mul_en_i,

  input logic accumulate_loopback_i,
  input logic accumulate_out_relu_i,

  input fp_t  mul_add_op_a_din_i,
  input fp_t  mul_add_op_b_din_i,
  input logic mul_add_op_a_en_i,
  input logic mul_add_op_b_en_i,

  input fp_t  accumulate_level_0_din_i,
  input       accumulate_level_0_en_i,

  input logic [1:0] accumulate_mode_0_en_i,
  input logic [1:0] accumulate_mode_1_en_i,
  input logic [1:0] accumulate_mode_2_en_i,
  output fp_t accumulate_o
);

  fp_t mul_val_op_q [ValArrayWidth][ValArrayHeight];
  fp_t param_val_op_q [ValArrayWidth][ValArrayHeight];

  for (genvar y = 0;y < ValArrayHeight; ++y) begin
    for (genvar x = 0;x < ValArrayWidth; ++x) begin
      always_ff @(posedge clk_i) begin
        if (param_write_i[x * ValArrayHeight +  y]) begin
          param_val_op_q[x][y] <= param_i;
        end
      end

      if (x == (ValArrayWidth - 1)) begin
        always_ff @(posedge clk_i) begin
          if (val_shift_i[y]) begin
            mul_val_op_q[x][y] <= val_i;
          end
        end
      end else begin
        always_ff @(posedge clk_i) begin
          if (val_shift_i[y]) begin
            mul_val_op_q[x][y] <= mul_val_op_q[x + 1][y];
          end
        end
      end
    end
  end

  fp_t mul_op_a   [ValArrayWidth];
  fp_t mul_op_b   [ValArrayWidth];
  fp_t mul_result [ValArrayWidth];

  for (genvar x = 0;x < ValArrayWidth; ++x) begin
    assign mul_op_a[x] = mul_row_sel_i ? mul_val_op_q[x][0]   : mul_val_op_q[x][1];
    assign mul_op_b[x] = mul_row_sel_i ? param_val_op_q[x][0] : param_val_op_q[x][1];

    fp_mul u_mul (
      .op_a_i(mul_op_a[x]),
      .op_b_i(mul_op_b[x]),
      .result_o(mul_result[x])
    );
  end

  logic [ValArrayWidth/2-1:0] mul_add_op_a_en, mul_add_op_b_en;
  fp_t mul_add_op_a_q [ValArrayWidth/2];
  fp_t mul_add_op_a_d [ValArrayWidth/2];

  fp_t mul_add_op_b_q [ValArrayWidth/2];
  fp_t mul_add_op_b_d [ValArrayWidth/2];

  fp_t mul_add_op_a   [ValArrayWidth/2];

  fp_t mul_add_result [ValArrayWidth/2];

  logic [ValArrayWidth/2-1:0] accumulate_level_0_en;
  fp_t                        accumulate_level_0_q [ValArrayWidth/2];
  fp_t                        accumulate_level_0_d [ValArrayWidth/2];

  for (genvar x = 0;x < ValArrayWidth/2; ++x) begin : g_accumulate_level_0_inner
    if (x == 1) begin
      assign mul_add_op_a_d[x] = mul_add_op_a_en_i ? mul_add_op_a_din_i : mul_result[x * 2];
      assign mul_add_op_b_d[x] = mul_add_op_b_en_i ? mul_add_op_b_din_i : mul_result[x * 2 + 1];

      assign mul_add_op_a_en[x] = mul_en_i | mul_add_op_a_en_i;
      assign mul_add_op_b_en[x] = mul_en_i | mul_add_op_b_en_i;

      assign mul_add_op_a[x] = accumulate_loopback_i ? accumulate_level_0_q[x] : mul_add_op_a_q[x];
    end else begin
      assign mul_add_op_a_d[x] = mul_result[x * 2];
      assign mul_add_op_b_d[x] = mul_result[x * 2 + 1];

      assign mul_add_op_a_en[x] = mul_en_i;
      assign mul_add_op_b_en[x] = mul_en_i;

      assign mul_add_op_a[x] = mul_add_op_a_q[x];
    end

    always_ff @(posedge clk_i) begin
      if (mul_add_op_a_en[x]) begin
        mul_add_op_a_q[x] <= mul_add_op_a_d[x];
      end
    end

    always_ff @(posedge clk_i) begin
      if (mul_add_op_b_en[x]) begin
        mul_add_op_b_q[x] <= mul_add_op_b_d[x];
      end
    end

    fp_add u_add (
      .op_a_i(mul_add_op_a[x]),
      .op_b_i(mul_add_op_b_q[x]),
      .result_o(mul_add_result[x])
    );

    if (x == 0) begin
      assign accumulate_level_0_en[x] = accumulate_level_0_en_i | accumulate_mode_0_en_i[0];

      assign accumulate_level_0_d[x] = accumulate_level_0_en_i ? accumulate_level_0_din_i :
                                                                 mul_add_result[x];
    end else begin
      assign accumulate_level_0_en[x] = accumulate_mode_2_en_i[0] | accumulate_mode_1_en_i[0] | accumulate_mode_0_en_i[0];
      assign accumulate_level_0_d[x]  = mul_add_result[x];
    end

    always_ff @(posedge clk_i) begin
      if (accumulate_level_0_en[x]) begin
        accumulate_level_0_q[x] <= accumulate_level_0_d[x];
      end
    end
  end

  fp_t                       accumulate_level_0_result;
  fp_t                       accumulate_level_1_q[ValArrayHeight];
  logic [ValArrayHeight-1:0] accumulate_level_1_en;

  fp_add u_add (
    .op_a_i(accumulate_level_0_q[0]),
    .op_b_i(accumulate_level_0_q[1]),
    .result_o(accumulate_level_0_result)
  );

  for (genvar y = 0;y < ValArrayHeight; ++y) begin
    assign accumulate_level_1_en[y] = (accumulate_mode_0_en_i[0] & (mul_row_sel_i == y));

    always_ff @(posedge clk_i) begin
      if (accumulate_level_1_en[y]) begin
        accumulate_level_1_q[y] <= accumulate_level_0_result;
      end
    end
  end

  fp_t  accumulate_final_q, accumulate_final_d, accumulate_final_result;
  logic accumulate_final_en;

  fp_add u_add_accumulate_final (
    .op_a_i(accumulate_level_1_q[0]),
    .op_b_i(accumulate_level_1_q[1]),
    .result_o(accumulate_final_result)
  );

  always_comb begin
    if (accumulate_mode_1_en_i[1]) begin
      if (accumulate_out_relu_i && accumulate_level_0_result.sgn) begin
        accumulate_final_d = FPZero;
      end else begin
        accumulate_final_d = accumulate_level_0_result;
      end
    end else if (accumulate_mode_2_en_i[1]) begin
      accumulate_final_d = mul_add_result[1];
    end else begin
      accumulate_final_d = accumulate_final_result;
    end
  end

  assign accumulate_final_en = accumulate_mode_0_en_i[1] | accumulate_mode_1_en_i[1] | accumulate_mode_2_en_i[1];

  always_ff @(posedge clk_i) begin
    if (accumulate_final_en) begin
      accumulate_final_q <= accumulate_final_d;
    end
  end

  assign accumulate_o = accumulate_final_q;
endmodule
