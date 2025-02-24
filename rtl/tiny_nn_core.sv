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

  input logic [1:0] accumulate_en_i,
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
  fp_t mul_result_d [ValArrayWidth];
  fp_t mul_result_q [ValArrayWidth];

  for (genvar x = 0;x < ValArrayWidth; ++x) begin
    assign mul_op_a[x] = mul_row_sel_i ? mul_val_op_q[x][0]   : mul_val_op_q[x][1];
    assign mul_op_b[x] = mul_row_sel_i ? param_val_op_q[x][0] : param_val_op_q[x][1];

    fp_mul u_mul (
      .op_a_i(mul_op_a[x]),
      .op_b_i(mul_op_b[x]),
      .result_o(mul_result_d[x])
    );

    always_ff @(posedge clk_i) begin
      if (mul_en_i) begin
        mul_result_q[x] <= mul_result_d[x];
      end
    end
  end

  fp_t accumulate_level_0_q [ValArrayWidth/2];
  fp_t accumulate_level_0_d [ValArrayWidth/2];

  for (genvar x = 0;x < ValArrayWidth/2; ++x) begin : g_accumulate_level_0_inner
    fp_add u_add (
      .op_a_i(mul_result_q[x * 2]),
      .op_b_i(mul_result_q[x * 2 + 1]),
      .result_o(accumulate_level_0_d[x])
    );

    always_ff @(posedge clk_i) begin
      if (accumulate_en_i[0]) begin
        accumulate_level_0_q[x] <= accumulate_level_0_d[x];
      end
    end
  end

  fp_t accumulate_level_1_q[ValArrayHeight];
  fp_t accumulate_level_1_d;

  fp_add u_add (
    .op_a_i(accumulate_level_0_q[0]),
    .op_b_i(accumulate_level_0_q[1]),
    .result_o(accumulate_level_1_d)
  );

  for (genvar y = 0;y < ValArrayHeight; ++y) begin
    always_ff @(posedge clk_i) begin
      if (accumulate_en_i[0] && (y == mul_row_sel_i)) begin
        accumulate_level_1_q[y] <= accumulate_level_1_d;
      end
    end
  end

  fp_t accumulate_final_q, accumulate_final_d;

  fp_add u_add_accumulate_final (
    .op_a_i(accumulate_level_1_q[0]),
    .op_b_i(accumulate_level_1_q[1]),
    .result_o(accumulate_final_d)
  );

  always_ff @(posedge clk_i) begin
    if (accumulate_en_i[1]) begin
      accumulate_final_q <= accumulate_final_d;
    end
  end

  assign accumulate_o = accumulate_final_q;
endmodule
