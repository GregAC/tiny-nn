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

  input logic accumulate_en_i,
  output fp_t accumulate_o
);

  fp_t mul_val_op_q [ValArrayWidth][ValArrayHeight];
  fp_t param_val_op_q [ValArrayWidth][ValArrayHeight];

  for (genvar y = 0;y < ValArrayHeight; ++y) begin
    for (genvar x = 0;x < ValArrayWidth; ++x) begin
      always_ff @(posedge clk_i) begin
        if (param_write_i[x + y * ValArrayWidth]) begin
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
  fp_t mul_result_held [ValArrayWidth][ValArrayHeight];

  for (genvar x = 0;x < ValArrayWidth; ++x) begin
    assign mul_op_a[x] = mul_row_sel_i ? mul_val_op_q[x][0]   : mul_val_op_q[x][1];
    assign mul_op_b[x] = mul_row_sel_i ? param_val_op_q[x][0] : param_val_op_q[x][1];

    fp_mul u_mul (
      .op_a_i(mul_op_a[x]),
      .op_b_i(mul_op_b[x]),
      .result_o(mul_result[x])
    );

    for (genvar y = 0;y < ValArrayHeight; ++y) begin
      always_ff @(posedge clk_i) begin
        if (mul_en_i && (mul_row_sel_i == y)) begin
          mul_result_held[x][y] <= mul_result[x];
        end
      end
    end
  end

  fp_t accumulate_level_0_q [ValArrayWidth/2][ValArrayHeight];
  fp_t accumulate_level_0_d [ValArrayWidth/2][ValArrayHeight];

  for (genvar y = 0;y < ValArrayHeight; ++y) begin : g_accumulate_level_0_outer
    for (genvar x = 0;x < ValArrayWidth/2; ++x) begin : g_accumulate_level_0_inner
      fp_add u_add (
        .op_a_i(mul_result_held[x * 2][y]),
        .op_b_i(mul_result_held[x * 2 + 1][y]),
        .result_o(accumulate_level_0_d[x][y])
      );

      always_ff @(posedge clk_i) begin
        if (accumulate_en_i) begin
          accumulate_level_0_q[x][y] <= accumulate_level_0_d[x][y];
        end
      end
    end
  end

  fp_t accumulate_level_1_q [ValArrayHeight];
  fp_t accumulate_level_1_d [ValArrayHeight];

  for (genvar y = 0;y < ValArrayHeight; ++y) begin : g_accumulate_level_1
    fp_add u_add (
      .op_a_i(accumulate_level_0_q[0][y]),
      .op_b_i(accumulate_level_0_q[1][y]),
      .result_o(accumulate_level_1_d[y])
    );

    always_ff @(posedge clk_i) begin
      if (accumulate_en_i) begin
        accumulate_level_1_q[y] <= accumulate_level_1_d[y];
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
    if (accumulate_en_i) begin
      accumulate_final_q <= accumulate_final_d;
    end
  end

  assign accumulate_o = accumulate_final_q;
endmodule

