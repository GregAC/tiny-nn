module tiny_nn_top import tiny_nn_pkg::*; #(
  // Design assumes val array is 4x2! Changing this will break the design.
  parameter int unsigned ValArrayWidth  = 4,
  parameter int unsigned ValArrayHeight = 2
) (
  input clk_i,
  input rst_ni,

  input  logic [15:0] data_i,
  output logic [7:0]  data_o
);
  localparam int unsigned ValArraySize = ValArrayWidth * ValArrayHeight;
  localparam int unsigned CountWidth   = 8;

  logic                    phase_q, phase_d;
  logic [CountWidth-1:0]   counter_q, counter_d;
  logic [CountWidth-1:0]   start_count_q, start_count_d;
  logic [ValArraySize-1:0] param_write_q, param_write_d;
  logic                    relu_q, relu_d;
  logic                    convolve_run;

  logic [ValArrayHeight-1:0] core_val_shift;
  logic                      core_mul_row_sel, core_mul_en;
  fp_t                       core_accumulate_result;
  logic                      core_mul_add_op_a_en;
  logic                      core_mul_add_op_b_en;
  logic [1:0]                core_accumulate_mode_0_en;
  logic [1:0]                core_accumulate_mode_1_en;
  logic                      core_accumulate_loopback;
  logic                      core_accumulate_out_relu;
  logic                      core_accumulate_level_0_en;

  typedef enum logic [3:0] {
    NNIdle              = 4'h0,
    NNConvolveParamIn   = 4'h1,
    NNConvolveExec      = 4'h2,
    NNConvolveExecEnd   = 4'h3,
    NNAccumulateBiasIn  = 4'h4,
    NNAccumulateExec    = 4'h5,
    NNAccumulateExecEnd = 4'h6,
    NNMulAccBiasIn      = 4'h7,
    NNMulAccExec        = 4'h8,
    NNMulAccExecEnd     = 4'h9,
    NNTestCount         = 4'hd,
    NNTestPulse         = 4'he,
    NNTestASCII         = 4'hf
  } state_e;

  state_e state_q, state_d;

  always_comb begin
    state_d       = state_q;
    counter_d     = counter_q;
    start_count_d = start_count_q;
    phase_d       = 1'b0;
    param_write_d = param_write_q;
    convolve_run  = 1'b0;
    relu_d        = relu_q;

    core_val_shift             = '0;
    core_mul_row_sel           = 1'b0;
    core_mul_en                = 1'b0;
    core_mul_add_op_a_en       = 1'b0;
    core_mul_add_op_b_en       = 1'b0;
    core_accumulate_mode_0_en  = '0;
    core_accumulate_mode_1_en  = '0;
    core_accumulate_loopback   = 1'b0;
    core_accumulate_out_relu   = 1'b0;
    core_accumulate_level_0_en = 1'b0;

    case (state_q)
      NNIdle: begin
        case (data_i[15:12])
          CmdOpConvolve: begin
            state_d   = NNConvolveParamIn;

            param_write_d    = '0;
            param_write_d[0] = 1'b1;
          end
          CmdOpAccumulate: begin
            start_count_d = data_i[CountWidth-1:0];
            counter_d     = CountWidth'(1'b1);
            state_d       = NNAccumulateBiasIn;
            relu_d        = data_i[8];
          end
          CmdOpTest: begin
            case (data_i[11:8])
              4'hf: begin
                state_d   = NNTestASCII;
                counter_d = 8'd3;
              end
              4'h1: begin
                state_d   = NNTestCount;
                counter_d = data_i[CountWidth-1:0];
              end
              4'h0: begin
                state_d   = NNTestPulse;
                counter_d = 8'd1;
              end
              default: ;
            endcase
          end
          CmdOpMulAcc: begin
            state_d       = NNMulAccBiasIn;
            relu_d        = data_i[8];
            phase_d       = 1'b0;
            counter_d     = CountWidth'(2'd3);
          end
          default: ;
        endcase
      end
      NNConvolveParamIn: begin
        if (param_write_q[ValArraySize-1]) begin
          // Last parameter input this cycle move to performing convolve
          param_write_d = '0;
          state_d = NNConvolveExec;
        end else begin
          // Move to next parameter for write
          param_write_d = {param_write_q[ValArraySize-2:0], 1'b0};
        end
      end
      NNConvolveExec, NNConvolveExecEnd: begin
        phase_d      = ~phase_q;
        convolve_run = 1'b1;

        core_val_shift[0] = ~phase_q;
        core_val_shift[1] =  phase_q;
        core_mul_row_sel  =  phase_q;
        core_mul_en       = 1'b1;

        core_accumulate_mode_0_en[0] = 1'b1;
        core_accumulate_mode_0_en[1] = phase_q;

        if (state_q == NNConvolveExec) begin
          if (data_i == FPStdNaN) begin
            state_d = NNConvolveExecEnd;
            counter_d = 8'd4;
          end
        end else begin
          if (counter_q != '0) begin
            counter_d = counter_q - 1'b1;
          end else begin
            state_d = NNIdle;
          end
        end
      end
      NNAccumulateBiasIn: begin
        core_accumulate_level_0_en = 1'b1;
        state_d                    = NNAccumulateExec;
      end
      NNAccumulateExec: begin
        core_accumulate_mode_1_en[0] = 1'b1;
        core_accumulate_out_relu     = relu_q;
        core_mul_add_op_b_en         = 1'b1;

        if (counter_q == '0) begin
          core_accumulate_mode_1_en[1] = 1'b1;
          core_accumulate_loopback     = 1'b0;
          counter_d                    = start_count_q;
        end else begin
          if (counter_q == CountWidth'(1'b1)) begin
            core_mul_add_op_a_en = 1'b1;
          end

          core_accumulate_loopback = 1'b1;
          counter_d                = counter_q - 1'b1;

          if (data_i == FPStdNaN) begin
            state_d   = NNAccumulateExecEnd;
            counter_d = 8'd2;
          end
        end
      end
      NNAccumulateExecEnd: begin
        core_accumulate_out_relu = relu_q;

        if (counter_q == 8'd2) begin
          core_accumulate_mode_1_en[1] = 1'b1;
        end else if (counter_q == '0) begin
          state_d = NNIdle;
        end

        counter_d = counter_q - 1'b1;
      end
      NNMulAccBiasIn: begin
        core_accumulate_level_0_en = 1'b1;
        core_mul_add_op_a_en       = 1'b1;
        state_d                    = NNMulAccExec;
      end
      NNMulAccExec: begin
        phase_d          = ~phase_q;
        core_mul_row_sel = 1'b1;

        // Don't loopback for first number!
        if (phase_q) begin
          core_val_shift[0]            = 1'b0;
          param_write_d[6]             = 1'b0;
          core_accumulate_mode_1_en[0] = 1'b1;
          core_accumulate_loopback     = counter_q == '0;
        end else begin
          core_val_shift[0] = 1'b1;
          param_write_d[6]  = 1'b1;

          core_mul_en = counter_q != CountWidth'(2'd3);

          if (counter_q != '0) begin
            counter_d = counter_q - 1'b1;
          end

          if (data_i == FPStdNaN) begin
            state_d   = NNMulAccExecEnd;
            counter_d = 8'd3;
          end
        end
      end
      NNMulAccExecEnd: begin
        if (counter_q == 8'd3) begin
          core_accumulate_loopback     = 1'b1;
          core_accumulate_mode_1_en[0] = 1'b1;
        end else if (counter_q == 8'd2) begin
          core_accumulate_mode_1_en[1] = 1'b1;
          core_accumulate_out_relu = relu_q;
        end else if (counter_q == '0) begin
          state_d = NNIdle;
        end

        counter_d = counter_q - 1'b1;
      end
      NNTestASCII: begin
        if (data_i[15:8] == 8'hff) begin
          if (counter_q == 0) begin
            counter_d = 8'd3;
          end else begin
            counter_d = counter_q - 1'b1;
          end
        end else begin
          state_d = NNIdle;
        end
      end
      NNTestPulse: begin
        if (data_i[15:8] == 8'hf0) begin
          counter_d = counter_q - 1'b1;
        end else begin
          state_d = NNIdle;
        end
      end
      NNTestCount: begin
        if (counter_q == '0) begin
          state_d = NNIdle;
        end else begin
          counter_d = counter_q - 1'b1;
        end
      end
      default: ;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= NNIdle;
    end else begin
      state_q <= state_d;
    end
  end

  always_ff @(posedge clk_i) begin
    counter_q     <= counter_d;
    start_count_q <= start_count_d;
    phase_q       <= phase_d;
    param_write_q <= param_write_d;
    relu_q        <= relu_d;
  end

  tiny_nn_core #(
    .ValArrayWidth(ValArrayWidth),
    .ValArrayHeight(ValArrayHeight)
  ) u_core (
    .clk_i,
    .rst_ni,

    .val_i      (fp_t'(data_i)),
    .val_shift_i(core_val_shift),

    .param_i      (fp_t'(data_i)),
    .param_write_i(param_write_q),

    .mul_row_sel_i(core_mul_row_sel),
    .mul_en_i     (core_mul_en),

    .accumulate_loopback_i(core_accumulate_loopback),
    .accumulate_out_relu_i(core_accumulate_out_relu),

    .mul_add_op_a_din_i('0),
    .mul_add_op_b_din_i(fp_t'(data_i)),
    .mul_add_op_a_en_i (core_mul_add_op_a_en),
    .mul_add_op_b_en_i (core_mul_add_op_b_en),

    .accumulate_level_0_din_i(fp_t'(data_i)),
    .accumulate_level_0_en_i (core_accumulate_level_0_en),

    .accumulate_mode_0_en_i(core_accumulate_mode_0_en),
    .accumulate_mode_1_en_i(core_accumulate_mode_1_en),

    .accumulate_o   (core_accumulate_result)
  );

  logic [7:0] test_out;

  always_comb begin
    test_out = '0;

    case (state_q)
      NNTestASCII: begin
        case (counter_q)
          8'd3:       test_out = 8'h54;
          8'd2:       test_out = 8'h2d;
          8'd0, 8'd1: test_out = 8'h4e;
          default:    test_out = '0;
        endcase
      end
      NNTestPulse: begin
        test_out = counter_q[0] ? 8'b1010_1010 :
                                  8'b0101_0101;
      end
      NNTestCount: begin
        test_out = counter_q;
      end
      default: ;
    endcase
  end

  always_comb begin
    data_o = '1;

    case (state_q)
      NNConvolveExec, NNConvolveExecEnd: begin
        data_o = phase_q ? core_accumulate_result[15:8] :
                           core_accumulate_result[7:0];
      end
      NNAccumulateExec: begin
        if (counter_q == start_count_q) begin
          data_o = core_accumulate_result[7:0];
        end else begin
          data_o = core_accumulate_result[15:8];
        end
      end
      NNAccumulateExecEnd: begin
        data_o = counter_q[0] ? core_accumulate_result[7:0] : core_accumulate_result[15:8];
      end
      NNMulAccExecEnd: begin
        data_o = counter_q[0] ? core_accumulate_result[7:0] : core_accumulate_result[15:8];
      end
      NNTestASCII, NNTestPulse, NNTestCount: begin
        data_o = test_out;
      end
      default: ;
    endcase
  end
endmodule
