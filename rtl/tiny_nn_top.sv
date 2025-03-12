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
  fp_t                     max_val_q, max_val_d;
  logic [7:0]              max_val_skid_q, max_val_skid_d;

  logic [ValArrayHeight-1:0] core_val_shift;
  logic                      core_mul_row_sel, core_mul_en;
  fp_t                       core_accumulate_result;
  logic                      core_mul_add_op_a_en;
  logic                      core_mul_add_op_b_en;
  logic [1:0]                core_accumulate_mode_0_en;
  logic [1:0]                core_accumulate_mode_1_en;
  logic [1:0]                core_accumulate_mode_2_en;
  logic                      core_accumulate_loopback;
  logic                      core_accumulate_out_relu;
  logic                      core_accumulate_level_0_en;

  logic data_i_q_is_greater;

  logic [15:0] data_i_q;
  logic [7:0]  data_o_d, data_o_q;

  typedef enum logic [4:0] {
    NNIdle               = 5'h00,

    NNConvolveParamIn    = 5'h01,
    NNConvolveExec       = 5'h02,
    NNConvolveExecEnd    = 5'h03,

    NNAccumulateBiasIn   = 5'h04,
    NNAccumulateExec     = 5'h05,
    NNAccumulateExecEnd  = 5'h06,

    NNMulAccBiasIn       = 5'h07,
    NNMulAccExec         = 5'h08,
    NNMulAccExecEnd      = 5'h09,

    NNFixedMulAccParamIn = 5'h0a,
    NNFixedMulAccExec    = 5'h0b,
    NNFixedMulAccExecEnd = 5'h0c,

    NNMaxPoolExec        = 5'h0d,
    NNMaxPoolExecEnd     = 5'h0e,

    NNTestCount          = 5'h1d,
    NNTestPulse          = 5'h1e,
    NNTestASCII          = 5'h1f
  } state_e;

  state_e state_q, state_d;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_i_q <= '0;
    end else begin
      data_i_q <= data_i;
    end
  end

  always_comb begin
    state_d       = state_q;
    counter_d     = counter_q;
    start_count_d = start_count_q;
    phase_d       = 1'b0;
    param_write_d = param_write_q;
    convolve_run  = 1'b0;
    relu_d        = relu_q;
    max_val_d     = max_val_q;

    core_val_shift             = '0;
    core_mul_row_sel           = 1'b0;
    core_mul_en                = 1'b0;
    core_mul_add_op_a_en       = 1'b0;
    core_mul_add_op_b_en       = 1'b0;
    core_accumulate_mode_0_en  = '0;
    core_accumulate_mode_1_en  = '0;
    core_accumulate_mode_2_en  = '0;
    core_accumulate_loopback   = 1'b0;
    core_accumulate_out_relu   = 1'b0;
    core_accumulate_level_0_en = 1'b0;

    case (state_q)
      NNIdle: begin
        case (data_i_q[15:12])
          CmdOpConvolve: begin
            state_d   = NNConvolveParamIn;

            param_write_d    = '0;
            param_write_d[0] = 1'b1;
          end
          CmdOpAccumulate: begin
            start_count_d = data_i_q[CountWidth-1:0];
            counter_d     = CountWidth'(1'b1);
            state_d       = NNAccumulateBiasIn;
            relu_d        = data_i_q[8];
          end
          CmdOpTest: begin
            case (data_i_q[11:8])
              4'hf: begin
                state_d   = NNTestASCII;
                counter_d = 8'd3;
              end
              4'h1: begin
                state_d   = NNTestCount;
                counter_d = data_i_q[CountWidth-1:0];
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
            relu_d        = data_i_q[8];
            phase_d       = 1'b0;
            counter_d     = CountWidth'(2'd3);
          end
          CmdOpFixedMulAcc: begin
            start_count_d    = data_i_q[CountWidth-1:0];
            counter_d        = CountWidth'(2'd2);
            state_d          = NNFixedMulAccParamIn;
            param_write_d[6] = 1'b1;
          end
          CmdOpMaxPool: begin
            max_val_d     = FPNegInf;
            start_count_d = data_i_q[CountWidth-1:0];
            counter_d     = data_i_q[CountWidth-1:0];
            state_d       = NNMaxPoolExec;
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
          if (data_i_q == FPStdNaN) begin
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

          if (data_i_q == FPStdNaN) begin
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

          if (data_i_q == FPStdNaN) begin
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
      NNFixedMulAccParamIn: begin
        param_write_d[6]     = 1'b0;
        state_d              = NNFixedMulAccExec;
      end
      NNFixedMulAccExec: begin
        core_mul_en                  = 1'b1;
        core_accumulate_mode_2_en[0] = 1'b1;
        core_val_shift[0]            = 1'b1;
        core_mul_row_sel             = 1'b1;

        if (counter_q == 8'd0) begin
          counter_d = start_count_q;
        end else begin
          core_accumulate_loopback = 1'b1;
          counter_d                = counter_q - 1'b1;

          if (counter_q == CountWidth'(1'b1)) begin
            core_accumulate_mode_2_en[1] = 1'b1;
            core_mul_add_op_a_en         = 1'b1;
          end else if (counter_q == CountWidth'(2'd2)) begin
            if (data_i_q == FPStdNaN) begin
              counter_d = CountWidth'(2'd3);
              state_d   = NNFixedMulAccExecEnd;
            end
          end
        end
      end
      NNFixedMulAccExecEnd: begin
        if (counter_q == 8'd2) begin
          core_mul_row_sel             = 1'b1;
          core_accumulate_mode_2_en[1] = 1'b1;
          core_accumulate_loopback     = 1'b1;
        end else if (counter_q == '0) begin
          state_d = NNIdle;
        end

        counter_d = counter_q - 1'b1;
      end
      NNMaxPoolExec: begin
        if (data_i_q == FPStdNaN) begin
          state_d = NNMaxPoolExecEnd;
        end

        if (counter_q == 8'd0) begin
          counter_d = start_count_q;
          max_val_d = FPNegInf;
        end else begin
          counter_d = counter_q - 1'b1;

          if (data_i_q_is_greater) begin
            max_val_d = data_i_q;
          end
        end
      end
      NNMaxPoolExecEnd: begin
        state_d = NNIdle;
      end
      NNTestASCII: begin
        if (data_i_q[15:8] == 8'hff) begin
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
        if (data_i_q[15:8] == 8'hf0) begin
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
    max_val_q     <= max_val_d;
  end

  tiny_nn_core #(
    .ValArrayWidth(ValArrayWidth),
    .ValArrayHeight(ValArrayHeight)
  ) u_core (
    .clk_i,
    .rst_ni,

    .val_i      (fp_t'(data_i_q)),
    .val_shift_i(core_val_shift),

    .param_i      (fp_t'(data_i_q)),
    .param_write_i(param_write_q),

    .mul_row_sel_i(core_mul_row_sel),
    .mul_en_i     (core_mul_en),

    .accumulate_loopback_i(core_accumulate_loopback),
    .accumulate_out_relu_i(core_accumulate_out_relu),

    .mul_add_op_a_din_i('0),
    .mul_add_op_b_din_i(fp_t'(data_i_q)),
    .mul_add_op_a_en_i (core_mul_add_op_a_en),
    .mul_add_op_b_en_i (core_mul_add_op_b_en),

    .accumulate_level_0_din_i(fp_t'(data_i_q)),
    .accumulate_level_0_en_i (core_accumulate_level_0_en),

    .accumulate_mode_0_en_i(core_accumulate_mode_0_en),
    .accumulate_mode_1_en_i(core_accumulate_mode_1_en),
    .accumulate_mode_2_en_i(core_accumulate_mode_2_en),

    .accumulate_o   (core_accumulate_result)
  );

  fp_cmp u_fp_cmp (
    .op_a_i(data_i_q),
    .op_b_i(max_val_q),
    .op_a_greater_o(data_i_q_is_greater),
    .invalid_nan_o()
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
    data_o_d       = '1;
    max_val_skid_d = max_val_skid_q;

    case (state_q)
      NNConvolveExec, NNConvolveExecEnd: begin
        data_o_d = phase_q ? core_accumulate_result[15:8] :
                           core_accumulate_result[7:0];
      end
      NNAccumulateExec, NNFixedMulAccExec: begin
        if (counter_q == start_count_q) begin
          data_o_d = core_accumulate_result[7:0];
        end else begin
          data_o_d = core_accumulate_result[15:8];
        end
      end
      NNAccumulateExecEnd, NNMulAccExecEnd, NNFixedMulAccExecEnd: begin
        data_o_d = counter_q[0] ? core_accumulate_result[7:0] : core_accumulate_result[15:8];
      end
      NNTestASCII, NNTestPulse, NNTestCount: begin
        data_o_d = test_out;
      end
      NNMaxPoolExec: begin
        if (counter_q == '0) begin
          if (data_i_q_is_greater) begin
            data_o_d       = data_i_q[7:0];
            max_val_skid_d = data_i_q[15:8];
          end else begin
            data_o_d       = max_val_q[7:0];
            max_val_skid_d = max_val_q[15:8];
          end
        end else begin
          data_o_d = max_val_skid_q;
        end
      end
      NNMaxPoolExecEnd: begin
        data_o_d = max_val_skid_q;
      end
      default: ;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      data_o_q <= '1;
    end else begin
      data_o_q <= data_o_d;
    end
  end

  always_ff @(posedge clk_i) begin
    max_val_skid_q <= max_val_skid_d;
  end

  assign data_o = data_o_q;
endmodule
