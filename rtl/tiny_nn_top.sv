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
  logic                    accumulate_run;
  logic                    core_result_skid_en;
  logic                    accumulate_loopback;
  logic                    accumulate_out_relu;
  logic [1:0]              accumulate_level_1_direct_en;

  typedef enum logic [3:0] {
    NNIdle               = 4'h0,
    NNConvolveParamIn    = 4'h1,
    NNConvolveExec       = 4'h2,
    NNConvolveExecEnd    = 4'h3,
    NNAccumulateBiasIn   = 4'h4,
    NNAccumulateExec     = 4'h5,
    NNAccumulateExecEnd1 = 4'h6,
    NNAccumulateExecEnd2 = 4'h7,
    NNTestCount          = 4'hd,
    NNTestPulse          = 4'he,
    NNTestASCII          = 4'hf
  } state_e;

  state_e state_q, state_d;

  always_comb begin
    state_d                      = state_q;
    counter_d                    = counter_q;
    start_count_d                = start_count_q;
    phase_d                      = 1'b0;
    param_write_d                = param_write_q;
    convolve_run                 = 1'b0;
    accumulate_run               = 1'b0;
    relu_d                       = relu_q;
    accumulate_loopback          = 1'b0;
    accumulate_out_relu          = 1'b0;
    core_result_skid_en          = 1'b0;
    accumulate_level_1_direct_en = '0;

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

            state_d = NNAccumulateBiasIn;

            relu_d = data_i[8];
          end
          CmdOpTest: begin
            case (data_i[11:8])
              4'hf: begin
                state_d   = NNTestASCII;
                counter_d = 8'd4;
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
        accumulate_level_1_direct_en[1] = 1'b1;
        state_d                         = NNAccumulateExec;
      end
      NNAccumulateExec: begin
        accumulate_run                  = 1'b1;
        accumulate_level_1_direct_en[0] = 1'b1;

        if (counter_q == '0) begin
          counter_d           = start_count_q;
          core_result_skid_en = 1'b1;
        end else begin
          if (counter_q == CountWidth'(1'b1)) begin
            accumulate_out_relu = relu_q;
          end

          accumulate_loopback = 1'b1;
          counter_d           = counter_q - 1'b1;

          if (data_i == FPStdNaN) begin
            state_d = NNAccumulateExecEnd1;
          end
        end
      end
      NNAccumulateExecEnd1: begin
        core_result_skid_en = 1'b1;
        state_d             = NNAccumulateExecEnd2;
      end
      NNAccumulateExecEnd2: begin
        state_d = NNIdle;
      end
      NNTestASCII: begin
        if (data_i[15:8] == 8'hff) begin
          if (counter_q == 0) begin
            counter_d = 8'd4;
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

  logic [ValArrayHeight-1:0] core_val_shift;
  logic                      core_mul_row_sel, core_mul_en;
  logic [1:0]                core_accumulate_en;
  fp_t                       core_accumulate_result;
  logic [7:0]                core_result_skid_q;
  fp_t                       accumulate_level_1_direct_din;

  assign core_val_shift[0]     = ~phase_q & convolve_run;
  assign core_val_shift[1]     =  phase_q & convolve_run;
  assign core_mul_row_sel      =  phase_q;
  assign core_mul_en           =  convolve_run;
  assign core_accumulate_en[0] =  convolve_run;
  assign core_accumulate_en[1] =  (phase_q & convolve_run) | accumulate_run;

  assign accumulate_level_1_direct_din = data_i;

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

    .accumulate_loopback_i(accumulate_loopback),
    .accumulate_out_relu_i(accumulate_out_relu),

    .accumulate_level_1_direct_din_i(accumulate_level_1_direct_din),
    .accumulate_level_1_direct_en_i(accumulate_level_1_direct_en),

    .accumulate_en_i(core_accumulate_en),
    .accumulate_o   (core_accumulate_result)
  );

  always_ff @(posedge clk_i) begin
    if (core_result_skid_en) begin
      core_result_skid_q <= core_accumulate_result[15:8];
    end
  end

  logic [7:0] test_out;

  always_comb begin
    test_out = '0;

    case (state_q)
      NNTestASCII: begin
        case (counter_q)
          8'd3, 8'd4: test_out = 8'h54;
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
        if (counter_q == '0) begin
          data_o = core_accumulate_result[7:0];
        end else begin
          data_o = core_result_skid_q;
        end
      end
      NNAccumulateExecEnd1: begin
        data_o = core_accumulate_result[7:0];
      end
      NNAccumulateExecEnd2: begin
        data_o = core_result_skid_q;
      end
      NNTestASCII, NNTestPulse, NNTestCount: begin
        data_o = test_out;
      end
      default: ;
    endcase
  end
endmodule
