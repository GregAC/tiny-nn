module tiny_nn_top import tiny_nn_pkg::*; #(
  parameter int unsigned CountWidth     = 12,
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

  logic                    phase_q, phase_d;
  logic [CountWidth-1:0]   counter_q, counter_d;
  logic [ValArraySize-1:0] param_write_q, param_write_d;
  logic                    convolve_run;

  typedef enum logic [2:0] {
    NNIdle            = 0,
    NNConvolveParamIn = 1,
    NNConvolveExec    = 2
  } state_e;

  state_e state_q, state_d;

  always_comb begin
    state_d       = state_q;
    counter_d     = counter_q;
    phase_d       = phase_q;
    param_write_d = param_write_q;
    convolve_run  = 1'b0;

    case (state_q)
      NNIdle: begin
        case (data_i[15:12])
          CmdOpConvolve: begin
            state_d   = NNConvolveParamIn;
            counter_d = data_i[CountWidth-1:0];

            param_write_d    = '0;
            param_write_d[0] = 1'b1;
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
          phase_d = 1'b0;
        end
      end
      NNConvolveExec: begin
        phase_d      = ~phase_d;
        convolve_run = 1'b1;

        if (phase_q) begin
          if (counter_q != '0) begin
            counter_d = counter_q - 1'b1;
          end else begin
            state_d = NNIdle;
          end
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
    phase_q       <= phase_d;
    param_write_q <= param_write_d;
  end

  logic [ValArrayHeight-1:0] core_val_shift;
  logic                      core_mul_row_sel, core_mul_en;
  logic [1:0]                core_accumulate_en;
  fp_t                       core_accumulate_result;

  assign core_val_shift[0]     = ~phase_q & convolve_run;
  assign core_val_shift[1]     =  phase_q & convolve_run;
  assign core_mul_row_sel      =  phase_q;
  assign core_mul_en           =  convolve_run;
  assign core_accumulate_en[0] =  convolve_run;
  assign core_accumulate_en[1] =  phase_q & convolve_run;

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

    .accumulate_en_i(core_accumulate_en),
    .accumulate_o   (core_accumulate_result)
  );

  always_comb begin
    data_o = '1;

    case (state_q)
      NNConvolveExec: begin
        data_o = phase_q ? core_accumulate_result[15:8] :
                           core_accumulate_result[7:0];
      end
      default: ;
    endcase
  end
endmodule
