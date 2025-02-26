package tiny_nn_pkg;
  parameter FPExpWidth  = 8;
  parameter FPMantWidth = 7;

  typedef struct packed {
    logic sgn;
    logic [FPExpWidth-1:0] exp;
    logic [FPMantWidth-1:0] mant;
  } fp_t;

  parameter fp_t FPZero   = '{sgn: 1'b0, exp: '0, mant: '0};
  parameter fp_t FPStdNaN = '{sgn: 1'b1, exp: '1, mant: '1};
  parameter fp_t FPPosInf = '{sgn: 1'b0, exp: '1, mant: '0};
  parameter fp_t FPNegInf = '{sgn: 1'b1, exp: '1, mant: '0};

  function logic is_nan(fp_t x);
    if (x.exp == '0) begin
      if (x.mant != '0) begin
        return 1'b1;
      end else if (x.sgn == 1'b1) begin
        return 1'b1;
      end else begin
        return 1'b0;
      end
    end else if (x.exp == '1) begin
      if (x.mant != '0) begin
        return 1'b1;
      end else begin
        return 1'b0;
      end
    end else begin
      return 1'b0;
    end
  endfunction

  function bit is_inf(fp_t x);
    return x == FPPosInf || x == FPNegInf;
  endfunction

  parameter logic [3:0] CmdOpConvolve   = 4'h1;
  parameter logic [3:0] CmdOpAccumulate = 4'h2;
endpackage
