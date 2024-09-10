package tiny_nn_pkg;
  parameter FPExpWidth  = 8;
  parameter FPMantWidth = 7;

  typedef struct packed {
    logic sgn;
    logic [FPExpWidth-1:0] exp;
    logic [FPMantWidth-1:0] mant;
  } fp_t;

  parameter fp_t FPZero = '{sgn: 1'b0, exp: '0, mant: '0};
endpackage
