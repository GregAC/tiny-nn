module tiny_nn_top_tb;
  logic clk;
  logic rst_n;

  initial begin
    clk = 1'b0;
    rst_n = 1'b1;
    #5;
    rst_n = 1'b0;
    #5;
    clk = 1'b1;
    #5;
    clk = 1'b0;
    #5;
    rst_n = 1'b1;
    forever begin
      #5;
      clk = ~clk;
    end
  end

  logic [15:0] test_vals [16];
  logic [31:0] counter;

  initial begin
    $dumpfile("sim.fst");
    $dumpvars;

    $readmemh("test_image.hex", test_vals);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      counter <= '0;
    end else begin
      counter <= counter + 32'd1;
    end
  end

  logic [15:0] test_data_in;

  always_comb begin
    test_data_in = '0;
    if (counter == 32'd5) begin
      test_data_in = {tiny_nn_pkg::CmdOpConvolve, 12'd32};
    end else if(counter >= 32'd6 && counter < 32'd14) begin
      test_data_in = 16'h3f00;
    end else begin
      if (counter < 32'd30) begin
        test_data_in = test_vals[counter - 14];
      end else if (counter < 32'd100) begin
        test_data_in = tiny_nn_pkg::FPZero;
      end else begin
        $finish();
      end
    end
  end

  tiny_nn_top dut (
    .clk_i(clk),
    .rst_ni(rst_n),

    .data_i(test_data_in),
    .data_o()
  );
endmodule
