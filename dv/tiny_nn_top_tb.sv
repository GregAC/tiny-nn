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

  logic [15:0] test_vals[$];

  logic [15:0] test_data_in;
  logic [7:0]  data_out;

  initial begin
    integer idle_wait_count;

    integer test_file;
    string  test_filename;

    integer out_file;
    string  out_filename;


    if (!$value$plusargs("test_data=%s", test_filename)) begin
      $display("No +test_data argument provided");
      $finish();
    end

    if (!$value$plusargs("out=%s", out_filename)) begin
      $display("No +out argument provided");
      $finish();
    end

    test_file = $fopen(test_filename, "r");
    if (test_file == 0) begin
      $display("Error opening %s", test_filename);
      $finish();
    end

    out_file = $fopen(out_filename, "w");
    if (out_file == 0) begin
      $display("Error opening %s", out_filename);
      $finish();
    end

    $dumpfile("sim.fst");
    $dumpvars;

    while (!$feof(test_file)) begin
      bit [15:0] val;

      $fscanf(test_file, "%x\n", val);
      test_vals.push_back(val);
    end

    test_data_in = '0;
    repeat(3) @(posedge clk);


    // TODO: Tidy up output into `out_file`. Need to manually replicate the $fwrite calls below in
    // the wait for idle and simulation exit. Better to have a separate process that just writes out
    // to the file every clock until stopped.
    foreach (test_vals[i]) begin
      test_data_in = test_vals[i];
      @(posedge clk);
      $fwrite(out_file, "%02x\n", data_out);
    end

    idle_wait_count = 0;

    forever begin
      @(posedge clk);
      $fwrite(out_file, "%02x\n", data_out);

      if (dut.state_q != dut.NNIdle) begin
        idle_wait_count++;
      end else begin
        break;
      end

      if (idle_wait_count == 100) begin
        $display("ERROR: Timeout awaiting idle state at end of simulation");
        break;
      end
    end

    @(posedge clk);

    $fclose(test_file);
    $fclose(out_file);
    $finish();
  end

  tiny_nn_top dut (
    .clk_i(clk),
    .rst_ni(rst_n),

    .data_i(test_data_in),
    .data_o(data_out)
  );
endmodule
