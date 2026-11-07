`timescale 1ns / 1ps
module tb_conv1_mac;
    reg clk = 0;
    reg reset = 1;
    reg start = 0;
    wire busy;
    wire done;
    wire signed [31:0] result_q16;
    conv1_mac dut(.clk(clk),.reset(reset),.start(start),
                  .busy(busy),.done(done),.result_q16(result_q16));
    localparam signed [31:0] EXPECTED_Q16 = 32'sd40836;
    always #5 clk = ~clk;
    initial begin
        repeat (3) @(negedge clk);
        reset = 0;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        wait(done);
        #1;
        $display("RESULT Q16.16 integer = %0d", result_q16);
        $display("RESULT approximate = %f", $itor(result_q16)/65536.0);
        $display("EXPECTED Q16.16 integer = %0d", EXPECTED_Q16);
        if ($signed(result_q16) !== $signed(EXPECTED_Q16)) begin
            $display("FAIL: result differs from reference");
            $fatal(1);
        end
        $display("PASS: RTL convolution dot-product matches fixed-point reference");
        $finish;
    end
    initial begin
        #100000;
        $display("TIMEOUT");
        $fatal(1);
    end
endmodule
