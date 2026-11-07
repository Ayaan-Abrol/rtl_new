`timescale 1ns/1ps
module tb_conv1_serial_q16;
`ifdef ALL_FILTERS
localparam integer FILTERS=64;
`else
localparam integer FILTERS=1;
`endif
reg clk=0, rst=1, start=0;
wire busy,done,output_valid,trigger;
wire [15:0] output_index;
wire signed [31:0] output_q16;
reg signed [31:0] golden [0:65535];
integer count=0, errors=0;
conv1_serial_q16 #(.FILTERS(FILTERS)) dut(
 .clk(clk),.rst(rst),.start(start),.busy(busy),.done(done),
 .output_valid(output_valid),.output_index(output_index),
 .output_q16(output_q16),.trigger(trigger));
always #5 clk = ~clk;
initial begin
 $readmemh("mem/golden_conv1_q16.mem",golden);
 repeat (4) @(negedge clk); rst=0;
 @(negedge clk); start=1;
 @(negedge clk); start=0;
end
always @(negedge clk) begin
 if (output_valid) begin
   if ($signed(output_q16) !== $signed(golden[output_index])) begin
     if(errors<8) $display("MISMATCH idx=%0d got=%0d expected=%0d",output_index,output_q16,golden[output_index]);
     errors=errors+1;
   end
   count=count+1;
 end
 if(done) begin
   $display("CHECKED %0d Conv1 outputs across %0d filters; errors=%0d",count,FILTERS,errors);
   if(errors==0 && count==FILTERS*1024) $display("PASS: Conv1 RTL agrees with Q16.16 software golden values");
   else $fatal(1,"FAIL: Conv1 RTL mismatch or incomplete outputs");
   $finish;
 end
end
initial begin
 #1000000000; $fatal(1,"TIMEOUT");
end
endmodule
