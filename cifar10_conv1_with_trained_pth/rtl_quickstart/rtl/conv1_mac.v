// First 3x3x3 convolution dot-product, using weights from trained conv1 filter 0.
// Q16.16 SIGNED FIXED POINT -- NOT IEEE-754 FP32.
// Each 32-bit word encodes signed real * 65536. 27 cycles per dot product.
module conv1_mac (
    input wire clk,
    input wire reset,
    input wire start,
    output reg busy,
    output reg done,
    output reg signed [31:0] result_q16
);
    reg signed [31:0] weight_rom [0:26];
    reg signed [31:0] patch_rom  [0:26];
    reg [5:0] idx;
    reg signed [63:0] accum;
    wire signed [63:0] product;

    initial begin
        $readmemh("mem/conv1_filter0_q16.mem", weight_rom);
        $readmemh("mem/test_patch_q16.mem", patch_rom);
    end

    assign product = $signed(weight_rom[idx]) * $signed(patch_rom[idx]);

    always @(posedge clk) begin
        if (reset) begin
            idx <= 0;
            accum <= 0;
            busy <= 0;
            done <= 0;
            result_q16 <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                idx <= 0;
                accum <= 0;
                busy <= 1;
            end else if (busy) begin
                if (idx == 26) begin
                    result_q16 <= (accum + product) >>> 16;
                    busy <= 0;
                    done <= 1;
                end else begin
                    accum <= accum + product;
                    idx <= idx + 1'b1;
                end
            end
        end
    end
endmodule
