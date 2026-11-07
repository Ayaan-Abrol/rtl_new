`timescale 1ns/1ps
// Sequential full Conv1: 3 input channels x 3x3 kernel x 64 filters x 32x32 image.
// Q16.16 SIGNED FIXED POINT; NOT IEEE-754 FP32.
// Produces one output every 27 busy clock cycles; output ordering filter-major, then row-major.
// This RTL is first intended for *simulation*. Board-specific BRAM, streaming I/O,
// clock/reset/trigger and physical capture interfaces still require integration.
module conv1_serial_q16 #(
    parameter integer FILTERS = 1  // 1 for fast check, 64 for complete Conv1
)(
    input wire clk, input wire rst, input wire start,
    output reg busy, output reg done,
    output reg output_valid,
    output reg [15:0] output_index,
    output reg signed [31:0] output_q16,
    output wire trigger
);
reg signed [31:0] weights [0:1727];
reg signed [31:0] biases [0:63];
reg signed [31:0] input_image [0:3071];
reg [5:0] filter_index;
reg [9:0] spatial_index;
reg [4:0] tap_index;
reg signed [63:0] acc;
reg signed [31:0] pixel;
reg signed [31:0] weight;
reg signed [63:0] product;
integer out_y, out_x, ch, ky, kx, src_y, src_x;
integer wi, xi;
initial begin
    $readmemh("mem/conv1_all_weights_q16.mem",weights);
    $readmemh("mem/conv1_all_bias_q16.mem",biases);
    $readmemh("mem/input_q16.mem",input_image);
end
// This is a simple demonstration data-access path; a physical FPGA design
// should pipeline synchronous ROM/BRAM addresses and register arithmetic.
always @* begin
    out_y = spatial_index / 32;
    out_x = spatial_index % 32;
    ch = tap_index / 9;
    ky = (tap_index % 9) / 3;
    kx = tap_index % 3;
    src_y = out_y + ky - 1;
    src_x = out_x + kx - 1;
    wi = filter_index*27 + tap_index;
    xi = ch*1024 + src_y*32 + src_x;
    weight = weights[wi];
    pixel = 32'sd0;
    if (src_y >= 0 && src_y < 32 && src_x >= 0 && src_x < 32)
        pixel = input_image[xi];
    product = pixel * weight;
end
assign trigger = busy;
always @(posedge clk) begin
    if (rst) begin
        busy <= 1'b0; done <= 1'b0; output_valid <= 1'b0;
        filter_index <= 0; spatial_index <= 0; tap_index <= 0;
        output_index <= 0; output_q16 <= 0; acc <= 0;
    end else begin
        done <= 0;
        output_valid <= 0;
        if (start && !busy) begin
            busy <= 1; filter_index <= 0; spatial_index <= 0;
            tap_index <= 0; acc <= 0;
        end else if (busy) begin
            if (tap_index == 26) begin
                output_q16 <= ((acc + product) >>> 16) + biases[filter_index];
                output_index <= filter_index*1024 + spatial_index;
                output_valid <= 1;
                acc <= 0;
                tap_index <= 0;
                if (spatial_index == 1023) begin
                    spatial_index <= 0;
                    if (filter_index == FILTERS - 1) begin
                        busy <= 0; done <= 1;
                    end else filter_index <= filter_index + 1'b1;
                end else spatial_index <= spatial_index + 1'b1;
            end else begin
                acc <= acc + product;
                tap_index <= tap_index + 1'b1;
            end
        end
    end
end
endmodule
