module Qxxmultiply # (parameter int DATA_WIDTH, FRAC_BITS) (
    input clk, reset,
    input logic signed [DATA_WIDTH-1:0] input_reg, 
    input logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [31:0] final_product
);

localparam qBitShift = FRAC_BITS;
logic signed [31:0] product;

always_ff @(posedge clk) begin
    if (reset) begin
        product <= '0;
        final_product <= '0;
    end else begin
        product <= input_reg * weight;
        final_product <= (product + (1 <<< (qBitShift-1))) >>> qBitShift; // I eliminated 2 cycles of latency, hopefully this doesn't break timeing - DJ
    end
end
endmodule

function automatic int Qxxmultiply_latency();
    int latency = 2;
    return latency;
endfunction