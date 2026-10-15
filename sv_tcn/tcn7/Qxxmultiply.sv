module Qxxmultiply # (parameter int DATA_WIDTH, FRAC_BITS) (
    input clk, reset,
    input logic signed [DATA_WIDTH-1:0] input_reg, 
    input logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [31:0] final_product
);

localparam qBitShift = FRAC_BITS;
logic signed [31:0] product;
logic signed [31:0] product_out;
logic signed [DATA_WIDTH-1:0] a;
logic signed [DATA_WIDTH-1:0] b;

always_ff @(posedge clk) begin
    if (reset) begin
        a <= '0;
        b <= '0;
        product <= '0;
        product_out <= '0;
        final_product <= '0;
    end else begin
        a <= input_reg;
        b <= weight;
        product <= a * b;
        product_out <= product;
        final_product <= (product_out + (1 <<< (qBitShift-1))) >>> qBitShift;
    end
end
endmodule

function automatic int Qxxmultiply_latency();
    int latency = 4;
    return latency;
endfunction