module adder_tree_block # (parameter int NUM, DATA_WIDTH, DEPTH=-1, MAX_ADDS=2) (
    input clk, reset,
    input logic signed [0:NUM-1][DATA_WIDTH-1:0] nums,
    output logic signed [31:0] sum
);
parameter int depth = (DEPTH >= 0) ? DEPTH: adder_tree_block_latency(NUM);
logic [31:0] total;

if (NUM < MAX_ADDS || (NUM == MAX_ADDS && depth == 1)) begin: base_case
    always_comb begin
        total = '0;
        foreach(nums[i]) begin
            total += nums[i];
        end
    end
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= total;
    end
end else begin: recursion
    localparam int floored = NUM / MAX_ADDS;
    localparam int remainder = NUM % MAX_ADDS;

    logic signed [0:MAX_ADDS-1][31:0] sub_sums;

    for (genvar i = 0; i < MAX_ADDS; i++) begin: inst_recurs
        localparam slice_width = (remainder > i && remainder != 0) ? floored + 1: floored;
        localparam start_index = (remainder > i) ? i*slice_width: remainder*(slice_width+1) + (i-remainder)*slice_width;
        localparam end_index = start_index + slice_width - 1;
        adder_tree_block # (.NUM(slice_width), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
        subAdder1 (
            .clk(clk), 
            .reset(reset), 
            .nums(nums[start_index:end_index]), 
            .sum(sub_sums[i])
        );
    end

    always_comb begin
        total = '0;
        foreach(sub_sums[i]) begin
            total += sub_sums[i];
        end
    end
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= total;
    end
end
endmodule

function automatic int adder_tree_block_latency(int num);
    localparam int max_adds = 2;
    int latency = (num <= 2) ? 1: $floor(($ln((real'(num) - 1.0) / real'(max_adds)) / $ln(real'(max_adds))) + 2);
    return latency;
endfunction