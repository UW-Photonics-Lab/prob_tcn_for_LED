module adder_tree_block # (parameter int NUM, DATA_WIDTH, DEPTH=-1, MAX_ADDS=2) (
    input clk, reset,
    input logic signed [0:NUM-1][DATA_WIDTH-1:0] nums,
    output logic signed [31:0] sum
);
parameter int depth = (DEPTH >= 0) ? DEPTH: adder_tree_block_latency(NUM, MAX_ADDS);
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
    localparam int default_slice_width = NUM / MAX_ADDS;
    localparam int remainder_count = NUM % MAX_ADDS;

    // Amortize the extra remainder additions by increasing the slice width by 1 remainder times
    localparam remainder_slice_width = default_slice_width + 1;

    logic signed [0:MAX_ADDS-1][31:0] sub_sums;

    for (genvar i = 0; i < MAX_ADDS; i++) begin: inst_recurs

        localparam int start_index = (i < remainder_count)
            ? i * remainder_slice_width
            : remainder_count * remainder_slice_width + (i - remainder_count) * default_slice_width;

        localparam int end_index = start_index +
            ((i < remainder_count) ? remainder_slice_width : default_slice_width) - 1;

        adder_tree_block # (.NUM(end_index - start_index + 1), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1), .MAX_ADDS(MAX_ADDS)) 
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

function automatic int adder_tree_block_latency(int num, int max_adds);
    // Num levels = latency = ceil(log_{max_adds}(num))
    // Use while loop to avoid floating point silent errors
    int latency = 0;
    int MAX_WIDTH = 1;
    while (MAX_WIDTH < num) begin
        MAX_WIDTH = MAX_WIDTH * max_adds;
        latency++;
    end
    return (latency < 1) ? 1 : latency; // Minimum latency of 1
endfunction