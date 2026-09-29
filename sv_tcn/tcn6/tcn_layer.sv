`include "hidden_channel_block.sv"

module tcn_layer # (parameter TEST, DATA_WIDTH, FRAC_BITS, DIALATION, IN_CH, OUT_CH, LAYER_NUM, KERNEL_SIZE, RESAMPLE, MODEL_TYPE) (
    input clk, reset,
    input signed [IN_CH-1:0][DATA_WIDTH-1:0] in,
    output signed [OUT_CH-1:0][DATA_WIDTH-1:0] out
);
logic signed [IN_CH-1:0][DATA_WIDTH-1:0] input_reg [0:(KERNEL_SIZE-1)*DIALATION];

//manage the input stream in a buffer
always_ff @(posedge clk) begin
    if (reset) begin
        //input_reg <= '0;
        foreach(input_reg[i]) begin
            input_reg[i] <= '0;
        end
    end else begin
        for (int i = 1; i < $size(input_reg); i++) begin
            input_reg[i] <= input_reg[i-1];
        end
        input_reg[0] <= in;
    end
end
//wire up hidden channels - Each hidden_channel_block gives you 1 output
logic signed [DATA_WIDTH-1:0] actual_input_reg [0:KERNEL_SIZE*IN_CH-1];
logic [OUT_CH-1:0][DATA_WIDTH-1:0] hc_output;

generate
    for (genvar i = 0; i < KERNEL_SIZE; i++) begin: timeSeries
        for (genvar j = 0; j < IN_CH; j++) begin: channelSeries
            assign actual_input_reg[j * KERNEL_SIZE + i] = input_reg[(KERNEL_SIZE-1-i)*DIALATION][j];
        end
    end
endgenerate

generate
    for (genvar i = 0; i < OUT_CH; i++) begin: HC_inst

        hidden_channel_block #(
            .NUM_TAPS(KERNEL_SIZE*IN_CH), 
            .DATA_WIDTH(DATA_WIDTH),
            .FRAC_BITS(FRAC_BITS), 
            .SKIPCONN(KERNEL_SIZE*i + (KERNEL_SIZE-1)), 
            .TEST(TEST), .LAYER_NUM(LAYER_NUM), 
            .HIDDEN_CH_NUM(i), .RESAMPLE(RESAMPLE), 
            .MODEL_TYPE(MODEL_TYPE)
        ) hc (.input_reg(actual_input_reg), .clk(clk), .reset(reset), .out(hc_output[i]));
            
    end
endgenerate

assign out = hc_output;
endmodule

function automatic int tcn_layer_latency(int KERNEL_SIZE, int HIDDEN_CHANNELS, int RESAMPLE);
    int latency;
    if (RESAMPLE == 0) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(KERNEL_SIZE*HIDDEN_CHANNELS + 1) + 3;
    else if (RESAMPLE == 1) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(KERNEL_SIZE + 1) + 3;
    else if (RESAMPLE == 2) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(HIDDEN_CHANNELS + 1) + 1;
    return latency;
endfunction