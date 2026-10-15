`include "hidden_channel_block.sv"

module tcn_layer # (parameter TEST, DATA_WIDTH, FRAC_BITS, DILATION, IN_CH, OUT_CH, LAYER_NUM, KERNEL_SIZE, RESAMPLE, MODEL_TYPE, THROUGHPUT, STRIDE) (
    input clk, reset,
    input logic signed [0:THROUGHPUT-1][IN_CH-1:0][DATA_WIDTH-1:0] input_vector,
    output logic signed [0:THROUGHPUT/STRIDE-1][OUT_CH-1:0][DATA_WIDTH-1:0] out
);

localparam HISTORY = (KERNEL_SIZE - 1) * DILATION;
logic signed [IN_CH-1:0][DATA_WIDTH-1:0] sample_history [0:HISTORY + THROUGHPUT - 1];


// If stride is nonzero, must check if stride divides throughput

if (THROUGHPUT % STRIDE != 0) begin : g_throughput_stride_check
    $error("Stride must cleanly divide throughput!");
end

//manage the input stream input_vector a buffer
always_ff @(posedge clk) begin
    if (reset) begin
        foreach(sample_history[i]) begin
            sample_history[i] <= '0;
        end
    end else begin
        for (int i = 0; i < HISTORY; i++) begin
            sample_history[i] <= sample_history[i + THROUGHPUT];
        end
        //sample_history[0] <= input_vector;
        for (int i = 0; i < THROUGHPUT; i++) begin
            sample_history[i + HISTORY] <= input_vector[i];
        end
    end
end
//wire up hidden channels - Each hidden_channel_block gives you {THROUGHPUT} output(s)


logic signed [0:KERNEL_SIZE*IN_CH-1][DATA_WIDTH-1:0] conv_window [THROUGHPUT/STRIDE]; // Decrease total througphut by stride
logic signed [0:THROUGHPUT/STRIDE-1][OUT_CH-1:0][DATA_WIDTH-1:0] hc_out;
generate
    for (genvar k = 0; k < THROUGHPUT/STRIDE; k++) begin: throughput
        for (genvar t = 0; t < KERNEL_SIZE; t++) begin: timeSeries
            for (genvar ch = 0; ch < IN_CH; ch++) begin: channelSeries
                localparam newest_sample_for_k = HISTORY + STRIDE * k + STRIDE - 1; // last sample of each stride group
                localparam tap_t_for_k = (KERNEL_SIZE-1 - t) * DILATION;
                assign conv_window[k][ch * KERNEL_SIZE + t] = sample_history[newest_sample_for_k - tap_t_for_k][ch]; // Pick out every stride sample 
            end
        end
    end
endgenerate

generate
    for (genvar k = 0; k < THROUGHPUT/STRIDE; k++) begin: throughput_iter_hc
        for (genvar ch = 0; ch < OUT_CH; ch++) begin: HC_inst

            hidden_channel_block #(
                .NUM_TAPS(KERNEL_SIZE*IN_CH), 
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS), 
                .SKIPCONN(KERNEL_SIZE*ch + (KERNEL_SIZE-1)), 
                .TEST(TEST), .LAYER_NUM(LAYER_NUM), 
                .HIDDEN_CH_NUM(ch), .RESAMPLE(RESAMPLE), 
                .MODEL_TYPE(MODEL_TYPE)
            ) hc (.input_reg(conv_window[k]), .clk(clk), .reset(reset), .out(hc_out[k][ch]));
                
        end
    end
endgenerate
assign out = hc_out;
endmodule

function automatic int tcn_layer_latency(int KERNEL_SIZE, int HIDDEN_CHANNELS, int RESAMPLE);
    int latency;
    if (RESAMPLE == 0) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(KERNEL_SIZE*HIDDEN_CHANNELS + 1, 2) + 3; //TODO: would it be possible to change these magic numnbers to named localparam int declarations? 
    else if (RESAMPLE == 1) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(KERNEL_SIZE + 1, 2) + 3;
    else if (RESAMPLE == 2) latency = 1 + Qxxmultiply_latency() + adder_tree_block_latency(HIDDEN_CHANNELS + 1, 2) + 1;
    return latency;
endfunction