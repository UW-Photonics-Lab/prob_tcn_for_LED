`include "tcn_layer.sv"

localparam int STRIDE_BITS = 5;

// For every layer n, divide that layer's throughput by that layer's stride
function automatic int tcn_layer_throughput(int THROUGHPUT, logic [255:0] STRIDES, int n);
    int throughput = THROUGHPUT;
    for (int i = 0; i < n; i++) begin
        throughput = throughput / STRIDES[STRIDE_BITS*i +: STRIDE_BITS];
    end
    return throughput;
endfunction

module tcn #(parameter TEST, DATA_WIDTH, FRAC_BITS, HIDDEN_CHANNELS, KERNEL_SIZE, LAYERS, DILATION_BASE, MODEL_TYPE, THROUGHPUT, STRIDES) (
    input clk, reset, data_valid_in,
    input logic signed [0:THROUGHPUT-1][DATA_WIDTH-1:0] in,
    output logic signed [0:tcn_layer_throughput(THROUGHPUT, STRIDES, LAYERS)-1][DATA_WIDTH-1:0] out,
    output data_valid_out
);
// conections connects every input and output between layers.
// logic [0:THROUGHPUT-1][HIDDEN_CHANNELS-1:0][DATA_WIDTH-1:0] connections [1:LAYERS];
logic [0:THROUGHPUT-1][0:0][DATA_WIDTH-1:0] first;
// logic [0:THROUGHPUT-1][0:0][DATA_WIDTH-1:0] readout;
//assign connections[0] = in; // these are not the same size, I am assuming in will be left padded with 0s, as well as a truncation of connections[0] when assigned to the first layer input
for (genvar i = 0; i < THROUGHPUT; i++) begin: assign_input
    assign first[i][0] = in[i];
end

genvar i;
generate
    for (i = 0; i <= LAYERS; i++) begin: LayerInstantiation
        localparam LAYER_THROUGHPUT = tcn_layer_throughput(THROUGHPUT, STRIDES, i);
        localparam LAYER_STRIDE = (i == LAYERS) ? 1 : STRIDES[i*STRIDE_BITS +: STRIDE_BITS];
        localparam IN_CH = (i == 0) ? 1 : HIDDEN_CHANNELS;
        localparam OUT_CH = (i == LAYERS) ? 1 : HIDDEN_CHANNELS;
        logic [0:LAYER_THROUGHPUT-1][IN_CH-1:0][DATA_WIDTH-1:0] layer_in;
        logic [0:LAYER_THROUGHPUT/LAYER_STRIDE-1][OUT_CH-1:0][DATA_WIDTH-1:0] layer_out;

        if (i == 0) begin : g_in
            assign layer_in = first;
        end else begin : g_in
            assign layer_in = LayerInstantiation[i-1].layer_out;
        end


        if (i == 0) begin : first_layer
            tcn_layer #(
                .TEST(TEST), 
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS), 
                .DILATION(DILATION_BASE ** i), 
                .IN_CH(1), 
                .OUT_CH(HIDDEN_CHANNELS), 
                .LAYER_NUM(i), 
                .KERNEL_SIZE(KERNEL_SIZE), 
                .RESAMPLE(1), 
                .MODEL_TYPE(MODEL_TYPE),
                .THROUGHPUT(LAYER_THROUGHPUT),
                .STRIDE(LAYER_STRIDE)
            ) layer (.input_vector(layer_in), .out(layer_out), .clk(clk), .reset(reset));
        end else if (i == LAYERS) begin : readout_layer
            tcn_layer #(
                .TEST(TEST), 
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS), 
                .DILATION(1), 
                .IN_CH(HIDDEN_CHANNELS), 
                .OUT_CH(1), 
                .LAYER_NUM(i), 
                .KERNEL_SIZE(1), 
                .RESAMPLE(2), 
                .MODEL_TYPE(MODEL_TYPE),
                .THROUGHPUT(LAYER_THROUGHPUT),
                .STRIDE(LAYER_STRIDE)
            ) layer (.input_vector(layer_in), .out(layer_out), .clk(clk), .reset(reset));
        end else begin : regular_layer
            tcn_layer #(
                .TEST(TEST), 
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS), 
                .DILATION(DILATION_BASE ** i), 
                .IN_CH(HIDDEN_CHANNELS), 
                .OUT_CH(HIDDEN_CHANNELS), 
                .LAYER_NUM(i), 
                .KERNEL_SIZE(KERNEL_SIZE), 
                .RESAMPLE(0), 
                .MODEL_TYPE(MODEL_TYPE),
                .THROUGHPUT(LAYER_THROUGHPUT),
                .STRIDE(LAYER_STRIDE)
            ) layer (.input_vector(layer_in), .out(layer_out), .clk(clk), .reset(reset));
        end
    end
endgenerate
localparam latency = tcn_layer_latency(KERNEL_SIZE, HIDDEN_CHANNELS, 1) + ((LAYERS-1) * tcn_layer_latency(KERNEL_SIZE, HIDDEN_CHANNELS, 0)) + tcn_layer_latency(KERNEL_SIZE, HIDDEN_CHANNELS, 2); 
initial begin
    $display("Network Latency: %0d", latency);
end
logic [latency-1:0] counter;
wire yeet;
assign yeet = counter[0]; //redundant

always_ff @(posedge clk) begin
    if (reset) begin
        counter <= '0;
    end else begin
        counter <= {data_valid_in, counter[latency-1:1]};
    end
end
assign data_valid_out = counter[0];
//assign out = connections[LAYERS+1][0];
for (genvar i = 0; i < tcn_layer_throughput(THROUGHPUT, STRIDES, LAYERS); i++) begin: assign_output
    assign out[i] = LayerInstantiation[LAYERS].layer_out[i][0];
end
endmodule