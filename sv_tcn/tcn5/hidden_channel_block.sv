// in theory this module should use 1 mac DSP slice on the FPGA
module hidden_channel_block # (parameter int NUM_TAPS, DATA_WIDTH, SKIPCONN, TEST, LAYER_NUM, HIDDEN_CH_NUM, RESAMPLE, parameter [0:55] MODEL_TYPE) (
    input logic [NUM_TAPS:0] counter,
    input logic signed [0:NUM_TAPS-1][DATA_WIDTH-1:0] input_reg,
    input clk, reset,
    output logic [DATA_WIDTH-1:0] out
);
logic signed [DATA_WIDTH-1:0] weights [NUM_TAPS];
logic signed [DATA_WIDTH-1:0] bias [0:0];
//get weights from file todo: add error handling for retrieving from a file - The order of the weights matters in this setup
initial begin
    string fileName;
    if (RESAMPLE == 2) begin
        fileName = $sformatf("TestingData/Test%0d/%s/readout_weight/channel%0d.mem", TEST, MODEL_TYPE, HIDDEN_CH_NUM);
        $readmemh(fileName, weights, 0, NUM_TAPS-1);

        fileName = $sformatf("TestingData/Test%0d/%s/readout_bias/channel%0d.mem", TEST, MODEL_TYPE, HIDDEN_CH_NUM);
        $readmemh(fileName, bias);
    end else begin
        fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_conv_weight/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
        $readmemh(fileName, weights, 0, NUM_TAPS-1);

        fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_conv_bias/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
        $readmemh(fileName, bias);
    end
    /*
    $display($sformatf("Test: %0d, Model Type: %s, Layer num: %0d, Hidden num: %0d", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM));
    foreach (weights[i]) begin
        $display($sformatf("W:  %h", weights[i]));
    end
    $display($sformatf("Bias:  %h", bias[0]));
    */
end

logic signed [DATA_WIDTH-1:0] weight_mux;
logic signed [DATA_WIDTH-1:0] in_mux;
always_comb begin
    weight_mux = 0;
    in_mux = 0;
    for (int i = 0; i < NUM_TAPS; i++) begin
        if (counter[i]) begin
            weight_mux = weights[i];
            in_mux     = input_reg[i];
        end
    end
end

int signed product;
int signed accumulator;
always_ff @(posedge clk) begin
    if (reset) begin
        accumulator <= bias[0];
        product <= 0;
    end else if (counter == 0) begin
        accumulator <= bias[0];
        product <= 0;
    end else begin
        product <= Q88multiply(weight_mux, in_mux);
        accumulator <= accumulator + product;
        /*
        $display("Layer %0d Channel %0d", LAYER_NUM, HIDDEN_CH_NUM);
        $display("      in_mux: %0d %f %h, weight_mux: %0d %f %h, Prod: %0d %f %h, Acc: %0d %f %h", 
            in_mux,              real'(in_mux) / 256.0,          in_mux, 
            weight_mux,          real'(weight_mux) / 256.0,      weight_mux, 
            product,                      real'(product) / 256.0,                  product, 
            accumulator,                  real'(accumulator) / 256.0,              accumulator
            );
        */
    end
end

generate
    string fileName;
    if (RESAMPLE == 1) begin: resample_weight_generate //generates this for the very first layer only
        logic signed [DATA_WIDTH-1:0] resample_weight [0:0];
        logic signed [DATA_WIDTH-1:0] resample_bias [0:0];
        logic signed [DATA_WIDTH-1:0] resampled_input;

        initial begin
            fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_resample_weight/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
            $readmemh(fileName, resample_weight);
            fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_resample_bias/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
            $readmemh(fileName, resample_bias);
        end

        assign resampled_input = Q88clip(Q88multiply(input_reg[NUM_TAPS-1], resample_weight[0]) + resample_bias[0]);

        assign out = (accumulator >= 0) ?  Q88clip(accumulator + resampled_input): (resampled_input); //relu and resampled input
    end else if (RESAMPLE == 2) begin
        assign out = Q88clip(accumulator); //no relu or skip connection for the readout
    end else begin // Below is the default generation for all other layers
        assign out = (accumulator >= 0) ?  Q88clip(accumulator + input_reg[SKIPCONN]): input_reg[SKIPCONN]; //relu and residual input added in
    end
endgenerate

// helper functions
function automatic int Q88multiply(int a, int b);
    localparam int qBitShift = DATA_WIDTH/2;
    int prod;
    prod = (a * b + (1 <<< (qBitShift-1))) >>> qBitShift;
    return prod;
endfunction
function automatic int Q88clip(int signed a);
    localparam int maximumVal = 2 ** (DATA_WIDTH - 1) - 1;
    localparam int minimumVal = -1 * (2 ** (DATA_WIDTH - 1));
    if (a > maximumVal) return maximumVal;
    else if (a < minimumVal) return minimumVal;
    else return a;
endfunction
endmodule

