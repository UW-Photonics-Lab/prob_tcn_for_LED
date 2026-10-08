`include "tcn_layer.sv"

module tcn #(parameter TEST, DATA_WIDTH, HIDDEN_CHANNELS, KERNEL_SIZE, LAYERS, DIALATION_BASE, parameter [0:55] MODEL_TYPE) (
    input clk, reset, update,
    input signed [DATA_WIDTH-1:0] in,
    output logic signed [DATA_WIDTH-1:0] out,
    output logic switch
);
// conections connects every input and output between layers.
wire [0:LAYERS+1][HIDDEN_CHANNELS-1:0][DATA_WIDTH-1:0] connections;
assign connections[0] = in; // these are not the same size, I am assuming in will be left padded with 0s, as well as a truncation of connections[0] when assigned to the first layer input

logic [0:LAYERS+1] update_arr;
assign update_arr[0] = update;
genvar i;
generate
    for (i = 0; i <= LAYERS; i++) begin: LayerInstantiation
        //localparam inch = (i == 0) ? 1 : HIDDEN_CHANNELS;
        //localparam ouch = (i == LAYERS-1) ? 1 : HIDDEN_CHANNELS;
        if (i == 0) begin : first_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(DIALATION_BASE ** i), .IN_CH(1), .OUT_CH(HIDDEN_CHANNELS), .LAYER_NUM(i), .KERNEL_SIZE(KERNEL_SIZE), .RESAMPLE(1), .MODEL_TYPE(MODEL_TYPE)) 
                        layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset), .update(update_arr[i]), .ready(update_arr[i+1]));
        end else if (i == LAYERS) begin : readout_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(1), .IN_CH(HIDDEN_CHANNELS), .OUT_CH(1), .LAYER_NUM(i), .KERNEL_SIZE(1), .RESAMPLE(2), .MODEL_TYPE(MODEL_TYPE)) 
                        layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset), .update(update_arr[i]), .ready(update_arr[i+1]));
        end else begin : regular_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(DIALATION_BASE ** i), .IN_CH(HIDDEN_CHANNELS), .OUT_CH(HIDDEN_CHANNELS), .LAYER_NUM(i), .KERNEL_SIZE(KERNEL_SIZE), .RESAMPLE(0), .MODEL_TYPE(MODEL_TYPE)) 
                            layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset), .update(update_arr[i]), .ready(update_arr[i+1]));
        end
    end
endgenerate

// set the out reg
logic update_posedge, update_reg;
always_ff @(posedge clk) begin
    if (reset) update_reg <= 0;
    else       update_reg <= update_arr[LAYERS+1];
end
assign update_posedge = update_arr[LAYERS+1] && !update_reg;

always_ff @(posedge clk) begin //todo: theres some bug with switch here that needs to be fixed
    if (reset) begin
        out <= '0;
        switch <= 0;
    end else if (update_posedge) begin
        out <= connections[LAYERS+1][0];
        switch <= !switch;
    end
end
endmodule