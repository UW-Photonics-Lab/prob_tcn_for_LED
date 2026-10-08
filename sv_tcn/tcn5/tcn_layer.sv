`include "hidden_channel_block.sv"

module tcn_layer # (parameter TEST, DATA_WIDTH, DIALATION, IN_CH, OUT_CH, LAYER_NUM, KERNEL_SIZE, RESAMPLE, parameter [0:55] MODEL_TYPE) (
    input signed [IN_CH-1:0][DATA_WIDTH-1:0] in,
    output logic signed [OUT_CH-1:0][DATA_WIDTH-1:0] out,
    input clk, reset, update,
    output wire ready
);
//logic signed [DATA_WIDTH-1:0] input_reg [INPUT_SAMPLES + (KERNEL_SIZE-1)*DIALATION];
logic signed [0:(KERNEL_SIZE-1)*DIALATION][IN_CH-1:0][DATA_WIDTH-1:0] input_reg;
//logic [KERNEL_SIZE:0] counter;
logic [KERNEL_SIZE*IN_CH:0] counter;
logic ready_reg;

// handle the posegde of update
logic update_reg, update_posedge;
always_ff @(posedge clk) begin
    if (reset) update_reg <= 1'b0;
    else    update_reg <= update; 
end
assign update_posedge = update && !update_reg;

//manage the input stream in a buffer
always_ff @(posedge clk) begin
    if (reset) begin
        input_reg <= '0;
        counter <= '0;
    end else if (update_posedge) begin
        //$display("Layer %0d UPDATED INPUT --------------------", LAYER_NUM);
        counter[0] <= 1'b1;
        for (int i = 1; i < $size(input_reg); i++) begin
            input_reg[i] <= input_reg[i-1];
        end
        input_reg[0] <= in;
    end else begin
        if (counter[KERNEL_SIZE*IN_CH] == 1'b1) begin
            //$display("Layer %0d IS DONE ----------------------", LAYER_NUM);
        end
        ready_reg <= counter[KERNEL_SIZE*IN_CH];
        for (int i = 1; i < $size(counter); i++) begin
            counter[i] <= counter[i-1];
        end
        counter[0] <= 1'b0;
    end
end
//wire up hidden channels - Each hidden_channel_block gives you 1 output
logic signed [0:KERNEL_SIZE*IN_CH-1][DATA_WIDTH-1:0] actual_input_reg;
logic [OUT_CH-1:0][DATA_WIDTH-1:0] hc_output;

// flatten the input and exclude time steps ignored due to DIALATION
// generate
//     for (genvar i = 0; i < (KERNEL_SIZE-1)*DIALATION+1; i=i+DIALATION) begin: timeSeries
//         for (genvar j = 0; j < IN_CH; j++) begin: channelSeries
//             assign actual_input_reg[((i/DIALATION)*IN_CH) + j * ] = input_reg[i][j];
//         end
//     end
// endgenerate

generate
    for (genvar i = 0; i < KERNEL_SIZE; i++) begin: timeSeries
        for (genvar j = 0; j < IN_CH; j++) begin: channelSeries
            assign actual_input_reg[j * KERNEL_SIZE + i] = input_reg[(KERNEL_SIZE-1-i)*DIALATION][j];
        end
    end
endgenerate

generate
    for (genvar i = 0; i < OUT_CH; i++) begin: HC_inst

        hidden_channel_block #(.NUM_TAPS(KERNEL_SIZE*IN_CH), .DATA_WIDTH(DATA_WIDTH), .SKIPCONN(KERNEL_SIZE*i + (KERNEL_SIZE-1)), .TEST(TEST), .LAYER_NUM(LAYER_NUM), .HIDDEN_CH_NUM(i), .RESAMPLE(RESAMPLE), .MODEL_TYPE(MODEL_TYPE)) 
                hc (.counter(counter), .input_reg(actual_input_reg), .clk(clk), .reset(reset), .out(hc_output[i]));
            
    end
endgenerate

assign ready = ready_reg;
assign out = (ready)? hc_output: '0;
endmodule