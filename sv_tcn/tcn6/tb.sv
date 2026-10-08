`include "tcn.sv"

module tb ();

parameter int DATA_WIDTH = 16;
parameter int FRAC_BITS = $ceil(DATA_WIDTH / 2);
parameter MODEL_TYPE = "encoder";
parameter int TEST = 1;
parameter int K = 5;
parameter int L = 3;
parameter int D = 2;
parameter int HC = 8;

parameter SAMPLES = 5;

localparam BUFFER_TIME = 500;
localparam CLK_TIME = 3;
localparam UPDATE_TIME = CLK_TIME*2; //calculates time before the next input is sent in
localparam FINISH_TIME = SAMPLES * UPDATE_TIME + BUFFER_TIME;

string filePath = "input_time_series.mem";

logic signed [DATA_WIDTH-1:0] all_samples [SAMPLES];
logic signed [DATA_WIDTH-1:0] allOutput [SAMPLES];
//logic signed [DATA_WIDTH-1:0] desiredOut;
logic signed [DATA_WIDTH-1:0] out;
logic signed [DATA_WIDTH-1:0] currentIn;
logic track_out;
logic switch;
// logic match;
logic clk;
logic update_cycle;
logic reset;

tcn #(
    .TEST(TEST), 
    .DATA_WIDTH(DATA_WIDTH), 
    .FRAC_BITS(FRAC_BITS),
    .KERNEL_SIZE(K), .LAYERS(L), 
    .HIDDEN_CHANNELS(HC), 
    .DIALATION_BASE(D), 
    .MODEL_TYPE(MODEL_TYPE)
) Network1 (
    .clk(clk), 
    .in(currentIn), 
    .out(out), 
    .reset(reset),
    .data_valid_in(switch),
    .data_valid_out(track_out)
);

initial begin
    $display("DATAWIDTH: %0d, MODELTYPE:%s, TEST:%0d", DATA_WIDTH, MODEL_TYPE, TEST);
    clk = 0;
    update_cycle = 0;
    reset = 1;
    switch <= 0;
    if (MODEL_TYPE == "decoder") begin
        filePath = "recieved_time.mem";
    end

    $readmemh(filePath, all_samples, 0, SAMPLES-1);

    //filePath = $sformatf("TestingData/Test%0d/py_output.mem", TEST);
    //$readmemh(filePath, desiredOutput, 0, SAMPLES-1);

    if (FINISH_TIME < 5000) begin
        $dumpfile("tb.vcd");
        $dumpvars(1, tb.currentIn, tb.out);
        $dumpvars(1, tb.clk, tb.reset, tb.switch, tb.update_cycle, tb.Network1.yeet);
        //$dumpvars(0, tb);
    end else begin
        $display("Not generating a waveform viewer because time units exceed 26634, at %0d time units.", FINISH_TIME);
    end
    

    #(CLK_TIME+1) //after the clock goes high, pull reset low
    reset = 0;
    #(UPDATE_TIME*2)


    #FINISH_TIME

    $writememh($sformatf("%s_output.mem", MODEL_TYPE), allOutput, 0, SAMPLES-1);
    $finish;
end

always begin
    #CLK_TIME
    clk = ~clk;
end
int samplenum = 0;
always begin
    #(UPDATE_TIME/2)
    if (samplenum < SAMPLES) begin
        update_cycle = ~update_cycle;
    end else begin
        #(UPDATE_TIME)
        update_cycle = 0;
        switch = 0;
    end
end

always @(posedge update_cycle) begin //attempting to simulate how data would come in
    for (int i = 0; i < SAMPLES-1; i++) begin
        all_samples[i] <= all_samples[i+1];
    end
    currentIn <= all_samples[0];
    samplenum++;
    switch <= 1;
end


always @(posedge clk) begin
    if (track_out) begin
        for (int i = 0; i < $size(allOutput)-1; i++) begin
            allOutput[i] <= allOutput[i+1];
        end
        allOutput[$size(allOutput)-1] <= out;
    end
end

endmodule: tb