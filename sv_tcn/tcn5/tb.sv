`include "tcn.sv"

module tb ();

parameter int DATA_WIDTH = 16;
parameter string MODEL_TYPE = "encoder";
parameter int TEST = 1;
localparam int K = 5;
localparam int L = 3;
localparam int D = 2;
localparam int HC = 8;

localparam BUFFER_TIME = 10;
parameter SAMPLES = 10;
localparam CLK_TIME = 3;
localparam UPDATE_TIME = ((K * HC + 1)*(CLK_TIME*2) + BUFFER_TIME); //calculates time before the next input is sent in
localparam FINISH_TIME = SAMPLES * UPDATE_TIME + (UPDATE_TIME*(L+1) + BUFFER_TIME);

string filePath = "input_time_series.mem";

logic signed [DATA_WIDTH-1:0] all_samples [SAMPLES];
logic signed [DATA_WIDTH-1:0] allOutput [SAMPLES];
//logic signed [DATA_WIDTH-1:0] desiredOut;
logic signed [DATA_WIDTH-1:0] out;
logic signed [DATA_WIDTH-1:0] currentIn;
logic track_out;
logic switch;
//logic match;

logic clk;
logic update_cycle;
logic reset;


tcn #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(K), .LAYERS(L), .HIDDEN_CHANNELS(HC), .DIALATION_BASE(D), .MODEL_TYPE(MODEL_TYPE)) 
            Network1 (.clk(clk), .in(currentIn), .out(out), .reset(reset), .update(update_cycle), .switch(switch));

initial begin
    $display("DATAWIDTH: %0d, MODELTYPE:%s, TEST:%0d", DATA_WIDTH, MODEL_TYPE, TEST);
    clk = 0;
    update_cycle = 0;
    reset = 1;
    track_out = '0;
    if (MODEL_TYPE == "decoder") begin
        filePath = "recieved_time.mem";
    end

    $readmemh(filePath, all_samples, 0, SAMPLES-1);

    //filePath = $sformatf("TestingData/Test%0d/py_output.mem", TEST);
    //$readmemh(filePath, desiredOutput, 0, SAMPLES-1);

    if (FINISH_TIME < 26634) begin
        $dumpfile("tb.vcd");
        //$dumpvars(1, tb.currentIn, tb.out);
        //$dumpvars(1, tb.clk, tb.reset, tb.update_cycle);
        $dumpvars(0, tb);
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
        update_cycle = 0;
    end
end

always @(posedge update_cycle) begin //attempting to simulate how data would come in
    for (int i = 0; i < SAMPLES-1; i++) begin
        all_samples[i] <= all_samples[i+1];
    end
    currentIn <= all_samples[0];
    samplenum++;
    //match <= (desiredOut == out) ? 1:0;
end


always @(posedge clk) begin
    if (track_out != switch && !$isunknown(out)) begin
        for (int i = 0; i < $size(allOutput)-1; i++) begin
            allOutput[i] <= allOutput[i+1];
        end
        allOutput[$size(allOutput)-1] <= out;
        track_out <= switch;
    end
end

/*logic signed [DATA_WIDTH-1:0] placeholder;
always @(posedge update_cycle) begin
    for (int i = 0; i < SAMPLES; i++) begin
        desiredOutput[i] <= desiredOutput[i+1];
    end
    placeholder <= desiredOutput[0];
    desiredOut <= placeholder;
end*/


endmodule: tb