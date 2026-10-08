import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(base_pth)

import subprocess
import torch
import json
import yaml
import zarr
import matplotlib.pyplot as plt
from modules.utils import save_mem_file
from modules.models import TCN, QxxTCN
from modules.grid_search import EncoderDecoderGridSearch
from modules.grid_search.encoder_decoder import ARCH_KEYS
from modules.grid_search.adapters import TCNAdapter
from modules.utils import q88_int_to_hex, q88_hex_to_float, float_to_q88_int, evm_pct, calculate_per_burst_rrmse_pct_loss 
from experiments.test_real_gridsearch import (
    OFDM_CONFIG, ENCODER_DECODER_GRID
)
FILES = ["input_time_series.mem", "encoder_output.mem", "recieved_time.mem", "decoder_output.mem"]

sim_directory = "sv_tcn/tcn7"
DATA_WIDTH = 16
TEST = 1
SAVE_PATH = f"{sim_directory}"
ED_MODEL = "data/experiments/test_real_gridsearch/encoder_decoder_20260925_1312/runs/Qxx_tcn_efd24442"
CHANNEL_PTH = "data/experiments/test_real_gridsearch/channel_models_20260925_1310/runs/tcn_b338d2dc"
TCN_NAME = QxxTCN
DATA_PATH = "data/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr/sent_burst"
WAVEFORMS = 4
TYPE = "Synthetic"
ed_gs = EncoderDecoderGridSearch(
            ENCODER_DECODER_GRID,
            channel_models=[],
            dataset_path="nothing",
)


def create_plots(data_width, sent_time, ShowTimeSeries=False, ed_model_pth=ED_MODEL, plot_title="Qx.x"):
    py_freq = []
    with open(os.path.join(base_pth, ed_model_pth, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
        p = config["params"]

        encoder = TCN_NAME(**p).to("cpu")
        checkpoint = torch.load(os.path.join(base_pth, ed_model_pth, "model.pt"), map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()

        #ed_gs.preamble = torch.tensor(band_limited_zc_preamble(256, OFDM_CONFIG.subcarrier_spacing*OFDM_CONFIG.baseband_fft_length, float(300000.0), float(7600000.0), 3.0), dtype=torch.float32, device="cpu").unsqueeze(0)
        in_time = sent_time
        #preamble, symbol = in_time[:, :256], in_time[:, 256:]
        py_freq.append(ed_gs._frame_to_freq(in_time, OFDM_CONFIG))


        outp = encoder(in_time)
        
        py_freq.append(ed_gs._frame_to_freq(outp, OFDM_CONFIG))
        recieved_time = send_through_channel(outp)
        py_freq.append(ed_gs._frame_to_freq(recieved_time, OFDM_CONFIG))

        decoder = TCN_NAME(**p).to("cpu")
        decoder.load_state_dict(checkpoint["decoder"])
        decoder.eval()

        decoded_time = decoder(recieved_time)
        #decoded_time = _decode_freq(encoder.eval(), decoder.eval(), channel_model, eval_sent_time, ofdm_config)
        
        py_freq.append(ed_gs._frame_to_freq(decoded_time, OFDM_CONFIG))

        #print(f"input av power: {torch.square(torch.mean(in_time))}")
        #print(f"encoded av power: {torch.square(torch.mean(outp))}")
        #print(f"recieved av power: {torch.square(torch.mean(recieved_time))}")
        #print(f"decoded av power: {torch.square(torch.mean(decoded_time))}")

        #print(f"encoded average: {torch.mean(outp)}")
        #print(f"encoded max: {torch.max(outp)}")
        #print(f"encoded min: {torch.min(outp)}")
        #print(f"decoded average: {torch.mean(decoded_time)}")
        #print(f"decoded max: {torch.max(decoded_time)}")
        #print(f"decoded min: {torch.min(decoded_time)}\n")
    
    all_time_series = []
    for file in FILES:
          with open(os.path.join(base_pth, SAVE_PATH, file), "r") as file:
                time = []
                for line in file:
                    line = line.strip()
                    if line[0:2] != "//":
                        num = q88_hex_to_float(line, data_width)
                        time.append(num)
                all_time_series.append(time)

    time_tensors = [torch.tensor(all_time_series[0]).view(WAVEFORMS, 940),  torch.tensor(all_time_series[1]).view(WAVEFORMS, 940),  torch.tensor(all_time_series[2]).view(WAVEFORMS, 940), torch.tensor(all_time_series[3]).view(WAVEFORMS, 940)]

    freq_tensors = [ed_gs._frame_to_freq(time_tensors[0], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[1], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[2], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[3], OFDM_CONFIG)]
    
    evm_sv = evm_pct(freq_tensors[0], freq_tensors[3]).item()
    rrmse = calculate_per_burst_rrmse_pct_loss(freq_tensors[0], freq_tensors[3])
    print("SystemVerilog TCN Performance Metrics:")
    print(f"EVM: {evm_sv}")
    print(f"RRMSE: {rrmse}\n")

    evm_py = evm_pct(py_freq[0], py_freq[3]).item()
    rrmse = calculate_per_burst_rrmse_pct_loss(py_freq[0], py_freq[3])
    print("Pytorch TCN Performance Metrics:")
    print(f"EVM: {evm_py}")
    print(f"RRMSE: {rrmse}")

    ed_gs._plot_constellation_enhanced_for_sv(run_dir=Path(os.path.join(base_pth, SAVE_PATH)), sv=freq_tensors, py=py_freq, freqs=OFDM_CONFIG.subcarrier_freqs_hz, evm_sv=evm_sv, evm_py=evm_py, channel_type=plot_title)

    if ShowTimeSeries:
        plt.plot(all_time_series[1][0:(940 * WAVEFORMS)], label="SystemVerilog", marker='o')
        plt.plot(list(outp.detach().numpy().flatten())[0:(940 * WAVEFORMS)], label="Pytorch", marker='s')
        plt.plot(list(in_time.detach().numpy().flatten())[0:(940 * WAVEFORMS)], label="input", marker='s')

        plt.xlabel("Time Series")
        plt.ylabel("Output")
        plt.title("Plotting Input and encoded")
        plt.legend()
        plt.show()

        plt.plot(all_time_series[3][0:(940 * WAVEFORMS)], label="SystemVerilog", marker='o')
        plt.plot(list(decoded_time.detach().numpy().flatten())[0:(940 * WAVEFORMS)], label="Pytorch", marker='s')
        plt.plot(list(recieved_time.detach().numpy().flatten())[0:(940 * WAVEFORMS)], label="input", marker='s')

        plt.xlabel("Time Series")
        plt.ylabel("Output")
        plt.title("Plotting Recived and Decoded")
        plt.legend()
        plt.show()

    return evm_sv, evm_py

def read_dataset(path, n_frames: int = 64) -> torch.Tensor:
    sent_arr = zarr.open(path, mode='r')
    sent_symbols = torch.from_numpy(sent_arr[:n_frames, :]).float()
    return sent_symbols

def create_sent_time():
    if TYPE == "Synthetic":
        return ed_gs._sample_batch(WAVEFORMS, len(OFDM_CONFIG.active_carrier_indices)*2, OFDM_CONFIG, "please add the preamble")[1]
    elif TYPE == "Real":
        return read_dataset(os.path.join(base_pth, DATA_PATH), n_frames=WAVEFORMS)
    elif TYPE == "Step":
        length = 940
        # 4 cycles (Frequency x4)
        single_waveform = (((torch.arange(length) * 2 * 1) // length) % 2 * 2.0 - 1.0).to(torch.float32)
        return single_waveform.repeat(WAVEFORMS, 1)
    elif TYPE == "Random Step":
        length = 940
        values = torch.tensor([0.0, 1.0, -1.0, 2.0, -3.0, 3.0], dtype=torch.float32)
        time_series = torch.empty((WAVEFORMS, length), dtype=torch.float32)

        for burst_idx in range(WAVEFORMS):
            position = 0
            while position < length:
                remaining = length - position
                min_step = min(80, remaining)
                max_step = min(300, remaining)
                step_length = int(torch.randint(min_step, max_step + 1, ()).item())
                value = values[torch.randint(0, len(values), ())].item()
                time_series[burst_idx, position:position + step_length] = value
                position += step_length

        return time_series
    else:
        raise ValueError("TYPE was not correctly specified.")

def writeSentTime(save_pth, datawidth):
    sent_time = create_sent_time()
    time_series = []
    tensor_time_series = sent_time.detach().cpu().numpy()
    for burst in tensor_time_series:    
            for point in burst:
                time_series.append(q88_int_to_hex(float_to_q88_int(point, data_width=datawidth), data_width=datawidth))
    save_mem_file(os.path.join(base_pth, save_pth, "input_time_series.mem"), time_series, "OFDM modulated time series - includes cyclic prefix and preamble")
    return sent_time

def read_model(read_pth, save_pth, datawidth):
    output_dir = os.path.join(base_pth, save_pth)
    os.makedirs(output_dir, exist_ok=True)
    model = torch.load(os.path.join(base_pth, read_pth), weights_only=True)

    for type in ["encoder", "decoder"]:
        e_or_d = model[type]
        keys = e_or_d.copy().keys()
        #print(f"{type} keys: {keys}")

        #pre processing step
        bn_list = []
        for key in keys:
            if "bn" in key:
                if key.endswith("num_batches_tracked"):
                    e_or_d.pop(key)
                else:
                    bn_list.append(key)
            if len(bn_list) == 4:
                gamma, beta, mu, var = [e_or_d[s] for s in bn_list]
                eps = 1e-05
                bn_weight = gamma / torch.sqrt(var + eps)
                bn_bias = beta - (bn_weight * mu)
                e_or_d[bn_list[0]] = bn_weight
                e_or_d[bn_list[1]] = bn_bias
                e_or_d.pop(bn_list[2])
                e_or_d.pop(bn_list[3])
                bn_list = []

        keys = e_or_d.keys()
        #print(f"{type} keys: {keys}")
        for key in keys:
            #print(f"Keyname: {key}")

            tensor = e_or_d[key]
            #print(tensor)

            for i, hidden_channel in enumerate(tensor):
                #print(f"    Channel{i}:")
                #print(hidden_channel)

                arr = hidden_channel.detach().cpu().numpy().flatten()
                save_arr = []
                for num in arr:
                    hexa = q88_int_to_hex(float_to_q88_int(num, datawidth), datawidth)
                    #print(f"        Weight/Bias: {num} Hex: {hexa}")
                    save_arr.append(hexa)

                safe_key = key.replace('.', '_')
                save_mem_file(os.path.join(output_dir, f"{type}", safe_key, f"channel{i}.mem"), save_arr, f"contains {len(arr)} values", printDebug=False)

def send_through_channel(sent_time):
    with open(os.path.join(base_pth, CHANNEL_PTH, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)
            params = config["params"]
            #print(f"Model config: {params}")
            adapter = TCNAdapter.load(params=params, checkpoint=os.path.join(base_pth, CHANNEL_PTH, "model.pt"), device="cpu")
            rec_time_tensor = adapter.model(sent_time)
            #print(rec_time_tensor)
            return rec_time_tensor[0] # 0 for noise, 1 for mean

def Execute_Channel(data_width, Channel_pth, Save_pth):
    with open(os.path.join(base_pth, Channel_pth, "config.yaml"), "r") as file:
        sent_time = [[]]
        with open(os.path.join(base_pth, Save_pth, "encoder_output.mem"), "r") as file:
            for line in file:
                line = line.strip()
                if line[0:2] != "//":
                    num = q88_hex_to_float(line, data_width)
                    sent_time[0].append(num)
        recieved_time = send_through_channel(torch.tensor(sent_time)).detach().cpu().numpy().flatten()
        words = []
        for sample in recieved_time:
            words.append(q88_int_to_hex(float_to_q88_int(sample, data_width), data_width))
        save_mem_file(os.path.join(base_pth, Save_pth, "recieved_time.mem"), words, f"Sent through the channel model")

if __name__ == "__main__":
    sent_time = writeSentTime(SAVE_PATH, DATA_WIDTH)
    read_model(os.path.join(base_pth, ED_MODEL, "model.pt"), os.path.join(base_pth, SAVE_PATH, "TestingData", f"Test{TEST}"), DATA_WIDTH)

    print("Running encoder...")
    result = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="encoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-P", f"tb.SAMPLES={940 * WAVEFORMS}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    result2 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    
    print("Running Channel Model...")
    Execute_Channel(DATA_WIDTH, CHANNEL_PTH, SAVE_PATH)
    
    print("Running decoder...")
    result3 = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="decoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-P", f"tb.SAMPLES={940 * WAVEFORMS}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    result4 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )

    print("Complete.\n")
    create_plots(DATA_WIDTH, sent_time, ShowTimeSeries=False, ed_model_pth=os.path.join(base_pth, ED_MODEL))