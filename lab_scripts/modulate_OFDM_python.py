import sys
module_dir = r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled'
if module_dir not in sys.path:
    sys.path.append(module_dir)
from lab_scripts.constellation_diagram import QPSK_Constellation
from lab_scripts.constellation_diagram import RingShapedConstellation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample_poly
from scipy import signal
from scipy.signal import find_peaks
import os
from training_state import STATE
import torch

# Get logging
from lab_scripts.logging_code import *

decode_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\decode_bits_log.txt")


def get_constellation(mode: str):
        if mode == "qpsk":
            constellation = QPSK_Constellation()
        elif mode == "m5_apsk_constellation":
            constellation = RingShapedConstellation(filename=r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\lab_scripts\saved_constellations\m5_apsk_constellation.npy')
        return constellation

def modulate_data_OFDM(mode: str, 
                       num_carriers: int, 
                       data: list[int], 
                       cyclic_prefix_length: int,
                       f_min: float,
                       f_max: float,
                       subcarrier_delta_f: float) -> tuple[list, list]:
    '''Creates and n x m symbol matrix to fill frequency band for OFDM
    
        Args:
            f_min: minumum frequency (Hz) 
            f_max: maximum frequency (Hz)
            mode: a string to determine how modulation occurs. e.g qpsk
            data: list of bits of 1, 0 to modulate
        
        Outputs:
            Returns a tuple containing two lists representing two 2d arrays
            of Real{Symbols} and Imaginary{Symbols}

            Each row is an indepedent OFDM frame to be transmitted    
    '''
    bits = "".join([str(x) for x in data])

    k_min = int(np.floor(f_min / subcarrier_delta_f))

    N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min)
    
    # Grab constellation object
    constellation = get_constellation(mode)
    
    encoded_symbols = constellation.bits_to_symbols(bits)
    # decode_logger.debug(f"Encoded symbols: {encoded_symbols}")
    true_bits = np.array(data)


    # Calculate frame length
    frame_length = N_data
    # Reshape such that each row is a frame to be transmitted
    if len(encoded_symbols) % frame_length != 0:
        # Padd for full frame
        zeros_to_add = frame_length - len(encoded_symbols) % frame_length
        padding_symbols = np.repeat(constellation.bits_to_symbols("00"), zeros_to_add)
        bits_added = np.array(list(constellation.symbols_to_bits(padding_symbols)))
        encoded_symbols = np.hstack((encoded_symbols, padding_symbols))
        true_bits = np.hstack((true_bits, bits_added))

    encoded_symbol_frames = encoded_symbols.reshape(-1, frame_length)

    # Add zeros on front equal to k_min 
    encoded_symbol_frames = np.hstack((np.zeros((encoded_symbol_frames.shape[0], k_min)), encoded_symbol_frames))
    encoded_symbol_frames_real = encoded_symbol_frames.real.copy()
    encoded_symbol_frames_imag = encoded_symbol_frames.imag.copy()


    # Reshape true bits by frame
    number_of_frames, _ = encoded_symbol_frames.shape
    grouped_true_bits = true_bits.reshape(number_of_frames, -1)

    grouped_true_bits_list = [[str(element) for element in row] for row in grouped_true_bits.tolist()]

    # Return as real and imaginary parts
    return encoded_symbol_frames_real, encoded_symbol_frames_imag, grouped_true_bits_list, int(N_data)


AWG_MEMORY_LENGTH = 16384
# AWG_MEMORY_LENGTH = 105

BARKER_LENGTH = int(0.01 * (AWG_MEMORY_LENGTH)) if int(0.01 * (AWG_MEMORY_LENGTH)) > 5 else 5
def symbols_to_xt(real_symbol_groups: list[list[float]], imag_symbol_groups: list[list[float]], cyclic_prefix_length: int) -> list[float]:
    '''Takes a symbol frame matrix and converts and x(t) frame matrix

    
        Args:
            symbol_groups: n x m matrix where n is number of frames and m in number of data carrier symbols

        Outputs:
            returns n x AWG_MEMORY_LENGTH matrix and the preable used for later correlation

    '''
    symbol_groups = np.array(real_symbol_groups) + np.array(imag_symbol_groups) * 1j
    barker_code = np.array([1, -1, 1, -1, 1], dtype=float)
    barker_code = np.repeat(barker_code, BARKER_LENGTH // len(barker_code)) # Set as 1%
    IFFT_LENGTH = int(AWG_MEMORY_LENGTH - len(barker_code))

    # Add DC offset, Nyquist carrier, and conjugate symmetry
    symbol_groups_conjugate_flipped = np.conj(symbol_groups)[:, ::-1] # Flip along axis=1
    DC_nyquist = np.zeros(shape=(symbol_groups.shape[0], 1)) # Both are set to 0
    full_symbols = np.hstack((DC_nyquist, symbol_groups, DC_nyquist, symbol_groups_conjugate_flipped))

    # Take ifft with time interpolation
    x_t_groups = np.real(np.fft.ifft(full_symbols, axis=1, n=IFFT_LENGTH))

    # Normalize x_t with epsilon to avoid division by zero
    epsilon = 1e-12  # Small value to prevent division by zero
    max_values = np.max(x_t_groups, axis=1, keepdims=True)
    x_t_groups = x_t_groups / (max_values + epsilon)

    # Add on preamble to each frame
    barker_code_grouped = np.tile(barker_code, (len(x_t_groups), 1))
    x_t_groups_with_preamble = np.hstack((barker_code_grouped, x_t_groups)).astype(float)
    x_t_groups_with_preamble = np.ascontiguousarray(x_t_groups_with_preamble)

    return x_t_groups_with_preamble.tolist(), barker_code




def demodulate_OFDM_one_symbol_frame(y_t:list,
                                     num_carriers: int,
                                     CP_length: int,
                                     freq_AWG: float,
                                     osc_sample_rate: float,
                                     record_length: int,
                                     preamble_sequence: list,
                                     mode: str,
                                     f_min: float,
                                     f_max: float,
                                     subcarrier_delta_f: float) -> list:
    '''Converts received y(t) into a bit string with optional debugging plots'''

    debug_plots = True

    # Define paths for saving logs and plots
    log_dir = r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'demodulation_log.txt')
    plot_file = os.path.join(log_dir, 'demodulation_plots.png')

    k_min = int(np.floor(f_min / subcarrier_delta_f))

    N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min)

    # Open log file for writing
    with open(log_file, 'w') as log:
        log.write("Demodulation Log\n")
        log.write("================\n")
        
        constellation = get_constellation(mode)

        if debug_plots:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Original received signal
            plt.subplot(321)
            plt.plot(y_t)
            plt.title(f'Original Received Signal y(t) Length: {len(y_t)}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

        # Upsample the preamble
        preamble = np.array(preamble_sequence)
        voltages = y_t
        time_OFDM_frame = 1 / freq_AWG
        time_preamble = (len(preamble) / (AWG_MEMORY_LENGTH)) * time_OFDM_frame
        num_points_preamble = osc_sample_rate * time_preamble
        num_points_frame = osc_sample_rate * time_OFDM_frame
        preamble_sequence_upsampled = resample_poly(preamble, up=int(num_points_preamble), down=len(preamble))

        # Correlation and peak detection
        corr = signal.correlate(voltages, preamble_sequence_upsampled, mode='valid')
        peaks, _ = find_peaks(corr, height=0.99*np.max(corr), distance=len(preamble_sequence_upsampled))

        log.write(f"Number of peaks detected: {len(peaks)}\n")
        log.write(f"Peaks: {peaks}\n")
    
        if debug_plots:
            # Plot 2: Resampled signal
            plt.subplot(322)
            plt.plot(voltages)
            plt.title(f'Resampled Signal Length: {len(voltages)}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

        if debug_plots:
            # Plot 3: Correlation output
            plt.subplot(323)
            plt.plot(corr)
            plt.title('Correlation with Preamble')
            plt.xlabel('Sample')
            plt.ylabel('Correlation')
                    # Add peak labels
            for i, peak in enumerate(peaks):
                plt.plot(peak, corr[peak], 'r^')  # Red triangle marker
                plt.annotate(f'Peak {i+1}\n({peak})', 
                            xy=(peak, corr[peak]),
                            xytext=(10, 10),
                            textcoords='offset points',
                            ha='left',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Parameters and frame extraction

        preamble_length = len(preamble_sequence_upsampled)
        ofdm_payload_length = int(num_points_frame - preamble_length)
        frames = []
    
        for peak in peaks:
            frame_start = peak + preamble_length
            frame_end = frame_start + ofdm_payload_length
            frame = voltages[frame_start:frame_end]
            frames.append(frame)

        if debug_plots and len(frames) > 0:
            # Plot 4: First extracted frame
            plt.subplot(324)
            plt.plot(frames[0])
            plt.title(f'First Extracted Frame Length: {len(frames[0])}')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')

        if len(frames) == 0:
            log.write("No valid frames detected\n")
            raise ValueError("No valid frames detected")
        

        # decode_logger.debug(f"Number of frames extracted: {len(frames)}\n")

        frame_y_t = frames[0]

        # Perform FFT with proper normalization
        Y_s = np.fft.fft(frame_y_t)

        if debug_plots:
            plt.subplot(325)
            limited_magnitude = np.abs(Y_s)

            # Plot the FFT magnitude
            plt.plot(limited_magnitude)
            plt.title('FFT Magnitude')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')

        # Extract both positive and negative frequency carriers
        num_data_carriers = N_data
        negative_carriers = Y_s[k_min + 1:num_data_carriers + k_min + 1]
        positive_carriers = Y_s[-num_data_carriers -k_min - 1: -k_min -1]
        data_subcarriers = negative_carriers
        data_subcarriers = data_subcarriers * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(data_subcarriers)))
        # decode_logger.debug(f"Received Positive Carriers: {data_subcarriers}")

        negative_carriers_t = Y_s[:2 * num_data_carriers]
        positive_carriers_t = Y_s[-2 * num_data_carriers:]

        negative_carriers_t = negative_carriers_t * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(negative_carriers_t)))
        positive_carriers_t = positive_carriers_t * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(positive_carriers_t)))

        # decode_logger.debug(f"Y pos: {positive_carriers_t}\n")
        # decode_logger.debug(f"Y neg: {negative_carriers_t}\n")


        if debug_plots:
            # Plot 6: Constellation Diagram

            # Calculate frequencies
            frequencies = np.arange(f_min+ subcarrier_delta_f, 
                                    f_min + subcarrier_delta_f * num_data_carriers + subcarrier_delta_f, 
                                    subcarrier_delta_f)

            normalized_frequencies = (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
            colors = plt.cm.viridis(normalized_frequencies)

            plt.subplot(326)
            scatter = plt.scatter(data_subcarriers.real, data_subcarriers.imag, c=frequencies, cmap='viridis', label='Received')
            constellation = np.array(list(constellation._symbols_to_bits_map.keys()))
            
            plt.scatter(constellation.real, constellation.imag, c='black', marker='x', label='Ideal')
            plt.title('Constellation Diagram')
            plt.xlabel('Real')
            plt.ylabel('Imaginary')
            plt.legend()
            plt.grid(True)

            cbar = plt.colorbar(scatter)
            cbar.set_label('Carrier Frequency (Hz)')

            plt.tight_layout()
            plt.show()
            plt.savefig(plot_file)
            plt.close()

    return data_subcarriers.real.tolist(), data_subcarriers.imag.tolist()


from encoder_decoder import update_weights
def decode_symbols_OFDM(real_symbols: list, imag_symbols: list, true_bits: list,  mode: str) -> list:
    # Create constellation for demodulation


    

    # Instead of grabbing from labview, directly take from decoder
    if 'decoder_out' in STATE:
        symbols = STATE['decoder_out']
    else:
        symbols = torch.tensor(real_symbols) + 1j * torch.tensor(imag_symbols)


    # Grab constellation object
    constellation = get_constellation(mode)

    # symbols = np.array(symbols)
    # Demap symbols to bits
    constellation_symbols = torch.tensor(
        list(constellation._symbols_to_bits_map.keys()),
        dtype=symbols.dtype,
        device=symbols.device
    )
    distances = abs(symbols.reshape(-1, 1) - constellation_symbols.reshape(1, -1))



    closest_idx = distances.argmin(axis=1)
    # decisions = constellation_symbols[closest_idx]
    # decided_bits = [constellation._symbols_to_bits_map[complex(symbol.real.item(), symbol.imag.item())] 
    #                 for symbol in decisions]


    constellation_symbols_list = list(constellation._symbols_to_bits_map.keys())
    decided_bits = [constellation._symbols_to_bits_map[constellation_symbols_list[idx]] for idx in closest_idx.cpu().numpy()]

    # Flatten decided bits into a 1D array
    decided_bits_flat = [int(bit) for symbol_bits in decided_bits for bit in symbol_bits]

    # Flatten true bits into a 1D array
    true_bits_flat = [int(bit) for symbols_bits in true_bits for bit in symbols_bits]

    # Convert to NumPy arrays for comparison
    true_bits_array = np.array(true_bits_flat, dtype=np.int32)
    decided_bits_flat_array = np.array(decided_bits_flat, dtype=np.int32)

    # Take minimum length to avoid shape mismatch
    min_len = min(len(true_bits_array), len(decided_bits_flat_array))
    true_bits_array = true_bits_array[:min_len]
    decided_bits_flat_array = decided_bits_flat_array[:min_len]

    # decode_logger.debug(f"True bit array: \n {true_bits_array} \n Decided bits array: \n {decided_bits_flat_array}")

    # Calculate BER
    BER = float(np.sum(true_bits_array != decided_bits_flat_array) / len(true_bits_array))

    # Log frame BER
    STATE['frame_BER'] = BER

       # Call backprop and log loss
    if 'encoder_out' in STATE and 'decoder_out' in STATE:
        update_weights()

    SNR, PowerFactor = float(0), float(0) 
    return decided_bits_flat, float(BER), SNR, PowerFactor