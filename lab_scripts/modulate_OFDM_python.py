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
    # No changes required till here -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    bits = "".join([str(x) for x in data]) # Takes the list of bits and concatenates them into one long string

    k_min = int(np.floor(f_min / subcarrier_delta_f)) # No of subcarriers zeropadded

    N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min) # No of data subcarriers
    
    # Grab constellation object
    constellation = get_constellation(mode)
    
    encoded_symbols = constellation.bits_to_symbols(bits)
    # decode_logger.debug(f"Encoded symbols: {encoded_symbols}")
    true_bits = np.array(data) # To compare after demodulation

    # Calculate frame length
    frame_length = N_data # Number of data subcarriers per OFDM symbol
    
    # Reshape such that each row is a frame to be transmitted
    if len(encoded_symbols) % frame_length != 0: # Checking if the number of symbols obtained from mapping the data = Num of data subcarriers - all OFDM symbols combined
        # Padd for full frame
        zeros_to_add = frame_length - (len(encoded_symbols) % frame_length) # Adding zeros to accommodate the rest data subcarriers
        padding_symbols = np.repeat(constellation.bits_to_symbols("00"), zeros_to_add) # Padding with zeros
        bits_added = np.array(list(constellation.symbols_to_bits(padding_symbols))) # saving the record of zero padded bits
        encoded_symbols = np.hstack((encoded_symbols, padding_symbols)) # Updating the encoded symbols with the new zero padded symbols
        true_bits = np.hstack((true_bits, bits_added)) # Updating the bits - for BER comparison after demodulation

    encoded_symbol_frames = encoded_symbols.reshape(-1, frame_length) # Converting 1D array into 2D with one side - number of data subcarriers and the other - number of OFDM frames

    # Add zeros on front equal to k_min 
    encoded_symbol_frames = np.hstack((np.zeros((encoded_symbol_frames.shape[0], k_min)), encoded_symbol_frames)) # Stacking the zeros corresponding to the zero subcarriers
    encoded_symbol_frames_real = encoded_symbol_frames.real.copy() # Complete 2D array - real
    encoded_symbol_frames_imag = encoded_symbol_frames.imag.copy() # Complete 2D array - imaginary


    # Reshape true bits by frame
    number_of_frames, _ = encoded_symbol_frames.shape # No of OFDM symbols
    grouped_true_bits = true_bits.reshape(number_of_frames, -1) # 1D array of bits --> 2D according to the number of OFDM symbols

    grouped_true_bits_list = [[str(element) for element in row] for row in grouped_true_bits.tolist()] # Converts the NumPy 2D array --> regular Python list of lists of integers -  for comparison at the reciver

    # Return as real and imaginary parts
    return encoded_symbol_frames_real, encoded_symbol_frames_imag, grouped_true_bits_list, int(N_data) # Returning the final 2D array symbols - real, imaginary, 2D bits grid, No of data subcarriers

    # In the above code - eventhough every row represents an OFDM frame, at a time only one is sent with a preamble - 
    # encoded_symbols --> each row --> zeros corresponding to the intial subcarriers and then the bits of data subcarriers

#----------------------------------------Done till here --------------------------------------------- 

#--------------------------------------- Replace the below with alternate 0s and 1s preamble

# AWG_MEMORY_LENGTH = 16384
# # AWG_MEMORY_LENGTH = 105

# BARKER_LENGTH = int(0.01 * (AWG_MEMORY_LENGTH)) if int(0.01 * (AWG_MEMORY_LENGTH)) > 5 else 5
# def symbols_to_xt(real_symbol_groups: list[list[float]], imag_symbol_groups: list[list[float]], cyclic_prefix_length: int) -> list[float]:
#     '''Takes a symbol frame matrix and converts and x(t) frame matrix

    
#         Args:
#             symbol_groups: n x m matrix where n is number of frames and m in number of data carrier symbols

#         Outputs:
#             returns n x AWG_MEMORY_LENGTH matrix and the preamble used for later correlation

#     '''
#     symbol_groups = np.array(real_symbol_groups) + np.array(imag_symbol_groups) * 1j
#     barker_code = np.array([1, -1, 1, -1, 1], dtype=float)
#     barker_code = np.repeat(barker_code, BARKER_LENGTH // len(barker_code)) # Set as 1%
#     IFFT_LENGTH = int(AWG_MEMORY_LENGTH - len(barker_code))

#     # Add DC offset, Nyquist carrier, and conjugate symmetry
#     symbol_groups_conjugate_flipped = np.conj(symbol_groups)[:, ::-1] # Flip along axis=1
#     DC_nyquist = np.zeros(shape=(symbol_groups.shape[0], 1)) # Both are set to 0
#     full_symbols = np.hstack((DC_nyquist, symbol_groups, DC_nyquist, symbol_groups_conjugate_flipped))

#     # Take ifft with time interpolation
#     x_t_groups = np.real(np.fft.ifft(full_symbols, axis=1, n=IFFT_LENGTH))

#     # Normalize x_t with epsilon to avoid division by zero
#     epsilon = 1e-12  # Small value to prevent division by zero
#     max_values = np.max(x_t_groups, axis=1, keepdims=True)
#     x_t_groups = x_t_groups / (max_values + epsilon)

#     # Add on preamble to each frame
#     barker_code_grouped = np.tile(barker_code, (len(x_t_groups), 1))
#     x_t_groups_with_preamble = np.hstack((barker_code_grouped, x_t_groups)).astype(float)
#     x_t_groups_with_preamble = np.ascontiguousarray(x_t_groups_with_preamble)

#     return x_t_groups_with_preamble.tolist(), barker_code

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################################### NEW #################### Additionally returns - N - FFT point to apply at the rx

def symbols_to_xt( 
    real_symbol_groups: list[list[float]],
    imag_symbol_groups: list[list[float]],
    cyclic_prefix_length: int
) -> tuple[list[list[float]], list[float]]:
    """
    Convert frequency-domain symbol groups into time-domain OFDM waveforms
    with an OFDM-symbol preamble constructed inline.

    Args:
      - real_symbol_groups: list of n lists, each length = N_data
      - imag_symbol_groups: same shape, imaginary parts
      - cyclic_prefix_length: CP length in samples for preamble

    Returns:
      - waveforms: list of n lists, each = [preamble_with_CP, data_symbol_time]
      - preamble_td: list of floats (time-domain preamble with CP)
    """
    # Combine real and imaginary into complex symbols
    symbol_groups = np.array(real_symbol_groups) + 1j * np.array(imag_symbol_groups)
    n_frames, N_data = symbol_groups.shape # N_data --> K_min zeros + symbols of datasubcarriers

    # Total OFDM size (N-point IFFT)
    N = 2 * N_data + 2  # DC + N_data + Nyquist + mirrored N_data
    
    # Prepare output list
    waveforms = []

                # --- Inline OFDM preamble generation ---
    # Use a fixed 64-point FFT preamble of 31 alternating bits
    # 1. Define 31-bit alternating pattern
    preamble_bits = np.array([0, 1] * 16)[:31]  # [0,1,0,1,...] length 31
    # 2. BPSK map: 0->e^{j*pi/2}, 1->e^{j*3*pi/2}
    preamble_symbols = np.exp(1j * (np.pi/2 + np.pi * preamble_bits)) # Check if this changes anything while computing the scaling factor
    # 3. Build full frequency-domain vector: DC + data + Nyquist + mirror
    dc = np.zeros(1, dtype=complex)
    mirror = np.conj(preamble_symbols)[::-1]
    preamble_fd = np.hstack((dc, preamble_symbols, dc, mirror))  # length = 64
    # 4. IFFT to time domain (64-point)
    preamble_td_no_cp = np.real(np.fft.ifft(preamble_fd, n=64))
    # 5. Add cyclic prefix (if desired, keep same cp as data)
    cp_section = preamble_td_no_cp[-16:]
    preamble_td = np.concatenate((cp_section, preamble_td_no_cp)).astype(float)

    # --- Process each OFDM frame --- 
    for frame in symbol_groups:
        # 1. Build full FD vector for data: DC, data, Nyquist, mirror
        dc = np.zeros(1, dtype=complex)
        mirror = np.conj(frame)[::-1]
        full_fd = np.hstack((dc, frame, dc, mirror))  # length = N

        # 2. IFFT to time domain (no CP for data)
        td = np.real(np.fft.ifft(full_fd, n=N))

        # 3. Prepend the preamble
        waveform = np.concatenate((preamble_td, td)).tolist()
        waveforms.append(waveform)

    return waveforms, preamble_td.tolist(), N # Returning the time domain waveforms (row - preamble + OFDM Frame), preamble, 
                                              # N - point IFFT - to apply same point FFT at the receiver


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

############################################################################### OLD ##################################################################################################

# def demodulate_OFDM_one_symbol_frame(y_t:list,
#                                      num_carriers: int,
#                                      CP_length: int,
#                                      freq_AWG: float,
#                                      osc_sample_rate: float,
#                                      record_length: int,
#                                      preamble_sequence: list,
#                                      mode: str,
#                                      f_min: float,
#                                      f_max: float,
#                                      subcarrier_delta_f: float) -> list:
#     '''Converts received y(t) into a bit string with optional debugging plots'''

#     debug_plots = False

#     # Define paths for saving logs and plots
#     log_dir = r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs'
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, 'demodulation_log.txt')
#     plot_file = os.path.join(log_dir, 'demodulation_plots.png')

#     k_min = int(np.floor(f_min / subcarrier_delta_f))

#     N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min)

#     # Open log file for writing
#     with open(log_file, 'w') as log:
#         log.write("Demodulation Log\n")
#         log.write("================\n")
        
#         constellation = get_constellation(mode)

#         if debug_plots:
#             plt.figure(figsize=(15, 10))
            
#             # Plot 1: Original received signal
#             plt.subplot(321)
#             plt.plot(y_t)
#             plt.title(f'Original Received Signal y(t) Length: {len(y_t)}')
#             plt.xlabel('Sample')
#             plt.ylabel('Amplitude')

#         # Upsample the preamble
#         preamble = np.array(preamble_sequence)
#         voltages = y_t
#         time_OFDM_frame = 1 / freq_AWG
#         time_preamble = (len(preamble) / (AWG_MEMORY_LENGTH)) * time_OFDM_frame
#         num_points_preamble = osc_sample_rate * time_preamble
#         num_points_frame = osc_sample_rate * time_OFDM_frame
#         preamble_sequence_upsampled = resample_poly(preamble, up=int(num_points_preamble), down=len(preamble))

#         # Correlation and peak detection
#         corr = signal.correlate(voltages, preamble_sequence_upsampled, mode='valid')
#         peaks, _ = find_peaks(corr, height=0.99*np.max(corr), distance=len(preamble_sequence_upsampled))

#         log.write(f"Number of peaks detected: {len(peaks)}\n")
#         log.write(f"Peaks: {peaks}\n")
    
#         if debug_plots:
#             # Plot 2: Resampled signal
#             plt.subplot(322)
#             plt.plot(voltages)
#             plt.title(f'Resampled Signal Length: {len(voltages)}')
#             plt.xlabel('Sample')
#             plt.ylabel('Amplitude')

#         if debug_plots:
#             # Plot 3: Correlation output
#             plt.subplot(323)
#             plt.plot(corr)
#             plt.title('Correlation with Preamble')
#             plt.xlabel('Sample')
#             plt.ylabel('Correlation')
#                     # Add peak labels
#             for i, peak in enumerate(peaks):
#                 plt.plot(peak, corr[peak], 'r^')  # Red triangle marker
#                 plt.annotate(f'Peak {i+1}\n({peak})', 
#                             xy=(peak, corr[peak]),
#                             xytext=(10, 10),
#                             textcoords='offset points',
#                             ha='left',
#                             va='bottom',
#                             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

#         # Parameters and frame extraction

#         preamble_length = len(preamble_sequence_upsampled)
#         ofdm_payload_length = int(num_points_frame - preamble_length)
#         frames = []
    
#         for peak in peaks:
#             frame_start = peak + preamble_length
#             frame_end = frame_start + ofdm_payload_length
#             frame = voltages[frame_start:frame_end]
#             frames.append(frame) # All the copies are stored

#         if debug_plots and len(frames) > 0:
#             # Plot 4: First extracted frame
#             plt.subplot(324)
#             plt.plot(frames[0])
#             plt.title(f'First Extracted Frame Length: {len(frames[0])}')
#             plt.xlabel('Sample')
#             plt.ylabel('Amplitude')

#         if len(frames) == 0:
#             log.write("No valid frames detected\n")
#             raise ValueError("No valid frames detected")
        

#         # decode_logger.debug(f"Number of frames extracted: {len(frames)}\n")

#         frame_y_t = frames[0] # Considering the first copy

#         # Perform FFT with proper normalization
#         Y_s = np.fft.fft(frame_y_t)

#         if debug_plots:
#             plt.subplot(325)
#             limited_magnitude = np.abs(Y_s)

#             # Plot the FFT magnitude
#             plt.plot(limited_magnitude)
#             plt.title('FFT Magnitude')
#             plt.xlabel('Frequency (Hz)')
#             plt.ylabel('Magnitude')

#         # Extract both positive and negative frequency carriers
#         num_data_carriers = N_data
#         negative_carriers = Y_s[k_min + 1:num_data_carriers + k_min + 1]
#         positive_carriers = Y_s[-num_data_carriers -k_min - 1: -k_min -1]
#         data_subcarriers = negative_carriers #### Will have to ask why? 
#         data_subcarriers = data_subcarriers * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(data_subcarriers)))
#         # decode_logger.debug(f"Received Positive Carriers: {data_subcarriers}")

#         negative_carriers_t = Y_s[:2 * num_data_carriers]
#         positive_carriers_t = Y_s[-2 * num_data_carriers:]

#         negative_carriers_t = negative_carriers_t * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(negative_carriers_t)))
#         positive_carriers_t = positive_carriers_t * (np.max(np.abs(constellation._complex_symbols)) / np.max(np.abs(positive_carriers_t)))

#         # decode_logger.debug(f"Y pos: {positive_carriers_t}\n")
#         # decode_logger.debug(f"Y neg: {negative_carriers_t}\n")


#         if debug_plots:
#             # Plot 6: Constellation Diagram

#             # Calculate frequencies
#             frequencies = np.arange(f_min+ subcarrier_delta_f, 
#                                     f_min + subcarrier_delta_f * num_data_carriers + subcarrier_delta_f, 
#                                     subcarrier_delta_f)

#             normalized_frequencies = (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
#             colors = plt.cm.viridis(normalized_frequencies)

#             plt.subplot(326)
#             scatter = plt.scatter(data_subcarriers.real, data_subcarriers.imag, c=frequencies, cmap='viridis', label='Received')
#             constellation = np.array(list(constellation._symbols_to_bits_map.keys()))
            
#             plt.scatter(constellation.real, constellation.imag, c='black', marker='x', label='Ideal')
#             plt.title('Constellation Diagram')
#             plt.xlabel('Real')
#             plt.ylabel('Imaginary')
#             plt.legend()
#             plt.grid(True)

#             cbar = plt.colorbar(scatter)
#             cbar.set_label('Carrier Frequency (Hz)')

#             plt.tight_layout()
#             plt.show()
#             plt.savefig(plot_file)
#             plt.close()

#     return data_subcarriers.real.tolist(), data_subcarriers.imag.tolist()
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#################### NEW ################# Additionally takes - grouped_true_bits_list, FFT_length
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import correlate, find_peaks

def demodulate_OFDM_one_symbol_frame(
    y_t: list[float],
    num_carriers: int, # What does this signify? 
    CP_length: int,  # Currently no CP for the data - 16 for the preamble
    freq_AWG: float,
    osc_sample_rate: float,
    record_length: int,
    preamble_sequence: list[float],
    mode: str,
    f_min: float,
    f_max: float,
    subcarrier_delta_f: float, grouped_true_bits_list: list[list[str]], FFT_length: int # Will have to input this as well as the OFDM frame length - IFFT pt
) -> tuple[list[np.ndarray], list[list[int]], list[np.ndarray]]:
    """
    1. Drift correction via spline interpolation.
    2. Resample to 10× AWG rate.
    3. Split into 10 oversampled “copies.”
    4. Preamble detection for synchronization:
       - correlate each copy with the known preamble
       - find valid start indices for full frames
    Returns:
        copies: list of 10 numpy arrays (the downsampled voltage traces)
        valid_indices: list of lists of start indices per copy
        corr_plots: list of numpy arrays of correlation magnitudes per copy
    """
    # Step 1: Drift correction
    delta_f      = 564.0 # Drift corresponding to 125Msa/sec and RL = 2M
    fs_nominal   = osc_sample_rate
    fs_corrected = fs_nominal + delta_f

    n            = np.arange(len(y_t))
    t_nominal    = n / fs_nominal
    t_corrected  = n / fs_corrected
    interp_func  = interp1d(t_corrected, y_t, kind='cubic', fill_value='extrapolate')
    voltages     = interp_func(t_nominal)

    # Step 2: Resample to 10× AWG rate
    fs_tx     = (len(preamble_sequence) + FFT_length)*(freq_AWG) # 80 - preamble length, FFT_length - OFDM Frame length
    fs_rx_new = fs_tx * 10
    t_rx      = np.arange(len(voltages)) / fs_nominal
    duration  = len(voltages) / fs_nominal

    N_new     = int(round(duration * fs_rx_new))
    t_new     = np.linspace(t_rx[0], t_rx[-1], N_new)
    voltages  = np.interp(t_new, t_rx, voltages)

    # Step 3: Organize into oversampled “copies”
    downsampling_factor = int(fs_rx_new / fs_tx)  # should be 10
    rows    = len(voltages) // downsampling_factor
    reshaped = np.zeros((rows, downsampling_factor), dtype=float)
    for i in range(downsampling_factor):
        reshaped[:, i] = voltages[i : i + rows*downsampling_factor : downsampling_factor]
    copies = [reshaped[:, j] for j in range(downsampling_factor)]
    
    # Precompute constants
    k_min          = int(np.floor(f_min / subcarrier_delta_f))
    N_data         = int(np.floor(f_max / subcarrier_delta_f) - k_min)


    # Step 4: Preamble detection for synchronization
    valid_indices = []
    corr_plots    = []
    L             = len(preamble_sequence)
    full_frame    = L + FFT_length

    for col in copies:
        # Correlate with known preamble
        corr = np.array([np.dot(col[i:i+L], preamble_sequence)
                         for i in range(len(col)-L+1)])
        corr_plots.append(corr)

        # Sort peaks by correlation magnitude
        sorted_idx = np.argsort(corr)[::-1]
        candidates = len(col) // full_frame
        max_check  = min(candidates, len(sorted_idx))

        # Keep only those that fit a full frame
        starts = [idx for idx in sorted_idx[:max_check]
                  if ((idx + full_frame) <= len(col))]
        valid_indices.append(starts)

    # Step 5: Extract and append payload copies per column

    rxDataAppended_by_col = []

    for j, rxSignalDs in enumerate(copies):
        valid_idx = valid_indices[j]  # start positions for this column
        if not valid_idx:
            # no valid frames in this copy
            rxDataAppended_by_col.append(None)
            continue

        # we’ll accumulate all OFDM‐symbol subcarrier data here
        rx_data_list = []

        for start in valid_idx:
            # 1) extract one full transmission: preamble + data
            window = rxSignalDs[start : start + full_frame]

            # 2) compute scaling from received preamble
            rx_preamble = window[:len(preamble_sequence)]
            # remove CP from preamble
            rx_preamble_no_cp = rx_preamble[16:]
            # FFT the received preamble (64 points)
            P = np.fft.fft(rx_preamble_no_cp, n=64)
            # data bins = indices 1…31
            preamble_data_bins = P[1:32]
            mean_rx_pre = np.mean(np.abs(preamble_data_bins))

            # --- now compute expected amplitude the same way ---
            # drop CP from the known preamble
            tx_pre_no_cp = np.array(preamble_sequence)[16:]
            # FFT the known preamble
            P_tx = np.fft.fft(tx_pre_no_cp, n=64)
            # take the same bins 1…31
            expected_pre = np.mean(np.abs(P_tx[1:32]))

            scaling = expected_pre / mean_rx_pre

            # 3) extract only the data portion (drop preamble)
            payload = window[len(preamble_sequence) : ]
            
            # 4) strip CP from each OFDM symbol (if you had CP on data)
            # here, no CP on data frames per your last spec; otherwise:
            # payload_no_cp = payload[CP_length : CP_length+N]
            payload_no_cp = payload

            # 5) # Single-symbol FFT
            #N = OFDM_Frame_Length 
            P = np.fft.fft(payload_no_cp, n=FFT_length)
            # only the actual data subcarriers, k_min+1…k_min+N_data
            data_bins = P[k_min+1 : k_min+1+N_data] * scaling

            # 6) append to list
            rx_data_list.append(data_bins)

        # concatenate all copies end-to-end along symbol axis
        rxDataAppended_by_col.append(np.hstack(rx_data_list))
        
        # Step 6: Demodulate each column, compute BER, pick best
        constellation = get_constellation(mode)
        sym_map       = constellation._symbols_to_bits_map
        points        = np.array(list(sym_map.keys()))

        # Bits of the first transmitted frame
        tx_frame_bits = [int(b) for b in grouped_true_bits_list[0]]

        best_ber   = float('inf')
        best_first = None

        for j, data_bins in enumerate(rxDataAppended_by_col):
            starts = valid_indices[j]
            if data_bins is None or not starts:
                continue

            # Flatten all symbols across all copies in this column
            all_syms = data_bins.flatten()

            # Hard‐decision demap
            dists   = np.abs(all_syms[:, None] - points[None, :])
            idxs    = np.argmin(dists, axis=1)
            bits_rx = np.concatenate([list(map(int, sym_map[points[i]])) for i in idxs])

            # Replicate transmitted bits per copy
            num_copies = len(starts)
            tx_bits_rep = tx_frame_bits * num_copies

            # Compute BER (lengths now match)
            L = len(tx_bits_rep)
            ber = np.sum(bits_rx != np.array(tx_bits_rep)) / L

            if ber < best_ber:
                best_ber   = ber
                # Return only the first OFDM symbol copy (first N_data bins)
                best_first = data_bins[:N_data]

        if best_first is None:
            return [], []
        return best_first.real.tolist(), best_first.imag.tolist()


#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

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
        cancel_run_early = update_weights()
    else:
        cancel_run_early = False

    SNR, PowerFactor = float(0), float(0) 
    
    return decided_bits_flat, float(BER), SNR, PowerFactor, cancel_run_early