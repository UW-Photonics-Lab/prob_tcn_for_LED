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

# decode_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\decode_bits_log.txt")
STATE['sent_symbols'] = []
STATE['received_symbols'] = []
STATE['channel_estimates'] = []
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
                       subcarrier_delta_f: float,
                       num_symbols_per_frame: int) -> tuple[list, list, list, int]:
    '''Creates and n x m symbol matrix to fill frequency band for OFDM
    
        Args:
            f_min: minumum frequency (Hz) 
            f_max: maximum frequency (Hz)
            mode: a string to determine how modulation occurs. e.g qpsk
            data: list of bits of 1, 0 to modulate
        
        Outputs:
            Returns a tuple containing two lists representing two 2d arrays
            of Real{Symbols} and Imaginary{Symbols}

            Each row is an indepedent OFDM symbol to be transmitted    
    '''
    bits = "".join([str(x) for x in data])

    # Account for frequency scaling with the AWG. Namely, if Nt symbols are sent successively,
    # each OFDM symbol's carriers are scaled by Nt. This can be acomplished by driving each frame at subcarrier_delta_f / Nt

    k_min = int(np.floor(f_min / subcarrier_delta_f))

    N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min)

    STATE['Nt'] = num_symbols_per_frame
    STATE['Nf'] = N_data
    
    # Grab constellation object
    constellation = get_constellation(mode)
    encoded_symbols = constellation.bits_to_symbols(bits)
    true_bits = np.array(data)
    if len(true_bits) % constellation.modulation_order != 0:
        raise ValueError(f"Number of bits not a multiple of modulation order {constellation.modulation_order}! | Allocated {len(true_bits)}")

    max_symbols_per_frame = N_data * num_symbols_per_frame
    if len(encoded_symbols) > max_symbols_per_frame:
        raise ValueError(f"Too many symbols alloacated to frame! Max{max_symbols_per_frame} | Allocated {len(encoded_symbols)}")
    
    # Calculate frame length
    frame_length = N_data * num_symbols_per_frame
    # Reshape such that each row is a symbol
    if len(encoded_symbols) < frame_length:
        # Padd for full frame
        zeros_to_add = frame_length - len(encoded_symbols) % frame_length
        padding_symbols = np.repeat(constellation.bits_to_symbols(constellation.modulation_order * "0"), zeros_to_add)
        bits_added = np.array(list(constellation.symbols_to_bits(padding_symbols)))
        encoded_symbols = np.hstack((encoded_symbols, padding_symbols))
        true_bits = np.hstack((true_bits, bits_added))

    encoded_symbol_frame = encoded_symbols.reshape(num_symbols_per_frame, N_data) # [Nt, Nf]

    # Add zeros on front equal to k_min 
    encoded_symbol_frame = np.hstack((np.zeros((encoded_symbol_frame.shape[0], k_min)), encoded_symbol_frame))
    encoded_symbol_frame_real = encoded_symbol_frame.real.copy()
    encoded_symbol_frame_imag = encoded_symbol_frame.imag.copy()


    # Reshape true bits by frame
    Nt, _ = encoded_symbol_frame.shape
    grouped_true_bits = true_bits.reshape(Nt, -1)

    grouped_true_bits_list = [[str(element) for element in row] for row in grouped_true_bits.tolist()]

    # Return as real and imaginary parts
    return encoded_symbol_frame_real, encoded_symbol_frame_imag, grouped_true_bits_list, int(N_data)


AWG_MEMORY_LENGTH = 16384
# AWG_MEMORY_LENGTH = 105
# AWG_MEMORY_LENGTH = 65536


BARKER_LENGTH = int(0.01 * (AWG_MEMORY_LENGTH)) if int(0.01 * (AWG_MEMORY_LENGTH)) > 5 else 5
CP_RATIO = 0.25
def symbols_to_xt(real_symbol_groups: list[list[float]], imag_symbol_groups: list[list[float]], cyclic_prefix_length: int) -> list[float]:
    '''Takes a symbol frame matrix and converts and x(t) frame matrix

    
        Args:
            symbol_groups: Nt x Nf matrix where Nt is number of symbols per frame and Nf in number of data carrirs per symbol

        Outputs:
            returns 1 x AWG_MEMORY_LENGTH matrix and the preable used for later correlation

    '''
    symbol_groups = np.array(real_symbol_groups) + np.array(imag_symbol_groups) * 1j
    STATE['sent_symbols'].append(torch.tensor(symbol_groups[:, -STATE['Nf']:]))
    N_t = symbol_groups.shape[0]
    barker_code = np.array([1, -1, 1, -1, 1], dtype=float)
    barker_code = np.repeat(barker_code, BARKER_LENGTH // len(barker_code)) # Set as 1%
    IFFT_LENGTH = int((AWG_MEMORY_LENGTH - len(barker_code)) // N_t) 

    '''CP'''
    IFFT_LENGTH = int((AWG_MEMORY_LENGTH - len(barker_code)) // (N_t * (1 + CP_RATIO)))
    cyclic_prefix_length = int(IFFT_LENGTH * CP_RATIO)

    # Add DC offset, Nyquist carrier, and conjugate symmetry
    symbol_groups_conjugate_flipped = np.conj(symbol_groups)[:, ::-1] # Flip along axis=1
    DC_nyquist = np.zeros(shape=(symbol_groups.shape[0], 1)) # Both are set to 0
    full_symbols = np.hstack((DC_nyquist, symbol_groups, DC_nyquist, symbol_groups_conjugate_flipped)) # [Nt, Nf]
    
    SYMBOL_LENGTH = int((AWG_MEMORY_LENGTH - len(barker_code)) / N_t)
    IFFT_LENGTH = int(SYMBOL_LENGTH * (1 - CP_RATIO))
    cyclic_prefix_length = int(SYMBOL_LENGTH * CP_RATIO)
    x_t_no_cp = np.real(np.fft.ifft(full_symbols, axis=1, n=IFFT_LENGTH))

    # Add cyclic prefix per symbol
    x_t_groups = []
    for row in x_t_no_cp:
        cp = row[-cyclic_prefix_length:]
        with_cp = np.concatenate([cp, row])
        x_t_groups.append(with_cp)

    x_t_groups = np.stack(x_t_groups)  # shape: [Nt, IFFT_LENGTH + CP_length]
  
    # Flatten to create frame in time domain
    x_t_frame = x_t_groups.flatten()
    max_abs_val = np.max(np.abs(x_t_frame)) + 1e-12  # avoid div-by-zero
    x_t_frame /= max_abs_val
    x_t_with_barker = np.concatenate([barker_code, x_t_frame])
    x_t_with_barker = np.ascontiguousarray(x_t_with_barker).astype(float)
    x_t_with_barker = x_t_with_barker[:AWG_MEMORY_LENGTH]  # truncate if needed
    x_t_with_barker = np.clip(x_t_with_barker, -1.0, 1.0)
    return x_t_with_barker.reshape(1, -1).tolist(), barker_code




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
                                     subcarrier_delta_f: float,
                                     Nt: int) -> list:
    '''Converts received y(t) into symbols with optional debugging plots'''

    debug_plots = False
    PLOT_SNR = False

    k_min = int(np.floor(f_min / subcarrier_delta_f))

    N_data = int(np.floor(f_max / subcarrier_delta_f) - k_min)

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
    if debug_plots:
        plt.subplot(322)
        plt.plot(voltages)
        plt.title(f'Resampled Signal Length: {len(voltages)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    if debug_plots:
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
        plt.subplot(324)
        plt.plot(frames[0])
        plt.title(f'First Extracted Frame Length: {len(frames[0])}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    if len(frames) == 0:
        raise ValueError("No valid frames detected")

    frame_y_t = np.array(frames[0])
    frame_len = len(frame_y_t)
    symbol_len = frame_len // Nt
    CP_length = int(symbol_len * CP_RATIO)
    fft_len = symbol_len - CP_length

    # Target sample centers for each symbol
    symbols = []
    for i in range(Nt):
        start = i * symbol_len
        end = start + symbol_len

        symbol_with_cp = frame_y_t[start:end]
        symbol = symbol_with_cp[CP_length:][:fft_len]
        # symbol /= np.max(np.abs(symbol)) + 1e-12
        symbols.append(symbol)

    Y_s_matrix = np.array([np.fft.fft(s) for s in symbols])
    if debug_plots:
        plt.subplot(325)
        limited_magnitude = np.abs(Y_s_matrix[0])

        # Plot the FFT magnitude
        plt.plot(limited_magnitude)
        plt.title('FFT Magnitude of 1st symbol')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

    # # Extract both positive and negative frequency carriers
    num_data_carriers = N_data

    data_subcarriers_all = []

    for symbol_fft in Y_s_matrix:
        negative_carriers = symbol_fft[k_min + 1:N_data + k_min + 1] # Account for DC = 0 being added
        # Optionally include positive_carriers if your OFDM symbol is Hermitian symmetric
        # positive_carriers = symbol_fft[-num_data_carriers - k_min - 1: -k_min - 1]

        data_subcarriers_all.append(negative_carriers)  

    normalized_data_subcarriers = []

    for carriers in data_subcarriers_all:
        scale = np.sqrt(np.mean(np.abs(constellation._complex_symbols)**2)) / np.sqrt(np.mean(np.abs(carriers)**2) + 1e-12)
        normalized_carriers = carriers * scale
        normalized_data_subcarriers.append(normalized_carriers)

    STATE['received_symbols'].append(torch.tensor(normalized_data_subcarriers))
    data_subcarriers = np.concatenate(normalized_data_subcarriers)

    '-----------------------------Channel Estimation ---------------------------------------------------------'
    X_k = np.array(STATE['sent_symbols'][-1])[0]
    Y_k = np.array(STATE['received_symbols'][-1])[0]
    H_k = Y_k / (X_k + 1e-12)  # Avoid divide-by-zero

    # Save current results to compare to later for noise estimate
    STATE['channel_estimates'].append(H_k)

    if debug_plots:
        # Plot 6: Constellation Diagram

        # Calculate frequencies
        frequencies_per_symbol = np.arange(f_min+ subcarrier_delta_f, 
                                f_min + subcarrier_delta_f * num_data_carriers + subcarrier_delta_f, 
                                subcarrier_delta_f)
        frequencies = np.tile(frequencies_per_symbol, Nt)
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
        plt.close()


    if debug_plots:
        # Compute magnitude and phase
        H_mag = np.abs(H_k)
        H_phase = np.unwrap(np.angle(H_k))
        freqs = frequencies_per_symbol  # Already defined earlier

        # Bode-like plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(freqs, 20 * np.log10(H_mag + 1e-12))  # dB scale
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title("Estimated Channel Frequency Response")

        ax2.plot(freqs, (180 / np.pi) * H_phase)
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")

        ax1.grid(True)
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    if PLOT_SNR and len(STATE['channel_estimates']) == 20:
        H_ks = np.array(STATE['channel_estimates'])
        H_ks_mean = np.mean(H_ks, axis=0)
        H_ks_var = np.mean(np.square(np.abs(H_ks - H_ks_mean)), axis=0)

        H_mag = np.abs(H_ks)
        H_mag_mean = np.abs(H_ks_mean)
        H_mag_std = np.std(H_mag, axis=0)

        H_phase = np.unwrap(np.angle(H_ks), axis=0)
        H_phase_mean = np.mean(H_phase, axis=0)
        H_phase_std = np.std(H_phase, axis=0)

        signal_power = np.abs(H_ks_mean)**2 * np.mean(np.abs(X_k)**2, axis=0)
        noise_power = H_ks_var * np.mean(np.abs(X_k)**2, axis=0)
        snr_est = signal_power / noise_power
        snr_dB = 10 * np.log10(snr_est)

        freqs = np.arange(f_min, f_max, subcarrier_delta_f)
        C_total = np.trapz(np.log2(1 + snr_est), freqs)

        # ---- Plotting ----
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Magnitude plot
        axs[0].plot(freqs, 20 * np.log10(H_mag_mean + 1e-12), label='Mean |H(f)| [dB]')
        axs[0].fill_between(freqs,
                            20 * np.log10(H_mag_mean - H_mag_std + 1e-12),
                            20 * np.log10(H_mag_mean + H_mag_std + 1e-12),
                            alpha=0.3, label='±1 STD')
        axs[0].set_ylabel("Magnitude (dB)")
        axs[0].set_title("Estimated Channel Magnitude ± STD")
        axs[0].grid(True)
        axs[0].legend()

        # Phase plot
        axs[1].plot(freqs, H_phase_mean * 180 / np.pi, label='Mean ∠H(f) [deg]')
        axs[1].fill_between(freqs,
                            (H_phase_mean - H_phase_std) * 180 / np.pi,
                            (H_phase_mean + H_phase_std) * 180 / np.pi,
                            alpha=0.3, label='±1 STD')
        axs[1].set_ylabel("Phase (degrees)")
        axs[1].set_title("Estimated Channel Phase ± STD")
        axs[1].grid(True)
        axs[1].legend()

        # SNR plot
        axs[2].plot(freqs, snr_dB, label='Estimated SNR [dB]', color='green')
        axs[2].set_xlabel("Subcarrier Index")
        axs[2].set_ylabel("SNR (dB)")
        axs[2].set_title(f"Estimated SNR per Subcarrier with Estimated C{C_total: .3e}")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        plt.close()
    return data_subcarriers.real.tolist(), data_subcarriers.imag.tolist()

def decode_symbols_OFDM(real_symbols: list, imag_symbols: list, true_bits: list,  mode: str) -> list:
    from encoder_decoder import update_weights
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

    # Calculate BER
    BER = float(np.sum(true_bits_array != decided_bits_flat_array) / len(true_bits_array))

    # Log frame BER
    STATE['frame_BER'] = BER

       # Call backprop and log loss
    if 'encoder_out' in STATE and 'decoder_out' in STATE:
        cancel_run_early = update_weights()
    elif 'cancel_channel_train' in STATE:
        cancel_run_early = STATE['cancel_channel_train']
    else:
        cancel_run_early = False

    SNR, PowerFactor = float(0), float(0) 
    
    return decided_bits_flat, float(BER), SNR, PowerFactor, cancel_run_early