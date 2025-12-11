import sys

import zarr
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
import wandb
from training_state import STATE
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.fft import irfft, rfft
from fractions import Fraction
from scipy.signal import hilbert, filtfilt, butter
from modules.utils import calculate_BER, evm_loss, save_validation_data
from constellation_diagram import get_constellation
from encoder_decoder import append_symbol_frame
# Get logging
from lab_scripts.logging_code import *
STATE['normalize_power'] = False
STATE['verify_synchronization'] = False
STATE['in_band_filter'] = False
decode_logger = setup_logger(log_file=r"C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\debug_logs\test3.txt")
STATE["frame_BER_accumulator"] = [] # Use to estimate BER over many frames
STATE['frame_evm_accumulator'] = []

# def resample_poly(signal, **kwargs):
#     rms_before = np.sqrt(np.mean(np.square(signal)))
#     resampled_signal = resample_poly(signal, **kwargs)
#     rms_after = np.sqrt(np.mean(np.square(resampled_signal)))
#     scaling_factor = rms_before / rms_after
#     corrected_signal = resampled_signal * scaling_factor
#     return corrected_signal

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

    num_zeros = int(np.floor(f_min / subcarrier_delta_f)) - 1
    STATE['num_zeros'] = num_zeros

    N_data = int(np.floor(f_max / subcarrier_delta_f) - num_zeros) - 1

    STATE['N_data'] = N_data

    STATE['Nt'] = num_symbols_per_frame
    STATE['Nf'] = N_data
    STATE['delta_f'] = subcarrier_delta_f
    STATE['frequencies'] = np.arange(f_min, f_max, subcarrier_delta_f)
    STATE['ks'] = (STATE['frequencies'] / subcarrier_delta_f).astype(int)
    
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

    # Add zeros on front equal to num_zeros 
    encoded_symbol_frame = np.hstack((np.zeros((encoded_symbol_frame.shape[0], num_zeros)), encoded_symbol_frame))
    # decode_logger.debug(f"Modulate Data Outshape {encoded_symbol_frame.shape}")
    encoded_symbol_frame_real = encoded_symbol_frame.real.copy()
    encoded_symbol_frame_imag = encoded_symbol_frame.imag.copy()


    # Reshape true bits by frame
    Nt, _ = encoded_symbol_frame.shape
    grouped_true_bits = true_bits.reshape(Nt, -1)

    grouped_true_bits_list = [[str(element) for element in row] for row in grouped_true_bits.tolist()]

    # Return as real and imaginary parts
    return encoded_symbol_frame_real, encoded_symbol_frame_imag, grouped_true_bits_list, int(N_data)


AWG_MEMORY_LENGTH_MAX = 16384
SCALING_FACTOR = 2
MESSAGE_LENGTH = 16384 // SCALING_FACTOR
BARKER_LENGTH = 384 / SCALING_FACTOR
CP_RATIO = 0.25 # Need Message length - barker cleanly divisible by 4

def symbols_to_time(X, upsampling_factor: int, num_leading_zeros):
    # Make Hermitian symmetric11
    Nt, Nf = X.shape
    num_padding_zeros = Nf * (upsampling_factor - 1)
    padding_zeros = np.zeros((Nt, num_padding_zeros))
    leading_zeros = np.zeros((Nt, num_leading_zeros))
    X = np.concatenate([leading_zeros, X, padding_zeros], axis=-1)

    DC_Nyquist = np.zeros((X.shape[0], 1))
    X_hermitian = np.conj(np.flip(X, axis=1))
    X_full = np.hstack([DC_Nyquist, X, DC_Nyquist, X_hermitian])

    # Convert to time domain
    x_time = np.fft.ifft(X_full, axis=-1, norm="ortho").real
    return x_time


def generate_zadoff_chu(length, root=1):
    """Generate Zadoff-Chu sequence"""
    num_pad = 6
    n = np.arange(length - 2 * num_pad)
    if length % 2 == 0:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        zc = np.exp(-1j * np.pi * root * n**2 / length)

    # add 5 zeros on either side
    padding_zeros = np.zeros(num_pad, dtype=np.complex128)
    zc = np.concatenate([padding_zeros, zc, padding_zeros])
    return zc

def nearest_prime(n):
    """Find nearest prime number"""
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True
    
    for offset in range(n):
        if is_prime(n + offset):
            return n + offset
        if n - offset > 0 and is_prime(n - offset):
            return n - offset
    return n


def symbols_to_xt(real_symbol_groups: list[list[float]], imag_symbol_groups: list[list[float]]) -> list[float]:
    '''Takes a symbol frame matrix and converts and x(t) frame matrix

    
        Args:
            symbol_groups: Nt x Nf matrix where Nt is number of symbols per frame and Nf in number of data carrirs per symbol

        Outputs:
            returns 1 x MESSAGE_LENGTH matrix and the preable used for later correlation

    '''
    symbol_groups = np.array(real_symbol_groups) + np.array(imag_symbol_groups) * 1j

    # decode_logger.debug(f"Symbol Groups shape {symbol_groups.shape}")

    avg_pow = np.mean(np.square(np.abs(symbol_groups)))
    # decode_logger.debug(f"S-> Xt Average Input power {avg_pow: .5f}")


    data_lo = STATE['num_zeros']       # index inside symbol_groups (no DC bin in this array)
    data_hi = data_lo + STATE['Nf']
    STATE['last_sent'] = torch.tensor(symbol_groups[:, data_lo:data_hi])
    N_t = symbol_groups.shape[0]
    # barker_code = 3 *  np.array([0, 0, 0, 1, -1, 1, -1, 1,-1, 1, -1, 1, 0, 0, 0], dtype=float)
    # barker_code = np.repeat(barker_code, BARKER_LENGTH // len(barker_code)) # Set as 1%

    zc_sequence = generate_zadoff_chu(BARKER_LENGTH, root=1)
    barker_code = np.real(zc_sequence) 
    barker_code /= np.max(np.abs(barker_code))
    barker_code = np.ascontiguousarray(3 * barker_code)
  
    SYMBOL_LENGTH = (MESSAGE_LENGTH - len(barker_code)) // N_t
    IFFT_LENGTH = round(SYMBOL_LENGTH * (1 - CP_RATIO))
    IFFT_LENGTH -= IFFT_LENGTH % 2 # Force to be even
    STATE['IFFT_LENGTH'] = IFFT_LENGTH

    # Upsample in multiples of two zeros on each side to satisy Hermitian symmetry
    M = IFFT_LENGTH // 2 - 1
    number_of_zeros = M - symbol_groups.shape[1]
    padding_zeros = np.zeros(shape=(symbol_groups.shape[0], number_of_zeros))
    symbol_groups = np.hstack((symbol_groups, padding_zeros))
    DC_nyquist = np.zeros(shape=(symbol_groups.shape[0], 1), dtype=np.complex64) # Both are set to 0
    half_spectrum = np.hstack((DC_nyquist, symbol_groups, DC_nyquist))
    assert half_spectrum.shape[1] == (IFFT_LENGTH // 2 + 1), "Half-spectrum length mismatch."

    cyclic_prefix_length = round(SYMBOL_LENGTH * CP_RATIO)
    # decode_logger.debug(f"IFFT padding zeros {number_of_zeros}")
 

    x_t_no_cp = irfft(half_spectrum, n=IFFT_LENGTH, axis=1, norm='ortho', workers=os.cpu_count()).astype(np.float32, copy=False)
    # Add cyclic prefix per symbol
    x_t_groups = []
    for row in x_t_no_cp:
        cp = row[-cyclic_prefix_length:]
        with_cp = np.concatenate([cp, row])
        x_t_groups.append(with_cp)

    x_t_groups = np.stack(x_t_groups)  # shape: [Nt, IFFT_LENGTH + CP_length]

    STATE['num_points_symbol'] = x_t_groups.shape[1]
    STATE['cp_length'] = cyclic_prefix_length

    # decode_logger.debug(f"CP {cyclic_prefix_length} | FFT {IFFT_LENGTH}")

    STATE['barker_code'] = barker_code
    # Flatten to create frame in time domain
    x_t_frame = x_t_groups.flatten()
    # decode_logger.debug(f"S->T output mean power{np.mean(np.square(x_t_frame))} | output shape: {x_t_frame.shape} | output max {np.max(np.abs(x_t_frame))} | IFFT len {IFFT_LENGTH}")
    return x_t_frame.reshape(1, -1).tolist(), barker_code

def add_preamble_and_upsample(x_t_frame, preamble):
    x_t_frame = np.array(x_t_frame).flatten()
    # decode_logger.debug(f"Num points frame {len(x_t_frame)}")
    # Upsample to fill AWG memory
    preamble = np.array(preamble).flatten()
    preamble_min = np.min(preamble)
    preamble_max = np.max(preamble)
    # decode_logger.debug(f"Preamble heights {preamble_min} | {preamble_max}")
    x_t_frame = np.clip(x_t_frame, preamble_min, preamble_max)
    x_t_with_barker = np.concatenate([preamble, x_t_frame])

    if False:
        preamble_fft = np.fft.fft(preamble)
        freqs_before = np.fft.fftfreq(len(preamble), 1/(STATE['IFFT_LENGTH'] * STATE['delta_f']))
        upsampled_preamble = resample_poly(preamble, up=SCALING_FACTOR, down=1, )
        # After upsampling
        upsampled_fft = np.fft.fft(upsampled_preamble)
        freqs_after = np.fft.fftfreq(len(upsampled_preamble), 1/(STATE['delta_f'] * STATE['IFFT_LENGTH']*SCALING_FACTOR))

        plt.subplot(211)
        plt.plot(freqs_before[:len(freqs_before)//2], np.abs(preamble_fft[:len(preamble_fft)//2]))
        plt.title('Preamble Spectrum Before Upsampling')

        plt.subplot(212)
        plt.plot(freqs_after[:len(freqs_after)//2], np.abs(upsampled_fft[:len(upsampled_fft)//2]))
        plt.title('Preamble Spectrum After Upsampling')
        plt.show()

    x_t_with_barker = resample_poly(x_t_with_barker, up=SCALING_FACTOR, down=1)
    x_t_with_barker = x_t_with_barker[:AWG_MEMORY_LENGTH_MAX]  # truncate if needed
    return x_t_with_barker.reshape(1, -1).tolist()

def find_start(peak, voltages, preamble_length, ofdm_payload_length, search_window=20):
    '''Based on rough estimates of peaks, use the cyclic prefix to find a better start to the frame'''
    num_symbols = STATE['Nt']
    curr_start = peak + preamble_length
    symbol_length = (ofdm_payload_length // num_symbols)
    cp_length = int(CP_RATIO * symbol_length)
    N_fft = symbol_length - cp_length
    best_start = curr_start
    best_corr = 0
    STATE['offs'], STATE['corrs'], STATE['phis'] = [], [], []
    for offset in range(-search_window, search_window + 1):
        # Test correlation of cyclic prefix and tail
        start = curr_start + offset
        end = start + symbol_length
        if start < 0 or end > len(voltages):
            continue # try different offset
        symbol = voltages[start: end]
        analytic_symbol = hilbert(symbol)
        cyclic_prefix_j = analytic_symbol[:cp_length]
        tail_j = analytic_symbol[-cp_length: ]
        corr = np.vdot(cyclic_prefix_j, tail_j) / (np.linalg.norm(cyclic_prefix_j) * np.linalg.norm(tail_j))
        STATE['offs'].append(offset)
        STATE['corrs'].append(np.abs(corr))
        STATE['phis'].append(np.angle(corr))
        if np.abs(corr) > best_corr:
            best_corr = np.abs(corr)
            best_start = curr_start + offset
            best_phi = np.angle(corr)
            omega_hat = best_phi / float(N_fft)
            epsilon_hat = best_phi / (2 * np.pi)
    if STATE['verify_synchronization']:
        # Plot correlation vs various offsets
        offs = np.array(STATE['offs'])
        corrs = np.array(STATE['corrs'])
        plt.plot(offs, corrs)
        plt.xlabel("Offset")
        plt.ylabel("Correlation")
        plt.show()

    return best_start, best_start - preamble_length, omega_hat, epsilon_hat


def in_band_filter(xt, in_band_indices, nfft):
    xt = torch.tensor(xt)
    mask = torch.zeros(nfft)
    neg_ks_indices = nfft - in_band_indices
    mask[in_band_indices] = 1.0
    mask[neg_ks_indices] = 1.0

    impulse_response = torch.fft.ifftshift(torch.fft.ifft(mask).real)
    h = impulse_response.view(1, 1, -1)
    filtered_x = F.conv1d(xt.unsqueeze(1), h, padding='same').squeeze(1)
    return filtered_x.numpy()


STATE['peaks'] = []
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
    num_zeros = STATE['num_zeros']

    N_data = STATE['N_data']
    constellation = get_constellation(mode)
    if debug_plots:
        plt.figure(figsize=(15, 10))
    
        # Plot original received signal
        plt.subplot(321)
        plt.plot(y_t)
        plt.title(f'Original Received Signal y(t) Length: {len(y_t)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    baseband_sample_rate = (freq_AWG * (MESSAGE_LENGTH)) 
    baseband_nyquist_rate = baseband_sample_rate / 2
    downsample_factor = baseband_sample_rate / osc_sample_rate
    frac = Fraction(downsample_factor).limit_denominator(10000)
    up = frac.numerator
    down = frac.denominator
    # Downsample received waveformdecd
    # decode_logger.debug(f"Downsample factor {downsample_factor}")
    y_t = resample_poly(y_t, up=up, down=down)
    preamble = np.array(preamble_sequence)

    DEBUG_ALIASING = False


    # Filter out high frequency content from preamble to avoid rollover
    # b, a = butter(N=16, Wn=cutoff, btype='low', fs=osc_sample_rate)
    # y_t_filtered = filtfilt(b, a, y_t)

    if DEBUG_ALIASING:
        # plot current received spectrum
        Yk = np.abs(np.fft.fft(y_t, norm='ortho'))
        freqs = np.fft.fftfreq(len(y_t), 1/osc_sample_rate)
        freq_mask = (freqs <= 2 * baseband_nyquist_rate) & (freqs > 0)
        plt.figure(figsize=(16, 8))
        half = len(y_t)//2
        plt.plot(freqs[freq_mask], Yk[freq_mask])
        plt.title("Spectrum of Received Signal")
        plt.xlabel("Freq")
        plt.ylabel("|Y|")
        plt.grid(True)
        plt.show()


    y_t_filtered = y_t
    if DEBUG_ALIASING:
        Yk = np.abs(np.fft.fft(y_t_filtered, norm='ortho'))
        freqs = np.fft.fftfreq(len(y_t), 1/osc_sample_rate)
        freq_mask = (freqs <= 2 * baseband_nyquist_rate) & (freqs > 0)
        plt.figure(figsize=(16, 8))
        half = len(y_t)//2
        plt.plot(freqs[freq_mask], Yk[freq_mask])
        plt.title("Spectrum of Filtered Signal")
        plt.xlabel("Freq")
        plt.ylabel("|Y|")
        plt.grid(True)
        plt.show()

    if DEBUG_ALIASING:
        Yk = np.abs(np.fft.fft(y_t, norm='ortho'))
        freqs = np.fft.fftfreq(len(y_t), 1/baseband_sample_rate)
        freq_mask = freqs > 0
        plt.figure(figsize=(16, 8))
        half = len(y_t)//2
        plt.plot(freqs[freq_mask], Yk[freq_mask])
        plt.title("Spectrum of Downsampled Signal")
        plt.xlabel("Freq")
        plt.ylabel("|Y|")
        plt.grid(True)
        plt.show()
        
    corr = signal.correlate(y_t, preamble, mode='valid')
    peaks, _ = find_peaks(corr, height=0.95 * np.max(np.abs(corr)), distance=len(preamble))




    # Get actual frame length
    if len(peaks) > 1:
        actual_frame_length = round(np.median(np.diff(peaks)))
        # decode_logger.debug(f"Calculated Message Length {actual_frame_length} | Ideal {MESSAGE_LENGTH} | CP {STATE['cp_length']} | Num S {STATE['num_points_symbol']}")
        actual_preamble_length = len(preamble)
        actual_payload_length = actual_frame_length - actual_preamble_length
        # decode_logger.debug(f"Received Symbol Length {actual_payload_length} | Preamble Generated {len(STATE['barker_code'])} | Preamble received {actual_preamble_length}")
    peak = peaks[0]

    if debug_plots:
        plt.subplot(322)
        plt.plot(y_t)
        plt.title(f'Resampled Signal Length: {len(y_t)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    if debug_plots:
        plt.subplot(323)
        plt.plot(corr[peak-20:peak+20])
        plt.title('Correlation with Preamble')
        plt.xlabel('Sample')
        plt.ylabel('Correlation')
                # Add peak labels
        # for i, peak in enumerate(peaks):
        plt.plot(20, corr[peak], 'r^')  # Red triangle marker
        i=0
        plt.annotate(f'Peak {i+1}\n({peak})', 
                    xy=(20, corr[peak]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


    # Parameters and frame extraction
    # decode_logger.debug(f"Peak : {peak}")
    best_start = peak + actual_preamble_length 
    frame_end = best_start + actual_payload_length
    frame = y_t[best_start:frame_end]

    if debug_plots:
        plt.subplot(324)
        plt.plot(frame)
        plt.title(f'First Extracted Frame Length: {len(frame)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    frame_y_t = np.array(frame)
    frame_len = len(frame_y_t)
    symbol_len = STATE['num_points_symbol'] 
    fft_len = STATE['IFFT_LENGTH']           
    CP_length = STATE['cp_length'] 

    # decode_logger.debug(f"Transmitted: IFFT={STATE['IFFT_LENGTH']}, CP={STATE['cp_length']}, Total={STATE['num_points_symbol']}")
    # decode_logger.debug(f"Received: FFT={fft_len}, CP={CP_length}, Total={symbol_len}")
    # decode_logger.debug(f"Mismatch: FFT={fft_len - STATE['IFFT_LENGTH']}, CP={CP_length - STATE['cp_length']}")

    # decode_logger.debug(f"CP Length {CP_length}")
    # decode_logger.debug(f"FFT Length {fft_len}")

    # Target sample centers for each symbol
    symbols = []
    symbols_with_cp = []
    for i in range(Nt):
        start = i * symbol_len
        end = start + symbol_len

        symbol_with_cp = frame_y_t[start:end]
        symbol = symbol_with_cp[CP_length:][:fft_len]
        window = 4
        if STATE['verify_synchronization']:
            fft_lens = range(fft_len - window, fft_len + window + 1, 2)
            for N in fft_lens:
                symbol_i = symbol_with_cp[CP_length:][:N]
                debug_Y_half = rfft(symbol_i, n=N, norm="ortho")
                mag = 20*np.log10(np.abs(debug_Y_half) / (np.max(np.abs(debug_Y_half))))
                k_peak = np.argmax(mag)
                bins = np.arange(len(mag)) - k_peak
                win = 50  # show ±50 bins around peak
                k0 = max(0, k_peak - win)
                k1 = min(len(mag), k_peak + win)
                plt.plot(bins[k0:k1], mag[k0:k1], alpha=0.6, label=f"N={N}")

            # plot here
            plt.xlabel("Bin offset from main lobe")
            plt.ylabel("Magnitude (dB, normalized)")
            plt.title("Dirichlet sidelobes vs FFT length (aligned)")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=3, fontsize=8)
            plt.show()
        # Account for CFO 
        # n = np.arange(fft_len, dtype= np.complex128)
        # symbol *= np.exp(-1j * best_omega * n)
        # symbol /= np.max(np.abs(symbol)) + 1e-12
        symbols.append(symbol)
        symbols_with_cp.append(symbol_with_cp)

    symbols_arr = np.ascontiguousarray(np.stack(symbols_with_cp).astype(np.float32))  # [Nt, fft_len + CP_len]
    symbols_arr -= np.mean(symbols_arr, axis=-1, keepdims=True) # Remove DC Bias

    STATE['last_time_symbol_received'] = symbols_arr
    # decode_logger.debug(f"Received time array power {np.mean(np.square(symbols_arr))} | shape {symbols_arr.shape}")

    # Apply time decoder here
    if 'run_model' in STATE and STATE['validate_model']:
        if STATE['run_model']:
            with torch.no_grad():
                symbols_arr = torch.tensor(symbols_arr, dtype=torch.float32)
                # decode_logger.debug(f"decoder in {symbols_arr.shape}")
                STATE['time_decoder_in'] = symbols_arr.detach().cpu().numpy()
                symbols_arr = STATE["decoder"](symbols_arr)
                # decode_logger.debug(f"decoder out {symbols_arr.shape}")
                STATE['time_decoder_out'] = symbols_arr.detach().cpu().numpy()
                symbols_arr = STATE['time_decoder_out']
        else:
            STATE['time_decoder_out'] = symbols_arr
            STATE['time_decoder_in'] = STATE['time_decoder_out']
    
    # perform in band filters
    if STATE['in_band_filter']:
        symbols_arr = in_band_filter(symbols_arr, STATE['ks'], STATE['IFFT_LENGTH'])

    # Remove cyclic prefix
    symbols_arr = symbols_arr[:, CP_length:]

    if DEBUG_ALIASING:
        Yk = np.abs(np.fft.fft(symbols_arr[0], norm='ortho'))
        freqs = np.fft.fftfreq(len(symbols_arr[0]), 1/baseband_sample_rate)
        freq_mask1 = (freqs > 0)
        freq_mask2 = (freqs > 0) & (freqs < int(100 * 1e3))
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        half = len(symbols_arr[0])//2
        ax[0].plot(freqs[freq_mask1], Yk[freq_mask1])
        ax[1].plot(freqs[freq_mask2], Yk[freq_mask2])
        ax[0].set_title("Spectrum of Symbol without CP")
        ax[1].set_title("Spectrum of Symbol without CP")
        ax[1].set_xlabel("Freq")
        ax[1].set_ylabel("|Y|")
        ax[0].set_xlabel("Freq")
        ax[0].set_ylabel("|Y|")
        plt.grid(True)
        plt.show()

    Y_half = rfft(symbols_arr, n=STATE['IFFT_LENGTH'], axis=1, norm="ortho")  # [Nt, fft_len//2+1
    data_subcarriers = Y_half[:, num_zeros+1 : num_zeros+1 + N_data]  # shape [Nt, N_data]
    STATE['last_freq_symbol_received'] = data_subcarriers

    if STATE['normalize_power']:
        power_normalization = np.sqrt(np.mean(np.square(np.abs(data_subcarriers)), axis=1, keepdims=True))
        data_subcarriers = data_subcarriers / power_normalization

    if debug_plots:
        plt.subplot(325)
        limited_magnitude = np.log10(np.abs(Y_half[0][1:]))

        # Plot the FFT magnitude
        plt.plot(limited_magnitude)
        plt.title('FFT Magnitude of 1st symbol')
        plt.xlabel('Frequency (10 kHz)')
        plt.ylabel('Magnitude Log10')

    STATE['last_received'] = torch.tensor(data_subcarriers)
    data_subcarriers = np.concatenate(data_subcarriers)

    # Channel Estimation
    X_k = np.array(STATE['last_sent'])[0]
    Y_k = np.array(STATE['last_received'])[0]
    H_k = Y_k / (X_k + 1e-12)  # Avoid divide-by-zero

    # Save current results to compare to later for noise estimate
    if debug_plots:
        # Plot Constellation Diagram
        frequencies_per_symbol = np.arange(f_min + subcarrier_delta_f, 
                                f_min + subcarrier_delta_f * N_data + subcarrier_delta_f, 
                                subcarrier_delta_f)
        frequencies = np.tile(frequencies_per_symbol, Nt)
        normalized_frequencies = (frequencies - np.min(frequencies)) / (np.max(frequencies) - np.min(frequencies))
        colors = plt.cm.viridis(normalized_frequencies)

        plt.subplot(326)
        scatter = plt.scatter(data_subcarriers.real, data_subcarriers.imag, c=frequencies, cmap='viridis', label='Received')
        constellation = np.array(list(constellation._symbols_to_bits_map.keys()))
        
        plt.scatter(constellation.real, constellation.imag, c='black', marker='x', label='Ideal')
        plt.title(f"Constellation Diagram MODEL {STATE['run_model']}")
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
    return data_subcarriers.real.tolist(), data_subcarriers.imag.tolist()

def decode_symbols_OFDM(real_symbols: list, imag_symbols: list, true_bits: list,  mode: str) -> list:
    # Instead of grabbing from labview, directly take from decoder

    # symbols = torch.tensor(real_symbols) + 1j * torch.tensor(imag_symbols)

    symbols = STATE['last_received']
    
    # # Grab constellation object
    constellation = get_constellation(mode)
    # true_bits_array = np.array(list(constellation.symbols_to_bits(STATE['last_sent'])))


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
    evm = torch.mean(torch.square(torch.abs(STATE['last_sent'] - STATE['last_received'])))
    # print("evm shapes:", STATE['last_sent'].shape, STATE['last_received'].shape)

    # Log frame BER
    STATE['frame_BER'] = BER

       # Call backprop and log loss
    if 'validate_model' in STATE and STATE['validate_model']:
        cancel_run_early = False
        if STATE['run_model']:
            wandb.log({"freq model evm loss": evm})
        else:
             wandb.log({"freq no model evm loss": evm})
        make_time_validate_plots(STATE['time_encoder_in'], STATE['time_encoder_out'],
                                 STATE['time_decoder_in'], STATE['time_decoder_out'], 
                                 STATE['frame_BER'], STATE['run_model'], STATE['frequencies'])
        

        # Save to Zarr file
        save_validation_data(
            STATE['last_sent'],
            STATE['last_freq_symbol_received'],
            STATE['frequencies'],
            STATE['time_encoder_in'],
            STATE['time_encoder_out'],
            STATE['time_decoder_in'],
            STATE['time_decoder_out'],
            zarr_path = f"C:/Users/Public_Testing/Desktop/peled_interconnect/mldrivenpeled/data/validation_measurements/{wandb.run.name}.zarr",
            metadata={'BER': BER, 'EVM':evm}
        )

    else:
        cancel_run_early = False

    SNR, PowerFactor = float(0), float(0)

    cancel_run_early = False
    return decided_bits_flat, float(BER), evm, PowerFactor, cancel_run_early

def make_time_validate_plots(encoder_in, encoder_out, decoder_in, decoder_out, frame_BER, run_model, freqs=None):
    encoder_in = np.asarray(encoder_in)
    encoder_out = np.asarray(encoder_out)
    decoder_in = np.asarray(decoder_in)
    decoder_out = np.asarray(decoder_out)


    enc_power_in = float(np.mean(np.square(encoder_in.flatten())))
    enc_power_out = float(np.mean(np.square(encoder_out.flatten())))
    enc_scale = enc_power_out / enc_power_in

    dec_power_in = float(np.mean(np.square(decoder_in.flatten())))
    dec_power_out = float(np.mean(np.square(decoder_out.flatten())))
    dec_scale = dec_power_out / dec_power_in

    mse_encoder = np.mean((encoder_in.flatten() - encoder_out.flatten()) ** 2)
    mse_decoder = np.mean((decoder_in.flatten() - decoder_out.flatten()) ** 2)
    min_len = min(len(encoder_in.flatten()), len(decoder_out.flatten()))
    mse_total = np.mean((encoder_in.flatten()[:min_len] - decoder_out.flatten()[:min_len]) ** 2)
    prefix = "validate/time_model_" if run_model else "validate/time_no_model_"
    wandb.log({f"{prefix}mse_loss": mse_total})
    wandb.log({f"{prefix}frame_BER": frame_BER})
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    zoom_samples = 200
    time_points = np.arange(zoom_samples)
    axes[0].plot(time_points, encoder_in.flatten()[2 * zoom_samples: 3 *zoom_samples], 'r', alpha=0.5, label='Encoder Input', linewidth=1)
    axes[0].plot(time_points, encoder_out.flatten()[2 * zoom_samples: 3 *zoom_samples], 'b', alpha=0.8, label='Encoder Output', linewidth=1)
    axes[0].set_title(f"Encoder Comparison (MSE: {mse_encoder:.2e}) | In {enc_power_in: .3f} | Out {enc_power_out: .3f} | Scale {enc_scale: .3f}")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(time_points, decoder_in.flatten()[2 * zoom_samples: 3 *zoom_samples], 'r', alpha=0.5, label='Decoder Input', linewidth=1)
    axes[1].plot(time_points, decoder_out.flatten()[2 * zoom_samples: 3 *zoom_samples], 'b', alpha=0.8, label='Decoder Output', linewidth=1)
    axes[1].set_title(f"Decoder Comparison (MSE: {mse_decoder:.2e}) | In {dec_power_in: .4f} | Out {dec_power_out:.3f} | Scale {dec_scale: .3f}")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(time_points, encoder_in.flatten()[2 * zoom_samples: 3 *zoom_samples], 'r', alpha=0.5, label='Original Input', linewidth=1)
    axes[2].plot(time_points, decoder_out.flatten()[2 * zoom_samples: 3 *zoom_samples], 'b', alpha=0.8, label='Final Output', linewidth=1)
    axes[2].set_title(f"End-to-End Comparison ({'Trained' if run_model else 'Untrained'})\nMSE: {mse_total:.2e}, BER: {frame_BER:.2f}")
    axes[2].legend()
    axes[2].grid(True)
    plot_path = f"wandb_time_domain/{prefix}signals.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    wandb.log({f"{prefix}time_signals": wandb.Image(plot_path)})
    plt.close(fig)

    freq_sent = np.asarray(STATE['last_sent']).flatten()
    freq_received = np.asarray(STATE['last_received']).flatten()
    fig_const, axc = plt.subplots(figsize=(6, 6))
    axc.scatter(freq_sent.real, freq_sent.imag, c='r', alpha=0.5, s=10, label='Sent')
    axc.scatter(freq_received.real, freq_received.imag, c='b', alpha=0.5, s=10, label='Received')
    axc.set_title("Constellation Diagram")
    axc.set_xlabel("In-Phase")
    axc.set_ylabel("Quadrature")
    axc.legend()
    axc.grid(True)
    plot_path_const = f"wandb_time_domain/{prefix}constellation.png"
    os.makedirs(os.path.dirname(plot_path_const), exist_ok=True)
    fig_const.tight_layout()
    fig_const.savefig(plot_path_const, dpi=150)
    wandb.log({f"{prefix}constellation": wandb.Image(plot_path_const)})
    plt.close(fig_const)

    min_len = min(len(freq_sent), len(freq_received))
    freq_sent = freq_sent[:min_len]
    freq_received = freq_received[:min_len]

    evm_per_freq = (np.abs(freq_received - freq_sent) ** 2) / (np.abs(freq_sent) ** 2 + 1e-12)

    freqs_axis = np.arange(min_len)
    fig_evm, axe = plt.subplots(figsize=(10, 4))
    axe.plot(freqs_axis, evm_per_freq, 'b-', linewidth=1)
    axe.set_title("EVM per Frequency")
    axe.set_xlabel("Subcarrier Index")
    axe.set_ylabel("EVM (dB)")
    axe.grid(True)
    plot_path_evm = f"wandb_time_domain/{prefix}evm_per_freq.png"
    os.makedirs(os.path.dirname(plot_path_evm), exist_ok=True)
    fig_evm.tight_layout()
    fig_evm.savefig(plot_path_evm, dpi=150)
    wandb.log({f"{prefix}evm_per_freq": wandb.Image(plot_path_evm)})
    plt.close(fig_evm)

    if os.path.exists(plot_path):
        os.remove(plot_path)
    # if STATE['run_model']:
    #     STATE['run_model'] = False
    # else:
    #     STATE['run_model'] = True
