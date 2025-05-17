import numpy as np
import pandas as pd
from lab_scripts.constellation_OLD import ConstellationPlot
from scipy.signal import resample_poly
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
qpsk_const = ConstellationPlot(qpsk_symbols)
SYMBOLS_TO_BITS = qpsk_const.symbols_to_bit_map
AWG_MEMORY_LENGTH = 16384 # full length
barker_code = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
BARKER_LENGTH = int(0.01 * (AWG_MEMORY_LENGTH))

def add_preamble_func(x_t: list) -> list:
    """Adds a known preamble sequence for frame synchronization
    
    Args:
        x_t: FODM time domain signal
        num_carriers: Number of carriers

    Returns:
        List of OFDM time domain signals with premable
        Barker code to send to demodulator

    """

    x_t_array = np.array(x_t)
    barker_code = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
    barker_code = np.repeat(barker_code, BARKER_LENGTH // len(barker_code)) # Set as 1%

    barker_repeated = np.tile(barker_code, (len(x_t_array), 1))
    
    # Stack horizontally
    output = np.hstack([barker_repeated, x_t_array])
    
    # Convert back to list of lists for LabVIEW compatibility
    return [list(np.real(row)) for row in output], list(barker_code), len(barker_code)

def OFDM_time_x_t(real_part: list, imag_part: list, num_carriers: int, cyclic_prefix: bool, add_preamble: bool) -> list:
    '''Converts I, Q baseband of symbols to x(t) OFDM symbol groupings
    
        Args:
            real_part: real part of symbols
            imag_part: imaginary part
            num_carriers: number of ODFM carriers
            cyclic_prefix: whether to add 25% cyclic prefix

        returns:
            An 2D array of dimensions (num_symbols, num_carriers) if the data transmitted
            cannot be cleanly reshaped into these dimensions, zero padding is employed.
    '''

    if len(real_part) != len(imag_part):
        raise ValueError("Length of real and imaginary inputs not equal")
    
    if num_carriers % 2 != 0:
        raise ValueError("Number of carriers must be even for Hermitian symmetry")

    
    # Convert to complex array
    complex_symbols = np.array(real_part) + 1j * np.array(imag_part)

    num_data_carriers = (num_carriers // 2) - 1 # -1 for DC offset and Nyquist

     # Calculate number of complete OFDM symbols we can make
    n_symbols = int(np.ceil(len(complex_symbols) / num_data_carriers))

    # Pad input to fit complete number of OFDM symbols
    target_length = n_symbols * num_data_carriers
    if len(complex_symbols) < target_length:
        pad_length = target_length - len(complex_symbols)
        complex_symbols = np.pad(complex_symbols, (0, pad_length))

    # Reshape to group symbols
    n_symbols = len(complex_symbols) // num_data_carriers
    data_groups = complex_symbols.reshape(n_symbols, num_data_carriers)
    
    # Create full OFDM symbol with Hermitian symmetry
    hermitian_symbol_groups = np.zeros((n_symbols, num_carriers), dtype=complex)
    
    for i in range(n_symbols):
        # DC term = 0
        hermitian_symbol_groups[i, 0] = 0
        # Data carriers
        hermitian_symbol_groups[i, 1:num_data_carriers+1] = data_groups[i]
        # Nyquist term = 0
        hermitian_symbol_groups[i, num_carriers//2] = 0
        # Conjugate symmetric part
        hermitian_symbol_groups[i, (num_carriers//2)+1:] = np.conj(np.flip(data_groups[i]))
    
    # Serial to parallel
    symbol_groups = hermitian_symbol_groups.reshape(-1, num_carriers)

    cyclic_prefix_len = AWG_MEMORY_LENGTH // 4 if cyclic_prefix else 0 # 25% CP
    
    ifft_length = int(AWG_MEMORY_LENGTH + -BARKER_LENGTH + -cyclic_prefix_len)
    # perform IDFT
    x_t_groups = np.fft.ifft(symbol_groups, n=ifft_length, axis=1) # Add zero padding 
    x_t_groups_normalized = x_t_groups * (1 / np.max(x_t_groups, axis=1).reshape(-1, 1)) # Normalize to ensure preamble on same scale as waveform

    if cyclic_prefix:
        # Extract CP from end of each symbol
        cp = x_t_groups_normalized[:, -cyclic_prefix_len:]
        # Concatenate CP to beginning of each symbol
        x_t_groups_normalized = np.hstack([cp, x_t_groups_normalized])

    output = []
    for row in np.real(x_t_groups_normalized):
        output.append(list(row))

    if add_preamble:
        output, preamble, _ = add_preamble_func(output)
    return output, preamble


# def demodulate_OFDM_one_symbol_frame(y_t:list,
#                                      num_carriers: int,
#                                      CP_length: int,
#                                      freq_AWG: float,
#                                      osc_sample_rate: float,
#                                      record_length: int,
#                                      preamble_sequence: list,
#                                      ) -> list:
#     '''Converts received y(t) into a bit string

#     '''


#     # Save the last frame for bugtesting
#     df = pd.DataFrame({
#         'y_t': y_t,
#         'num_carriers': [num_carriers] * len(y_t),
#         'CP_length': [CP_length] * len(y_t),
#         'freq_AWG': [freq_AWG]* len(y_t),
#         'osc_sample_rate': [osc_sample_rate]* len(y_t),
#         'record_length': [record_length]* len(y_t),
#     })
#     df.to_csv('C:/Users/Public_Testing/Desktop/peled_interconnect/mldrivenpeled/data/last_ofdm_voltages.csv', index=False, mode='w')

#     t_frame = 1 / freq_AWG
#     scope_samples_per_frame = int(osc_sample_rate * t_frame)
#     number_of_frames = record_length / scope_samples_per_frame

#     num_output_points = int((num_carriers + len(preamble_sequence)) * number_of_frames)
#     resampled_voltages = resample_poly(y_t, up=num_output_points, down=len(y_t))

#     preamble_sequence = np.array(preamble_sequence)
#     corr = signal.correlate(resampled_voltages, preamble_sequence, mode='valid')

#     # Find correlation peaks
#     peaks, _ = find_peaks(corr, height=0.9*np.max(corr), distance=len(preamble_sequence))

#     # Parameters
#     ofdm_frame_length = num_carriers  # Length of each OFDM frame
#     preamble_length = len(preamble_sequence)  # Length of Barker sequence

#     # Extract frames after each preamble
#     frames = []
#     for peak in peaks:
#         # Calculate start and end positions for frame extraction
#         frame_start = peak + preamble_length  # Start after preamble
#         frame_end = frame_start + ofdm_frame_length
        
#         # Extract the frame
#         frame = resampled_voltages[frame_start:frame_end]
#         frames.append(frame)

#     num_data_carriers = (num_carriers // 2) - 1
    
#     frame_y_t = frames[0]

#     # Remove Cyclic Prefix
#     y_t = y_t[CP_length:]

#     # Perform FFT to convert to frequency domain
#     Y_s = np.fft.fft(y_t)

#     # Extract data subcarriers
#     num_data_carriers = (num_carriers // 2) - 1
#     data_subcarriers = Y_s[1:num_data_carriers + 1]

#     # Demap symbols to bits
#     constellation = np.array(list(SYMBOLS_TO_BITS.keys()))
#     distances = abs(data_subcarriers.reshape(-1, 1) - constellation.reshape(1, -1))
#     closest_idx = distances.argmin(axis=1)
#     decisions = constellation[closest_idx]
#     decided_bits = [SYMBOLS_TO_BITS[symbol] for symbol in decisions]
#     return decided_bits


# This is the same version as above except there is no downsampling before taking FFT
# So preamble is upsampled instead

def demodulate_OFDM_one_symbol_frame(y_t:list,
                                     num_carriers: int,
                                     CP_length: int,
                                     freq_AWG: float,
                                     osc_sample_rate: float,
                                     record_length: int,
                                     preamble_sequence: list, 
                                     true_bits: list) -> list:
    '''Converts received y(t) into a bit string with optional debugging plots'''


    # Save the last frame for bugtesting
    df = pd.DataFrame({
        'y_t': y_t,
        'num_carriers': [num_carriers] * len(y_t),
        'CP_length': [CP_length] * len(y_t),
        'freq_AWG': [freq_AWG]* len(y_t),
        'osc_sample_rate': [osc_sample_rate]* len(y_t),
        'record_length': [record_length]* len(y_t),
    })
    df.to_csv('C:/Users/Public_Testing/Desktop/peled_interconnect/mldrivenpeled/data/last_ofdm_voltages.csv', index=False, mode='w')
    
    debug_plots = False
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
    peaks, _ = find_peaks(corr, height=0.9*np.max(corr), distance=len(preamble_sequence_upsampled))

    
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
        plt.title('First Extracted Frame')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

    if len(frames) == 0:
        raise ValueError("No valid frames detected")

    frame_y_t = frames[0]

    # Perform FFT with proper normalization
    Y_s = np.fft.fft(frame_y_t)

    if debug_plots:
        # Plot 5: FFT magnitude
        plt.subplot(325)
        plt.plot(np.abs(Y_s))
        plt.title('FFT Magnitude')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude')

    # Extract both positive and negative frequency carriers
    num_data_carriers = (num_carriers // 2) - 1
    positive_carriers = Y_s[1:num_data_carriers + 1]
    negative_carriers = Y_s[-num_data_carriers:]
    data_subcarriers = positive_carriers
    data_subcarriers = data_subcarriers * (np.max(np.abs(qpsk_const._complex_symbols)) / np.max(np.abs(data_subcarriers)))
    
    if debug_plots:
        # Plot 6: Constellation Diagram
        plt.subplot(326)
        plt.scatter(data_subcarriers.real, data_subcarriers.imag, c='b', label='Received')
        constellation = np.array(list(SYMBOLS_TO_BITS.keys()))
        plt.scatter(constellation.real, constellation.imag, c='r', marker='x', label='Ideal')
        plt.title('Constellation Diagram')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    # Demap symbols to bits
    constellation = np.array(list(SYMBOLS_TO_BITS.keys()))
    distances = abs(data_subcarriers.reshape(-1, 1) - constellation.reshape(1, -1))
    closest_idx = distances.argmin(axis=1)
    decisions = constellation[closest_idx]
    decided_bits = [SYMBOLS_TO_BITS[symbol] for symbol in decisions]

    # Better bit handling - process individual bits
    decided_bits_flat = []
    for symbol_bits in decided_bits:
        # Handle each bit in the symbol's bit string individually
        for bit in symbol_bits:
            decided_bits_flat.append(int(bit))
    decided_bits_flat = np.array(decided_bits_flat, dtype=np.int32)

    # Convert true bits similarly
    true_bits_flat = []
    for symbols_bits in true_bits:
        for bit in symbols_bits:
            true_bits_flat.append(int(bit))
    true_bits = np.array(true_bits_flat, dtype=np.int32)

    # Take minimum length to avoid shape mismatch
    min_len = min(len(true_bits), len(decided_bits_flat))
    true_bits = true_bits[:min_len]
    decided_bits_flat = decided_bits_flat[:min_len]


    # Save arrays for debugging with fixed path
    debug_dir = r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled\data'
    debug_output = os.path.join(debug_dir, 'debug_bits.txt')
    
    # Create directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)
    
    with open(debug_output, 'w') as f:
        f.write("True bits array:\n")
        f.write(','.join(map(str, true_bits)))
        f.write("\n\nDecided bits array:\n")
        f.write(','.join(map(str, decided_bits_flat)))
        f.write("\n\nComparison (1 means different):\n")
        f.write(','.join(map(str, (true_bits != decided_bits_flat).astype(int))))
        f.write(f"\n\nNumber of differences: {np.sum(true_bits != decided_bits_flat)}")
        f.write(f"\nTotal bits compared: {len(true_bits)}")
        f.write(f"\nBER: {float(np.sum(true_bits != decided_bits_flat) / len(true_bits))}")

    # Calculate BER
    BER = float(np.sum(true_bits != decided_bits_flat) / len(true_bits))
    
    return decided_bits, BER