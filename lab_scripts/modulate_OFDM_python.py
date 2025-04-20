import sys
module_dir = r'C:\Users\Public_Testing\Desktop\peled_interconnect\mldrivenpeled'
if module_dir not in sys.path:
    sys.path.append(module_dir)
from lab_scripts.constellation_diagram import QPSK_Constellation
import numpy as np

def modulate_data_OFDM(mode: str, num_carriers: int, data: list[int]) -> tuple[list, list]:
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

    # Calculate the maximum number of frequency carriers
    # N_max = np.floor(f_max / f_min)
    # For now, I will just assume a set amount of carriers
    N_max = num_carriers

    # Calculate amount that can actually carry data
    N_data = int(N_max // 2) - 1
    
    # Grab constellation object
    if mode == "qpsk":
        constellation = QPSK_Constellation()
    
    encoded_symbols = constellation.bits_to_symbols(bits)
    true_bits = np.array(data)
    
    # Reshape such that each row is a frame to be transmitted
    if len(encoded_symbols) % N_data != 0:
        # Padd for full frame
        zeros_to_add = N_data - len(encoded_symbols) % N_data
        bits_added = constellation.symbols_to_bits(np.zeros(zeros_to_add))
        encoded_symbols = np.hstack((encoded_symbols, np.zeros(zeros_to_add)))
        true_bits = np.hstack((true_bits, bits_added))

    encoded_symbol_frames = encoded_symbols.reshape(-1, N_data)
    encoded_symbol_frames_real = encoded_symbol_frames.real.copy()
    encoded_symbol_frames_imag = encoded_symbol_frames.imag.copy()


    # Reshape true bits by frame
    number_of_frames, _ = encoded_symbol_frames.shape
    grouped_true_bits = true_bits.reshape(number_of_frames, -1)

    grouped_true_bits_list = [[str(element) for element in row] for row in grouped_true_bits.tolist()]

    # Return as real and imaginary parts
    return encoded_symbol_frames_real, encoded_symbol_frames_imag, grouped_true_bits_list