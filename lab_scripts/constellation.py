'''
Version 2/18/25
This module aims to provide tools to be called in labview to generate a
train of complex symbols to send through the AWG
'''

import numpy as np
import matplotlib.pyplot as plt

'''
Helper functions
'''
def generate_bit_map(symbols):
    num_symbols = len(symbols)
    modulation_order = np.log2(num_symbols)
    def build_bits(order):
        '''
        Builds bit string combinations recursively
        '''
        if order == 0:
            return [""]
        else:
            sub_combinations = build_bits(order - 1)
            return ["1" + c for c in sub_combinations] + ["0" + c for c in sub_combinations]
        
    bits = build_bits(modulation_order)
    bit_mapping = dict(zip(symbols, bits))
    return bit_mapping

class ConstellationPlot:
    def __init__(self, complex_train):
        self._complex_symbols = complex_train
        self.symbols_to_bit_map = self.generate_bit_map(self._complex_symbols)

        self.bits_to_symbols_map = {v : k for k, v in self.symbols_to_bit_map.items()}

    def generate_bit_map(self, symbols):
        num_symbols = len(symbols)
        modulation_order = np.log2(num_symbols)
        def build_bits(order):
            '''
            Builds bit string combinations recursively
            '''
            if order == 0:
                return [""]
            else:
                sub_combinations = build_bits(order - 1)
                return ["1" + c for c in sub_combinations] + ["0" + c for c in sub_combinations]
            
        bits = build_bits(modulation_order)
        bit_mapping = dict(zip(symbols, bits))
        return bit_mapping
    

    def visualize(self):
        reals = np.real(self._complex_symbols)
        imags = np.imag(self._complex_symbols)
        magnitudes = np.sqrt(np.real(self._complex_symbols.conj() *  self._complex_symbols))
        max_mag = np.max(magnitudes)

        plt.figure(figsize=(4, 4))
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.scatter(reals, imags)
        plt.xlim(-1.5 * max_mag, 1.5 * max_mag)  # Adjust limits to center at origin
        plt.ylim(-1.5 * max_mag, 1.5 * max_mag)
        plt.gca().set_aspect('equal', adjustable='datalim')  # Forces equal spacing
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.title("Complex Symbols")
        plt.show()

    def create_training_data_Ts(self, num_symbols, random=True):
        bit_strings = np.array(list(self.symbols_to_bit_map.values()))
        if random:
            random_indices = np.random.choice(
                len(self._complex_symbols), size=num_symbols)
            random_symbols = self._complex_symbols[random_indices]
            random_bits = bit_strings[random_indices]
            return random_symbols, random_bits
        else:
            return np.resize(self._complex_symbols, num_symbols), np.resize(bit_strings, num_symbols)

    
'''
Functions that can be called in LabView
'''

def generate_qpsk_training_data(num_symbols: int, random: bool=False) -> tuple[np.array, np.array, list[str]]:
    '''
    Returns a tuple with (real, imag, associated_bits)
    '''
    qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    qpsk_const = ConstellationPlot(qpsk_symbols)
    complex_symbols, bits = qpsk_const.create_training_data_Ts(num_symbols, random)
    real_parts = complex_symbols.real
    imag_parts = complex_symbols.imag
    # LabView doesn't support non-contiguous arrays
    real_parts = np.ascontiguousarray(real_parts)
    imag_parts = np.ascontiguousarray(imag_parts)
    bits = [str(b) for b in bits]
    return real_parts, imag_parts, bits