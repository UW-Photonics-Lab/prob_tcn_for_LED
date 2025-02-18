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
        self.bit_map = generate_bit_map(self._complex_symbols)


    def visualize(self):
        reals = np.real(symbols)
        imags = np.imag(symbols)
        magnitudes = np.sqrt(np.real(symbols.conj() * symbols))
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
        bit_array = np.array(list(self.bit_map.values()))
        if random:
            random_indices = np.random.choice(
                len(self._complex_symbols), size=num_symbols)
            random_symbols = self._complex_symbols[random_indices]
            print(random_indices)
            random_bits = bit_array[random_indices]
            return random_symbols, random_bits
        else:
            return np.resize(self._complex_symbols, num_symbols), np.resize(bit_array, num_symbols)