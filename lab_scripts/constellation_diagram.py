import numpy as np
import matplotlib.pyplot as plt


from abc import ABC, abstractmethod

class ConstellationDiagram(ABC):

    def __init__(self, complex_symbols: np.array):
        self._complex_symbols = complex_symbols
        self._symbols_to_bits_map = self._generate_bit_map(complex_symbols)
    
    @abstractmethod
    def _generate_bit_map(self, complex_symbols: np.array) -> dict[complex, str]:
        pass
    
    @abstractmethod
    def symbols_to_bits(self, complex_symbols: np.array) -> str:
        pass

    @abstractmethod
    def bits_to_symbols(self, bits: str) -> np.array:
        pass
    
    @abstractmethod
    def visualize_constellation(self):
        pass


class QPSK_Constellation(ConstellationDiagram):
    def __init__(self):
        qpsk_symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        super().__init__(qpsk_symbols)
        self.modulation_order = 2

    def _generate_bit_map(self, complex_symbols):
        num_symbols = len(complex_symbols)
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
        bit_mapping = dict(zip(complex_symbols, bits))
        return bit_mapping
    
    def bits_to_symbols(self, bits: str) -> np.array:
        # Group strings into size by modulation order
        if len(bits) % 2 != 0:
            # Padd by a zero
            bits = bits + "0"
        grouped_bits = [bits[x : x + self.modulation_order] for x in range(0, len(bits), self.modulation_order)]
        bits_to_symbol_map = {v : k for k, v in self._symbols_to_bits_map.items()}
        symbols = np.array([bits_to_symbol_map[group] for group in grouped_bits])
        return symbols
    
    def symbols_to_bits(self, complex_symbols):
        bit_groups = [self._symbols_to_bits_map[s] for s in complex_symbols]
        bits = "".join(bit_groups)
        return bits
    
    def visualize_constellation(self):
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