import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
class ConstellationDiagram(ABC):

    def __init__(self, complex_symbols: np.array):
        # Normalize to unit average power
        avg_power = np.mean(np.abs(complex_symbols) ** 2)
        normalized_symbols = complex_symbols / np.sqrt(avg_power)
        self._complex_symbols = normalized_symbols
        self._symbols_to_bits_map = self._generate_bit_map(self._complex_symbols)

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


from typing import Union
class RingShapedConstellation(ConstellationDiagram):
    def __init__(self, filename: Union[str, np.ndarray]):
        if isinstance(filename, str):
            symbols = np.load(filename)
        else:
            symbols = filename

        assert len(symbols) & (len(symbols) - 1) == 0, "Number of symbols must be a power of 2"
        self.modulation_order = int(np.log2(len(symbols)))
        super().__init__(symbols)

    def _generate_bit_map(self, complex_symbols: np.ndarray) -> dict[complex, str]:
        """Use deterministic order (e.g., radius then angle) to assign bits."""
        num_symbols = len(complex_symbols)
        bit_width = int(np.log2(num_symbols))

        def int_to_bin(i, width):
            return format(i, f'0{width}b')

        # Sort by radius, then angle
        sort_key = lambda s: (np.round(np.abs(s), 8), np.angle(s))
        sorted_symbols = sorted(complex_symbols, key=sort_key)
        bit_strings = [int_to_bin(i, bit_width) for i in range(num_symbols)]

        return dict(zip(sorted_symbols, bit_strings))

    def bits_to_symbols(self, bits: str) -> np.ndarray:
        if len(bits) % self.modulation_order != 0:
            bits = bits.ljust(len(bits) + (self.modulation_order - len(bits) % self.modulation_order), '0')
        grouped = [bits[i:i+self.modulation_order] for i in range(0, len(bits), self.modulation_order)]
        inv_map = {v: k for k, v in self._symbols_to_bits_map.items()}
        return np.array([inv_map[b] for b in grouped])

    def symbols_to_bits(self, complex_symbols: np.ndarray) -> str:
        return "".join([self._symbols_to_bits_map[s] for s in complex_symbols])

    def visualize_constellation(self):
        symbols = self._complex_symbols
        reals = symbols.real
        imags = symbols.imag
        plt.figure(figsize=(6, 6))
        plt.scatter(reals, imags, edgecolors='k', s=60)
        for s, b in self._symbols_to_bits_map.items():
            plt.text(s.real + 0.02, s.imag + 0.02, b, fontsize=8)
        plt.axhline(0, color='gray', linewidth=1)
        plt.axvline(0, color='gray', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal')
        plt.title("Loaded Ring-Based Constellation")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.show()

def get_constellation(mode: str):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base = os.path.join(PROJECT_ROOT, "modules", "saved_constellations")
    if mode == "qpsk":
        constellation = QPSK_Constellation()
    if mode == "m5_apsk_constellation":
        return RingShapedConstellation(filename=os.path.join(base, "m5_apsk_constellation.npy"))
    if mode == "m6_apsk_constellation":
        return RingShapedConstellation(filename=os.path.join(base, "m6_apsk_constellation.npy"))
    if mode == "m7_apsk_constellation":
        return RingShapedConstellation(filename=os.path.join(base, "m7_apsk_constellation.npy"))
    return None