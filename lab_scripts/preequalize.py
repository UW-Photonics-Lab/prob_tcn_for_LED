'''
Author: Dylan Jones

Purpose: Apply prequalization to waveform before transmitting in LabView

'''


DAMAGE_MAX = 10
DAMAGE_MIN = 0
import numpy as np

def apply_pre_equalization(wave_form: tuple[list, list]) -> tuple[list, list]:
    reals = np.array(wave_form[0])
    imag = np.array(wave_form[1])
    return reals.tolist(), imag.tolist()


def apply_linear_pre_distortion(wave_form: list, check_domain: bool) -> tuple[list, list]:
    voltages = np.array(wave_form)


    # Check domain
    max_linear_voltage = 5.896
    min_linear_voltage = 2.933

    if check_domain:
        if (voltages > max_linear_voltage).any():
            error_voltage = np.max(voltages)
            raise ValueError(f"Input voltages must be lower than max permitted voltage: {max_linear_voltage} Current bad voltage: {error_voltage}")

        elif (voltages < min_linear_voltage).any():
            error_voltage = np.min(voltages)
            raise ValueError(f"Input voltages must be greater than min permitted voltage: {min_linear_voltage} Current bad voltage: {error_voltage}")
    
    # Apply linear map
    m, b = 0.856835, -2.513462
    linear_map = lambda v: m * v + b


    f_i_coeffs = {0: -0.055763157637478265,
                    1: 0.47570864380437466,
                    2: -1.6506370266105774,
                    3: 3.0830393596413117,
                    4: -3.288791762193383,
                    5: 3.0183795735904764,
                    6: 2.5161904989663246}


    f_i_numpy = np.poly1d(list(f_i_coeffs.values()))

    f_i = lambda i: f_i_numpy(i).astype(float)
    shifted_voltages = f_i(linear_map(voltages))

        # Check if clipping will occur
    if np.any(shifted_voltages > DAMAGE_MAX) or np.any(shifted_voltages < DAMAGE_MIN):
        print(f"Warning: Some voltages were clipped to stay within [{DAMAGE_MIN}, {DAMAGE_MAX}]V")
    
    safe_voltages = np.clip(shifted_voltages, DAMAGE_MIN, DAMAGE_MAX)

    safe_voltages = list(shifted_voltages)
    return safe_voltages
    

def apply_digital_zobel_network(wave_form: list, check_domain: bool) -> tuple[list, list]:
    A1 = 10
    A2 = 5
    f0 = 8e6

    voltages = np.array(wave_form)


    def H_dzn(f: float, A1: float, A2: float, f0: float, RL=50) -> float:
        '''Transfer function for Zobel Network:
        
        Args: 
            f: frequency at which the transfer function is calculated

            Use "A Gb/s VLC Transmission Using Hardware Preequalization Circit" as a reference
            for component values and meaning

            f0: bandwidth frequency
            A1: dB gain at 0 frequency 
            A2: dB gain at f0/2
            RL: Load resistor (50 Ohm default)


        '''
        def R4_func(A1: float, RL) -> float:
            '''Calculates R4

                Args:
                    A1: desired gain in (dB) at 0 frequency
            '''
            denom = (10 ** (A1 / 20)) - 1
            return RL / denom
        
        
        def L(f0: float, A1: float, A2: float, RL) -> float:
            R4 = R4_func(A1, RL)
            denom = (10 ** (A2 / 10) - 1) * ((4 * np.pi * f0) / 3) ** 2
            num = (R4 + RL) ** 2 - (10 ** (A2 / 10)) * (R4 ** 2)
            num = (R4 + RL) ** 2 - (10 ** (A2 / 10) * (R4 ** 2))
            return np.sqrt(num / denom)
        
            
        def C(f0, A1, A2, RL) -> float:
            return 1 / ((L(f0, A1, A2, RL)) * np.square(2 * np.pi * f0))


        C1 = C(f0, A1, A2, RL)
        L1 = L(f0, A1, A2, RL)
        R4 = R4_func(A1, RL)


        omega = 2 * np.pi * f
        denom_1 = 1 - ((omega ** 2) * C1 * L1) + 1e-12
        num_1 = 1j * omega * L1

        denom_2 = R4 + num_1 / denom_1
        
        denom_3 = (1 + RL / denom_2)

        return 1 / denom_3
    

    def get_x_prime(x_original: np.array, H=H_dzn) -> np.array:
        '''Applies Zobel network in the time domain

            Args:
                x_original: transmitted voltages to be equalized
                f_bandwidth: max bandwidth frequency of signal
        '''
        X_f = np.fft.fft(x_original)
        N = len(x_original)
        freqs = np.fft.fftfreq(N, d=1/N)
        H_at_freqs = H_dzn(freqs, A1, A2, f0, RL=50)
        X_f_prime = H_at_freqs * X_f
        x_prime = np.real(np.fft.ifft(X_f_prime))
        return x_prime


    voltages_prime = get_x_prime(voltages)
    # Check if clipping will occur
    if np.any(voltages_prime > DAMAGE_MAX) or np.any(voltages_prime < DAMAGE_MIN):
        print(f"Warning: Some voltages were clipped to stay within [{DAMAGE_MIN}, {DAMAGE_MAX}]V")
    
    safe_voltages = np.clip(voltages_prime, DAMAGE_MIN, DAMAGE_MAX)

    safe_voltages = list(voltages_prime)
    return safe_voltages