'''
Author: Dylan Jones

Purpose: Apply prequalization to waveform before transmitting in LabView

'''

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

    f_i = lambda i: f_i_numpy(i).astype(float).tolist()

    shifted_voltages = list(f_i(linear_map(voltages)))
    return shifted_voltages
    