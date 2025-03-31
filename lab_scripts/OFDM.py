import numpy as np

def add_preamble_func(x_t: list, num_carriers: int) -> list:
    """Adds a known preamble sequence for frame synchronization
    
    Args:
        x_t: FODM time domain signal
        num_carriers: Number of carriers

    Returns:
        List of OFDM time domain signals with premable

    """

    x_t_array = np.array(x_t)
    barker_code = np.array([1, 1, -1, -1, 1, 1])

    barker_repeated = np.tile(barker_code, (len(x_t_array), 1))
    
    # Stack horizontally
    output = np.hstack([barker_repeated, x_t_array])
    
    # Convert back to list of lists for LabVIEW compatibility
    return [list(np.real(row)) for row in output]

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
    
    # perform IDFT
    x_t_groups = np.fft.ifft(symbol_groups, axis=1)

    if cyclic_prefix:
        cyclic_prefix_len = num_carriers // 4  # 25% CP
        # Extract CP from end of each symbol
        cp = x_t_groups[:, -cyclic_prefix_len:]
        # Concatenate CP to beginning of each symbol
        x_t_groups = np.hstack([cp, x_t_groups])

    output = []
    for row in np.real(x_t_groups):
        output.append(list(row))

    if add_preamble:
        output = add_preamble_func(output, num_carriers)
    return output