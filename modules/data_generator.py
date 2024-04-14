import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from tqdm import tqdm


class WaveformGenerator(ABC):
    def __init__(self, freq: float, dt: float):
        """
        :freq: frequency of the generated waveform
        :dt: discrete time step
        """
        self.freq = freq
        self.dt = dt

    @abstractmethod
    def generate(self, duration: float):
        raise NotImplementedError()

    def get_time_array(self, duration: float) -> np.array:
        """Get the time array, where the time difference between two adjacent time slots is the delta time"""
        samples = np.ceil(duration / self.dt)
        return np.arange(0, (samples + 1) * self.dt, self.dt)


class SineWaveGenerator(WaveformGenerator):
    def __init__(self, amplitude: float, freq: float, dc: float, dt: float):
        super(SineWaveGenerator, self).__init__(freq, dt)
        self.amplitude = amplitude
        self.freq = freq
        self.dc = dc

    def generate(self, duration: float) -> np.array:
        t = self.get_time_array(duration)
        return self.amplitude * np.sin(2 * np.pi * self.freq * t) + self.dc


class SquareWaveGenerator(WaveformGenerator):
    def __init__(self, amplitude: float, dc: float, duty_cycle: float, freq: float = 1, dt: float = 1e-6):
        assert 0 < duty_cycle < 1

        super(SquareWaveGenerator, self).__init__(freq, dt)
        self.amplitude = amplitude
        self.freq = freq
        self.dc = dc
        self.duty_cycle = duty_cycle

    def generate(self, duration: float):
        t = self.get_time_array(duration)
        period = int(1 / self.freq // self.dt)
        high_period = int(period * self.duty_cycle)
        low_period = period - high_period

        p, q = 0, 0
        output = np.zeros(t.size)
        while q < t.size:
            while q - p < high_period and q < t.size:
                output[q] = 1
                q += 1
            q += low_period
            p = q

        return self.amplitude * output + self.dc


class NaiveRCCircuit:
    """
    Naive series RC circuit where a capacitor and a resistor. The resistance of the resistor depends on the accumulated
    charge flux through the resistor. In Latex format, R(t) = R_0 + \alpha \int_{0}^{t} i(\tau)\,d\tau, where alpha is
    a coefficient, and R_0 is the base resistance of the resistor.

    The output voltage is measured on the capacitor. It can be calculated using the following differential equations.

    Latex format:
        - R(t) = R_0 + \alpha \int_{0}^{t} i(\tau)\,d\tau
        - i(t) = C\frac{\,dV_{out}(t)}{\,dt}
        - v_{in}(t) - v_{out}(t) = i(t)R(t)
        - Initial condition: v_{in}(0) = 0 and v_{out}(0) = 0
    """

    def __init__(self, c: float, r0: float, alpha: float):
        """
        Initialize the circuit system
        :c: capacitance of the capacitor
        :r0: base resistance of the resistor
        :alpha: coefficient of the charge accumulation
        """
        self.c = c
        self.r0 = r0
        self.alpha = alpha

    def simulate(self, v_in: np.array, dt: float) -> np.array:
        """
        Given input waveform starting from t=0, calculate output waveform
        :v_in: input waveform as 1-D numpy array
        :dt: delta time between two adjacent data points of the input waveform
        """
        v_out = np.zeros(v_in.size, dtype=np.float64)
        for i in range(1, v_out.size):
            tmp = self.r0 + self.alpha * self.c * v_out[i - 1]
            v_out[i] = v_out[i - 1] + dt * (v_in[i - 1] - v_out[i - 1]) / self.c / tmp

        return v_out


def run_input_output_experiment(
        circuit_system: NaiveRCCircuit,
        waveform_generator: WaveformGenerator,
        duration: float) -> (np.array, np.array):
    """
    Run an experiment and achieve the output with the given circuit system and the waveform generator
    :circuit_system: the given RC circuit
    :waveform_generator: the waveform generator
    :duration: the time span of the experiment
    :return: a tuple of an (input, output) pair
    """
    v_in = waveform_generator.generate(duration)
    v_out = circuit_system.simulate(v_in, waveform_generator.dt)
    return v_in, v_out


def calculate_response(v_in: np.array, v_out: np.array, freq: float, dt: float,
                       min_periods_to_equilibrium: int = 20) -> float:
    """
    Calculate the response from input and output waveforms in dB, assuming the out waveform reach the equilibrium state
    :v_in: input waveform
    :v_out: output waveform
    :freq: frequency of the input waveform
    :dt: the discrete time step between two adjacent elements in the input and output waveforms
    :min_periods_to_equilibrium: the number of periods when the output can reach equilibrium
    """
    assert v_in.size == v_out.size
    assert v_in.size * dt >= (min_periods_to_equilibrium + 1) / freq

    start_index = int(min_periods_to_equilibrium / freq / dt) + 1
    diff_out = max(v_out[start_index:]) - min(v_out[start_index:])
    diff_in = max(v_in[start_index:]) - min(v_in[start_index:])

    return 20 * (np.log(diff_out) - np.log(diff_in))


def run_frequency_scanning_experiment(
        circuit_system: NaiveRCCircuit,
        waveform_generator: WaveformGenerator,
        start_freq: float,
        end_freq: float,
        num_freq_samples: int,
        input_output_cycles: int,
        min_periods_to_equilibrium: int = 20) -> (np.array, np.array):
    """
    Run an experiment and achieve the with the given circuit system and the waveform generator
    :circuit_system: the given RC circuit
    :waveform_generator: the waveform generator
    :start_freq: start frequency
    :end_freq: end frequency
    :num_freq_samples: the number of frequency samples
    :input_output_cycles: the number of period cycles in each input-output experiment
    :min_periods_to_equilibrium: the number of periods when the output can reach equilibrium
    :return: an array of (frequency, frequency response) pair. The response is in dB
    """
    assert start_freq >= 1  # only support frequency no less than 1 Hz
    assert input_output_cycles > 20  # assume the output waveform will reach equilibrium after at least 20 cycles
    freq = np.linspace(start_freq, end_freq, num=num_freq_samples)
    responses = np.zeros(freq.size)
    for i in tqdm(range(freq.size)):
        waveform_generator.freq = freq[i]
        waveform_generator.dt = 1 / freq[i] / 1e3
        duration = 1 / freq[i] * input_output_cycles
        v_in, v_out = run_input_output_experiment(circuit_system, waveform_generator, duration)
        responses[i] = calculate_response(v_in, v_out, freq[i], waveform_generator.dt, min_periods_to_equilibrium)

    return freq, responses


def search_low_pass_3db_frequency(
        circuit_system: NaiveRCCircuit,
        waveform_generator: WaveformGenerator,
        start_freq: float,
        end_freq: float,
        input_output_cycles: int,
        freq_resolution: float = 1e-6,
        min_periods_to_equilibrium: int = 20) -> float:
    """
    Calculate the -3dB frequency of the given circuit.
    :circuit_system: the given RC circuit
    :waveform_generator: the waveform generator
    :start_freq: start frequency
    :end_freq: end frequency
    :input_output_cycles: the number of period cycles in each input-output experiment
    :freq_resolution: the resolution of the -3dB frequency
    :min_periods_to_equilibrium: the number of periods when the output can reach equilibrium
    :return: the -3dB frequency in Hz. If not found between the start and end frequency, return -1
    """
    assert start_freq >= 1  # only support frequency no less than 1 Hz
    assert input_output_cycles > 20  # assume the output waveform will reach equilibrium after at least 20 cycles

    left = start_freq
    right = end_freq
    # assume the circuit is low pass filter, use binary search to search for the -3dB frequency
    while left <= right:
        mid = (left + right) / 2
        waveform_generator.freq = mid
        waveform_generator.dt = 1 / mid / 1e3
        duration = 1 / mid * input_output_cycles
        v_in, v_out = run_input_output_experiment(circuit_system, waveform_generator, duration)
        response = calculate_response(v_in, v_out, mid, waveform_generator.dt, min_periods_to_equilibrium)

        if response == -3:
            return mid
        elif response > -3:
            left = mid + freq_resolution
        else:
            right = mid - freq_resolution

    if left > end_freq:
        return -1

    return left


if __name__ == '__main__':
    circuit = NaiveRCCircuit(c=1e-3, r0=1, alpha=3e3)

    # Example: running input-output experiment
    print('Running input-output experiment')
    freq = 3e2
    dt = 1 / freq / 1e3
    duration = 1 / freq * 50
    generator = SquareWaveGenerator(amplitude=1, dc=0, duty_cycle=0.5, freq=freq, dt=1e-6)
    v_in, v_out = run_input_output_experiment(circuit, generator, duration)
    t = generator.get_time_array(duration)
    plt.plot(t, v_in)
    plt.plot(t, v_out)
    plt.title('Input-Output Experiment Example')
    plt.xlabel('time (sec)')
    plt.ylabel('voltage (V)')
    plt.grid()
    plt.show()

    # Example: running frequency scanning experiment
    print('Running frequency scan experiment')
    start_freq = 1
    end_freq = 3e2
    n_samples = 500
    n_cycles = 50
    generator = SquareWaveGenerator(amplitude=1, dc=0, duty_cycle=0.5)
    frequencies, responses = run_frequency_scanning_experiment(circuit, generator, start_freq, end_freq, n_samples,
                                                               n_cycles, min_periods_to_equilibrium=20)
    plt.plot(frequencies, responses)
    plt.title('Frequency Scanning Experiment Example')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('response (dB)')
    plt.grid()
    plt.show()

    # Example: running find -3dB experiment
    print('Running search -3dB experiment')
    duty_cycle = 0.5
    start_freq = 1
    end_freq = 1e3
    n_cycles = 50
    generator = SquareWaveGenerator(amplitude=1, dc=0, duty_cycle=duty_cycle)
    cutoff_freq = search_low_pass_3db_frequency(circuit, generator, start_freq, end_freq,
                                                n_cycles, min_periods_to_equilibrium=20)
    if cutoff_freq < 0:
        print(f'Cannot find -3dB frequency between {start_freq} Hz and {end_freq} Hz')
    else:
        print(f'-3dB frequency: {cutoff_freq} Hz')
