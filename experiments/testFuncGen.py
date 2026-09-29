import os
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
import matplotlib.pyplot as plt
import numpy as np
import time

CONFIG_FILE = "testFuncGen.yml"

# set the cwd to be parent folder and not workspace root
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# create blocks for sending sine wave a gathering waveform
class SendSine(ActionBlock):
    def __init__(self, fs, freq, fn_generator_driver):
        super().__init__(fs)
        self.driver = fn_generator_driver
        self.freq = freq
    
    def action(self, x: Signal):
        self.driver.apply_output_fun(
            waveform="SIN",
            frequency=self.freq,
            amplitude="1.0",
            offset="0.0")
        

class SendArbWaveform(ActionBlock):
    def __init__(self, fs, fn_generator_driver, waveform_data: np.ndarray, freq: float,
                 amplitude: float, offset: float):
        super().__init__(fs)
        self.driver = fn_generator_driver
        self.waveform_data = waveform_data
        self.freq = freq
        self.amplitude = amplitude
        self.offset = offset
        assert len(waveform_data) <= 16_384, "Agilent 33250A max arbitrary waveform length is 16384 points"
    
    def action(self, x: Signal):
        self.driver.send_arbitrary(
            samples=self.waveform_data,
            freq=self.freq,
            amplitude=self.amplitude,
            offset=self.offset)
        #print(f"Errors: {self.driver.drain_errors()}")
        
class MeasureSine(ActionBlock):
    def __init__(self, fs, osc_driver, input_signal_frequency, channel):
        super().__init__(fs)
        self.driver = osc_driver
        self.input_signal_frequency = input_signal_frequency
        self.channel = channel

    def action(self, x: Signal) -> Signal:
        _, x.data = self.driver.measure_waveform(channel=self.channel)
        x.sampling_rate = self.driver.input_sample_rate

class DisplayWaveform(FunctionalBlock):
    def __init__(self, fs):
        super().__init__(fs)

    def transform(self, x):
        plt.plot(x.data)
        plt.title("Measured Waveform")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude (Volts)")
        plt.grid()
        plt.show()
        return x

class ResampleWaveform(ResamplingBlock):
    def __init__(self, fs_in, fs_out):
        super().__init__(fs_in, fs_out)

    def resample(self, x: Signal) -> Signal:
        '''Pass here as sending sine does not produce a signal with data,
           so resampling is not needed. In a real use case, this block would
           resample the input signal from fs_in to fs_out.
        '''
        x.sampling_rate = self.fs_out
        return x

if __name__ == '__main__':
    '''Without the main guard, every child process will rerun the code'''
    """
    - test using higher baud
    - test using hardware flow control
    - test using software flow control
        - test using both binary and ascii

    - impliment a timer so you can see how long each process took 
    """
    
    
    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE) as Exp:

        FREQS = [1e4, 2e4] # Hz

        fn_gen = Exp.Agilent_33250A

        for f in FREQS:

            Exp.log(f"Starting Sine sending at {f}!")

            N = 16384  # max arbitrary waveform length for Agilent 33250A
            t = np.linspace(0, 1, N, endpoint=False)
            waveform_data = 1 - 2 * np.abs(2 * t - 1)
            
            send = SendArbWaveform(fs=fn_gen.output_sample_rate,
                              freq=f,
                              fn_generator_driver=fn_gen,
                              waveform_data=waveform_data,
                              amplitude=1.0,
                              offset=0.0
                            )
            x = Signal(data=None, sampling_rate=fn_gen.output_sample_rate)

            start = time.perf_counter()
            send.run(x)
            total_time = time.perf_counter() - start
            print(f"Finished in {total_time} seconds")