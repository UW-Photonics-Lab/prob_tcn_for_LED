import numpy as np

def make_sinc() -> list:
    t = np.linspace(-1, 1, 1000)
    return np.sinc((2 * np.pi * t))