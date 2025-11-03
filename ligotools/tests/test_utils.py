import numpy as np
from ligotools import utils as utils
from scipy.io import wavfile # Need this for the write_wavfile test
import os # Need this for the write_wavfile test

def test_whiten_logic():
    """
    Tests the whiten function with a perfectly flat "mock" PSD.
    
    If the Power Spectral Density (PSD) is 1 everywhere, "whitening"
    (dividing by sqrt(PSD)) should just be a scaling operation.
    """
    print("Running test_whiten_logic...")
    
    # 1. Setup simple inputs
    dt = 0.01
    t = np.arange(0, 10, dt) # 1000 points
    strain = np.cos(2 * np.pi * 1 * t) # Simple 1 Hz cosine wave
    
    # 2. Create a mock PSD function that always returns 1.0
    # This simulates a perfectly flat noise spectrum.
    def mock_psd(freqs):
        return np.ones_like(freqs)
        
    # 3. Calculate the expected output
    # From the function: norm = 1./np.sqrt(1./(dt*2))
    norm = np.sqrt(2 * dt)
    # If PSD is 1, the output should just be the input scaled by 'norm'
    expected_output = strain * norm
    
    # 4. Run the function
    white_ht = utils.whiten(strain, mock_psd, dt)
    
    # 5. Check the result
    # We use np.allclose for floating-point comparisons
    assert np.allclose(white_ht, expected_output), "Whitening with flat PSD failed scaling check"
    
    print("...test_whiten_logic PASSED")


def test_freq_shift_logic():
    """
    Tests the reqshift function.
    
    It creates a simple sine wave at a known frequency, shifts it,
    and then finds the peak frequency of the output to confirm it moved.
    """
    print("\nRunning test_freq_shift_logic...")

    # 1. Setup simple inputs
    sample_rate = 4096
    T = 1.0 # 1 second of data
    f_in = 100.0 # Input frequency
    f_shift = 50.0 # Frequency to shift by
    f_expected = f_in + f_shift # 150.0 Hz
    
    # 2. Create the input signal
    t = np.arange(0, T, 1.0/sample_rate)
    data_in = np.sin(2 * np.pi * f_in * t)
    
    # 3. Run the function
    data_out = utils.reqshift(data_in, fshift=f_shift, sample_rate=sample_rate)
    
    # 4. Check the output
    # Find the peak frequency in the output signal
    fft_out = np.fft.rfft(data_out)
    freqs = np.fft.rfftfreq(len(data_out), 1.0/sample_rate)
    peak_index = np.argmax(np.abs(fft_out))
    peak_freq = freqs[peak_index]
    
    # 5. Check the result
    # The frequency resolution is 1/T = 1 Hz, so we check within 1 Hz.
    assert np.isclose(peak_freq, f_expected, atol=1.0), f"Expected freq {f_expected}, but got {peak_freq}"

    print("...test_freq_shift_logic PASSED")


if __name__ == "__main__":
    print("--- Starting Simple Tests ---")
    test_whiten_logic()
    test_freq_shift_logic()
    print("\n--- All Simple Tests Passed! ---")