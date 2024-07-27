import numpy as np
import matplotlib.pyplot as plt

def analyze_signal_fft(f1, f2, fs, duration):
    # Create time array
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Generate the signal
    s = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    
    # Compute the FFT
    fft_result = np.fft.fft(s)
    
    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(len(t), 1/fs)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:len(freqs)//2], magnitude_spectrum[:len(freqs)//2])
    plt.title('Magnitude Spectrum of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    
    return freqs, magnitude_spectrum

# Parameters from the image
f1 = 50  # Hz
f2 = 120  # Hz
fs = 1000  # Hz (1 kHz sampling rate)
duration = 1  # second

# Call the function
freqs, magnitude_spectrum = analyze_signal_fft(f1, f2, fs, duration)

# Print the frequencies with the highest magnitudes
sorted_indices = np.argsort(magnitude_spectrum)[::-1]
print("Top frequency components:")
for i in range(4):  # Print top 4 frequencies
    freq = abs(freqs[sorted_indices[i]])
    mag = magnitude_spectrum[sorted_indices[i]]
    print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")