import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to calculate Fourier coefficients for sine and cosine components
def fourier_coefficients(data, f0, harmonics, t):
    N = len(data)
    a_n = []
    b_n = []

    for n in range(harmonics + 1):
        cos_term = np.cos(2 * np.pi * n * f0 * t)
        sin_term = np.sin(2 * np.pi * n * f0 * t)
        a_n.append(2 * np.dot(data, cos_term) / N)
        b_n.append(2 * np.dot(data, sin_term) / N)

    return a_n, b_n

# Function to plot the original waveform, fundamental, 1st harmonic, and sum of first 10 harmonics
def plot_2x2_waveforms(t, data, a_n, b_n, f0, n_harmonics):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # (1) Original waveform
    axs[0, 0].plot(t, data, label="Original Waveform", color='blue')
    axs[0, 0].set_title("Original Waveform")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    # axs[0, 0].legend()

    # (2) Fundamental frequency
    fundamental = a_n[1] * np.cos(2 * np.pi * 1 * f0 * t) + b_n[1] * np.sin(2 * np.pi * 1 * f0 * t)
    axs[0, 1].plot(t, fundamental, label="Fundamental Frequency", color='orange')
    axs[0, 1].set_title(f"Fundamental Frequency f0={f0:.0f}Hz (n=1)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Amplitude")
    # axs[0, 1].legend()

    # (3) 1st harmonic (n=2)
    first_harmonic = a_n[2] * np.cos(2 * np.pi * 2 * f0 * t) + b_n[2] * np.sin(2 * np.pi * 2 * f0 * t)
    axs[1, 0].plot(t, first_harmonic, label="1st Harmonic", color='green')
    axs[1, 0].set_title(f"1st Harmonic f={2*f0:.0f}Hz (n=2)")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Amplitude")
    # axs[1, 0].legend()

    # (4) Sum of fundamental + next 9 harmonics
    sum_of_harmonics = np.zeros(len(t))
    for i in range(1, n_harmonics):
        sum_of_harmonics += a_n[i] * np.cos(2 * np.pi * i * f0 * t) + b_n[i] * np.sin(2 * np.pi * i * f0 * t)
    axs[1, 1].plot(t, sum_of_harmonics, label=f"Fundamental + {n_harmonics-1} Harmonics", color='red')
    axs[1, 1].plot(t, data, label="Original Waveform", color='blue', linestyle='dotted')
    axs[1, 1].set_title(f"Fundamental + first {n_harmonics-1} Harmonics")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Amplitude")
    # axs[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_fourier_coefficients_3d(a_n, b_n):
    # Generate n values (harmonic numbers)
    n_values = np.arange(len(a_n))  # n values from 0 to the number of harmonics
    
    # Create a 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a_n and b_n in 3D space
    ax.scatter(n_values, a_n, b_n, color='purple', label="Fourier Coefficients")

    # Set labels
    ax.set_xlabel("Harmonic Number (n)")
    ax.set_ylabel("Cosine Coefficient (a_n)")
    ax.set_zlabel("Sine Coefficient (b_n)")
    ax.set_title("3D Plot of Fourier Coefficients ( a_n ) and ( b_n )")

    # Display legend and plot
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App
st.title("Fourier Analysis of a WAV File")

# File uploader for WAV file
uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

# If file is uploaded, process the file
if uploaded_file is not None:
    # Read the WAV file
    sample_rate, data = wavfile.read(uploaded_file)
    
    # Ensure mono by selecting first channel if stereo
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Set up time vector
    N = len(data)
    T = N / sample_rate
    t = np.linspace(0, T, N, endpoint=False)

    # FFT to estimate fundamental frequency
    fft_values = np.fft.fft(data)
    frequencies = np.fft.fftfreq(N, 1/sample_rate)
    f0 = abs(frequencies[np.argmax(np.abs(fft_values[:N // 2]))])

    # Slider for adjusting the fundamental frequency
    fundamental_divider = st.slider("Fundamental Divider", 1, 10, 1)
    f0 = f0 / fundamental_divider

    # Slider for selecting the number of harmonics to plot
    n_harmonics = st.slider("Number of Harmonics", 1, 20, 10)

    # Calculate Fourier coefficients
    a_n, b_n = fourier_coefficients(data, f0, n_harmonics, t)

    # Plot the 2x2 waveforms
    plot_2x2_waveforms(t, data, a_n, b_n, f0, n_harmonics)
    
    # Plot Fourier coefficients
    plot_fourier_coefficients_3d(a_n, b_n)
