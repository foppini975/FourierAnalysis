import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.io import wavfile
import streamlit.components.v1 as components

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

def display_fourier_coefficients_table(a_n, b_n, f0):
    # Calculate the number of harmonics
    n_harmonics = len(a_n)
    
    # Prepare data for the table
    data = {
        "Harmonic Frequency (i * f0)": np.arange(1, n_harmonics + 1) * f0,
        "a_n (i)": a_n,
        "b_n (i)": b_n
    }
    
    # Create a DataFrame for better display in Streamlit
    df = pd.DataFrame(data=data, index=np.arange(1, n_harmonics + 1))
    
    # Display the table in Streamlit
    st.table(df)

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
    axs[1, 1].set_title(f"Fundamental + first {n_harmonics} Harmonics")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Amplitude")
    # axs[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_harmonic_magnitudes(a_n, b_n):
    # Calculate the magnitude for each harmonic
    magnitudes = np.sqrt(np.array(a_n)**2 + np.array(b_n)**2)
    
    # Generate x-axis values (harmonic numbers)
    harmonics = np.arange(len(a_n))
    
    # Plot the magnitudes
    figure = plt.figure(figsize=(8, 5))
    markerline, stemlines, baseline = plt.stem(harmonics, magnitudes)
    plt.setp(markerline, color='blue', marker='o', markersize=6)
    plt.setp(stemlines, color='skyblue')
    plt.setp(baseline, color='gray', linewidth=0.5)
    
    plt.xlabel("Harmonic Number (n)")
    plt.ylabel("Magnitude ( sqrt{a_n^2 + b_n^2} )")
    plt.title("Magnitude of Fourier Coefficients for Each Harmonic")
    plt.tight_layout()
    st.pyplot(figure)

# Function to plot interactive 3D plot of Fourier coefficients
def plot_fourier_coefficients_3d(a_n, b_n):
    # Generate n values (harmonic numbers)
    n_values = np.arange(len(a_n))  # n values from 0 to the number of harmonics
    
    # Create an interactive 3D scatter plot with Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=n_values, y=a_n, z=b_n,
        mode='markers',
        marker=dict(size=5, color=n_values, colorscale='Viridis', opacity=0.8),
        text=[f"Harmonic {n}" for n in n_values]  # Hover text for each point
    )])

    # Update plot layout
    fig.update_layout(
        title="Interactive 3D Plot of Fourier Coefficients (a_n and b_n)",
        scene=dict(
            xaxis_title="Harmonic Number (n)",
            yaxis_title="Cosine Coefficient (a_n)",
            zaxis_title="Sine Coefficient (b_n)"
        ),
        template="plotly_white",
        width=1200,  # chart width
        height=800   # chart height
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

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

    st.header("Waveform Fourier Analysis", divider="gray")
    st.caption(f"The fundamental frequency (f0) automatically detected using the FFT (Fast Fourier Transform) is: {f0:.2f} Hz")
    st.caption("It may happen that FFT returns an harmonic frequency, \
        instead of the fundamental frequency. \
        Therefore you can here divide the detected f0 by an integer factor \
        to adjust its value and improve waveform Fourier analysis.")
    # Slider for adjusting the fundamental frequency
    fundamental_divider = st.slider("Fundamental Divider", 1, 10, 1)
    f0 = f0 / fundamental_divider

    # Slider for selecting the number of harmonics to plot
    n_harmonics = st.slider("Number of Harmonics", 1, 20, 10)

    st.divider()

    # Calculate Fourier coefficients
    a_n, b_n = fourier_coefficients(data, f0, n_harmonics, t)

    st.header("Calculated Fourier Coefficients", divider="gray")

    # Display the Fourier coefficients table
    display_fourier_coefficients_table(a_n, b_n, f0)

    st.header("Waveform Charts", divider="gray")

    # Plot the 2x2 waveforms
    plot_2x2_waveforms(t, data, a_n, b_n, f0, n_harmonics)
    
    st.header("Fourier Coefficients Charts", divider="gray")
    
    # Plot harmonic magnitude
    plot_harmonic_magnitudes(a_n, b_n)

    # Plot Fourier coefficients
    plot_fourier_coefficients_3d(a_n, b_n)
    
