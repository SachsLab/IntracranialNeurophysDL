import numpy as np


def generate_pink_noise(n_samples, n_channels=1):
    uneven = n_samples % 2
    n_fft = n_samples // 2 + 1 + uneven
    X = np.random.randn(n_fft, n_channels) + 1j * np.random.randn(n_fft, n_channels)
    S = np.sqrt(np.arange(n_fft) + 1.)  # +1 to avoid divide by zero
    y = (np.fft.irfft(X / S[:, np.newaxis], axis=0)).real
    if uneven:
        y = y[:-1, :]
    return y * np.sqrt(1 / np.mean(y**2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    FS = 30000      # samples per second
    DURATION = 5.0  # seconds
    NCHANNELS = 128
    SIG_FREQ = FS / 4

    t = np.arange(0.0, DURATION, 1 / FS)
    n_samples = len(t)
    signal = np.zeros((n_samples, NCHANNELS))
    signal[:, 0] = 0.1 * np.sin(SIG_FREQ * np.pi * t)
    pink_noise = generate_pink_noise(n_samples, NCHANNELS)
    signal += pink_noise

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

    # plot time signal:
    axes[0].set_title("Signal")
    axes[0].plot(t, signal[:, 0])
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")
    axes[1].set_title("Log. Magnitude Spectrum")
    axes[1].magnitude_spectrum(signal[:, 0], Fs=FS, scale='dB')

    fig.tight_layout()
    plt.show()
