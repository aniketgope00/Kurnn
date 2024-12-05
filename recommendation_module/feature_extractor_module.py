import numpy as np
import os
import torchaudio
import librosa
from gammatone.gtgram import gtgram
import soundfile as sf

def extract_audio_features(file_path):
    # Load audio file using soundfile (supports more formats)
    waveform, sample_rate = sf.read(file_path)
    if waveform.ndim > 1:  # Convert to mono if stereo
        waveform = waveform.mean(axis=1)

    # Resample if necessary
    target_sr = 22050
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    # Calculate features
    features = {}

    # 1. Low-level signal parameters
    features["rms_level"] = librosa.feature.rms(y=waveform).mean()
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate).mean()
    features["bandwidth"] = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate).mean()
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y=waveform).mean()

    # Band energy ratio (e.g., low freq vs high freq split at 2000 Hz)
    fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(waveform), d=1/sample_rate)
    low_energy = np.sum(np.abs(fft[freqs < 2000]))
    high_energy = np.sum(np.abs(fft[freqs >= 2000]))
    features["band_energy_ratio"] = low_energy / (high_energy + 1e-8)

    # Delta spectrum magnitude
    spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
    delta_spectral_contrast = librosa.feature.delta(spectral_contrast).mean()
    features["delta_spectrum_magnitude"] = delta_spectral_contrast

    # Pitch and pitch strength
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
    features["pitch"] = pitches.mean()
    features["pitch_strength"] = magnitudes.mean()

    # 2. MFCC coefficients
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    features["mfcc_mean"] = mfcc.mean(axis=1).tolist()  # List for each MFCC coefficient
    features["mfcc_std"] = mfcc.std(axis=1).tolist()

    # 3. Psychoacoustic Features
    # Roughness (placeholder using a simple approximation, not a full psychoacoustic model)
    def calculate_roughness(signal):
        return np.var(np.abs(np.diff(signal)))  # Approximation based on signal variability

    features["average_roughness"] = calculate_roughness(waveform)
    features["std_roughness"] = np.std(np.abs(np.diff(waveform)))

    # Loudness modulation energies
    def modulation_energy(signal, sr, low, high):
        """
        Compute modulation energy for a specified frequency range.
        Dynamically adjusts n_mels if the range is narrow.
        """
    # Ensure the frequency range is valid
        low = max(0, low)  # fmin cannot be negative
        high = min(sr // 2, high)  # fmax must be less than Nyquist frequency

        if high <= low:
            raise ValueError(f"Invalid frequency range: fmin={low}, fmax={high}")

    # Dynamically adjust n_mels based on the frequency range
        freq_range = high - low
        n_mels = max(1, int(freq_range / 100))  # Example heuristic: one mel filter per 100 Hz

        mel_filter = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels, fmin=low, fmax=high)
        fft = np.abs(np.fft.rfft(signal, n=2048))  # FFT with n_fft=2048
        mel_energy = mel_filter @ fft[:mel_filter.shape[1]]  # Apply mel filter
        return mel_energy.mean()



    features["one_2hz_loudness"] = modulation_energy(waveform, sample_rate, 1, 2)
    features["three_15hz_loudness"] = modulation_energy(waveform, sample_rate, 3, 15)
    features["twenty_43hz_loudness"] = modulation_energy(waveform, sample_rate, 20, 43)

    # Sharpness (simple proxy using spectral centroid normalized to high frequencies)
    features["sharpness"] = features["spectral_centroid"] / (np.max(freqs) + 1e-8)

    # 4. Auditory Filterbank Temporal Envelopes
    num_channels = 18
    window_time = 0.025
    hop_time = 0.010
    f_min = 26

    gtgram_output = gtgram(waveform, sample_rate, window_time, hop_time, num_channels, f_min)

    # Summarize temporal envelopes into modulation energy bands
    modulation_bands = {
    "0Hz": None,  # Handle this separately
    "3-15Hz": (3, 15),
    "20-150Hz": (20, 150),
    "150-1000Hz": (150, 1000),
    }

    for band_name, range_values in modulation_bands.items():
        if range_values is None:  # Special case for "0Hz"
            band_energy = gtgram_output[:, 0].mean()
        else:
            low, high = range_values
            band_energy = gtgram_output[:, low:high].sum(axis=1).mean()
        features[f"gamma_{band_name}_energy"] = band_energy
    return features

if __name__ == "__main__":
    # Example usage
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    file_path = "data_loader_spotify/test_audio/Jason Mraz - I Won't Give Up.wav"
    features = extract_audio_features(file_path)
    print(features)
