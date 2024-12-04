import torchaudio

file_path = "data_loader_spotify/test_audio/Jason Mraz - I Won't Give Up.wav"
waveform, sample_rate = torchaudio.load(file_path, normalize=True)
transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                       n_mfcc=20,
                                       melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},)
mfcc = transform(waveform)
print(mfcc)