import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from NeuronRuntimeHelper import NeuronContext

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.96  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 15360 samples

# color mapping
COLORS = {
    "background": "gray",
    "water": "blue",
    "singing": "yellow",
    "abnormal": "red",
    "unknown": "black"
}

def load_and_split_wav(wav_path):
    audio, sr = sf.read(wav_path, dtype='float32')
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # If stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Break into segments
    chunks = []
    num_segments = int(np.ceil(len(audio) / CHUNK_SIZE))
    for i in range(num_segments):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = audio[start:end]

        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)), mode='constant')

        chunks.append(chunk)
    return chunks

def preprocess_audio_to_mel(waveform):
    if len(waveform) < int(15600):
        pad_length = int(15600) - len(waveform)
        waveform = np.pad(waveform, (0, pad_length))
    mel_spec = librosa.feature.melspectrogram(y=waveform,
                                              sr=SAMPLE_RATE,
                                              n_fft=400,
                                              hop_length=160,
                                              n_mels=64,
                                              center=False,
                                              power=2.0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = mel_spec_db.astype(np.float32)
    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=0)  # (1, 64, 96)
    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1) # (1, 64, 96, 1)
    return mel_spec_norm

def run_inference(model_path, wav_path):
    # Load model
    context = NeuronContext(model_path)
    if not context.Initialize():
        raise RuntimeError("Failed to initialize NeuronRuntime")

    labels = ["background", "water", "singing", "abnormal"]

    # Load audio chunks
    chunks = load_and_split_wav(wav_path)
    print(f"Processing {len(chunks)} segments...")

    results = []  # store (waveform, label)
    for idx, waveform in enumerate(chunks, start=1):
        mel_input = preprocess_audio_to_mel(waveform)
        context.SetInputBuffer(mel_input, 0)
        context.Execute()
        real_output = context.GetOutputBuffer(0).flatten()

        top_index = np.argmax(real_output)
        label = labels[top_index]
        results.append((waveform, label))

        print(f"Segment {idx}: {label} ({real_output[top_index]:.4f})")

    # Plot the entire audio with color-coded segments
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title("Audio Classification by Segment")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1, 1)

    for i, (wave, label) in enumerate(results):
        color = COLORS.get(label, COLORS["unknown"])
        x = np.linspace(i * CHUNK_DURATION, (i + 1) * CHUNK_DURATION, len(wave))
        ax.plot(x, wave, color=color)

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bathnet.dla')
    parser.add_argument('--wav', type=str, default='example.wav')
    args = parser.parse_args()

    run_inference(args.model, args.wav)
