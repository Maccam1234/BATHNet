import threading
import queue
import numpy as np
import sounddevice as sd
import librosa
import time
from NeuronRuntimeHelper import NeuronContext
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from collections import deque


SAMPLE_RATE = 16000
CHUNK_DURATION = 0.96  # seconds
CHUNK_SIZE = 15360
AUDIO_QUEUE = queue.Queue(maxsize=3)  # holds 3 chunks at most to avoid memory bloat


# scrolling wave history
HISTORY_SECONDS = 30
NUM_CHUNKS = int(HISTORY_SECONDS / CHUNK_DURATION)
BUFFER = deque(maxlen=NUM_CHUNKS)  # stores (waveform, label)
BUFFER_LOCK = threading.Lock()


# color mapping
COLORS = {
    "background": "gray",
    "water": "blue",
    "singing": "yellow",
    "abnormal": "red",
    # fallback
    "unknown": "black"
}



def record_loop():
    while True:
        audio = sd.rec(CHUNK_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        waveform = audio.flatten()
        AUDIO_QUEUE.put(waveform)




def preprocess_audio_to_mel(waveform):
    if len(waveform) < 15600:
        waveform = np.pad(waveform, (0, 15600 - len(waveform)))


    mel_spec = librosa.feature.melspectrogram(y=waveform,
                                               sr=SAMPLE_RATE,
                                               n_fft=400,
                                               hop_length=160,
                                               n_mels=64,
                                               center=False,
                                               power=2.0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = mel_spec_db.astype(np.float32)
    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=0)     # (1, 64, 96)
    mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)    # (1, 64, 96, 1)
    return mel_spec_norm




def inference_loop(context, labels):
    while True:
        waveform = AUDIO_QUEUE.get()  # waits until data available
        mel_input = preprocess_audio_to_mel(waveform)


        #execute model
        context.SetInputBuffer(mel_input, 0)
        context.Execute()
        real_output = context.GetOutputBuffer(0).flatten()
        

        top_indices = np.argsort(real_output)[-4:][::-1]
        print("\nTop predictions:")
        for i in top_indices:
            print(f"{labels[i]}: {real_output[i]:.4f}")


        top_index = np.argmax(real_output)
        with BUFFER_LOCK:
            timestamp = time.time()
            BUFFER.append((timestamp, waveform.copy(), labels[top_index]))




def plot_loop():
    fig, ax = plt.subplots()
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, HISTORY_SECONDS)
    ax.set_title("Live Audio Waveform with Classification")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    legend_handles = [
    mpatches.Patch(color="gray",   label="Background"),
    mpatches.Patch(color="blue",   label="Water"),
    mpatches.Patch(color="yellow", label="Speech/Singing"),
    mpatches.Patch(color="red",    label="Abnormal")
    ]
    
    ax.legend(handles=legend_handles, loc="upper right")


    def update(frame):
        now = time.time()
        with BUFFER_LOCK:
            buffer_copy = list(BUFFER)


        ax.clear()
        ax.set_ylim(-1, 1)
        ax.set_xlim(-HISTORY_SECONDS, 0)  # Show last 30 seconds
        ax.set_title("Live Audio Waveform with Classification")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(handles=legend_handles, loc="upper right")


        for timestamp, wave, label in buffer_copy:
            time_offset = timestamp - now  # Should be negative
            x = np.linspace(time_offset, time_offset + CHUNK_DURATION, len(wave))
            if x[-1] < -HISTORY_SECONDS or x[0] > 0:
                continue  # Outside plot range
            color = COLORS.get(label, COLORS["unknown"])
            ax.plot(x, wave, color=color)


        return []

    ani = animation.FuncAnimation(fig, update, interval=120)
    plt.show()




def main(model_path):
    context = NeuronContext(model_path)
    if not context.Initialize():
        raise RuntimeError("Failed to initialize NeuronRuntime")

    labels = ["background", "water", "singing", "abnormal"]

    # Start recorder thread
    recorder = threading.Thread(target=record_loop, daemon=True)
    recorder.start()

    # Start inference loop in a thread
    inference_thread = threading.Thread(target=inference_loop, args=(context, labels), daemon=True)
    inference_thread.start()


    # Run plot in main thread
    try:
        plot_loop()
    except KeyboardInterrupt:
        print("\nExiting...")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bathnet.dla')
    args = parser.parse_args()

    main(args.model)











