import numpy as np
import pandas as pd
import os
import tensorflow as tf
import librosa

# read the whole directory of data and produce all waveforms + labels into a dataframe
def parseAudioData(data_path):
    print("Parsing Data...")
    audio_data = []
    for i in os.listdir(data_path):
        filename = data_path+i
        filename = filename.format(i=i)
        for j in os.listdir(filename):
            path = os.path.join(filename, j)
            if os.path.isfile(path) and path.lower().endswith('.wav'):
                try:
                    waveform, sr = librosa.load(path, sr=16000)
                    audio_data.append([waveform, i])
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    audio_dataframe = pd.DataFrame(audio_data, columns=["audio_data", "class"])
    return audio_dataframe

# pad waveform to target length
def pad_waveform(waveform, target_length):
    if len(waveform) < target_length:
        return np.pad(waveform, (0, target_length - len(waveform)))
    return waveform

# array of waveforms --> array of spectorgrams
def getMelSpectrograms(audio_data):
    sample_rate = 16000
    n_fft = 400          # ~25ms window
    hop_length = 160     # ~10ms stride (gives 96 frames over 0.96s)
    n_mels = 64
    target_length = 15600

    spectrograms = []
    for waveform in audio_data:
        waveform = pad_waveform(waveform, target_length)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0  # power=2.0 for power spectrogram 
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        spectrograms.append(log_mel_spec)

    return spectrograms

# one hot encoder for labels
def OneHotEncoder(df):
    class_names = ["background", "water", "singing", "abnormal"]
    
    # Create a lookup table
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(class_names, tf.range(len(class_names), dtype=tf.int64)),
        num_oov_buckets=1
    )
    
    # Convert the "class" column to a Tensor
    label_tensor = tf.constant(df["class"].to_list())
    
    label_indices = table.lookup(label_tensor)
    return tf.one_hot(label_indices, depth=len(class_names))


# audio dataframe --> features/labels --> melspectrogram/onehot --> tf dataset
def preprocess(audio_dataframe):
    print("Preprocessing data...")
    audio_dataframe["audio_data"] = audio_dataframe["audio_data"].apply(lambda x: pad_waveform(x, 15360))
    audio_dataframe["audio_data"] = audio_dataframe["audio_data"].apply(lambda x: x[:15360])
    audio_data = np.array(audio_dataframe["audio_data"].to_list())
    spectrograms = getMelSpectrograms(audio_data)
    onehot = OneHotEncoder(audio_dataframe)
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, onehot))
    return dataset









