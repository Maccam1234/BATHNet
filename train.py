import utils
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def main(data_path, save_path):
    try:
        audio_dataframe = utils.parseAudioData(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
    
    dataset = utils.preprocess(audio_dataframe)
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(2)
    dataset_batched = dataset.batch(16)

    val_size = int(0.2 * len(dataset_batched))
    val_ds = dataset_batched.take(val_size)
    train_ds = dataset_batched.skip(val_size)

    # load yamnet
    try:
        yamnet1024 = tf.keras.models.load_model('yamnet_1024_64x96_tl.h5/')
    except Exception as e:
        print(f"Error loading YAMNet: {e}")

    # Keep all layers exept the classification head
    new_input = yamnet1024.input
    x = yamnet1024.get_layer("permute")(new_input)
    x = yamnet1024.get_layer("model")(x)

    # CUSTOM classification head
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=new_input, outputs=outputs)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # train
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping])

    tf.saved_model.save(model, save_path)
    print('Model saved')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='training_data/')
    parser.add_argument('--output', type=str, default='models/bathnet')
    args = parser.parse_args()

    main(args.input, args.output)