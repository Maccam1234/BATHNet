import tensorflow as tf

def main(filename):
    converter = tf.lite.TFLiteConverter.from_saved_model(f"models/{filename}")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(f"models/tflite/{filename}.tflite", "wb") as f:
        f.write(tflite_quant_model)

    print('File converted')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='bathnet')
    args = parser.parse_args()

    main(args.input)