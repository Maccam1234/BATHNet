# BATHNet

BATHNet is a bathroom safety audio classifier used to identify any abnormal(falls, screams etc.) sounds in the bathroom. This model employs the [Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable convolution architecture, and was trained via transfer learning on [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md), a pretrained general audio classifier.

This directory contains the original YAMNet file, training data, training scripts, inference scripts, final model as well as exploration notebooks.

This model was trained as part of [VIA Technologies](https://www.viatech.com/en/) Summer Internship 2025. 

## Usage

The model was made to run the VIA VAB5000, a Pico-ITX Single Board Computer with an onboard MDLA 3.0 NPU.

Thus:
* The model is saved as a tflite file.
* `inference.py` and `live_inference.py` can only run on VIA products powered by MediaTek Genio 700, as they utilize VIA's `NeuronRuntimeHelper` package.

To run the model on the VAB5000, make sure whichever inference file you use is in the same directory as `bathnet.tflite` on the VAB5000, then compile `bathnet.tflite` into a `.dla` file using the MediaTek NeuronSDK:
```shell
sudo ncc-tflite -arch mdla3.0 --relax-fp32 --opt-accuracy bathnet.tflite
```

This will create a `bathnet.dla` file that will be ran in the inference files.

`inference.py` can be ran with any `.wav` file:
```shell
python3 inference.py --wav <arg>
```

`live_inference.py` can be ran as long as there is a microphone connected:
```shell
python3 live_inference.py
```

### [Model Demonstration](https://www.youtube.com/watch?v=bbdnsictTL4)

## About the Model

### Classes
BATHNet classifies 0.96s second audio inputs into 1 of 4 classes:
* Background - environment noise inside the bathroom and other indoor spaces
* Water - running water from baths, showers and sinks
* Speech/Singing
* Abnormal - falls, screams and various noises from object/human collisions

As the model was built for use in personal bathrooms, these 4 categories cover majority of noises that occur.

### Structure

BATHNet uses YAMNet's pretrained backbone to feature extract 1024D embeddings from 96x64 log mel spectrogram inputs. While YAMNet allows for variable length audio inputs and pools its embeddings for classification, BATHNet only accepts a single mel spectrogram(0.96s audio input) so that a single embedding is extracted. From this, the embedding is classified by our newly trained classification head. The structure can be understood as:

0.96s wav `--preprocess-->` 96x64 mel spectrogram `--YAMNet-feature-extraction-->` 1024D embedding `--BATHNet-classification-head-->` label

More deatils about preprocessing and feature extraction can be found at the [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md) repository.

### Data
Sources:
* [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) - speech.
* [Freesound Oneshot Percussive Sounds](https://zenodo.org/records/4687854) - hand picked for abnormal sounds.
* Own recordings - water, background and abnormal sounds.

I experimented with various datasets and self-collected data. The final model was trained from the aformentioned data, where each class has 200-300 examples.
