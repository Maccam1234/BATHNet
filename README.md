# BATHNet

BATHNet is a bathroom saftey audio classifier used to identify any abnormal(falls, screams etc.) sounds in the bathroom. 
This model employs the [Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable convolution architecture, 
and was trained via transfer learning on [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md),
a pretrained general audio classifier.

This model was trained as part of [VIA Technologies](https://www.viatech.com/en/) Summer Internship 2025. 

# Usage

The model was made to run the VIA VAB5000, a Pico-ITX Single Board Computer with an onboard MDLA 3.0 NPU.

Thus:
* The model is saved as a tflite file.
* `live_inference.py` can only run the model on the VAB5000, as it utilizes VIA's `NeuronRuntimeHelper` package.

To run the model on the VAB5000, make sure both `bathnet.tflite` and `live_inference.py` are in the same directory on the
VAB5000, then compile `bathnet.tflite` into a `.dla` file using the Mediatek NeuronSDK:

```shell
sudo ncc-tflite -arch mdla3.0 --relax-fp32 --opt-accuracy bathnet.tflite
```
This will create a `bathnet.dla` file that will be ran in `live_inference.py`.
