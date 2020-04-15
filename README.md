# Generative_Architectures
Implementation of several GANs and Auto-encoders models

## Introduction
In the last few years, deep learning based generative models have gained more and more interest due to (and implying) some amazing improvements in the field. Relying on huge amount of data, well-designed networks architectures and smart training techniques, deep generative models have shown an incredible ability to produce highly realistic pieces of content of various kind, such as images, texts and sounds. Among these deep generative models, two major families stand out and deserve a special attention: Generative Adversarial Networks (GANs) and Autoencoders (AEs).

## Project's parts
In this project, I implemented a simple GAN model along with three different families of Auto-encoders: traditional auto-encoder(AE), variational auto-encoder(VAE) and adversarial auto-encoder(AAE), also I showed some image alignement methods which could be useful as a pre-processing step before trying to learn the distribution of a given images sequence. In my project, I had a sequence of 90000 images taken by a fixed monocular camera but there might exist some vibrations caused by different weather conditions.

### Image Alignement
In this project, I showed three different methods: one based on features extraction, and the other two methods on the correlation between two input images. First of all, the feature-based method uses the features extracted from two images using a state-of-the art detector like: SIFT, SURF or ORB and matches them to obtain a homography matrix which represent the transformation between the two images (translation, rotation, scale ...) using RANSAC which is an iterative algorithm for the robust estimation of parameters from a subset of inliers from the complete data set. The second one is a wide used method called: Enhanced Correlation Coefficient (ECC) maximization, this method can be used for 2D affine or 3D transformations. And at the end, the FFT correlation method is the simplest one and it ised to calculate the shift between the two input images.

### GANs
Generative Adversarial Networks (GANs for short) have had a huge success. GANs are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. They are used widely in image generation, video generation and voice generation.

### AutoEncoders
The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding-decoding scheme using an iterative optimisation process. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the networks.
Thus, intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part of the information can go through and be reconstructed. Looking at our general framework, the family E of considered encoders is defined by the encoder network architecture, the family D of considered decoders is defined by the decoder network architecture and the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.

### Required python packages:

    OpenCV
    numpy
    skimage
    pytorch

## Installation
To install the project's files, use the following command:
```bash
git clone https://github.com/Ali-Sahili/Generative_Architectures.git
```
## Usage
The image alignement methods are used as pre-processing, thus you can used on your sequence before trying a generative model.
Put the following command in your shell in the Image_Alignement directory:
```python
python3 main.py
```

To generate your own model, put the following one in your terminal and choose your desired architecture:
```python
python3 main.py
```
You can change the parameters of the model and the paths of the input and the output data in the Param.py file.
