# DeepFake-Audio-Recognition

![App](./images/app_run.png)

## Introduction

This is a app that uses Convolutional Neural Network (CNN) to detect real vs deepfake audio. It has acheived a test accuracy of ~95%.

The model accepts mel spectrograms of 128 mels and 5.0 seconds duration (matrix dimensions: 128x216) and gives a single output between 0 and 1, 0 indicating real audio and 1 indicating deepfake audio.

## Dataset

The model is trained on this [Kaggle Dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition)
It contains Real(8) and Transformed/DeepFake(32) audios from 8 celebrities/politicians. Due to data imbalance, model trained on all the real audios and 8 to 16 randomly chosen deepfakes.

## Model Specifications

Size of model: **~66 MB**

No. of model parameters: **~8 M**

## Limitations

The model is light-weight and trained on a small portion of the dataset. Hence its accuracy to unforeseen data is very limited. It is still a work in progress, and hence, the size, data and features can be extended with time. Current capacity of the model is quite constrained. 

## Deployment

The app is deployed on Streamlit Cloud: <https://deepfake-audio-recognition.streamlit.app>
