# Automatic Speech Recognition (ASR) System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CMake](https://img.shields.io/badge/build-CMake-brightgreen)](https://cmake.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/w/cpp/17)

An end-to-end Automatic Speech Recognition system implementing CNN/LSTM architecture with CTC loss, trained on LibriSpeech dataset.

## Features

- Audio preprocessing (MFCC feature extraction)
- Deep learning architecture (CNN + BiLSTM)
- Connectionist Temporal Classification (CTC) loss
- Both training and inference modes
- LibriSpeech dataset compatibility

## Prerequisites

- CMake ≥ 3.10
- C++17 compatible compiler
- Libsndfile (`libsndfile1-dev` on Ubuntu)
- Make/Build essentials

## Installation

```bash
git clone https://github.com/Pihtaw/asr.git
cd asr
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```


# 1. Training the Model (Default Mode)
The main program is configured to train on LibriSpeech data by default:

```bash
./bin/asr
```
Configuration:

- Update dataset_path in main.cpp to point to your LibriSpeech data
- Adjust hyperparameters in asr.h (epochs, learning rate, etc.)

# 2. Inference Mode (Speech Recognition)
To use the pre-trained model for transcription:

Uncomment the inference block in `main.cpp`:

```cpp
int main() {
    ASRModel model;
    model.load("trained_model.bin");
    
    auto spectrogram = get_processed_audio("test.wav");
    string text = model.recognize(spectrogram);
    cout << "Recognized: " << text << endl;
}
```
Rebuild and run:

```bash
make && ./bin/asr
```
