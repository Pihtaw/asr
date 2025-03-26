// asr.h
#ifndef ASR_H
#define ASR_H

#include <vector>
#include <cmath>
#include <fstream>
#include <complex>
#include <string>
#include "lstm.h"
#include "ctc.h"

// Конфигурация
constexpr int SAMPLE_RATE = 16000;
constexpr int FRAME_SIZE = 512;
constexpr int HOP_LENGTH = 160;
constexpr int N_MELS = 80;

// Прототипы функций обработки аудио
std::vector<float> read_wav(const std::string& filename);
std::vector<std::vector<float>> compute_mel_spectrogram(const std::vector<float>& audio);
std::vector<std::vector<std::vector<float>>> get_processed_audio(const std::string& filename);
std::vector<float> read_flac(const std::string& filename);


// Класс CNN слоя
class ConvLayer {
private:
    std::vector<std::vector<std::vector<float>>> kernels; // [out_channels][3][3]
    std::vector<float> biases;
    int in_channels;
    int out_channels;

public:
    ConvLayer(int in_ch, int out_ch);
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input);

    void save(std::ofstream& file) const;
    void load(std::ifstream& file);
};

// Класс LSTM слоя
class ASRModel {
private:
    ConvLayer conv1;
    ConvLayer conv2;
    LSTMLayer lstm1;
    LSTMLayer lstm2;
    std::vector<std::vector<float>> linear_weights;
    SGDOptimizer optimizer;
    std::string alphabet;

public:
    ASRModel();
    
    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<std::vector<float>>>& input
    );
    
    void train(
        const std::vector<std::vector<std::vector<float>>>& spectrogram,
        const std::vector<int>& transcript
    );
    
    std::string recognize(
        const std::vector<std::vector<std::vector<float>>>& spectrogram
    );
    
    void save(const std::string& path);
    void load(const std::string& path);
};

#endif
