// ASR_MODEL_CPP

#include "../include/asr.h"
#include "../include/ctc.h"

#include <numeric>
#include <fstream>
#include <random>
#include <algorithm>
#include <iostream>

using namespace std;

// Конструктор с инициализацией всех слоев
ASRModel::ASRModel() 
    : conv1(1, 32),
      conv2(32, 64),
      lstm1(64, 128),
      lstm2(128, 128),
      linear_weights(128, std::vector<float>(29)),
      optimizer(0.001f),
      alphabet("abcdefghijklmnopqrstuvwxyz' ") 
{
    // Инициализация весов линейного слоя
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& row : linear_weights) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

std::vector<std::vector<float>> ASRModel::forward(
    const std::vector<std::vector<std::vector<float>>>& input) 
{
    // 1. CNN слои
    auto cnn_out1 = conv1.forward(input);
    auto cnn_out2 = conv2.forward(cnn_out1);
    
    // 2. Подготовка для LSTM
    int time_steps = cnn_out2[0].size();
    int features = cnn_out2.size();
    
    std::vector<std::vector<float>> lstm_input(time_steps, std::vector<float>(features));
    for (int t = 0; t < time_steps; ++t) {
        for (int c = 0; c < features; ++c) {
            lstm_input[t][c] = cnn_out2[c][t][0];
        }
    }
    
    // 3. Первый LSTM
    auto lstm1_out = lstm1.forward_sequence(lstm_input);
    
    // 4. Второй LSTM 
    auto lstm2_out = lstm2.forward_sequence(lstm1_out);
    
    // 4. Линейный слой
    std::vector<std::vector<float>> output(time_steps, std::vector<float>(29));
    for (int t = 0; t < time_steps; ++t) {
        for (int c = 0; c < 29; ++c) {
            output[t][c] = 0.0f;
            for (int f = 0; f < 128; ++f) {
                output[t][c] += lstm2_out[t][f] * linear_weights[f][c];
            }
        }
    }
    
    return output;
}

void ASRModel::train(
    const std::vector<std::vector<std::vector<float>>>& spectrogram,
    const std::vector<int>& transcript)
{
    // Forward pass
    auto output = forward(spectrogram);
    
    // Backward pass
    std::vector<std::vector<float>> grad_output(output.size(), 
        std::vector<float>(29, 0.0f));
    
    // 1. Градиент для CTC loss
    for (size_t t = 0; t < output.size(); ++t) {
        int target = (t < transcript.size()) ? transcript[t] : 0;
        grad_output[t][target] = output[t][target] - 1.0f;
    }
    
    // 2. Градиент для линейного слоя -> LSTM2
    std::vector<std::vector<float>> lstm2_grad(output.size(), 
        std::vector<float>(128, 0.0f));
    
    for (size_t t = 0; t < output.size(); ++t) {
        for (int f = 0; f < 128; ++f) {
            for (int c = 0; c < 29; ++c) {
                lstm2_grad[t][f] += grad_output[t][c] * linear_weights[f][c];
            }
        }
    }
    
    // 3. Backward через LSTM2 -> LSTM1
    std::vector<std::vector<float>> lstm1_grad = lstm2.backward(lstm2_grad);
    lstm1.backward(lstm1_grad);
    
    // 4. Обновление весов линейного слоя
    std::vector<std::vector<float>> lstm2_out = lstm2.get_last_output();
    for (int f = 0; f < 128; ++f) {
        for (int c = 0; c < 29; ++c) {
            float grad = 0.0f;
            for (size_t t = 0; t < output.size(); ++t) {
                grad += lstm2_out[t][f] * grad_output[t][c];
            }
            linear_weights[f][c] -= 0.001f * grad;
        }
    }
}

/*
void ASRModel::train(
    const std::vector<std::vector<std::vector<float>>>& spectrogram,
    const std::vector<int>& transcript)
{
    // Forward pass
    auto output = forward(spectrogram);
    
    // Сохраняем выход LSTM для backward pass
    //auto lstm_out = lstm1.get_last_output(); // Добавьте этот метод в LSTM
    // Градиент для второго LSTM
    std::vector<std::vector<float>> lstm2_grad = lstm2.backward(output);
    
    // Градиент для первого LSTM
    std::vector<std::vector<float>> lstm1_grad = lstm1.backward(lstm2_grad);

    auto lstm2_out = lstm2.get_last_output();
    for (int f = 0; f < 128; ++f) {
        for (int c = 0; c < 29; ++c) {
            float grad = 0.0f;
            for (size_t t = 0; t < output.size(); ++t) {
                grad += lstm2_out[t][f] * output[t][c];
            }
            linear_weights[f][c] -= 0.001f * grad;
        }
    }
    
    // Backward pass
    std::vector<std::vector<float>> grad_output(output.size(), 
        std::vector<float>(29, 0.0f));
    
    // Вычисляем градиент для CTC loss
    for (size_t t = 0; t < output.size(); ++t) {
        int target = (t < transcript.size()) ? transcript[t] : 0;
        grad_output[t][target] = output[t][target] - 1.0f;
    }
    
    // Преобразуем градиент к размерности LSTM
    std::vector<std::vector<float>> lstm2_grad(output.size(), 
        std::vector<float>(128, 0.0f));
    
    for (size_t t = 0; t < output.size(); ++t) {
        for (int f = 0; f < 128; ++f) {
            for (int c = 0; c < 29; ++c) {
                lstm2_grad[t][f] += grad_output[t][c] * linear_weights[f][c];
            }
        }
    }
    
    // Backward через LSTM
    auto lstm1_grad = lstm2.backward(lstm2_grad);
    lstm1.backward(lstm1_grad);  // Теперь размерности совпадают
    
    // 4. Обновление весов линейного слоя
    auto lstm2_out = lstm2.get_last_output();
    for (int f = 0; f < 128; ++f) {
        for (int c = 0; c < 29; ++c) {
            float grad = 0.0f;
            for (size_t t = 0; t < output.size(); ++t) {
                grad += lstm2_out[t][f] * grad_output[t][c];
            }
            linear_weights[f][c] -= 0.001f * grad;
        }
    }
    
    // Обновление весов линейного слоя
    for (int f = 0; f < 128; ++f) {
        for (int c = 0; c < 29; ++c) {
            float grad = 0.0f;
            for (size_t t = 0; t < output.size(); ++t) {
                grad += lstm2_out[t][f] * grad_output[t][c]; // Теперь lstm_out доступен
            }
            linear_weights[f][c] -= 0.001f * grad;
        }
    }
}
*/


// Функция распознавания
std::string ASRModel::recognize(
    const std::vector<std::vector<std::vector<float>>>& spectrogram)
{
    auto output = forward(spectrogram);
    return CTCDecoder::greedy_decode(output, alphabet);
}

void ASRModel::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    conv1.save(file);
    conv2.save(file);
    lstm1.save(file);
    lstm2.save(file);
}

void ASRModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    conv1.load(file);
    conv2.load(file);
    lstm1.load(file);
    lstm2.load(file);
}