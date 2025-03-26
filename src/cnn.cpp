// CNN_CPP

#include "../include/asr.h"

#include <random>

ConvLayer::ConvLayer(int in_ch, int out_ch) 
    : in_channels(in_ch), out_channels(out_ch) {
    
    std::default_random_engine generator;
    float limit = sqrt(6.0f / (in_channels * 9 + out_channels * 9));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    kernels.resize(out_channels);
    for (auto& kernel : kernels) {
        kernel.resize(3, std::vector<float>(3));
        for (auto& row : kernel) {
            for (auto& val : row) {
                val = distribution(generator);
            }
        }
    }
    biases.resize(out_channels, 0.1f);
}

std::vector<std::vector<std::vector<float>>> ConvLayer::forward(
    const std::vector<std::vector<std::vector<float>>>& input) {
    
    int height = input[0].size() - 2;
    int width = input[0][0].size() - 2;
    
    std::vector<std::vector<std::vector<float>>> output(
        out_channels,
        std::vector<std::vector<float>>(
            height,
            std::vector<float>(width, 0.0f)
        )
    );

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float sum = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            sum += input[ic][h + kh][w + kw] * kernels[oc][kh][kw];
                        }
                    }
                }
                
                output[oc][h][w] = std::max(0.0f, sum + biases[oc]); // ReLU
            }
        }
    }
    
    return output;
}

void ConvLayer::save(std::ofstream& file) const {
    for (const auto& kernel : kernels) {
        for (const auto& row : kernel) {
            for (float val : row) {
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }
    file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

void ConvLayer::load(std::ifstream& file) {
    for (auto& kernel : kernels) {
        for (auto& row : kernel) {
            for (float& val : row) {
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
            }
        }
    }
    file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
}
