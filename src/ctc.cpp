// CTC_CPP

#include "../include/ctc.h"

#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

// Жадное декодирование
std::string CTCDecoder::greedy_decode(
    const std::vector<std::vector<float>>& logits,
    const std::string& alphabet) 
{
    std::string text;
    int prev_char = -1;
    int blank_id = alphabet.size() - 1;
    
    for (const auto& frame : logits) {
        int best_char = std::distance(frame.begin(), 
                                    std::max_element(frame.begin(), frame.end()));
        
        // CTC правила декодирования:
        if (best_char != blank_id && best_char != prev_char) {
            text += alphabet[best_char];
            prev_char = best_char;
        }
    }
    
    // Простая постобработка
    if (text.empty()) return "";
    
    // Добавляем пробелы (эвристика)
    std::string result;
    for (size_t i = 0; i < text.size(); ++i) {
        result += text[i];
        if (i % 5 == 4) result += ' ';
    }
    
    return result;
}

// CTC Loss (упрощенная версия)
float CTCDecoder::compute_loss(
    const vector<vector<float>>& logits,
    const vector<int>& targets
) {
    float loss = 0.0f;
    int T = logits.size(); // Временные шаги
    int C = logits[0].size(); // Классы
    
    for (int t = 0; t < T; ++t) {
        float max_val = *max_element(logits[t].begin(), logits[t].end());
        float sum_exp = 0.0f;
        
        for (int c = 0; c < C; ++c) {
            sum_exp += exp(logits[t][c] - max_val);
        }
        
        int target = (t < targets.size()) ? targets[t] : 0;
        loss += (log(sum_exp) - (logits[t][target] - max_val));
    }
    
    return loss / T;
}

// SGD оптимизатор
void SGDOptimizer::update(
    vector<vector<float>>& weights,
    const vector<vector<float>>& grads
) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * grads[i][j];
        }
    }
}
