// LSTM_CPP

#include "../include/lstm.h"

#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>


 LSTMLayer::LSTMLayer(int input_size, int hidden_size) 
    : input_size(input_size), hidden_size(hidden_size) {
    
    int total_size = input_size + hidden_size;
    std::default_random_engine gen;
    float limit = sqrt(6.0f / (input_size + hidden_size));
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    // Инициализация весов как матриц
    auto init_matrix = [&](std::vector<std::vector<float>>& m) {
        m.resize(hidden_size, std::vector<float>(total_size));
        for (auto& row : m) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
    };
    
    init_matrix(Wf);
    init_matrix(Wi);
    init_matrix(Wo);
    init_matrix(Wc);
    
    // Инициализация смещений
    bf.resize(hidden_size, 0.1f);
    bi.resize(hidden_size, 0.1f);
    bo.resize(hidden_size, 0.1f);
    bc.resize(hidden_size, 0.0f);
    
    // Инициализация градиентов
    Wf_grad.resize(hidden_size, std::vector<float>(total_size, 0));
    Wi_grad.resize(hidden_size, std::vector<float>(total_size, 0));
    Wo_grad.resize(hidden_size, std::vector<float>(total_size, 0));
    Wc_grad.resize(hidden_size, std::vector<float>(total_size, 0));
    
    bf_grad.resize(hidden_size, 0);
    bi_grad.resize(hidden_size, 0);
    bo_grad.resize(hidden_size, 0);
    bc_grad.resize(hidden_size, 0);
    
    // Инициализация начальных состояний
    h0.resize(hidden_size, 0);
    c0.resize(hidden_size, 0);
}

// Скалярные версии
float LSTMLayer::tanh(float x) {
    return std::tanh(x);
}

float LSTMLayer::sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Векторные версии
void LSTMLayer::tanh(std::vector<float>& x) {
    for (auto& v : x) v = std::tanh(v);
}

void LSTMLayer::sigmoid(std::vector<float>& x) {
    for (auto& v : x) v = 1.0f / (1.0f + expf(-v));
}

// Один шаг LSTM
void LSTMLayer::forward_step(
    const std::vector<float>& x_t,
    std::vector<float>& h_prev,
    std::vector<float>& c_prev,
    std::vector<float>& h_next,
    std::vector<float>& c_next) {
    
    // Объединяем вход и предыдущее скрытое состояние
    std::vector<float> concat(input_size + hidden_size);
    std::copy(x_t.begin(), x_t.end(), concat.begin());
    std::copy(h_prev.begin(), h_prev.end(), concat.begin() + input_size);
    
    // Вычисляем гейты
    std::vector<float> fg(hidden_size), ig(hidden_size), 
                       og(hidden_size), cg(hidden_size);
    
    auto mat_vec_mult = [](const std::vector<std::vector<float>>& m, 
                      const std::vector<float>& v,
                      std::vector<float>& out) {
		for (size_t i = 0; i < m.size(); ++i) {
			out[i] = std::inner_product(m[i].begin(), m[i].end(), v.begin(), 0.0f);
		}
	};
    
    // Forget gate
    mat_vec_mult(Wf, concat, fg);
    for (int i = 0; i < hidden_size; ++i) fg[i] += bf[i];
    sigmoid(fg);
    
    // Input gate
    mat_vec_mult(Wi, concat, ig);
    for (int i = 0; i < hidden_size; ++i) ig[i] += bi[i];
    sigmoid(ig);
    
    // Output gate
    mat_vec_mult(Wo, concat, og);
    for (int i = 0; i < hidden_size; ++i) og[i] += bo[i];
    sigmoid(og);
    
    // Cell gate
    mat_vec_mult(Wc, concat, cg);
    for (int i = 0; i < hidden_size; ++i) cg[i] += bc[i];
    tanh(cg);
    
    // Сохраняем промежуточные значения для backward pass
    f_gate.push_back(fg);
    i_gate.push_back(ig);
    o_gate.push_back(og);
    c_candidate.push_back(cg);
    input_cache.push_back(concat);
    
    // Обновляем состояние ячейки
    c_next.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        c_next[i] = fg[i] * c_prev[i] + ig[i] * cg[i];
    }
    
    // Обновляем скрытое состояние
    h_next.resize(hidden_size);
    std::vector<float> tanh_c(c_next);
    tanh(tanh_c);
    for (int i = 0; i < hidden_size; ++i) {
        h_next[i] = og[i] * tanh_c[i];
    }
    
    // Сохраняем состояния
    h_cache.push_back(h_next);
    c_cache.push_back(c_next);
}

// Обработка всей последовательности
std::vector<std::vector<float>> LSTMLayer::forward_sequence(
    const std::vector<std::vector<float>>& input_sequence) {
    
    // Очищаем кэши перед новой последовательностью
    f_gate.clear();
    i_gate.clear();
    o_gate.clear();
    c_candidate.clear();
    input_cache.clear();
    h_cache.clear();
    c_cache.clear();
    
    std::vector<std::vector<float>> outputs;
    std::vector<float> h_prev = h0;
    std::vector<float> c_prev = c0;
    
    for (const auto& x_t : input_sequence) {
        std::vector<float> h_next, c_next;
        forward_step(x_t, h_prev, c_prev, h_next, c_next);
        outputs.push_back(h_next);
        h_prev = h_next;
        c_prev = c_next;
    }
    
    return outputs;
}

std::vector<std::vector<float>> LSTMLayer::backward(
    const std::vector<std::vector<float>>& grad_output) 
{
    if (grad_output.empty() || grad_output[0].size() != hidden_size) {
        std::cout << "ERROR" << std::endl;
    }
    if (grad_output.empty()) {
        std::cout <<"Empty grad_output in LSTM backward" << std::endl;
    }
    
    if (grad_output[0].size() != hidden_size) {
        std::cout << "Error: grad_output[0].size() = " << grad_output[0].size() 
                  << ", but hidden_size = " << hidden_size << std::endl;
    }
    int T = grad_output.size(); // Временные шаги
    int H = hidden_size;
    int I = input_size;
    
    // Инициализация градиентов
    std::vector<std::vector<float>> grad_input(T, std::vector<float>(I, 0.0f));
    std::vector<float> dh_next(H, 0.0f);
    std::vector<float> dc_next(H, 0.0f);
    
    // Временные переменные
    std::vector<float> dho(H), dco(H), dc(H), di(H), df(H), dg(H);

    for (int t = T-1; t >= 0; --t) {
        const auto& h_prev = (t > 0) ? h_cache[t-1] : h0;
        const auto& c_prev = (t > 0) ? c_cache[t-1] : c0;

        // Градиент по выходным воротам
        for (int i = 0; i < H; ++i) {
            float tanh_c = std::tanh(c_cache[t][i]); // Используем std::tanh напрямую
            dho[i] = grad_output[t][i] * tanh_c;
        }
        
        // Градиент по состоянию ячейки
        for (int i = 0; i < H; ++i) {
            float tanh_c = std::tanh(c_cache[t][i]);
            dco[i] = grad_output[t][i] * o_gate[t][i] * (1 - tanh_c * tanh_c);
            dc[i] = dco[i] + dc_next[i];
        }
        
        // Градиенты по воротам
        for (int i = 0; i < H; ++i) {
            float i_g = i_gate[t][i];
            float f_g = f_gate[t][i];
            float o_g = o_gate[t][i];
            float c_cand = c_candidate[t][i];
            float c_prev_i = c_prev[i];
            
            di[i] = dc[i] * c_cand * i_g * (1 - i_g);
            df[i] = dc[i] * c_prev_i * f_g * (1 - f_g);
            dg[i] = dc[i] * i_g * (1 - c_cand * c_cand);
        }
        
        // Градиенты по весам
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < I+H; ++j) {
                float x = (j < I) ? input_cache[t][j] : h_prev[j-I];
                Wf_grad[i][j] += df[i] * x;
                Wi_grad[i][j] += di[i] * x;
                Wo_grad[i][j] += dho[i] * x;
                Wc_grad[i][j] += dg[i] * x;
            }
            bf_grad[i] += df[i];
            bi_grad[i] += di[i];
            bo_grad[i] += dho[i];
            bc_grad[i] += dg[i];
        }
        
        // Градиент по входу
        for (int j = 0; j < I; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < H; ++i) {
                sum += df[i] * Wf[i][j] +
                      di[i] * Wi[i][j] +
                      dho[i] * Wo[i][j] +
                      dg[i] * Wc[i][j];
            }
            grad_input[t][j] = sum;
        }
        
        // Градиент по предыдущему состоянию
        for (int j = 0; j < H; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < H; ++i) {
                sum += df[i] * Wf[i][I+j] +
                      di[i] * Wi[i][I+j] +
                      dho[i] * Wo[i][I+j] +
                      dg[i] * Wc[i][I+j];
            }
            dh_next[j] = sum;
        }
        
        // Обновляем градиент состояния ячейки
        for (int i = 0; i < H; ++i) {
            dc_next[i] = dc[i] * f_gate[t][i];
        }
    }
    
    update_weights();
    return grad_input;
}

void LSTMLayer::update_weights() {
    const float lr = 0.001f;
    
    // Обновляем матрицы весов
    for (size_t i = 0; i < hidden_size; ++i) {
        // Обновляем Wf, Wi, Wo, Wc
        for (size_t j = 0; j < input_size + hidden_size; ++j) {
            Wf[i][j] -= lr * Wf_grad[i][j];
            Wi[i][j] -= lr * Wi_grad[i][j];
            Wo[i][j] -= lr * Wo_grad[i][j];
            Wc[i][j] -= lr * Wc_grad[i][j];
        }
        
        // Обновляем смещения
        bf[i] -= lr * bf_grad[i];
        bi[i] -= lr * bi_grad[i];
        bo[i] -= lr * bo_grad[i];
        bc[i] -= lr * bc_grad[i];
    }
    
    // Сбрасываем градиенты
    auto reset_grads = [](auto& grads) {
        for (auto& row : grads) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
    };
    
    reset_grads(Wf_grad);
    reset_grads(Wi_grad);
    reset_grads(Wo_grad);
    reset_grads(Wc_grad);
    
    std::fill(bf_grad.begin(), bf_grad.end(), 0.0f);
    std::fill(bi_grad.begin(), bi_grad.end(), 0.0f);
    std::fill(bo_grad.begin(), bo_grad.end(), 0.0f);
    std::fill(bc_grad.begin(), bc_grad.end(), 0.0f);
}

void LSTMLayer::save(std::ofstream& file) const {
    // Сохраняем все веса и смещения
    auto save_matrix = [&](const std::vector<std::vector<float>>& m) {
        for (const auto& row : m) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
    };

    save_matrix(Wf);
    save_matrix(Wi);
    save_matrix(Wo);
    save_matrix(Wc);
    file.write(reinterpret_cast<const char*>(bf.data()), bf.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bi.data()), bi.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bo.data()), bo.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(bc.data()), bc.size() * sizeof(float));
}

void LSTMLayer::load(std::ifstream& file) {
    // Загружаем все веса и смещения
    auto load_matrix = [&](std::vector<std::vector<float>>& m) {
        for (auto& row : m) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
        }
    };

    load_matrix(Wf);
    load_matrix(Wi);
    load_matrix(Wo);
    load_matrix(Wc);
    file.read(reinterpret_cast<char*>(bf.data()), bf.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(bi.data()), bi.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(bo.data()), bo.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(bc.data()), bc.size() * sizeof(float));
}