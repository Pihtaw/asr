// lstm.h
#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <cmath>
#include <random>

class LSTMLayer {
private:
    // Размерности
    int input_size;
    int hidden_size;
    
    // Весовые матрицы [hidden_size, input_size + hidden_size]
    std::vector<std::vector<float>> Wf, Wi, Wo, Wc; // Веса для forget, input, output, cell
    std::vector<float> bf, bi, bo, bc; // Смещения
    
    // Кэши для forward pass
    std::vector<std::vector<float>> h_cache, c_cache;
    std::vector<std::vector<float>> f_gate, i_gate, o_gate, c_candidate;
    std::vector<std::vector<float>> input_cache;
    std::vector<float> h0, c0;
    
    // Градиенты весов
    std::vector<std::vector<float>> Wf_grad, Wi_grad, Wo_grad, Wc_grad;
    std::vector<float> bf_grad, bi_grad, bo_grad, bc_grad;
    
    // Вспомогательные функции
    // Изменяем в lstm.h
	float tanh(float x);  // Для скалярных значений
	void tanh(std::vector<float>& x);  // Для векторов
	float sigmoid(float x);
	void sigmoid(std::vector<float>& x);
    
    void update_weights();
    
public:
    LSTMLayer(int input_size, int hidden_size);
    
    // Прямой проход
    void forward_step(
        const std::vector<float>& x_t,     // Вход на текущем шаге
        std::vector<float>& h_prev,        // Скрытое состояние
        std::vector<float>& c_prev,        // Состояние ячейки
        std::vector<float>& h_next,        // Новое скрытое состояние
        std::vector<float>& c_next         // Новое состояние ячейки
    );
    
    // Пакетная обработка
    std::vector<std::vector<float>> forward_sequence(
        const std::vector<std::vector<float>>& input_sequence
    );
    
    std::vector<std::vector<float>> backward(
    const std::vector<std::vector<float>>& grad_output);
    const std::vector<std::vector<float>>& get_last_output() const { return h_cache; }
    void save(std::ofstream& file) const;
    void load(std::ifstream& file);
};

#endif
