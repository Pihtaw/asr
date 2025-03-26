#ifndef CTC_H
#define CTC_H

#include <vector>
#include <string>

// CTC декодер
class CTCDecoder {
public:
    static std::string greedy_decode(
        const std::vector<std::vector<float>>& logits,
        const std::string& alphabet
    );
    
    static float compute_loss(
        const std::vector<std::vector<float>>& logits,
        const std::vector<int>& targets
    );
};

// Оптимизатор
class SGDOptimizer {
public:
    SGDOptimizer(float lr) : learning_rate(lr) {}
    
    void update(
        std::vector<std::vector<float>>& weights,
        const std::vector<std::vector<float>>& grads
    );
    
private:
    float learning_rate;
};

#endif
