#include "activations.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

std::vector<float> sigmoid(const std::vector<float>& logits) {
    std::vector<float> out;
    for (float val : logits) {
        out.push_back(1.0f / (1.0f + std::exp(-val)));
    }
    return out;
}

std::vector<float> sigmoid_derivative(const std::vector<float>& in) {
    std::vector<float> out;
    for (size_t i = 0; i < in.size(); i++) {
        out.push_back(in[i] * (1.0f - in[i]));
    }
    return out;
}

std::vector<float> relu(const std::vector<float>& logits) {
    std::vector<float> out;
    for (float val : logits) {
        out.push_back(std::max(0.0f, val));
    }
    return out;
}

std::vector<float> relu_derivative(const std::vector<float>& in) {
    std::vector<float> out;
    for (float val : in) {
        out.push_back((val > 0) ? 1.0f : 0.0f);
    }
    return out;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float sum_of_exponentials = 0.0f;

    float max_logit = *std::max_element(logits.begin(), logits.end());

    for (size_t i = 0; i < logits.size(); i++) {
        float exp_val = std::exp(logits[i] - max_logit);
        probabilities[i] = exp_val;
        sum_of_exponentials += exp_val;
    }

    if (sum_of_exponentials > 0) {
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] /= sum_of_exponentials;
        }
    }

    return probabilities;
}
