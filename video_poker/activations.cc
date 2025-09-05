#include "activations.h"

#include <cmath>
#include <vector>

std::vector<float> sigmoid(const std::vector<float>& in) {
    std::vector<float> out;
    for (float val : in) {
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

std::vector<float> relu(const std::vector<float>& in) {
    std::vector<float> out;
    for (float val : in) {
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
