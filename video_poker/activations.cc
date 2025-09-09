#include "activations.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

void sigmoid(const std::vector<float>& logitsBuffer, int numNeurons, std::vector<float>& out) {
    for (int i = 0; i < numNeurons; i++) {
        out[i] = 1.0f / (1.0f + std::exp(-logitsBuffer[i]));
    }
}

void sigmoid_derivative(const std::vector<float>& in, std::vector<float>& out) {
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = in[i] * (1.0f - in[i]);
    }
}

void relu(const std::vector<float>& logitsBuffer, int numNeurons, std::vector<float>& out) {
    for (int i = 0; i < numNeurons; i++) {
        out[i] = std::max(0.0f, logitsBuffer[i]);
    }
}

void relu_derivative(const std::vector<float>& in, std::vector<float>& out) {
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = (in[i] > 0) ? 1.0f : 0.0f;
    }
}

void softmax(const std::vector<float>& logitsBuffer, int numNeurons, std::vector<float>& out) {
    float sum_of_exponentials = 0.0f;
    float max_logit = *std::max_element(logitsBuffer.begin(), logitsBuffer.begin()+numNeurons);
    for (int i = 0; i < numNeurons; i++) {
        float exp_val = std::exp(logitsBuffer[i] - max_logit);
        out[i] = exp_val;
        sum_of_exponentials += exp_val;
    }
    if (sum_of_exponentials > 0) {
        for (int i = 0; i < numNeurons; i++) {
            out[i] /= sum_of_exponentials;
        }
    }
}
