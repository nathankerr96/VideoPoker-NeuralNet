#include "activations.h"

#include <cmath>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float relu(float x) {
    return std::max(0.0f, x);
}

float relu_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}
