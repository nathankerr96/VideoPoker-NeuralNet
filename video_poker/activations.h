#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

std::vector<float> sigmoid(const std::vector<float>& in);
std::vector<float> sigmoid_derivative(const std::vector<float>& in);

std::vector<float> relu(const std::vector<float>& in);
std::vector<float> relu_derivative(const std::vector<float>& in);

#endif