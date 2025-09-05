#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

std::vector<float> sigmoid(const std::vector<float>& logits);
std::vector<float> sigmoid_derivative(const std::vector<float>& outputs);

std::vector<float> relu(const std::vector<float>& logits);
std::vector<float> relu_derivative(const std::vector<float>& outputs);

std::vector<float> softmax(const std::vector<float>& logits);

#endif