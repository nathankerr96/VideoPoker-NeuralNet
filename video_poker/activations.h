#pragma once

#include <vector>

void sigmoid(const std::vector<float>& logits, int numNeurons, std::vector<float>& out);
void sigmoid_derivative(const std::vector<float>& outputs, std::vector<float>& out);

void relu(const std::vector<float>& logits, int numNeurons, std::vector<float>& out);
void relu_derivative(const std::vector<float>& outputs, std::vector<float>& out);

void softmax(const std::vector<float>& logits, int numNeurons, std::vector<float>& out);