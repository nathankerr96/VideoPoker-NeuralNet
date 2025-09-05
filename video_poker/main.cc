#include "neural.h"
#include "activations.h"
#include "poker.h"
#include "agent.h"

#include <iostream>
#include <random>
#include <vector>
#include <cassert>

#define INPUT_SIZE 85
#define LEARNING_RATE 0.002
#define TRAINING_ITERATIONS 100000
#define EVAL_ITERATIONS 10000

std::vector<LayerSpecification> SOFTMAX_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {32, Activation::SOFTMAX},
};

std::vector<LayerSpecification> SIGMOID_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {5, Activation::SIGMOID},
};

int main() {
    std::random_device rd {};
    Agent agent {SOFTMAX_TOPOLOGY, rd()};
    agent.randomEval(EVAL_ITERATIONS);
    agent.targetedEval();
    agent.train(TRAINING_ITERATIONS, LEARNING_RATE);
    agent.randomEval(EVAL_ITERATIONS);
    agent.targetedEval();

    return 0;
}
