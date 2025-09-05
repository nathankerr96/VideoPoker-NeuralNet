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
#define TRAINING_ITERATIONS 1000000
#define EVAL_ITERATIONS 100000

int main() {
    std::vector<LayerSpecification> topology {
        {INPUT_SIZE, Activation::LINEAR},
        {170, Activation::RELU},
        {170, Activation::RELU},
        {5, Activation::SIGMOID},
    };
    std::random_device rd {};
    Agent agent {topology, rd()};
    agent.randomEval(EVAL_ITERATIONS);
    agent.targetedEval();
    agent.train(TRAINING_ITERATIONS, LEARNING_RATE);
    agent.randomEval(EVAL_ITERATIONS);
    agent.targetedEval();

    return 0;
}
