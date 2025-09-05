#pragma once

#include "neural.h"
#include "poker.h"

#include <random>
#include <vector>

class Agent {
public:
    Agent(const std::vector<LayerSpecification>& topology, unsigned int seed);
    void train(int iterations, float learningRate);
    void randomEval(int iterations);
    void targetedEval();

private:
    NeuralNet mNet;
    VideoPoker mPoker;
    std::mt19937 mRng;
};