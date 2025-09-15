#pragma once

#include "neural.h"
#include "baseline.h"

#include <string>
#include <vector>

// One-hot encoding (13 Ranks + 4 Suits) * 5 Cards = 85 Input Neurons
constexpr int INPUT_SIZE = 85;

struct HyperParameters {
    std::string name;

    std::vector<LayerSpecification> actorTopology;
    float actorLearningRate;

    BaselineCalculatorType baselineCalculatorType;
    std::vector<LayerSpecification> criticTopology;
    float criticLearningRate;

    int numWorkers;
    int numInBatch;

    int getBatchSize() const {
        return numWorkers * numInBatch;
    }
};

inline std::vector<LayerSpecification> SOFTMAX_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {32, Activation::SOFTMAX},
};

inline std::vector<LayerSpecification> SIGMOID_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {5, Activation::SIGMOID},
};

inline std::vector<LayerSpecification> CRITIC_NETWORK_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {85, Activation::RELU},
    {1, Activation::LINEAR},
};

const HyperParameters Softmax_CriticNetwork_Batched {
    .name = "170-170-Softmax-Critic_Network-32_Batch",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.03f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.04f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_SingleBatch {
    .name = "170-170-Softmax-Critic_Network-1_Batch",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.002f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.003f,
    .numWorkers = 1,
    .numInBatch = 1,
};