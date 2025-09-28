#pragma once

#include "neural.h"
#include "baseline.h"
#include "optimizer.h"

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
    OptimizerType criticOptimizerType = SDG;
    float criticBeta;

    OptimizerType optimizerType = SDG;
    float momentum_coeff;

    float entropy_coeff = 0.0f;

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


const HyperParameters NoEntropy {
    .name = "NoEntropy",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0005f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .momentum_coeff = 0.95f,
    .entropy_coeff = 0.0f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters LowEntropy {
    .name = "LowEntropy",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0005f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .momentum_coeff = 0.95f,
    .entropy_coeff = 0.001f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters MedEntropy {
    .name = "MedEntropy",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0005f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .momentum_coeff = 0.95f,
    .entropy_coeff = 0.005f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters HighEntropy {
    .name = "HighEntropy",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0005f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .momentum_coeff = 0.95f,
    .entropy_coeff = 0.01f,
    .numWorkers = 8,
    .numInBatch = 4,
};

inline std::vector<HyperParameters> AvailableConfigs {
    NoEntropy,
    LowEntropy,
    MedEntropy,
    HighEntropy,
};