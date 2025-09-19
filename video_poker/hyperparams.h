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
    float beta;

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

const HyperParameters Softmax_CriticNetwork_32Batch_Med_Momentum {
    .name = "170-170-Softmax-Critic_Network-32_Batch-Med-Momentum",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.001f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .beta = 0.95f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_32Batch_Slow {
    .name = "170-170-Softmax-Critic_Network-32_Batch-Slow",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.002f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.003f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_32Batch_Med {
    .name = "170-170-Softmax-Critic_Network-32_Batch-Med",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.01f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_32Batch_Fast {
    .name = "170-170-Softmax-Critic_Network-32_Batch-Fast",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.064f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.096f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_1Batch {
    .name = "170-170-Softmax-Critic_Network-1_Batch",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.002f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.003f,
    .numWorkers = 1,
    .numInBatch = 1,
};

inline std::vector<HyperParameters> AvailableConfigs {
    Softmax_CriticNetwork_32Batch_Slow,
    Softmax_CriticNetwork_32Batch_Med,
    Softmax_CriticNetwork_32Batch_Med_Momentum,
    Softmax_CriticNetwork_32Batch_Fast,
    Softmax_CriticNetwork_1Batch,
};