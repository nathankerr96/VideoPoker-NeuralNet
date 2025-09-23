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

const HyperParameters Softmax_CriticNetwork_32Batch_LowMomentum {
    .name = "170-170-Softmax-Critic_Network-32_Batch-LowMomentum",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.002f, // SDG rate divided by 1 / (1-beta)
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .beta = 0.8f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_32Batch_MedMomentum {
    .name = "170-170-Softmax-Critic_Network-32_Batch-MedMomentum",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0005f, // SDG rate divided by 1 / (1-beta)
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .beta = 0.95f,
    .numWorkers = 8,
    .numInBatch = 4,
};

const HyperParameters Softmax_CriticNetwork_32Batch_HighMomentum {
    .name = "170-170-Softmax-Critic_Network-32_Batch-HighMomentum",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.0001f, // SDG rate divided by 1 / (1-beta)
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
    .optimizerType = MOMENTUM,
    .beta = 0.99f,
    .numWorkers = 8,
    .numInBatch = 4,
};


const HyperParameters Softmax_CriticNetwork_32Batch_SDG {
    .name = "170-170-Softmax-Critic_Network-32_Batch-SDG",
    .actorTopology = SOFTMAX_TOPOLOGY,
    .actorLearningRate = 0.01f,
    .baselineCalculatorType = CRITIC_NETWORK,
    .criticTopology = CRITIC_NETWORK_TOPOLOGY,
    .criticLearningRate = 0.015f,
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
    Softmax_CriticNetwork_32Batch_SDG,
    Softmax_CriticNetwork_32Batch_LowMomentum,
    Softmax_CriticNetwork_32Batch_MedMomentum,
    Softmax_CriticNetwork_32Batch_HighMomentum,
    Softmax_CriticNetwork_1Batch,
};