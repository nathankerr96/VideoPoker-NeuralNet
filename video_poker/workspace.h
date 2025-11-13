#pragma once

#include "neural.h"

#include <vector>

class InferenceWorkspace {
public:
    InferenceWorkspace(NeuralNet* net);
    void feedForward(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;
    const std::vector<std::vector<float>>& getActivations() const;

private:
    NeuralNet* mNet;
    std::vector<float> mLogitsBuffer;
    std::vector<std::vector<float>> mActivations;
};

class TrainingWorkspace {
public:
    TrainingWorkspace(NeuralNet* net);
    std::vector<double> getLayerGradientNormsSquared() const;
    void feedForward(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;
    void backpropagate(const std::vector<float>& errors);
    void aggregate(TrainingWorkspace& other);
    void batch(int batchSize);
    void reset();
    std::vector<std::vector<float>>& getTotalWeightGradients();
    std::vector<std::vector<float>>& getTotalBiasGradients();

private:
    NeuralNet* mNet;
    InferenceWorkspace mInferenceWorkspace;
    std::vector<std::vector<float>> mTotalWeightGradients;
    std::vector<std::vector<float>> mTotalBiasGradients;
    std::vector<float> mBlameBufferA;
    std::vector<float> mBlameBufferB;
    std::vector<float> mDeltaBuffer;
    std::vector<float> mOutputDerivativesBuffer;
};
