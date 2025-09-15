#pragma once

#include "neural.h"

#include <vector>

class Trainer {
public:
    Trainer(NeuralNet* net);
    std::vector<double> getLayerGradientNormsSquared() const;

    void feedForward(const std::vector<float>& inputs);
    void backpropagate(const std::vector<float>& errors);
    void aggregate(Trainer& other);
    void batch(int batchSize);
    void reset();
    const std::vector<float>& getOutputs();
    std::vector<std::vector<float>>& getTotalWeightGradients();
    std::vector<std::vector<float>>& getTotalBiasGradients();

private:
    NeuralNet* mNet;

    std::vector<std::vector<float>> mTotalWeightGradients;
    std::vector<std::vector<float>> mTotalBiasGradients;
    std::vector<float> mLogitsBuffer;
    std::vector<std::vector<float>> mActivations;
    std::vector<float> mBlameBufferA;
    std::vector<float> mBlameBufferB;
    std::vector<float> mDeltaBuffer;
    std::vector<float> mOutputDerivativesBuffer;
};
