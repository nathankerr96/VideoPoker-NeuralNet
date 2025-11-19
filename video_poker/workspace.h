#pragma once

#include <vector>

struct LayerSpecification;

class InferenceWorkspace {
    friend class TrainingWorkspace;
    friend class NeuralNet;
public:
    InferenceWorkspace(const std::vector<LayerSpecification>& topology);
private:
    std::vector<float> mLogitsBuffer;
    std::vector<std::vector<float>> mActivations;
};

class TrainingWorkspace {
    friend class NeuralNet;
public:
    TrainingWorkspace(const std::vector<LayerSpecification>& topology);
    InferenceWorkspace& getInferenceWorkspace();
    std::vector<double> getLayerGradientNormsSquared() const;
    const std::vector<std::vector<float>>& getActivations() const;
    void aggregate(TrainingWorkspace& other);
    void batch(int batchSize);
    void reset();
    std::vector<std::vector<float>>& getTotalWeightGradients();
    std::vector<std::vector<float>>& getTotalBiasGradients();
private:
    InferenceWorkspace mInferenceWorkspace;
    std::vector<std::vector<float>> mTotalWeightGradients;
    std::vector<std::vector<float>> mTotalBiasGradients;
    std::vector<float> mBlameBufferA;
    std::vector<float> mBlameBufferB;
    std::vector<float> mDeltaBuffer;
    std::vector<float> mOutputDerivativesBuffer;
};
