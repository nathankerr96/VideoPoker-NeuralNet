#pragma once

#include <vector>

struct LayerSpecification;

class InferenceWorkspace {
public:
    InferenceWorkspace(const std::vector<LayerSpecification>& topology);
    const std::vector<float>& getOutputs() const;
    const std::vector<std::vector<float>>& getActivations() const;
// TODO: private:
    std::vector<float> mLogitsBuffer;
    std::vector<std::vector<float>> mActivations;
};

class TrainingWorkspace {
public:
    TrainingWorkspace(const std::vector<LayerSpecification>& topology);
    std::vector<double> getLayerGradientNormsSquared() const;
    const std::vector<float>& getOutputs() const;
    void aggregate(TrainingWorkspace& other);
    void batch(int batchSize);
    void reset();
    std::vector<std::vector<float>>& getTotalWeightGradients();
    std::vector<std::vector<float>>& getTotalBiasGradients();
// TODO private:
    InferenceWorkspace mInferenceWorkspace;
    std::vector<std::vector<float>> mTotalWeightGradients;
    std::vector<std::vector<float>> mTotalBiasGradients;
    std::vector<float> mBlameBufferA;
    std::vector<float> mBlameBufferB;
    std::vector<float> mDeltaBuffer;
    std::vector<float> mOutputDerivativesBuffer;
};
