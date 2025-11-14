#include "workspace.h"

#include "neural.h"

#include <vector>

InferenceWorkspace::InferenceWorkspace(const std::vector<LayerSpecification>& topology) {
    mActivations.resize(topology.size());
    mActivations[0].resize(topology[0].numNeurons);
    int maxNeurons = 0;
    for (size_t i = 1; i < topology.size(); i++) {
        mActivations[i].resize(topology[i].numNeurons);
        if (topology[i].numNeurons > maxNeurons) {
            maxNeurons = topology[i].numNeurons;
        }
    }
    mLogitsBuffer.resize(maxNeurons, 0.0f);
}

const std::vector<float>& InferenceWorkspace::getOutputs() const {
    return mActivations.back();
}

const std::vector<std::vector<float>>& InferenceWorkspace::getActivations() const {
    return mActivations;
}

TrainingWorkspace::TrainingWorkspace(const std::vector<LayerSpecification>& topology) : mInferenceWorkspace(topology) {
    mTotalWeightGradients.resize(topology.size()-1);
    mTotalBiasGradients.resize(topology.size()-1);
    int maxNeurons = 0;
    for (size_t i = 1; i < topology.size(); i++) {
        mTotalWeightGradients[i-1].resize(topology[i-1].numNeurons*topology[i].numNeurons, 0.0f);
        mTotalBiasGradients[i-1].resize(topology[i].numNeurons, 0.0f);
        if (topology[i].numNeurons > maxNeurons) {
            maxNeurons = topology[i].numNeurons;
        }
    }
    mBlameBufferA.resize(maxNeurons, 0.0f);
    mBlameBufferB.resize(maxNeurons, 0.0f);
    mDeltaBuffer.resize(maxNeurons, 0.0f);
    mOutputDerivativesBuffer.resize(maxNeurons, 0.0f);
}

void TrainingWorkspace::aggregate(TrainingWorkspace& other) {
    const std::vector<std::vector<float>>& weightGradients = other.getTotalWeightGradients();
    const std::vector<std::vector<float>>& biasGradients = other.getTotalBiasGradients();
    for (size_t l = 0; l < weightGradients.size(); l++) {
        for (size_t w = 0; w < weightGradients[l].size(); w++) {
            mTotalWeightGradients[l][w] += weightGradients[l][w];
        }
        for (size_t w = 0; w < biasGradients[l].size(); w++) {
            mTotalBiasGradients[l][w] += biasGradients[l][w];
        }
    }
}

void TrainingWorkspace::batch(int batchSize) {
    for (size_t l = 0; l < mTotalWeightGradients.size(); l++) {
        for (size_t w = 0; w < mTotalWeightGradients[l].size(); w++) {
            mTotalWeightGradients[l][w] /= batchSize;
        }
        for (size_t w = 0; w < mTotalBiasGradients[l].size(); w++) {
            mTotalBiasGradients[l][w] /= batchSize;
        }
    }
}

void TrainingWorkspace::reset() {
    for(size_t i = 0; i < mTotalWeightGradients.size(); i++) {
        std::fill(mTotalWeightGradients[i].begin(), mTotalWeightGradients[i].end(), 0.0f);
        std::fill(mTotalBiasGradients[i].begin(), mTotalBiasGradients[i].end(), 0.0f);
    }
}


const std::vector<float>& TrainingWorkspace::getOutputs() const {
    return mInferenceWorkspace.getOutputs();
}

std::vector<std::vector<float>>& TrainingWorkspace::getTotalWeightGradients() {
    return mTotalWeightGradients;
}

std::vector<std::vector<float>>& TrainingWorkspace::getTotalBiasGradients() {
    return mTotalBiasGradients;
}

std::vector<double> TrainingWorkspace::getLayerGradientNormsSquared() const {
    std::vector<double> ret;
    for (size_t l = 0; l < mTotalWeightGradients.size(); l++) {
        double layerSum = 0.0;
        for (float b : mTotalBiasGradients[l]) {
            layerSum += b * b;
        }
        for (float w : mTotalWeightGradients[l]) {
            layerSum += w * w;
        }
        ret.push_back(layerSum);
    }
    return ret;
}
