#include "workspace.h"

#include <vector>

InferenceWorkspace::InferenceWorkspace(NeuralNet* net) : mNet(net) {
    const std::vector<Layer>& layers = net->getLayers();
    mActivations.resize(layers.size()+1); // +1 since the first "activation" is the input.
    mActivations[0].resize(layers[0].getNumInputs());
    int maxNeurons = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        mActivations[i+1].resize(layers[i].getNumNeurons());
        if (layers[i].getNumNeurons() > maxNeurons) {
            maxNeurons = layers[i].getNumNeurons();
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

TrainingWorkspace::TrainingWorkspace(NeuralNet* net) : mNet(net), mInferenceWorkspace(net) {
    const std::vector<Layer>& layers = net->getLayers();
    mTotalWeightGradients.resize(layers.size());
    mTotalBiasGradients.resize(layers.size());
    int maxNeurons = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        mTotalWeightGradients[i].resize(layers[i].getNumInputs()*layers[i].getNumNeurons(), 0.0f);
        mTotalBiasGradients[i].resize(layers[i].getNumNeurons(), 0.0f);
        if (layers[i].getNumNeurons() > maxNeurons) {
            maxNeurons = layers[i].getNumNeurons();
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
    for(size_t i = 0; i < mNet->getLayers().size(); i++) {
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
