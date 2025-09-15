#include "trainer.h"

#include <vector>

Trainer::Trainer(NeuralNet* net) : mNet(net) {
    const std::vector<Layer>& layers = net->getLayers();
    mTotalWeightGradients.resize(layers.size());
    mTotalBiasGradients.resize(layers.size());
    mActivations.resize(layers.size()+1); // +1 since the first "activation" is the input.
    mActivations[0].resize(layers[0].getNumInputs());
    int maxNeurons = 0;
    for (size_t i = 0; i < layers.size(); i++) {
        mTotalWeightGradients[i].resize(layers[i].getNumInputs()*layers[i].getNumNeurons(), 0.0f);
        mTotalBiasGradients[i].resize(layers[i].getNumNeurons(), 0.0f);
        mActivations[i+1].resize(layers[i].getNumNeurons());
        if (layers[i].getNumNeurons() > maxNeurons) {
            maxNeurons = layers[i].getNumNeurons();
        }
    }
    mLogitsBuffer.resize(maxNeurons, 0.0f);
    mBlameBufferA.resize(maxNeurons, 0.0f);
    mBlameBufferB.resize(maxNeurons, 0.0f);
    mDeltaBuffer.resize(maxNeurons, 0.0f);
    mOutputDerivativesBuffer.resize(maxNeurons, 0.0f);
}

void Trainer::backpropagate(const std::vector<float>& errors) {
    std::vector<float>* upstreamGradient = nullptr;
    std::vector<float>* downstreamGradient = &mBlameBufferA; 
    const std::vector<Layer>& layers = mNet->getLayers();
    int last = layers.size() - 1;
    layers[last].backpropagate(errors, 
                               mActivations[last],
                               mActivations[last+1],
                               mDeltaBuffer, 
                               mOutputDerivativesBuffer, 
                               mTotalWeightGradients[last], 
                               mTotalBiasGradients[last], 
                               *downstreamGradient);
    for (int i = last-1; i >= 0; i--) {
        upstreamGradient = downstreamGradient;
        downstreamGradient = (upstreamGradient == &mBlameBufferA ? &mBlameBufferB : &mBlameBufferA);
        layers[i].backpropagate(*upstreamGradient, 
                                mActivations[i],
                                mActivations[i+1],
                                mDeltaBuffer, 
                                mOutputDerivativesBuffer, 
                                mTotalWeightGradients[i], 
                                mTotalBiasGradients[i], 
                                *downstreamGradient);
    }
}

void Trainer::aggregate(Trainer& other) {
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

void Trainer::batch(int batchSize) {
    for (size_t l = 0; l < mTotalWeightGradients.size(); l++) {
        for (size_t w = 0; w < mTotalWeightGradients[l].size(); w++) {
            mTotalWeightGradients[l][w] /= batchSize;
        }
        for (size_t w = 0; w < mTotalBiasGradients[l].size(); w++) {
            mTotalBiasGradients[l][w] /= batchSize;
        }
    }
}

void Trainer::reset() {
    for(size_t i = 0; i < mNet->getLayers().size(); i++) {
        std::fill(mTotalWeightGradients[i].begin(), mTotalWeightGradients[i].end(), 0.0f);
        std::fill(mTotalBiasGradients[i].begin(), mTotalBiasGradients[i].end(), 0.0f);
    }
}

void Trainer::feedForward(const std::vector<float>& inputs) {
    mActivations[0] = inputs;
    const std::vector<Layer>& layers = mNet->getLayers();
    layers[0].fire(mActivations[0], mLogitsBuffer, mActivations[1]);
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].fire(mActivations[i], mLogitsBuffer, mActivations[i+1]);
    }
}

const std::vector<float>& Trainer::getOutputs() {
    return mActivations.back();
}

std::vector<std::vector<float>>& Trainer::getTotalWeightGradients() {
    return mTotalWeightGradients;
}

std::vector<std::vector<float>>& Trainer::getTotalBiasGradients() {
    return mTotalBiasGradients;
}

std::vector<double> Trainer::getLayerGradientNormsSquared() const {
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
