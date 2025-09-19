#include "optimizer.h"

void SDGOptimizer::step(NeuralNet* net, Trainer& trainer, float learningRate) {
    net->update(learningRate, trainer.getTotalWeightGradients(), trainer.getTotalBiasGradients());
}

MomentumOptimizer::MomentumOptimizer(NeuralNet* net, float beta) : mBeta(beta) {
    const std::vector<Layer>& layers = net->getLayers();
    mWeightVelocity.resize(layers.size());
    mBiasVelocity.resize(layers.size());
    for (size_t i = 0; i < layers.size(); i++) {
        mWeightVelocity[i].resize(layers[i].getNumInputs()*layers[i].getNumNeurons(), 0.0f);
        mBiasVelocity[i].resize(layers[i].getNumNeurons(), 0.0f);
    }
}

void MomentumOptimizer::step(NeuralNet* net, Trainer& trainer, float learningRate) {
    const std::vector<std::vector<float>>& weightGradients = trainer.getTotalWeightGradients();
    const std::vector<std::vector<float>>& biasGradients = trainer.getTotalBiasGradients();
    for (size_t l = 0; l < weightGradients.size(); l++) {
        for (size_t w = 0; w < weightGradients[l].size(); w++) {
            mWeightVelocity[l][w] = (mBeta * mWeightVelocity[l][w]) + weightGradients[l][w];
        }
        for (size_t w = 0; w < biasGradients[l].size(); w++) {
            mBiasVelocity[l][w] = (mBeta * mBiasVelocity[l][w]) + biasGradients[l][w];
        }
    }
    net->update(learningRate, mWeightVelocity, mBiasVelocity);
}