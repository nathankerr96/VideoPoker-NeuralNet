#include "baseline.h"

#include "neural.h"

#include <vector>

float FlatBaseline::predict(const std::vector<float>& inputs) {
    return 0.1f;
}

void FlatBaseline::train(int score) { /*No-op*/ }


float RunningAverageBaseline::predict(const std::vector<float>& inputs) {
    if (mCount == 0) {
        return 0.33f; // EV of random action
    }
    return mTotalScore / mCount;
}

void RunningAverageBaseline::train(int score) {
    mTotalScore += score;
    mCount += 1;
}

CriticNetworkBaseline::CriticNetworkBaseline(std::vector<LayerSpecification> topology, float learningRate)
        : mNet(topology), mLearningRate(learningRate) {}

float CriticNetworkBaseline::predict(const std::vector<float>& inputs) {
    mNet.feedForward(inputs);
    mPrediction = mNet.getOutputs()[0];
    return mPrediction;
}

void CriticNetworkBaseline::train(int score) {
    float error = mPrediction - score;
    std::vector<std::vector<float>> layeredWeightGradients;
    std::vector<std::vector<float>> layeredBiasGradients;
    mNet.backpropagate({error}, layeredWeightGradients, layeredBiasGradients);
    mNet.update(mLearningRate, layeredWeightGradients, layeredBiasGradients);
}
