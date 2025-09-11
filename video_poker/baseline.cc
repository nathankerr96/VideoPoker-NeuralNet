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
        : mNet(std::make_unique<NeuralNet>(topology)), 
          mTrainer(mNet.get()),
          mLearningRate(learningRate) {}

float CriticNetworkBaseline::predict(const std::vector<float>& inputs) {
    mTrainer.feedForward(inputs);
    mPrediction = mTrainer.getOutputs()[0];
    return mPrediction;
}

void CriticNetworkBaseline::train(int score) {
    float error = mPrediction - score;
    mTrainer.backpropagate({error});
    mNet->update(mLearningRate, mTrainer.getTotalWeightGradients(), mTrainer.getTotalBiasGradients());
}
