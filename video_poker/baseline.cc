#include "baseline.h"

#include "neural.h"

#include <vector>

float FlatBaseline::predict(const std::vector<float>& inputs) {
    return 0.1f;
}

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

CriticNetworkBaseline::CriticNetworkBaseline(NeuralNet* net, float learningRate)
        : mNet(net), 
          mTrainer(net),
          mLearningRate(learningRate) {}

float CriticNetworkBaseline::predict(const std::vector<float>& inputs) {
    mTrainer.feedForward(inputs);
    mPrediction = mTrainer.getOutputs()[0];
    return mPrediction;
}

void CriticNetworkBaseline::train(int score) {
    float error = mPrediction - score;
    mTrainer.backpropagate({error});
}

void CriticNetworkBaseline::update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) {
    for (size_t i = 1; i < otherCalcs.size(); i++) {
        // Icky encasulation breaking :( -- Crash if wrong type (bad_cast exception)
        CriticNetworkBaseline* otherCriticBaseline = dynamic_cast<CriticNetworkBaseline*>(otherCalcs[i].get());
        if (otherCriticBaseline == nullptr) {
            std::cerr << "Received wrong baseline calculator type in Critic Network Update" << std::endl;
            throw std::bad_cast();
        }
        mTrainer.aggregate(otherCriticBaseline->mTrainer);
        otherCriticBaseline->mTrainer.reset();
    }
    mTrainer.batch(batchSize);
    mNet->update(mLearningRate, mTrainer.getTotalWeightGradients(), mTrainer.getTotalBiasGradients());
    mTrainer.reset();
}