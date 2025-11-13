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

CriticNetworkBaseline::CriticNetworkBaseline(NeuralNet* net, float learningRate, std::unique_ptr<Optimizer> optimizer)
        : mNet(net), 
          mTrainingWorkspace(net),
          mLearningRate(learningRate),
          mOptimizer(std::move(optimizer)) {}

float CriticNetworkBaseline::predict(const std::vector<float>& inputs) {
    mTrainingWorkspace.feedForward(inputs);
    mPrediction = mTrainingWorkspace.getOutputs()[0];
    return mPrediction;
}

void CriticNetworkBaseline::train(int score) {
    float error = mPrediction - score;
    mTrainingWorkspace.backpropagate({error});
}

void CriticNetworkBaseline::update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) {
    for (size_t i = 1; i < otherCalcs.size(); i++) {
        // Icky encasulation breaking :( -- Crash if wrong type (bad_cast exception)
        CriticNetworkBaseline* otherCriticBaseline = dynamic_cast<CriticNetworkBaseline*>(otherCalcs[i].get());
        if (otherCriticBaseline == nullptr) {
            std::cerr << "Received wrong baseline calculator type in Critic Network Update" << std::endl;
            throw std::bad_cast();
        }
        mTrainingWorkspace.aggregate(otherCriticBaseline->mTrainingWorkspace);
        otherCriticBaseline->mTrainingWorkspace.reset();
    }
    mTrainingWorkspace.batch(batchSize);
    mOptimizer->step(mNet, mTrainingWorkspace, mLearningRate);
    // mNet->update(mLearningRate, mTrainer.getTotalWeightGradients(), mTrainer.getTotalBiasGradients());
    mTrainingWorkspace.reset();
}