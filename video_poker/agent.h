#pragma once

#include "neural.h"
#include "poker.h"
#include "decision.h"
#include "baseline.h"

#include <random>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <fstream>

class Agent {
public:
    Agent(const std::vector<LayerSpecification>& topology, 
          std::string fileName, 
          unsigned int seed, 
          float learningRate,
          std::unique_ptr<BaselineCalculator> baselineCalc);
    void train(const std::atomic<bool>& stopSignal, float learningRate);
    void randomEval(int iterations);
    void targetedEval();
    int getNumTrainingIterations();

private:
    std::unique_ptr<NeuralNet> mNet;
    VideoPoker mPoker;
    std::mt19937 mRng;
    std::unique_ptr<DecisionStrategy> mDiscardStrategy;
    std::unique_ptr<BaselineCalculator> mBaselineCalculator;
    std::atomic<int> mIterations;
    std::atomic<int> mTotalScore;
    std::atomic<int> mRecentTotal;
    std::ofstream mLogFile;

    std::vector<float> translateHand(const Hand& hand);
    void logAndPrintNorms(const Trainer& trainer);

};