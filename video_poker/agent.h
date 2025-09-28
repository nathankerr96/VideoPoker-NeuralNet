#pragma once

#include "neural.h"
#include "poker.h"
#include "decision.h"
#include "baseline.h"
#include "trainer.h"
#include "hyperparams.h"

#include <random>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <fstream>
#include <chrono>

class Agent {
public:
    Agent(const HyperParameters& config,
          std::string fileName, 
          unsigned int seed, 
          std::function<std::unique_ptr<BaselineCalculator>()> baselineFactory);
    void train(const std::atomic<bool>& stopSignal);
    void randomEval(int iterations, std::mt19937& rng) const;
    void targetedEval(std::mt19937& rng) const;
    int getNumTrainingIterations() const;

private:
    HyperParameters mConfig;
    std::unique_ptr<NeuralNet> mNet;
    std::unique_ptr<Optimizer> mOptimizer;
    std::vector<std::mt19937> mRngs; // Per worker RNG engine
    std::unique_ptr<DecisionStrategy> mDiscardStrategy;
    std::function<std::unique_ptr<BaselineCalculator>()> mBaselineFactory;
    std::ofstream mLogFile;
    // Agent-level RNG and Poker client for sample hands and Evals. Worker threads have separate copies.
    std::mt19937 mRng;
    VideoPoker mVideoPoker;
    // Progress indicators
    std::atomic<int> mTotalScore = 0;
    std::atomic<int> mRecentTotal = 0;
    std::atomic<int> mIterations = 0;
    int mNumBatches = 0; // Only called from single-threaded completion step.
    std::chrono::duration<double> mTotalTrainingTime {};

    std::vector<float> translateHand(const Hand& hand) const;
    float calculateEntropy(const std::vector<float>& policy);
    // Should be called after gradient aggregation but before reset! (Else gradient norm == 0)
    void logProgress(Trainer& t, BaselineCalculator* baselineCalc);
    void logAndPrintNorms(const Trainer& trainer);
};