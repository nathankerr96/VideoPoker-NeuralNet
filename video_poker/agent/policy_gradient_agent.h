#pragma once

#include "agent/base_agent.h"
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

class PolicyGradientAgent : public BaseAgent {
public:
    PolicyGradientAgent(const HyperParameters& config,
          std::string fileName, 
          unsigned int seed, 
          std::function<std::unique_ptr<BaselineCalculator>()> baselineFactory);
    void train(const std::atomic<bool>& stopSignal) override;
    std::vector<float> predict(const std::vector<float>& input) const override;
    int getNumTrainingIterations() const;
private:
    HyperParameters mConfig;
    std::unique_ptr<NeuralNet> mNet;
    std::unique_ptr<Optimizer> mOptimizer;
    std::vector<std::mt19937> mRngs; // Per worker RNG engine
    std::function<std::unique_ptr<BaselineCalculator>()> mBaselineFactory;
    std::ofstream mLogFile;
    // Agent-level RNG and Poker client for sample hands and Evals. Worker threads have separate copies.
    std::mt19937 mRng;
    VideoPoker mVideoPoker;
    // Progress indicators
    std::atomic<int> mTotalScore = 0;
    std::atomic<int> mRecentTotal = 0;
    std::atomic<float> mRecentEntropy = 0.0f;
    std::atomic<int> mIterations = 0;
    int mNumBatches = 0; // Only called from single-threaded completion step.
    std::chrono::duration<double> mTotalTrainingTime {};

    float calculateEntropy(const std::vector<float>& policy);
    // Should be called after gradient aggregation but before reset! (Else gradient norm == 0)
    void logProgress(Trainer& t, BaselineCalculator* baselineCalc);
    void logAndPrintNorms(const Trainer& trainer);
    NeuralNet* getNet() const override;
};