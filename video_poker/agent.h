#pragma once

#include "neural.h"
#include "poker.h"
#include "decision.h"

#include <random>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <fstream>

class Agent {
public:
    Agent(const std::vector<LayerSpecification>& topology, std::string fileName, unsigned int seed, float learningRate);
    void train(const std::atomic<bool>& stopSignal, float learningRate);
    void randomEval(int iterations);
    void targetedEval();
    int getNumTrainingIterations();

private:
    NeuralNet mNet;
    VideoPoker mPoker;
    std::mt19937 mRng;
    std::unique_ptr<DecisionStrategy> mDiscardStrategy;
    int mIterations;
    int mTotalScore;
    std::ofstream mLogFile;

    std::vector<float> translateHand(const Hand& hand);
    void logAndPrintNorms();

};