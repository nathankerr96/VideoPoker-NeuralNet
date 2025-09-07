#include "agent.h"

#include "neural.h"
#include "poker.h"

#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <atomic>
#include <string>


Agent::Agent(const std::vector<LayerSpecification>& topology, std::string fileName, unsigned int seed, float learningRate)
        : mNet(topology),
          mPoker(VideoPoker()),
          mRng(seed),
          mIterations(0),
          mTotalScore(0),
          mLogFile(fileName) {
    assert(topology[0].numNeurons == 85); // Hard dependency by hand translation layer.
    int outputSize = topology.back().numNeurons;
    switch (outputSize) {
        case 5:
            mDiscardStrategy = std::make_unique<FiveNeuronStrategy>();
            break;
        case 32:
            mDiscardStrategy = std::make_unique<ThirtyTwoNeuronStrategy>();
            break;
        default:
            std::cerr << "Output Layer Size: " << outputSize;
            throw std::invalid_argument("Unsupported Output Layer Size");
    }

    if (!mLogFile.is_open()) {
        std::cerr << "Could not open Log file!" << std::endl;
    } else {
        mLogFile << "Topology " << std::endl << topology << std::endl;
        mLogFile << "Learning Rate," << learningRate << std::endl << std::endl;
        mLogFile << "Iterations,TotalAvgScore,RecentAvgScore,GlobalWeightNorm,GlobalGradientNorm,";
        for (size_t i = 1; i < topology.size(); i++) {
            mLogFile << "Layer" << i << "WeightNorm,";
        }
        for (size_t i = 1; i < topology.size(); i++) {
            mLogFile << "Layer" << i << "GradientNorm,";
        }
        mLogFile << std::endl;
    }
}

std::vector<float> Agent::translateHand(const Hand& hand) {
    std::vector<float> ret(85, 0.0f);
    for (int i=0; i < 5; i++) {
        Card c = hand[i];
        ret[(i*17)+c.suit] = 1.0f;
        ret[(i*17)+4+(c.rank-2)] = 1.0f;
    }
    return ret;
}


void Agent::train(const std::atomic<bool>& stopSignal, float learningRate) {
    int recentTotal = 0;
    while (true) {
        Hand h = mPoker.deal();
        std::vector<float> input = translateHand(h);
        mNet.feedForward(input);
        const std::vector<float>& output = mNet.getOutputs();
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, true);
        h = mPoker.exchange(exchanges);

        int score = mPoker.score(mPoker.getHandType(h));
        mIterations += 1;
        mTotalScore += score;
        recentTotal += score;

        float baseline = float(mTotalScore) / mIterations;
        float advantage = (score - baseline);
        std::vector<float> errors = mDiscardStrategy->calculateError(output, exchanges, advantage);
        mNet.backpropagate(errors);

        if (mIterations % 1000 == 0) {
            float averageTotalScore = float(mTotalScore) / mIterations;
            float averageRecentScore = float(recentTotal) / 1000;
            std::cout << "Games Played: " << mIterations << ", Average Score: " << averageTotalScore << std::endl;
            std::cout << "  Average of last 1000: " << averageRecentScore << std::endl;
            std::cout << "  Sample Outputs: " << mNet.getOutputs() << std::endl;
            mLogFile << mIterations << ",";
            mLogFile << averageTotalScore << ",";
            mLogFile << averageRecentScore << ",";
            logAndPrintNorms();
            mLogFile << std::endl;
            std::cout << std::endl;
            recentTotal = 0;
            if (stopSignal) break; // Only stops on even multiples of 1000
        }

        mNet.update(learningRate);
    }
}

int Agent::getNumTrainingIterations() {
    return mIterations;
}

void Agent::randomEval(int iterations) {
    std::cout << "---Starting Eval, " <<  iterations << " iterations.---" << std::endl;
    int total_score = 0;
    for (int i = 0; i < iterations; i++) {
        Hand h = mPoker.deal();
        std::vector<float> input = translateHand(h);
        mNet.feedForward(input);
        const std::vector<float>& output = mNet.getOutputs();
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, false);
        h = mPoker.exchange(exchanges);
        if ((i+1) % 10000 == 0) {
            std::cout << "Games Played: " << (i+1) << ", Total Score: " << total_score << std::endl;
        }
        int score = mPoker.score(mPoker.getHandType(h));
        total_score += score;
    }
    std::cout << "---Average Score: " << float(total_score) / iterations << "---" << std::endl << std::endl;
}


void Agent::targetedEval() {
    std::vector<std::pair<std::string, Hand>> hands {
        {"Junk", {{{{CLUB, 2}, {SPADE, 7}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"Pair", {{{{CLUB, 2}, {SPADE, 2}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 3}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 12}}}}},
        {"Two Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Trips", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Quads", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 12}}}}}
    };

    for (const auto& h : hands) {
        mNet.feedForward(translateHand(h.second));
        std::vector<float> output = mNet.getOutputs();
        std::cout << h.first << ": " << h.second << std::endl;
        std::cout << "Outputs: " << mNet.getOutputs() << std::endl;
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, false);
        std::cout << "Decision: " << exchanges << std::endl;
    }
}

void Agent::logAndPrintNorms() {
    std::vector<double> weightNormsSquared = mNet.getLayerWeightNormsSquared();
    std::cout << "Weight Norms:" << std::endl;
    double totalWeightNormSquared = 0.0;
    for (size_t i = 0; i < weightNormsSquared.size(); i++) {
        std::cout << "Layer " << i << ": " << std::sqrt(weightNormsSquared[i]) << std::endl;
        totalWeightNormSquared += weightNormsSquared[i];
    }
    double globalWeightNorm = std::sqrt(totalWeightNormSquared);
    std::cout << "Overal Weight Norm: " <<  globalWeightNorm << std::endl;

    std::vector<double> gradientNormsSquared = mNet.getLayerGradientNormsSquared();
    std::cout << "Gradient Norms:" << std::endl;
    double totalGradientNormSquared = 0.0;
    for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
        std::cout << "Layer " << i << ": " << std::sqrt(gradientNormsSquared[i]) << std::endl;
        totalGradientNormSquared += gradientNormsSquared[i];
    }
    double globalGradientNorm = std::sqrt(totalGradientNormSquared);
    std::cout << "Overal Gradient Norm: " << globalGradientNorm << std::endl;

    mLogFile << globalWeightNorm << ",";
    mLogFile << globalGradientNorm << ",";
    for (size_t i = 0; i < weightNormsSquared.size(); i++) {
        mLogFile << std::sqrt(weightNormsSquared[i]) << ",";
    }
    for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
        mLogFile << std::sqrt(gradientNormsSquared[i]) << ",";
    }
}