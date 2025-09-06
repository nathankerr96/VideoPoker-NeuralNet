#include "agent.h"

#include "neural.h"
#include "poker.h"

#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <atomic>


Agent::Agent(const std::vector<LayerSpecification>& topology, unsigned int seed)
        : mNet(topology),
          mPoker(VideoPoker()),
          mRng(seed),
          mIterations(0),
          mTotalScore(0) {
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


int Agent::trainOneHand(float learningRate, float baseline) {
    Hand h = mPoker.deal();
    std::vector<float> input = translateHand(h);
    mNet.feedForward(input);
    const std::vector<float>& output = mNet.getOutputs();
    std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, true);
    h = mPoker.exchange(exchanges);

    int score = mPoker.score(mPoker.getHandType(h));
    float advantage = (score - baseline);
    std::vector<float> errors = mDiscardStrategy->calculateError(output, exchanges, advantage);
    mNet.backpropagate(errors);
    mNet.update(learningRate);

    return score;
}

void Agent::train(const std::atomic<bool>& stopSignal, float learningRate) {
    int game1000Total = 0;
    while (true) {
        mIterations += 1;
        int score = trainOneHand(learningRate, float(mTotalScore) / mIterations);
        mTotalScore += score;
        game1000Total += score;
        if (mIterations % 1000 == 0) {
            std::cout << "Games Played: " << mIterations << ", Average Score: " << float(mTotalScore) / mIterations << std::endl;
            std::cout << "  Average of last 1000: " << float(game1000Total) / 1000 << std::endl;
            std::cout << "  Sample Outputs: " << mNet.getOutputs() << std::endl;
            game1000Total = 0;
            if (stopSignal) break; // Only stops on even multiples of 1000
        }
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
    std::cout << "---Average Score: " << float(total_score) / iterations << "---" << std::endl;
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
