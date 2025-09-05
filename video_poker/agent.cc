#include "agent.h"

#include "neural.h"
#include "poker.h"

#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>

void printExchanges(const std::vector<bool>& exchanges) {
    std::cout << "Exchanges: [";
    for (bool e : exchanges) {
        std::cout << e << ", ";
    }
    std::cout << "]" << std::endl;
}

void printErrors(const std::vector<float>& errors) {
    std::cout << "Errors: [";
    for (float e : errors) {
        std::cout << e << ", ";
    }
    std::cout << "]" << std::endl;
}

std::vector<float> translateHand(Hand hand) {
    std::vector<float> ret(85, 0.0f);
    for (int i=0; i < 5; i++) {
        Card c = hand[i];
        ret[(i*17)+c.suit] = 1.0f;
        ret[(i*17)+4+(c.rank-2)] = 1.0f;
    }
    return ret;
}

Agent::Agent(const std::vector<LayerSpecification>& topology, unsigned int seed)
        : mNet(topology),
          mPoker(VideoPoker()),
          mRng(seed) {
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

int Agent::trainOneHand(float learningRate, float baseline) {
    Hand h = mPoker.deal();
    // std::cout << "Starting Hand: " << h << std::endl;
    std::vector<float> input = translateHand(h);
    mNet.feedForward(input);
    const std::vector<float>& output = mNet.getOutputs();
    // std::cout << nn << std::endl;
    std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, true);

    h = mPoker.exchange(exchanges);
    // std::cout << "Ending Hand: " << h << std::endl;

    int score = mPoker.score(mPoker.getHandType(h));

    float advantage = (score - baseline);

    std::vector<float> errors = mDiscardStrategy->calculateError(output, exchanges, advantage);
    mNet.backpropagate(errors);
    mNet.update(learningRate);

    return score;
}

void Agent::train(int iterations, float learningRate) {

    int totalScore = 0;
    int game1000Total = 0;
    
    // for (int i = 0; i < 100000000; i++) {
    for (int i = 0; i < iterations; i++) {
        int score = trainOneHand(learningRate, float(totalScore) / (i+1));
        totalScore += score;
        game1000Total += score;
        if ((i+1) % 1000 == 0) {
            std::cout << "Games Played: " << (i+1) << ", Average Score: " << float(totalScore) / (i+1) << std::endl;
            std::cout << "Total of last 1000: " << game1000Total << std::endl;
            game1000Total = 0;
            std::cout << mNet << std::endl;
        }
    }
}


void Agent::randomEval(int iterations) {
    std::cout << "Starting Eval, " <<  iterations << " iterations." << std::endl;
    int total_score = 0;
    for (int i = 0; i < iterations; i++) {
        Hand h = mPoker.deal();
        // std::cout << "Starting Hand: " << h << std::endl;
        std::vector<float> input = translateHand(h);
        mNet.feedForward(input);
        const std::vector<float>& output = mNet.getOutputs();
        // std::cout << nn << std::endl;
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, false);
        h = mPoker.exchange(exchanges);
        // std::cout << "Ending Hand: " << h << std::endl;
        if (i % 1000 == 0) {
            std::cout << "Games Played: " << i << ", Total Score: " << total_score << std::endl;
        }

        int score = mPoker.score(mPoker.getHandType(h));
        total_score += score;
    }

    std::cout << "Games played: " << iterations << ", Total Score: " << total_score << std::endl;
    std::cout << "Average Score: " << float(total_score) / iterations << std::endl;
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
        std::cout << "Outputs: ";
        mNet.printOutput();
        std::cout << std::endl;
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, false);
        std::cout << "Decision: ";
        printExchanges(exchanges);
        std::cout << std::endl;
    }
}
