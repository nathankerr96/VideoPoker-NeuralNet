#include "neural.h"
#include "activations.h"
#include "poker.h"

#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>

#define INPUT_SIZE 85
#define TRAINING_RATE 0.002
#define TRAINING_ITERATIONS 2000000
#define EVAL_ITERATIONS 100000

void printExchanges(const std::vector<bool>& exchanges) {
    std::cout << "Exchanges: [";
    for (bool e : exchanges) {
        std::cout << e << ", ";
    }
    std::cout << "]" << std::endl;
}


int selectDiscardCombination(const std::vector<float>& output, std::mt19937& rng) {
    assert(output.size() == 32);
    std::uniform_real_distribution<float> uniform_zero_to_one {0.0f, 1.0f};
    float target = uniform_zero_to_one(rng);
    for (size_t i = 0; i < output.size(); i++) {
        target -= output[i];
        if (target < 0) {
            return i;
        }
    }
    assert(false);
}

std::vector<bool> calcExchangeVector(int val) {
    assert(val >= 0);
    std::vector<bool> exchanges;
    for (int i = 0; i < 5; i++) {
        exchanges.push_back(val & 1);
        val >>= 1;
    }
    return exchanges;
}

void printErrors(const std::vector<float>& errors) {
    std::cout << "Errors: [";
    for (float e : errors) {
        std::cout << e << ", ";
    }
    std::cout << "]" << std::endl;
}

std::vector<float> translateHand(Hand hand) {
    std::vector<float> ret(INPUT_SIZE, 0.0f);
    for (int i=0; i < 5; i++) {
        Card c = hand[i];
        ret[(i*17)+c.suit] = 1.0f;
        ret[(i*17)+4+(c.rank-2)] = 1.0f;
    }
    return ret;
}

void finalEval(NeuralNet& nn) {

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
        nn.feedForward(translateHand(h.second));
        std::vector<float> output = nn.getOutputs();
        std::cout << h.first << ": " << h.second << std::endl;
        std::cout << "Outputs: ";
        nn.printOutput();
        std::cout << std::endl;
        int exchangeDecision = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        std::vector<bool> exchanges = calcExchangeVector(exchangeDecision);
        std::cout << "Decision: ";
        printExchanges(exchanges);
        std::cout << std::endl;
    }
    
}


std::vector<bool> calcExchangesIndividualRandom(const std::vector<float>& output, std::mt19937& rng) {
    assert(output.size() == 5);
    std::uniform_real_distribution<float> uniform_zero_to_one {0.0f, 1.0f};
    return std::vector<bool> {
        output[0] > uniform_zero_to_one(rng),
        output[1] > uniform_zero_to_one(rng),
        output[2] > uniform_zero_to_one(rng),
        output[3] > uniform_zero_to_one(rng),
        output[4] > uniform_zero_to_one(rng)
    };
}

std::vector<bool> calcExchangesIndividualFixed(const std::vector<float>& output) {
    assert(output.size() == 5);
    return std::vector<bool> {
        output[0] > 0.5f,
        output[1] > 0.5f,
        output[2] > 0.5f,
        output[3] > 0.5f,
        output[4] > 0.5f
    };
}

void train(VideoPoker& vp, NeuralNet& nn, int iterations) {
    std::random_device rd;
    std::mt19937 mRandomGenerator {rd()};

    int total_score = 0;
    int games_played = 0;
    int game1000Total = 0;
    
    // for (int i = 0; i < 100000000; i++) {
    for (int i = 0; i < iterations; i++) {
        Hand h = vp.deal();
        // std::cout << "Starting Hand: " << h << std::endl;
        std::vector<float> input = translateHand(h);
        nn.feedForward(input);
        const std::vector<float>& output = nn.getOutputs();
        // std::cout << nn << std::endl;
        int exchangeDecision = selectDiscardCombination(output, mRandomGenerator);
        std::vector<bool> exchanges = calcExchangeVector(exchangeDecision);

        h = vp.exchange(exchanges);
        // std::cout << "Ending Hand: " << h << std::endl;

        int score = vp.score(vp.getHandType(h));
        total_score += score;
        game1000Total += score;
        games_played += 1;
        float average_score = float(total_score) / games_played;
        // std::cout << "Score: " << score <<  ", Games Played: " << games_played << ", Average Score: " << average_score << std::endl;
        if (games_played % 1000 == 0) {
            std::cout << "Games Played: " << games_played << ", Average Score: " << average_score << std::endl;
            std::cout << "Total of last 1000: " << game1000Total << std::endl;
            game1000Total = 0;
            std::cout << nn << std::endl;
        }

        float advantage = (score - average_score);
        std::vector<float> errors = output;
        errors[exchangeDecision] -= 1.0f;
        for (size_t i = 0; i < errors.size(); i++) {
            errors[i] *= advantage;
        } 

        // std::vector<float> errors {
        //     (exchanges[0] - output[0]) * (score - average_score),
        //     (exchanges[1] - output[1]) * (score - average_score),
        //     (exchanges[2] - output[2]) * (score - average_score),
        //     (exchanges[3] - output[3]) * (score - average_score),
        //     (exchanges[4] - output[4]) * (score - average_score),
        // };
        // printErrors(errors);
        nn.backpropagate(errors);
        nn.update(TRAINING_RATE);
    }
}

void evaluate(VideoPoker& vp, NeuralNet& nn, int iterations) {
    std::cout << "Starting Eval, " <<  iterations << " iterations." << std::endl;
    int total_score = 0;
    for (int i = 0; i < iterations; i++) {
        Hand h = vp.deal();
        // std::cout << "Starting Hand: " << h << std::endl;
        std::vector<float> input = translateHand(h);
        nn.feedForward(input);
        const std::vector<float>& output = nn.getOutputs();
        // std::cout << nn << std::endl;
        int exchangeDecision = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        std::vector<bool> exchanges = calcExchangeVector(exchangeDecision);
        h = vp.exchange(exchanges);
        // std::cout << "Ending Hand: " << h << std::endl;
        if (i % 1000 == 0) {
            std::cout << "Games Played: " << i << ", Total Score: " << total_score << std::endl;
        }

        int score = vp.score(vp.getHandType(h));
        total_score += score;
    }

    std::cout << "Games played: " << iterations << ", Total Score: " << total_score << std::endl;
    std::cout << "Average Score: " << float(total_score) / iterations << std::endl;
}

int main() {
    std::vector<LayerSpecification> topology {
        {INPUT_SIZE, Activation::LINEAR},
        {170, Activation::RELU},
        {170, Activation::RELU},
        {32, Activation::SOFTMAX},
    };
    NeuralNet nn {topology};
    std::cout << nn << std::endl;
    VideoPoker vp {};
    evaluate(vp, nn, EVAL_ITERATIONS);
    finalEval(nn);
    train(vp, nn, TRAINING_ITERATIONS);
    evaluate(vp, nn, EVAL_ITERATIONS);
    finalEval(nn);

    return 0;
}
