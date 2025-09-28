#include "decision.h"

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>
#include <iterator>

std::vector<bool> FiveNeuronStrategy::selectAction(
        const std::vector<float>& netOutputs, 
        std::mt19937& rng, bool random) {
    assert(netOutputs.size() == 5);
    if (random) {
        std::uniform_real_distribution<float> uniform_zero_to_one {0.0f, 1.0f};
        return std::vector<bool> {
            netOutputs[0] > uniform_zero_to_one(rng),
            netOutputs[1] > uniform_zero_to_one(rng),
            netOutputs[2] > uniform_zero_to_one(rng),
            netOutputs[3] > uniform_zero_to_one(rng),
            netOutputs[4] > uniform_zero_to_one(rng)
        };
    } else {
        return std::vector<bool> {
            netOutputs[0] > 0.5f,
            netOutputs[1] > 0.5f,
            netOutputs[2] > 0.5f,
            netOutputs[3] > 0.5f,
            netOutputs[4] > 0.5f
        };
    }
}

std::vector<float> FiveNeuronStrategy::calculateError(
        const std::vector<float>& netOutputs, 
        const std::vector<bool>& actionTaken, float advantage) {
    assert(netOutputs.size() == 5);
    assert(actionTaken.size() == 5);
    std::vector<float> errors {
        (netOutputs[0] - actionTaken[0]) * advantage,
        (netOutputs[1] - actionTaken[1]) * advantage,
        (netOutputs[2] - actionTaken[2]) * advantage,
        (netOutputs[3] - actionTaken[3]) * advantage,
        (netOutputs[4] - actionTaken[4]) * advantage,
    };
    return errors;
}

std::vector<bool> ThirtyTwoNeuronStrategy::selectAction(
        const std::vector<float>& netOutputs, 
        std::mt19937& rng, bool random) {
    int exchangeDecision = selectDiscardCombination(netOutputs, rng, random);
    return calcExchangeVector(exchangeDecision);
}

std::vector<float> ThirtyTwoNeuronStrategy::calculateError(
        const std::vector<float>& netOutputs, 
        const std::vector<bool>& actionTaken, float advantage) {
    assert(netOutputs.size() == 32);
    assert(actionTaken.size() == 5);
    int indexOfAction = calcIndexFromAction(actionTaken);
    std::vector<float> errors = netOutputs;
    errors[indexOfAction] -= 1.0f;
    for (size_t i = 0; i < errors.size(); i++) {
        errors[i] *= advantage;
    } 
    return errors;
}

int ThirtyTwoNeuronStrategy::selectDiscardCombination(const std::vector<float>& netOutputs, std::mt19937& rng, bool random) {
    assert(netOutputs.size() == 32);
    if (random) {
        std::uniform_real_distribution<float> uniform_zero_to_one {0.0f, 1.0f};
        float target = uniform_zero_to_one(rng);
        for (size_t i = 0; i < netOutputs.size(); i++) {
            target -= netOutputs[i];
            if (target <= 0) {
                return i;
            }
        }
        std::cerr << "Reached end of selection loop. Remaining target: " << target << std::endl;
        return netOutputs.size() -1;
    } else {
        return std::distance(netOutputs.begin(), std::max_element(netOutputs.begin(), netOutputs.end()));
    }
}

std::vector<float> ThirtyTwoNeuronStrategy::calculateEntropyError(const std::vector<float>& netOutputs, float entropy, float beta) {
    assert(netOutputs.size() == 32);
    std::vector<float> errors(netOutputs.size());
    for (size_t i = 0; i < netOutputs.size(); i++) {
        if (netOutputs[i] > 0) {
            // TODO: More complex than it looks, come back to this and derive by hand.
            errors[i] = beta * netOutputs[i] * (std::log(netOutputs[i]) + entropy);
        } else {
            errors[i] = 0.0f;
        }
    }
    return errors;
}

std::vector<bool> ThirtyTwoNeuronStrategy::calcExchangeVector(int val) {
    assert(val >= 0);
    std::vector<bool> exchanges;
    for (int i = 0; i < 5; i++) {
        exchanges.push_back(val & 1);
        val >>= 1;
    }
    return exchanges;
}

int ThirtyTwoNeuronStrategy::calcIndexFromAction(const std::vector<bool>& actionTaken) {
    int index = 0;
    for (int i = 4; i >= 0; i--) {
        index <<= 1;
        index |= actionTaken[i];
    }
    return index;
}