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

class BaseAgent {
public:
    virtual ~BaseAgent() = default;
    virtual void train(const std::atomic<bool>& stopSignal) = 0;
    void randomEval(int iterations, std::mt19937& rng) const;
    void targetedEval(std::mt19937& rng) const;
protected:
    std::vector<float> translateHand(const Hand& hand) const;
    std::unique_ptr<DecisionStrategy> mDiscardStrategy;
    virtual NeuralNet* getNet() const = 0;
};
