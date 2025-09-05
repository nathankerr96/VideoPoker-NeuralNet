#pragma once

#include <vector>
#include <random>

class DecisionStrategy {
public:
    virtual ~DecisionStrategy() = default;
    virtual std::vector<bool> selectAction(const std::vector<float>& netOutputs, std::mt19937& rng, bool random) = 0;
    virtual std::vector<float> calculateError(const std::vector<float>& netOutputs, const std::vector<bool>& actionTaken, float advantage) = 0;
};

class FiveNeuronStrategy : public DecisionStrategy {
public:
    std::vector<bool> selectAction(const std::vector<float>& netOutputs, std::mt19937& rng, bool random) override;
    std::vector<float> calculateError(const std::vector<float>& netOutputs, const std::vector<bool>& actionTaken, float advantage) override;
};

class ThirtyTwoNeuronStrategy : public DecisionStrategy {
public:
    std::vector<bool> selectAction(const std::vector<float>& netOutputs, std::mt19937& rng, bool random) override;
    std::vector<float> calculateError(const std::vector<float>& netOutputs, const std::vector<bool>& actionTaken, float advantage) override;
private:
    int selectDiscardCombination(const std::vector<float>& output, std::mt19937& rng, bool random);
    std::vector<bool> calcExchangeVector(int val);
    int calcIndexFromAction(const std::vector<bool>& actionTaken);
};