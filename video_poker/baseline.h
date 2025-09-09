#pragma once

#include "neural.h"

#include <vector>
#include <string>
#include <memory>

class BaselineCalculator {
public:
    virtual ~BaselineCalculator() = default;
    virtual float predict(const std::vector<float>& inputs) = 0;
    virtual void train(int score) = 0;
    virtual std::string getName() = 0;
};

class FlatBaseline : public BaselineCalculator {
public:
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override;
    virtual std::string getName() { return "Flat"; }
};

class RunningAverageBaseline : public BaselineCalculator {
public:
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override;
    virtual std::string getName() { return "Running Average"; }

private:
    double mTotalScore = 0.0;
    int mCount = 0;
};

class CriticNetworkBaseline : public BaselineCalculator {
public:
    CriticNetworkBaseline(std::vector<LayerSpecification> topology, float learningRate);
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override;
    virtual std::string getName() { return "Critic Network"; }
private:
    std::unique_ptr<NeuralNet> mNet;
    Trainer mTrainer;
    float mPrediction;
    float mLearningRate;
};