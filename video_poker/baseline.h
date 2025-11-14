#pragma once

#include "neural.h"
#include "workspace.h"
#include "optimizer.h"

#include <vector>
#include <string>
#include <memory>

enum BaselineCalculatorType {
    FLAT,
    RUNNING_AVERAGE,
    CRITIC_NETWORK,
};

class BaselineCalculator {
public:
    virtual ~BaselineCalculator() = default;
    virtual float predict(const std::vector<float>& inputs) = 0;
    virtual void train(int score) = 0;
    virtual void update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) = 0;
    virtual std::string getName() = 0;
};

class FlatBaseline : public BaselineCalculator {
public:
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override { /*No-Op*/ };
    virtual void update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) override { /*No-Op*/ }
    virtual std::string getName() { return "Flat"; }
};

class RunningAverageBaseline : public BaselineCalculator {
public:
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override;
    // For simplicity, let each worker thread keep it's own running average. 
    virtual void update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) override { /* No-Op */ };
    virtual std::string getName() { return "Running Average"; }

private:
    double mTotalScore = 0.0;
    int mCount = 0;
};

class CriticNetworkBaseline : public BaselineCalculator {
public:
    CriticNetworkBaseline(NeuralNet* net, const std::vector<LayerSpecification>& criticTopology, float learningRate, std::unique_ptr<Optimizer> optimizer);
    virtual float predict(const std::vector<float>& inputs) override;
    virtual void train(int score) override;
    // Aggregates gradients and updates underlying net. Must only be called from *one* calculator.
    virtual void update(std::vector<std::unique_ptr<BaselineCalculator>>& otherCalcs, int batchSize) override;
    virtual std::string getName() { return "Critic Network"; }
private:
    NeuralNet* mNet;
    TrainingWorkspace mTrainingWorkspace;
    float mPrediction;
    float mLearningRate;
    std::unique_ptr<Optimizer> mOptimizer;
};