#pragma once

#include "neural.h"
#include "trainer.h"

enum OptimizerType {
    SDG,
    MOMENTUM,
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(NeuralNet* net, Trainer& trainer, float learningRate) = 0;
};

class SDGOptimizer : public Optimizer {
    virtual void step(NeuralNet* net, Trainer& trainer, float learningRate) override;
};

class MomentumOptimizer : public Optimizer {
public:
    MomentumOptimizer(NeuralNet* net, float beta);
    virtual void step(NeuralNet* net, Trainer& trainer, float learningRate) override;

private:
    float mBeta;
    std::vector<std::vector<float>> mWeightVelocity;
    std::vector<std::vector<float>> mBiasVelocity;
};
