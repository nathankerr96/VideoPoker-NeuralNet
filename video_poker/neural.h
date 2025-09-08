#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <iostream>
#include <functional>
#include <string>

enum class Activation {
    SIGMOID,
    RELU,
    SOFTMAX,
    LINEAR
};

class Layer {
public:
    Layer(int num_neurons, 
          int num_inputs, 
          Activation activationtype);
    void fire(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;
    int getNumInputs() const;
    int getNumNeurons() const;
    void backpropagate(const std::vector<float>& errors,
                       std::vector<float>& weightGradientOut,
                       std::vector<float>& biasGradientOut,
                       std::vector<float>& downstreamGradientOut);
    void update(float learningRate,
                const std::vector<float>& weightGradient, 
                const std::vector<float>& biasGradient);
    const std::vector<float>& getBlame() const;
    double getWeightNormSquared() const;

private:
    int mNumNeurons;
    int mNumInputs;
    std::vector<float> mWeights;
    std::vector<float> mBiases;

    Activation mActivationType;
    std::vector<float> mOutputs;
    std::vector<float> mLastInputs;
};

struct LayerSpecification {
    int numNeurons;
    Activation activationType;
};

class NeuralNet {
public:
    NeuralNet(const std::vector<LayerSpecification>& topology);
    void feedForward(const std::vector<float>& inputs);
    void backpropagate(const std::vector<float>& errors);
    void update(float learningRate);
    const std::vector<float>& getOutputs() const;
    std::vector<double> getLayerWeightNormsSquared() const;
    std::vector<double> getLayerGradientNormsSquared(
            const std::vector<std::vector<float>>& netBiasGraidents,
            const std::vector<std::vector<float>>& netWeightGradients) const;


    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);
private:
    std::vector<Layer> mLayers;
    std::vector<std::vector<float>> mWeightGradients;
    std::vector<std::vector<float>> mBiasGradients;
};

std::ostream& operator<<(std::ostream& os, const std::vector<float>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v);
std::ostream& operator<<(std::ostream& os, const LayerSpecification& l);
std::ostream& operator<<(std::ostream& os, const std::vector<LayerSpecification>& v);

#endif