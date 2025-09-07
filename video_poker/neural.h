#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <iostream>
#include <functional>
#include <string>

class Neuron {
public:
    Neuron(int num_inputs);
    void fire(const std::vector<float>& inputs);
    void backpropagate(float gradient, const std::vector<float>& inputs);
    void update(float learningRate);
    float getLogit() const;
    const std::vector<float>& getBlame() const;
    int getNumInputs() const;
    double getWeightNormSquared() const;
    double getGradientNormSquared() const;

private:
    float mBias;
    float mBiasChange;
    std::vector<float> mWeights;
    std::vector<float> mWeightChanges; // Separate update step for batching.
    std::vector<float> mBlame;
    float mLogit;
};

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
    void backpropagate(const std::vector<float>& errors);
    void update(float learningRate);
    const std::vector<float>& getBlame() const;
    double getWeightNormSquared() const;
    double getGradientNormSquared() const;

private:
    std::vector<Neuron> mNeurons;
    std::vector<float> mBlame;
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
    std::vector<double> getLayerGradientNormsSquared() const;


    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);
private:
    std::vector<Layer> mLayers;
};

std::ostream& operator<<(std::ostream& os, const std::vector<float>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v);
std::ostream& operator<<(std::ostream& os, const LayerSpecification& l);
std::ostream& operator<<(std::ostream& os, const std::vector<LayerSpecification>& v);

#endif