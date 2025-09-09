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
    void fire(const std::vector<float>& inputs,
              std::vector<float>& logitsBuffer,
              std::vector<float>& outputs) const;
    const std::vector<float>& getOutputs() const;
    int getNumInputs() const;
    int getNumNeurons() const;
    void backpropagate(const std::vector<float>& errors,
                       const std::vector<float>& layerInputs,
                       const std::vector<float>& layerActivations,
                       std::vector<float>& deltaBuffer,
                       std::vector<float>& outputDerivativesBuffer,
                       std::vector<float>& weightGradientOut,
                       std::vector<float>& biasGradientOut,
                       std::vector<float>& downstreamGradientOut) const;
    void update(float learningRate,
                const std::vector<float>& weightGradient, 
                const std::vector<float>& biasGradient);
    double getWeightNormSquared() const;

private:
    int mNumNeurons;
    int mNumInputs;
    std::vector<float> mWeights;
    std::vector<float> mBiases;
    Activation mActivationType;
};

struct LayerSpecification {
    int numNeurons;
    Activation activationType;
};

class NeuralNet {
public:
    NeuralNet(const std::vector<LayerSpecification>& topology);
    void feedForward(const std::vector<float>& inputs) const;
    void update(float learningRate,
        const std::vector<std::vector<float>>& weightGradients,
        const std::vector<std::vector<float>>& biasGradients);
    std::vector<double> getLayerWeightNormsSquared() const;
    const std::vector<Layer>& getLayers();


    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);
private:
    std::vector<Layer> mLayers;
};

class Trainer {
public:
    Trainer(NeuralNet* net);
    // void trainOneHand();
    std::vector<double> getLayerGradientNormsSquared() const;

    // TODO: Encapsulate training logic and make these private.
    void feedForward(const std::vector<float>& inputs);
    void backpropagate(const std::vector<float>& errors);
    const std::vector<float>& getOutputs();
    const std::vector<std::vector<float>>& getWeightGradients();
    const std::vector<std::vector<float>>& getBiasGradients();

private:

    NeuralNet* mNet;

    std::vector<std::vector<float>> mWeightGradients;
    std::vector<std::vector<float>> mBiasGradients;
    std::vector<float> mLogitsBuffer;
    std::vector<std::vector<float>> mActivations;
    std::vector<float> mBlameBufferA;
    std::vector<float> mBlameBufferB;
    std::vector<float> mDeltaBuffer;
    std::vector<float> mOutputDerivativesBuffer;
};


std::ostream& operator<<(std::ostream& os, const std::vector<float>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v);
std::ostream& operator<<(std::ostream& os, const LayerSpecification& l);
std::ostream& operator<<(std::ostream& os, const std::vector<LayerSpecification>& v);

#endif