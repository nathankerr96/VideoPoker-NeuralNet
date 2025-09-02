#include <vector>
#include <iostream>
#include <functional>

class Neuron {
public:
    Neuron(int num_inputs, 
        std::function<float(float)> activation, 
        std::function<float(float)> activation_derivative);
    void fire(const std::vector<float>& inputs);
    float getOutput();
    int getNumInputs() const;

private:
    float mBias;
    std::function<float(float)> mActivation;
    std::function<float(float)> mActivationDerivative;

    std::vector<float> mWeights;
    float mOutput;
};

class Layer {
public:
    Layer(int num_neurons, 
          int num_inputs, 
          std::function<float(float)> activation, 
          std::function<float(float)> activation_derivative);
    void fire(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;
    int getNumInputs() const;
    int getNumNeurons() const;

private:
    std::vector<Neuron> mNeurons;
    std::vector<float> mOutputs;
};

struct LayerSpecification {
    int numNeurons;
    std::function<float(float)> activation;
    std::function<float(float)> activation_derivative;
};

class NeuralNet {
public:
    NeuralNet(const std::vector<LayerSpecification>& topology);
    void feedForward(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;

    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);
private:
    std::vector<Layer> mLayers;
};
