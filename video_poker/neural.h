#include <vector>
#include <iostream>

class Neuron {
public:
    Neuron(int num_inputs);
    void fire(const std::vector<float>& inputs);
    float getOutput();
    int getNumInputs() const;

private:
    float activate(float x);

    std::vector<float> mWeights;
    float mBias;
    float mOutput;
};

class Layer {
public:
    Layer(int num_neurons, int num_inputs);
    void fire(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;
    int getNumInputs() const;
    int getNumNeurons() const;

private:
    std::vector<Neuron> mNeurons;
    std::vector<float> mOutputs;
};

class NeuralNet {
public:
    NeuralNet(const std::vector<int>& topology);
    void feedForward(const std::vector<float>& inputs);
    const std::vector<float>& getOutputs() const;

    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& net);
private:
    std::vector<Layer> mLayers;
};
