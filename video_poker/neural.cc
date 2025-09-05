#include "neural.h"
#include "activations.h"

#include <random>
#include <memory>
#include <stdexcept>
#include <cmath>

Neuron::Neuron(int num_inputs)
               : mBias(0),
                 mWeightChanges(std::vector<float>(num_inputs, 0.0f)),
                 mBlame(std::vector<float>(num_inputs, 0.0f)) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    mWeights.reserve(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        mWeights.push_back(dis(generator) / sqrt(num_inputs));
    }
}

void Neuron::fire(const std::vector<float>& inputs) {
    if (inputs.size() != mWeights.size()) {
        std::cerr << "Inputs: " << inputs.size() << ", Weights: " << mWeights.size() << std::endl;
        throw std::invalid_argument("Inputs != Weights");
    }
    float sum = mBias;
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * mWeights[i];
    }
    mLogit = sum;
}

void Neuron::backpropagate(float gradient, const std::vector<float>& inputs) {
    for (size_t i = 0; i < mWeights.size(); i++) {
        mBlame[i] = mWeights[i] * gradient;
    }
    for (size_t i = 0; i < mWeights.size(); i++) {
        mWeightChanges[i] += gradient * inputs[i];
    }
}

void Neuron::update(float learningRate) {
    for (size_t i = 0; i < mWeights.size(); i++) {
        mWeights[i] = mWeights[i] - (learningRate * mWeightChanges[i]);
    }
    std::fill(mWeightChanges.begin(), mWeightChanges.end(), 0);
}

float Neuron::getLogit() const {
    return mLogit;
}

const std::vector<float>& Neuron::getBlame() const {
    return mBlame;
}

Layer::Layer(int num_neurons, 
             int num_inputs,
             Activation activationType)
             : mBlame(std::vector<float>(num_inputs, 0.0f)), mActivationType(activationType) {
    for (int i = 0; i < num_neurons; i++) {
        mNeurons.push_back(Neuron(num_inputs));
    }
    mOutputs.resize(num_neurons);
}

void Layer::fire(const std::vector<float>& inputs) {
    mLastInputs = inputs;
    std::vector<float> mLogits;
    for (Neuron& n : mNeurons) {
        n.fire(inputs);
        mLogits.push_back(n.getLogit());
    }
    switch (mActivationType) {
        case Activation::LINEAR:
            mOutputs = mLogits;
            break;
        case Activation::RELU:
            mOutputs = relu(mLogits);
            break;
        case Activation::SIGMOID:
            mOutputs = sigmoid(mLogits);
            break;
        // case Activation::SOFTMAX:
        //     mOutputs = softmax(mLogits);
        //     break;
    }
}

void Layer::backpropagate(const std::vector<float>& errors) {
    std::vector<float> outputDerivatives;
    switch (mActivationType) {
        case Activation::LINEAR:
            outputDerivatives = std::vector<float>(mOutputs.size(), 1.0f);
            break;
        case Activation::RELU:
            outputDerivatives = relu_derivative(mOutputs);
            break;
        case Activation::SIGMOID:
            outputDerivatives = sigmoid_derivative(mOutputs);
            break;
        // case Activation::SOFTMAX:
        //     mOutputs = softmax(mLogits);
        //     break;
    }
    for (size_t i = 0; i < mNeurons.size(); i++) {
        float gradient = outputDerivatives[i] * errors[i];
        mNeurons[i].backpropagate(gradient, mLastInputs);
        const std::vector<float>& neuronBlame = mNeurons[i].getBlame();
        for (size_t j = 0; j < mBlame.size(); j++) {
            mBlame[j] += neuronBlame[j];
        }
    }
}

void Layer::update(float learningRate) {
    for (Neuron& n : mNeurons) {
        n.update(learningRate);
    }
    std::fill(mBlame.begin(), mBlame.end(), 0);
}

const std::vector<float>& Layer::getOutputs() const {
    return mOutputs;
}

const std::vector<float>& Layer::getBlame() const {
    return mBlame;
}

NeuralNet::NeuralNet(const std::vector<LayerSpecification>& topology) {
    for (size_t i = 1; i < topology.size(); i++) {
        mLayers.push_back(Layer(topology[i].numNeurons, 
                                topology[i-1].numNeurons, 
                                topology[i].activationType));
    }
}

void NeuralNet::feedForward(const std::vector<float>& inputs) {
    mLayers[0].fire(inputs);
    for (size_t i = 1; i < mLayers.size(); i++) {
        mLayers[i].fire(mLayers[i-1].getOutputs());
    }
}

void NeuralNet::backpropagate(const std::vector<float>& errors) {
    mLayers[mLayers.size()-1].backpropagate(errors);
    for (int i = mLayers.size()-2; i >= 0; i--) {
        mLayers[i].backpropagate(mLayers[i+1].getBlame());
    }
}

void NeuralNet::update(float learningRate) {
    for (int i = mLayers.size()-1; i >= 0; i--) {
        mLayers[i].update(learningRate);
    }
}

const std::vector<float>& NeuralNet::getOutputs() const {
    return mLayers.back().getOutputs();
}

int Neuron::getNumInputs() const {
    return mWeights.size();
}

int Layer::getNumInputs() const {
    if (mNeurons.empty()) {
        return 0;
    }
    return mNeurons[0].getNumInputs();
}

int Layer::getNumNeurons() const {
    return mNeurons.size();
}

void NeuralNet::printOutput() {
    std::cout << "[";
    for (float f : mLayers.back().getOutputs()) {
        std::cout << f << ", ";
    }
    std::cout << "]";
}

std::ostream& operator<<(std::ostream& os, const NeuralNet& net) {
    os << "NeuralNet Topology: ";
    if (net.mLayers.empty()) {
        os << "[No layers]" << std::endl;
        return os;
    }

    // Reconstruct topology
    os << net.mLayers[0].getNumInputs(); // Input layer size
    for (const auto& layer : net.mLayers) {
        os << " -> " << layer.getNumNeurons();
    }
    os << std::endl;

    // Print output
    os << "Output: [ ";
    const auto& outputs = net.getOutputs();
    if (outputs.empty()) {
        os << "(not computed yet)";
    } else {
        for (size_t i = 0; i < outputs.size(); ++i) {
            os << outputs[i] << (i == outputs.size() - 1 ? "" : ", ");
        }
    }
    os << " ]";

    return os;
}
