#include "neural.h"

#include <random>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>

Neuron::Neuron(int num_inputs, 
               std::function<float(float)> activation, 
               std::function<float(float)> activation_derivative) 
               : mBias(0),
                 mActivation(activation),
                 mActivationDerivative(activation_derivative),
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
    if (inputs.size() != mWeights.size()) throw std::invalid_argument("Inputs != Weights");
    float sum = mBias;
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * mWeights[i];
    }
    mOutput = mActivation(sum);
}

void Neuron::backpropagate(float error, const std::vector<float>& inputs) {
    float gradient = error * mActivationDerivative(mOutput);
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

float Neuron::getOutput() const {
    return mOutput;
}

const std::vector<float>& Neuron::getBlame() const {
    return mBlame;
}

Layer::Layer(int num_neurons, 
             int num_inputs,
             std::function<float(float)> activation, 
             std::function<float(float)> activation_derivative) 
             : mBlame(std::vector<float>(num_inputs, 0.0f)) {
    for (int i = 0; i < num_neurons; i++) {
        mNeurons.push_back(Neuron(num_inputs, activation, activation_derivative));
    }
    mOutputs.resize(num_neurons);
}

void Layer::fire(const std::vector<float>& inputs) {
    mLastInputs = inputs;
    for (size_t i = 0; i < mNeurons.size(); i++) {
        mNeurons[i].fire(inputs);
        mOutputs[i] = mNeurons[i].getOutput();
    }
}

void Layer::backpropagate(const std::vector<float>& errors) {
    for (size_t i = 0; i < mNeurons.size(); i++) {
        mNeurons[i].backpropagate(errors[i], mLastInputs);
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
                                topology[i].activation, 
                                topology[i].activation_derivative));
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

    // TODO: Implement batching here
    for (int i = mLayers.size()-1; i >= 0; i--) {
        mLayers[i].update(0.01f);
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
