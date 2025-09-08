#include "neural.h"
#include "activations.h"

#include <random>
#include <memory>
#include <stdexcept>
#include <cmath>

Layer::Layer(int num_neurons, 
             int num_inputs,
             Activation activationType)
             : mNumNeurons(num_neurons),
               mNumInputs(num_inputs),
               mBiases(std::vector<float>(num_neurons, 0.0f)), // Biases can start at 0 since weights break symmetry
               mActivationType(activationType),
               mLogits(num_neurons),
               mDelta(num_neurons),
               mOutputDerivatives(num_neurons) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    mWeights.reserve(num_neurons * num_inputs);
    for (int i = 0; i < num_neurons * num_inputs; i++) {
        mWeights.push_back(dis(generator) / sqrt(num_inputs));
    }
    mOutputs.resize(num_neurons);
}

void Layer::fire(const std::vector<float>& inputs) {
    if (int(inputs.size()) != mNumInputs) {
        std::cerr << "Inputs: " << inputs.size() << ", Neurons: " << mNumNeurons << std::endl;
        throw std::invalid_argument("Inputs != Weights");
    }
    mLastInputs = inputs;
    for (int n = 0; n < mNumNeurons; n++) {
        float sum = mBiases[n];
        for (size_t i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * mWeights[n*mNumInputs+i];
        }
        mLogits[n] = sum;
    }
    switch (mActivationType) {
        case Activation::LINEAR:
            mOutputs = mLogits;
            break;
        case Activation::RELU:
            relu(mLogits, mOutputs);
            break;
        case Activation::SIGMOID:
            sigmoid(mLogits, mOutputs);
            break;
        case Activation::SOFTMAX:
            softmax(mLogits, mOutputs);
            break;
    }
}

void Layer::backpropagate(const std::vector<float>& upstreamGradient,
                          std::vector<float>& weightGradientOut,
                          std::vector<float>& biasGradientOut,
                          std::vector<float>& downstreamGradientOut) {
    if (mActivationType == Activation::SOFTMAX) {
        mDelta = upstreamGradient;
    } else {
        switch (mActivationType) {
            case Activation::LINEAR:
                // mOutputDerivatives = std::vector<float>(mOutputs.size(), 1.0f);
                std::fill(mOutputDerivatives.begin(), mOutputDerivatives.end(), 1.0f);
                break;
            case Activation::RELU:
                relu_derivative(mOutputs, mOutputDerivatives);
                break;
            case Activation::SIGMOID:
                sigmoid_derivative(mOutputs, mOutputDerivatives);
                break;
            case Activation::SOFTMAX:
                // Errors vector is already final gradient, handled above.
                break;
        }
        for (int n = 0; n < mNumNeurons; n++) {
            mDelta[n] = mOutputDerivatives[n] * upstreamGradient[n];
        }
    }
    for (int n = 0; n < mNumNeurons; n++) {
        for (int i = 0; i < mNumInputs; i++) {
            weightGradientOut[n*mNumInputs+i] = mDelta[n] * mLastInputs[i];
        }
    }
    biasGradientOut = mDelta;

    for (int i = 0; i < mNumInputs; i++) {
        float sum = 0.0f;
        for (int n = 0; n < mNumNeurons; n++) {
            sum += mWeights[n*mNumInputs+i] * mDelta[n];
        }
        downstreamGradientOut[i] = sum;
    }
}

void Layer::update(float learningRate, 
                   const std::vector<float>& weightGradient, 
                   const std::vector<float>& biasGradient) {
    for (int n = 0; n < mNumNeurons; n++) {
        for (int i = 0; i < mNumInputs; i++) {
            mWeights[n*mNumInputs+i] = mWeights[n*mNumInputs+i] - (learningRate * weightGradient[n*mNumInputs+i]);
        }
        mBiases[n] = mBiases[n] - (learningRate * biasGradient[n]);
    }
}

const std::vector<float>& Layer::getOutputs() const {
    return mOutputs;
}

double Layer::getWeightNormSquared() const {
    double sum = 0.0;
    for (float b : mBiases) {
        sum += b * b;
    }
    for (float w : mWeights) {
        sum += w * w;
    }
    return sum;
}

NeuralNet::NeuralNet(const std::vector<LayerSpecification>& topology) {
    for (size_t i = 1; i < topology.size(); i++) {
        mLayers.push_back(Layer(topology[i].numNeurons, 
                                topology[i-1].numNeurons, 
                                topology[i].activationType));
    }
    mWeightGradients.resize(mLayers.size());
    mBiasGradients.resize(mLayers.size());
    int maxNeurons = 0;
    for (size_t i = 0; i < mLayers.size(); i++) {
        mWeightGradients[i].resize(mLayers[i].getNumInputs()*mLayers[i].getNumNeurons());
        mBiasGradients[i].resize(mLayers[i].getNumNeurons());
        if (mLayers[i].getNumNeurons() > maxNeurons) {
            maxNeurons = mLayers[i].getNumNeurons();
        }
    }
    mBlameBufferA.resize(maxNeurons, 0.0f);
    mBlameBufferB.resize(maxNeurons, 0.0f);
}

void NeuralNet::feedForward(const std::vector<float>& inputs) {
    mLayers[0].fire(inputs);
    for (size_t i = 1; i < mLayers.size(); i++) {
        mLayers[i].fire(mLayers[i-1].getOutputs());
    }
}

void NeuralNet::backpropagate(const std::vector<float>& errors) {
    std::vector<float>* upstreamGradient = nullptr;
    std::vector<float>* downstreamGradient = &mBlameBufferA; 
    int last = mLayers.size() - 1;
    mLayers[mLayers.size()-1].backpropagate(errors, mWeightGradients[last], mBiasGradients[last], *downstreamGradient);
    for (int i = last-1; i >= 0; i--) {
        upstreamGradient = downstreamGradient;
        downstreamGradient = (upstreamGradient == &mBlameBufferA ? &mBlameBufferB : &mBlameBufferA);
        mLayers[i].backpropagate(*upstreamGradient, mWeightGradients[i], mBiasGradients[i], *downstreamGradient);
    }
}

void NeuralNet::update(float learningRate) {
    for (int i = mLayers.size()-1; i >= 0; i--) {
        mLayers[i].update(learningRate, mWeightGradients[i], mBiasGradients[i]);
    }
}

const std::vector<float>& NeuralNet::getOutputs() const {
    return mLayers.back().getOutputs();
}

int Layer::getNumInputs() const {
    return mNumInputs;
}

int Layer::getNumNeurons() const {
    return mNumNeurons;
}

std::vector<double> NeuralNet::getLayerWeightNormsSquared() const {
    std::vector<double> ret;
    for (size_t i = 0; i < mLayers.size(); i++) {
        ret.push_back(mLayers[i].getWeightNormSquared());
    }
    return ret;
}

std::vector<double> NeuralNet::getLayerGradientNormsSquared() const {
    std::vector<double> ret;
    for (size_t l = 0; l < mWeightGradients.size(); l++) {
        double layerSum = 0.0;
        for (float b : mBiasGradients[l]) {
            layerSum += b * b;
        }
        for (float w : mWeightGradients[l]) {
            layerSum += w * w;
        }
        ret.push_back(layerSum);
    }
    return ret;
}

std::ostream& operator<<(std::ostream& os, const std::vector<float>& v) {
    os << "[ ";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    os << " ]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) {
    os << "[ ";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    os << " ]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const LayerSpecification& l) {
    os << "Neurons-" << l.numNeurons << ", Activation-";
    switch (l.activationType) {
        case Activation::SIGMOID:
            os << "SIGMOID";
            break;
        case Activation::RELU:
            os << "RELU";
            break;
        case Activation::SOFTMAX:
            os << "SOFTMAX";
            break;
        case Activation::LINEAR:
            os << "LINEAR";
            break;
        default:
            os << "UNKNOWN";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<LayerSpecification>& v) {
    for (size_t i = 0; i < v.size(); i++) {
        os << "Layer " << i << ", " << v[i];
    }
    return os;
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


    return os;
}
