#include "neural.h"
#include "activations.h"

#include <random>
#include <memory>
#include <stdexcept>
#include <cmath>

double getNeuronWeightNormSquared(float bias, std::vector<float> weights) {
    double sum = bias * bias;
    for (float w : weights) {
        sum += w * w;
    }
    return sum;
}

Layer::Layer(int num_neurons, 
             int num_inputs,
             Activation activationType)
             : mBiases(std::vector<float>(num_neurons, 0.0f)), // Biases can start at 0 since weights break symmetry
               mActivationType(activationType) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    mWeights.reserve(num_neurons);
    for (int i = 0; i < num_neurons; i++) {
        std::vector<float> weights;
        weights.reserve(num_inputs);
        for (int j = 0; j < num_inputs; j++) {
            weights.push_back(dis(generator) / sqrt(num_inputs));
        }
        mWeights.push_back(weights);
    }
    mOutputs.resize(num_neurons);
}

void Layer::fire(const std::vector<float>& inputs) {
    if (inputs.size() != mWeights[0].size()) {
        std::cerr << "Inputs: " << inputs.size() << ", Weights: " << mWeights[0].size() << std::endl;
        throw std::invalid_argument("Inputs != Weights");
    }
    mLastInputs = inputs;
    std::vector<float> logits;
    for (size_t n = 0; n < mWeights.size(); n++) {
        float sum = mBiases[n];
        for (size_t i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * mWeights[n][i];
        }
        logits.push_back(sum);
    }
    switch (mActivationType) {
        case Activation::LINEAR:
            mOutputs = logits;
            break;
        case Activation::RELU:
            mOutputs = relu(logits);
            break;
        case Activation::SIGMOID:
            mOutputs = sigmoid(logits);
            break;
        case Activation::SOFTMAX:
            mOutputs = softmax(logits);
            break;
    }
}

void Layer::backpropagate(const std::vector<float>& upstreamGradient,
                          std::vector<std::vector<float>>& weightGradientOut,
                          std::vector<float>& biasGradientOut,
                          std::vector<float>& downstreamGradientOut) {
    std::vector<float> delta(mWeights.size());

    if (mActivationType == Activation::SOFTMAX) {
        delta = upstreamGradient;
    } else {
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
            case Activation::SOFTMAX:
                // Errors vector is already final gradient, handled above.
                break;
        }
        for (size_t n = 0; n < mWeights.size(); n++) {
            delta[n] = outputDerivatives[n] * upstreamGradient[n];
        }
    }
    for (size_t n = 0; n < mWeights.size(); n++) {
        std::vector<float> neuronGradient;
        for (size_t i = 0; i < mWeights[n].size(); i++) {
            neuronGradient.push_back(delta[n] * mLastInputs[i]);
        }
        weightGradientOut.push_back(neuronGradient);
    }
    biasGradientOut = delta;

    for (size_t i = 0; i < mWeights[0].size(); i++) {
        float sum = 0.0f;
        for (size_t n = 0; n < mWeights.size(); n++) {
            sum += mWeights[n][i] * delta[n];
        }
        downstreamGradientOut.push_back(sum);
    }
}

void Layer::update(float learningRate, 
                   const std::vector<std::vector<float>>& weightGradient, 
                   const std::vector<float>& biasGradient) {
    for (size_t n = 0; n < mWeights.size(); n++) {
        for (size_t i = 0; i < mWeights[n].size(); i++) {
            mWeights[n][i] = mWeights[n][i] - (learningRate * weightGradient[n][i]);
        }
        mBiases[n] = mBiases[n] - (learningRate * biasGradient[n]);
    }
}

const std::vector<float>& Layer::getOutputs() const {
    return mOutputs;
}

double Layer::getWeightNormSquared() const {
    double sum = 0.0;
    for (size_t n = 0; n < mWeights.size(); n++) {
        sum += getNeuronWeightNormSquared(mBiases[n], mWeights[n]);
    }
    return sum;
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

void NeuralNet::backpropagate(const std::vector<float>& errors,
                              std::vector<std::vector<std::vector<float>>>& weightGradientsOut,
                              std::vector<std::vector<float>>& biasGradientsOut) {
    std::vector<std::vector<float>> weightGradients;
    std::vector<float> biasGradient;
    std::vector<float> downstreamGradient; 
    mLayers[mLayers.size()-1].backpropagate(errors, weightGradients, biasGradient, downstreamGradient);
    weightGradientsOut.push_back(weightGradients);
    biasGradientsOut.push_back(biasGradient);
    for (int i = mLayers.size()-2; i >= 0; i--) {
        std::vector<float> upstreamGradient = downstreamGradient;
        weightGradients.clear();
        biasGradient.clear();
        downstreamGradient.clear();
        mLayers[i].backpropagate(upstreamGradient, weightGradients, biasGradient, downstreamGradient);
        weightGradientsOut.push_back(weightGradients);
        biasGradientsOut.push_back(biasGradient);
    }
}

void NeuralNet::update(float learningRate,
                       const std::vector<std::vector<std::vector<float>>>& weightGradients,
                       const std::vector<std::vector<float>>& biasGradients) {
    for (size_t i = 0; i < mLayers.size(); i++) {
        int index = mLayers.size() - 1 - i;
        mLayers[index].update(learningRate, weightGradients[i], biasGradients[i]);
    }
}

const std::vector<float>& NeuralNet::getOutputs() const {
    return mLayers.back().getOutputs();
}

int Layer::getNumInputs() const {
    if (mWeights.empty()) {
        return 0;
    }
    return mWeights[0].size();
}

int Layer::getNumNeurons() const {
    return mWeights.size();
}

std::vector<double> NeuralNet::getLayerWeightNormsSquared() const {
    std::vector<double> ret;
    for (size_t i = 0; i < mLayers.size(); i++) {
        ret.push_back(mLayers[i].getWeightNormSquared());
    }
    return ret;
}

std::vector<double> NeuralNet::getLayerGradientNormsSquared(
            const std::vector<std::vector<float>>& netBiasGraidents,
            const std::vector<std::vector<std::vector<float>>>& netWeightGradients) const {
    std::vector<double> ret;
    for (size_t l = 0; l < netWeightGradients.size(); l++) {
        double layerSum = 0.0;
        for (size_t n = 0; n < netWeightGradients[l].size(); n++) {
            double neuronSum = netBiasGraidents[l][n] * netBiasGraidents[l][n];
            for (float c : netWeightGradients[l][n]) {
                neuronSum += c * c;
            }
            layerSum += neuronSum;
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
