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
               mActivationType(activationType) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    mWeights.reserve(num_neurons * num_inputs);
    for (int i = 0; i < num_neurons * num_inputs; i++) {
        mWeights.push_back(dis(generator) / sqrt(num_inputs));
    }
}

void Layer::fire(const std::vector<float>& inputs,
                 std::vector<float>& logitsBuffer,
                 std::vector<float>& activationsOut) const {
    if (int(inputs.size()) != mNumInputs) {
        std::cerr << "Inputs: " << inputs.size() << ", Neurons: " << mNumInputs << std::endl;
        throw std::invalid_argument("Inputs != Weights");
    }
    for (int n = 0; n < mNumNeurons; n++) {
        float sum = mBiases[n];
        for (size_t i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * mWeights[n*mNumInputs+i];
        }
        logitsBuffer[n] = sum;
    }
    switch (mActivationType) {
        case Activation::LINEAR:
            activationsOut = logitsBuffer;
            break;
        case Activation::RELU:
            relu(logitsBuffer, mNumNeurons, activationsOut);
            break;
        case Activation::SIGMOID:
            sigmoid(logitsBuffer, mNumNeurons, activationsOut);
            break;
        case Activation::SOFTMAX:
            softmax(logitsBuffer, mNumNeurons, activationsOut);
            break;
    }
}

void Layer::backpropagate(const std::vector<float>& upstreamGradient,
                          const std::vector<float>& layerInputs,
                          const std::vector<float>& layerActivations,
                          std::vector<float>& deltaBuffer,
                          std::vector<float>& outputDerivativesBuffer,
                          std::vector<float>& weightGradientOut,
                          std::vector<float>& biasGradientOut,
                          std::vector<float>& downstreamGradientOut) const {
    if (mActivationType == Activation::SOFTMAX) {
        deltaBuffer = upstreamGradient;
    } else {
        switch (mActivationType) {
            case Activation::LINEAR:
                std::fill(outputDerivativesBuffer.begin(), outputDerivativesBuffer.end(), 1.0f);
                break;
            case Activation::RELU:
                // TODO: May need to pass # of inputs/neurons here?
                relu_derivative(layerActivations, outputDerivativesBuffer);
                break;
            case Activation::SIGMOID:
                sigmoid_derivative(layerActivations, outputDerivativesBuffer);
                break;
            case Activation::SOFTMAX:
                // Errors vector is already final gradient, handled above.
                break;
        }
        for (int n = 0; n < mNumNeurons; n++) {
            deltaBuffer[n] = outputDerivativesBuffer[n] * upstreamGradient[n];
        }
    }
    for (int n = 0; n < mNumNeurons; n++) {
        for (int i = 0; i < mNumInputs; i++) {
            weightGradientOut[n*mNumInputs+i] += deltaBuffer[n] * layerInputs[i];
        }
        biasGradientOut[n] += deltaBuffer[n];
    }

    for (int i = 0; i < mNumInputs; i++) {
        float sum = 0.0f;
        for (int n = 0; n < mNumNeurons; n++) {
            sum += mWeights[n*mNumInputs+i] * deltaBuffer[n];
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
}

// void NeuralNet::feedForward(const std::vector<float>& inputs) const {
//     mLayers[0].fire(inputs);
//     for (size_t i = 1; i < mLayers.size(); i++) {
//         mLayers[i].fire(mLayers[i-1].getOutputs());
//     }
// }

const std::vector<Layer>& NeuralNet::getLayers() {
    return mLayers;
}

void NeuralNet::update(float learningRate, 
        const std::vector<std::vector<float>>& weightGradients,
        const std::vector<std::vector<float>>& biasGradients) {
    for (int i = mLayers.size()-1; i >= 0; i--) {
        mLayers[i].update(learningRate, weightGradients[i], biasGradients[i]);
    }
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
        os << "Layer " << i << "," << v[i] << ",";
    }
    return os;
}
