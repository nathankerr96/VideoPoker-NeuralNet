#include "neural.h"
#include "activations.h"
#include "poker.h"
#include "agent.h"

#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <atomic>
#include <thread>

#define INPUT_SIZE 85
#define LEARNING_RATE 0.02
#define TRAINING_ITERATIONS 100000
#define EVAL_ITERATIONS 10000

std::vector<LayerSpecification> SOFTMAX_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {32, Activation::SOFTMAX},
};

std::vector<LayerSpecification> SIGMOID_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {170, Activation::RELU},
    {170, Activation::RELU},
    {5, Activation::SIGMOID},
};

int main() {
    std::random_device rd {};
    Agent agent {SOFTMAX_TOPOLOGY, rd()};

    std::string input;
    std::cout << "Enter command: ";
    while (std::getline(std::cin, input)) {
        if (input == "train") {
            std::atomic<bool> stopSignal(false);
            std::thread t = std::thread([&agent, &stopSignal](){agent.train(stopSignal, LEARNING_RATE);});
            std::string unused;
            std::getline(std::cin, unused);
            stopSignal = true;
            t.join();
            std::cout << "Agent Iterations: " << agent.getNumTrainingIterations() << std::endl;
        } else if (input == "eval") {
            agent.randomEval(EVAL_ITERATIONS);
            agent.targetedEval();
        } else if (input == "exit") {
            break;
        } else {
            std::cout << "Unrecognized command: " << input << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Enter command: ";
    }

    return 0;
}
