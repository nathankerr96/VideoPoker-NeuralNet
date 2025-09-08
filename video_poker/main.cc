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
#include <chrono>
#include <ctime>

#define INPUT_SIZE 85
#define LEARNING_RATE 0.002
#define EVAL_ITERATIONS 100000
#define LOGS_DIR "logs/"
#define LOG_NAME  "TimingTest"
#define CRITIC_NETWORK_LEARNING_RATE 0.005

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

std::vector<LayerSpecification> CRITIC_NETWORK_TOPOLOGY {
    {INPUT_SIZE, Activation::LINEAR},
    {85, Activation::RELU},
    {1, Activation::LINEAR},
};

std::string getLogName() {
    const auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string timeString = std::to_string(now_c);
    return std::string(LOGS_DIR) + std::string(LOG_NAME) + "-" + timeString + ".csv";
}

std::unique_ptr<BaselineCalculator> getFlatBaseline() {
    return std::make_unique<FlatBaseline>();
}

std::unique_ptr<BaselineCalculator> getRunningAverageBaseline() {
    return std::make_unique<RunningAverageBaseline>();
}

std::unique_ptr<BaselineCalculator> getCriticNetworkBaseline() {
    return std::make_unique<CriticNetworkBaseline>(CRITIC_NETWORK_TOPOLOGY, CRITIC_NETWORK_LEARNING_RATE);
}

int main() {
    std::random_device rd {};

    Agent agent {
        SOFTMAX_TOPOLOGY, 
        getLogName(), 
        rd(), 
        LEARNING_RATE,
        // getRunningAverageBaseline()
        getCriticNetworkBaseline(),
    };

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
