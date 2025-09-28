#include "neural.h"
#include "activations.h"
#include "poker.h"
#include "agent.h"
#include "hyperparams.h"

#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <atomic>
#include <thread>
#include <chrono>
#include <ctime>

#define EVAL_ITERATIONS 100000
#define LOGS_DIR "logs/"

std::string getLogName(std::string actorName) {
    const auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string timeString = std::to_string(now_c);
    return std::string(LOGS_DIR) + actorName + "-" + timeString + ".csv";
}

std::unique_ptr<BaselineCalculator> getFlatBaseline() {
    return std::make_unique<FlatBaseline>();
}

std::unique_ptr<BaselineCalculator> getRunningAverageBaseline() {
    return std::make_unique<RunningAverageBaseline>();
}

std::unique_ptr<BaselineCalculator> getCriticNetworkBaseline(NeuralNet* net, const HyperParameters& config) {
    std::unique_ptr<Optimizer> optimizer;
    switch (config.criticOptimizerType) {
        case SDG:
            optimizer = std::make_unique<SDGOptimizer>();
            break;
        case MOMENTUM:
            optimizer = std::make_unique<MomentumOptimizer>(net, config.criticMomentumCoeff);
            break;
    }
    return std::make_unique<CriticNetworkBaseline>(net, config.criticLearningRate, std::move(optimizer));
}

int main() {
    std::random_device rd {};
    std::mt19937 rng {rd()};

    std::cout << "Select Config:" << std::endl;
    for (size_t i = 0; i < AvailableConfigs.size(); i++) {
        std::cout << "\t" << i << ": " << AvailableConfigs[i].name << std::endl;
    }
    int selection = -1;
    std::cin >> selection;
    std::cin.ignore();
    if (selection < 0 || selection > std::ssize(AvailableConfigs)-1) {
        std::cout << "Invalid selection: " << selection << std::endl;
        exit(1);
    }
    HyperParameters config = AvailableConfigs[selection];
    std::cout << "Loading " << config.name << std::endl;

    // TODO: Create all Neural Nets in the same place (i.e. main or agent).
    std::unique_ptr<NeuralNet> criticNetwork = std::make_unique<NeuralNet>(CRITIC_NETWORK_TOPOLOGY);
    std::function<std::unique_ptr<BaselineCalculator>()> baselineFactory;
    switch(config.baselineCalculatorType) {
        case FLAT:
            baselineFactory = getFlatBaseline;
            break;
        case RUNNING_AVERAGE:
            baselineFactory = getRunningAverageBaseline;
            break;
        case CRITIC_NETWORK:
            baselineFactory = std::bind(getCriticNetworkBaseline, criticNetwork.get(), config);
            break;
    }

    Agent agent {
        config,
        getLogName(config.name), 
        rd(), 
        baselineFactory,
    };

    std::string input;
    std::cout << "Enter command: ";
    while (std::getline(std::cin, input)) {
        if (input == "train") {
            std::atomic<bool> stopSignal(false);
            std::thread t = std::thread([&agent, &stopSignal](){agent.train(stopSignal);});
            std::string unused;
            std::getline(std::cin, unused);
            stopSignal = true;
            t.join();
            std::cout << "Agent Iterations: " << agent.getNumTrainingIterations() << std::endl;
        } else if (input == "eval") {
            agent.randomEval(EVAL_ITERATIONS, rng);
            agent.targetedEval(rng);
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
