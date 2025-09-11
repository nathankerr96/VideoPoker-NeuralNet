#include "agent.h"

#include "neural.h"
#include "poker.h"
#include "trainer.h"

#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <atomic>
#include <string>
#include <barrier>
#include <thread>
#include <mutex>

constexpr int NUM_WORKERS = 8;
constexpr int NUM_IN_BATCH = 5;

Agent::Agent(const std::vector<LayerSpecification>& topology, 
             std::string fileName, 
             unsigned int seed, 
             float learningRate, 
             std::unique_ptr<BaselineCalculator> baselineCalc)
        : mNet(std::make_unique<NeuralNet>(topology)),
          mBaselineCalculator(std::move(baselineCalc)),
          mIterations(0),
          mTotalScore(0),
          mLogFile(fileName) {
    assert(topology[0].numNeurons == 85); // Hard dependency by hand translation layer.
    int outputSize = topology.back().numNeurons;
    switch (outputSize) {
        case 5:
            mDiscardStrategy = std::make_unique<FiveNeuronStrategy>();
            break;
        case 32:
            mDiscardStrategy = std::make_unique<ThirtyTwoNeuronStrategy>();
            break;
        default:
            std::cerr << "Output Layer Size: " << outputSize;
            throw std::invalid_argument("Unsupported Output Layer Size");
    }

    if (!mLogFile.is_open()) {
        std::cerr << "Could not open Log file!" << std::endl;
    } else {
        mLogFile << "Topology " << std::endl << topology << std::endl;
        mLogFile << "Learning Rate," << learningRate << std::endl << std::endl;
        mLogFile << "Baseline Calculator, " << mBaselineCalculator->getName() << std::endl;
        mLogFile << "Iterations,TotalAvgScore,RecentAvgScore,GlobalWeightNorm,GlobalGradientNorm,";
        for (size_t i = 1; i < topology.size(); i++) {
            mLogFile << "Layer" << i << "WeightNorm,";
        }
        for (size_t i = 1; i < topology.size(); i++) {
            mLogFile << "Layer" << i << "GradientNorm,";
        }
        mLogFile << std::endl;
    }

    // RNG engines for the worker threads (so they aren't dealt the same hands)
    std::seed_seq seq {seed};
    std::vector<uint32_t> seeds(NUM_WORKERS);
    seq.generate(seeds.begin(), seeds.end());
    for (int i = 0; i < NUM_WORKERS; i++) {
        mRngs.push_back(std::mt19937(seeds[i]));
    }
}

std::vector<float> Agent::translateHand(const Hand& hand) const {
    std::vector<float> ret(85, 0.0f);
    for (int i=0; i < 5; i++) {
        Card c = hand[i];
        ret[(i*17)+c.suit] = 1.0f;
        ret[(i*17)+4+(c.rank-2)] = 1.0f;
    }
    return ret;
}

// Stored in place in the first vector of the trainers
void calculateAggregatedGradients(std::vector<Trainer>& trainers) {
        std::vector<std::vector<float>>& aggregatedWeightGradients = trainers[0].getTotalWeightGradients();
        std::vector<std::vector<float>>& aggregatedBiasGradients = trainers[0].getTotalBiasGradients();
        for (size_t i = 1; i < trainers.size(); i++) {
            const std::vector<std::vector<float>>& weightGradients = trainers[i].getTotalWeightGradients();
            const std::vector<std::vector<float>>& biasGradients = trainers[i].getTotalBiasGradients();
            for (size_t l = 0; l < weightGradients.size(); l++) {
                for (size_t w = 0; w < weightGradients[l].size(); w++) {
                    aggregatedWeightGradients[l][w] += weightGradients[l][w];
                }
                for (size_t w = 0; w < biasGradients[l].size(); w++) {
                    aggregatedBiasGradients[l][w] += biasGradients[l][w];
                }
            }
        }
        for (size_t l = 0; l < aggregatedWeightGradients.size(); l++) {
            int batchSize = NUM_WORKERS * NUM_IN_BATCH;
            for (size_t w = 0; w < aggregatedWeightGradients[l].size(); w++) {
                aggregatedWeightGradients[l][w] /= batchSize;
            }
            for (size_t w = 0; w < aggregatedBiasGradients[l].size(); w++) {
                aggregatedBiasGradients[l][w] /= batchSize;
            }
        }
}


void Agent::train(const std::atomic<bool>& stopSignal, float learningRate) {

    std::mutex vectorMutex;
    std::vector<Trainer> trainers(NUM_WORKERS, Trainer(mNet.get()));

    auto completionStep = [&]() {
        calculateAggregatedGradients(trainers);
        mNet->update(learningRate, trainers[0].getTotalWeightGradients(), trainers[0].getTotalBiasGradients());
    };


    std::barrier barrier(NUM_WORKERS, completionStep);


    auto trainingLoop = [&](int workerId) {
        VideoPoker vp;
        Trainer& t = trainers[workerId];

        while (true) { // Break when stopSignal is set.
            t.reset(); // Clear accumulated gradients

            for (int i = 0; i < NUM_IN_BATCH; i++) {
                Hand h = vp.deal();
                std::vector<float> input = translateHand(h);
                float baseline = mBaselineCalculator->predict(input);
                t.feedForward(input);
                const std::vector<float>& output = t.getOutputs();
                std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRngs[workerId], true);
                Hand e = vp.exchange(exchanges);

                int score = vp.score(vp.getHandType(e));
                mBaselineCalculator->train(score);
                mTotalScore += score;
                mRecentTotal += score;
                if (mIterations.fetch_add(1) % 10000 == 0) {
                    int recentTotalLocal = mRecentTotal.exchange(0); // Minor race condition, but only used for monitoring.
                    float averageTotalScore = float(mTotalScore) / mIterations;
                    std::cout << "Thread: " << std::this_thread::get_id() << "--- ";
                    std::cout << "Games Played: " << mIterations << ", Average Score: " << averageTotalScore << std::endl;
                    float averageRecentScore = float(recentTotalLocal) / 10000;
                    std::cout << "Average of last 10000: " << averageRecentScore << std::endl;
                    std::cout << "Sample Hand: " << h << std::endl;
                    std::cout << "Baseline: " << baseline << std::endl;
                    std::cout << "Outputs: " << t.getOutputs() << std::endl;
                    std::cout << "Prediction: " << exchanges << std::endl;
                    std::cout << "Ending Hand: " << e << std::endl;
                    std::cout << "Score: " << score << std::endl;
                    mLogFile << mIterations << ",";
                    mLogFile << averageTotalScore << ",";
                    mLogFile << averageRecentScore << ",";
                    logAndPrintNorms(t);
                    mLogFile << std::endl;
                    std::cout << std::endl;
                    mRecentTotal = 0;
                }

                float advantage = (score - baseline);
                std::vector<float> errors = mDiscardStrategy->calculateError(output, exchanges, advantage);
                t.backpropagate(errors);
            }

            barrier.arrive_and_wait(); // Runs completionStep once all threads arrive.
            if (stopSignal) {
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_WORKERS; i++) {
        threads.emplace_back(std::thread(trainingLoop, i));
    }
    for (std::thread& t: threads) {
        // Main thread will set stopSignal to end training.
        t.join();
    }
}

int Agent::getNumTrainingIterations() const {
    return mIterations;
}

void Agent::randomEval(int iterations, std::mt19937& rng) const {
    VideoPoker vp;
    std::cout << "---Starting Eval, " <<  iterations << " iterations.---" << std::endl;
    int total_score = 0;
    Trainer trainer(mNet.get()); // TODO: This shouldn't have to go through trainer.
    for (int i = 0; i < iterations; i++) {
        Hand h = vp.deal();
        std::vector<float> input = translateHand(h);
        trainer.feedForward(input);
        const std::vector<float>& output = trainer.getOutputs();
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, rng, false);
        h = vp.exchange(exchanges);
        if ((i+1) % 10000 == 0) {
            std::cout << "Games Played: " << (i+1) << ", Total Score: " << total_score << std::endl;
        }
        int score = vp.score(vp.getHandType(h));
        total_score += score;
    }
    std::cout << "---Average Score: " << float(total_score) / iterations << "---" << std::endl << std::endl;
}


void Agent::targetedEval(std::mt19937& rng) const {
    std::vector<std::pair<std::string, Hand>> hands {
        {"Junk", {{{{CLUB, 2}, {SPADE, 7}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"Pair", {{{{CLUB, 2}, {SPADE, 2}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 3}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 12}}}}},
        {"Two Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Trips", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Quads", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 12}}}}}
    };
    Trainer trainer(mNet.get());
    for (const auto& h : hands) {
        trainer.feedForward(translateHand(h.second));
        std::vector<float> output = trainer.getOutputs();
        std::cout << h.first << ": " << h.second << std::endl;
        std::cout << "Outputs: " << trainer.getOutputs() << std::endl;
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, rng, false);
        std::cout << "Decision: " << exchanges << std::endl;
    }
}

void Agent::logAndPrintNorms(const Trainer& trainer) {
    std::vector<double> weightNormsSquared = mNet->getLayerWeightNormsSquared();
    std::cout << "Weight Norms:" << std::endl;
    double totalWeightNormSquared = 0.0;
    for (size_t i = 0; i < weightNormsSquared.size(); i++) {
        std::cout << "Layer " << i << ": " << std::sqrt(weightNormsSquared[i]) << std::endl;
        totalWeightNormSquared += weightNormsSquared[i];
    }
    double globalWeightNorm = std::sqrt(totalWeightNormSquared);
    std::cout << "Overall Weight Norm: " <<  globalWeightNorm << std::endl;

    // TODO: Move Norm Calculation to Batch Completion Step
    // std::vector<double> gradientNormsSquared = trainer.getLayerGradientNormsSquared();
    // std::cout << "Gradient Norms:" << std::endl;
    // double totalGradientNormSquared = 0.0;
    // for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
    //     std::cout << "Layer " << i << ": " << std::sqrt(gradientNormsSquared[i]) << std::endl;
    //     totalGradientNormSquared += gradientNormsSquared[i];
    // }
    // double globalGradientNorm = std::sqrt(totalGradientNormSquared);
    // std::cout << "Overall Gradient Norm: " << globalGradientNorm << std::endl;

    mLogFile << globalWeightNorm << ",";
    // mLogFile << globalGradientNorm << ",";
    for (size_t i = 0; i < weightNormsSquared.size(); i++) {
        mLogFile << std::sqrt(weightNormsSquared[i]) << ",";
    }
    // for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
    //     mLogFile << std::sqrt(gradientNormsSquared[i]) << ",";
    // }
}