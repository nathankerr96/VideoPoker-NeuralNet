#include "agent.h"

#include "neural.h"
#include "poker.h"
#include "trainer.h"
#include "hyperparams.h"

#include <random>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <atomic>
#include <string>
#include <barrier>
#include <thread>
#include <chrono>

#define LOG_STEP 2000

Agent::Agent(const HyperParameters& config,
             std::string fileName, 
             unsigned int seed, 
             std::function<std::unique_ptr<BaselineCalculator>()> baselineFactory)
        : mConfig(config),
          mNet(std::make_unique<NeuralNet>(config.actorTopology)),
          mBaselineFactory(baselineFactory),
          mLogFile(fileName),
          mRng(seed),
          mVideoPoker(mRng)
{
    assert(config.actorTopology[0].numNeurons == 85); // Hard dependency by hand translation layer.
    int outputSize = config.actorTopology.back().numNeurons;
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

    switch (config.optimizerType) {
        case SDG:
            mOptimizer = std::make_unique<SDGOptimizer>();
            break;
        case MOMENTUM:
            mOptimizer = std::make_unique<MomentumOptimizer>(mNet.get(), config.momentumCoeff);
            break;
    }

    if (!mLogFile.is_open()) {
        std::cerr << "Could not open Log file!" << std::endl;
    } else {
        mLogFile << config << std::endl;
        mLogFile << std::endl;
        // mLogFile << "Baseline Calculator, " << mBaselineCalculator->getName() << std::endl;
        mLogFile << "Batches,Hands,TotalAvgScore,RecentAvgScore,RecentAvgEntropy,GlobalWeightNorm,GlobalGradientNorm,";
        for (size_t i = 1; i < config.actorTopology.size(); i++) {
            mLogFile << "Layer" << i << "WeightNorm,";
        }
        for (size_t i = 1; i < config.actorTopology.size(); i++) {
            mLogFile << "Layer" << i << "GradientNorm,";
        }
        mLogFile << std::endl;
    }

    // RNG engines for the worker threads (so they aren't dealt the same hands)
    std::seed_seq seq {seed};
    std::vector<uint32_t> seeds(mConfig.numWorkers);
    seq.generate(seeds.begin(), seeds.end());
    for (int i = 0; i < mConfig.numWorkers; i++) {
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

float Agent::calculateEntropy(const std::vector<float>& policy) {
    float entropy = 0.0f;
    for (float p : policy) {
        if (p > 0) {
            entropy -= p * std::log(p);
        }
    }    
    return entropy;
}

// TODO: These params should be made const, either by directly referencing the underlying NeuralNet or
// adding const equivalent functions (default feedforward saves activations for backprop).
void Agent::logProgress(Trainer& t, BaselineCalculator* baselineCalc) {
    float averageTotalScore = float(mTotalScore) / mIterations;
    std::cout << "Thread: " << std::this_thread::get_id() << "--- ";
    std::cout << "Batches: " << mNumBatches << ", Hands: " << mIterations << ", Average Score: " << averageTotalScore << std::endl;
    float averageRecentScore = float(mRecentTotal) / (LOG_STEP * mConfig.getBatchSize());
    float averageRecentEntropy = mRecentEntropy.exchange(0.0f) / (LOG_STEP * mConfig.getBatchSize());
    std::cout << "Average over last " << LOG_STEP << " batches: " << averageRecentScore << ", Entropy: " << averageRecentEntropy << std::endl;
    mRecentTotal = 0; // Reset for next N batches.

    // Run and log an example hand without making any updates
    Hand h = mVideoPoker.deal();
    std::cout << "Sample Hand: " << h << std::endl;
    std::vector<float> input = translateHand(h);
    float baseline = baselineCalc->predict(input);
    std::cout << "Baseline: " << baseline << std::endl;
    t.feedForward(input);
    const std::vector<float>& output = t.getOutputs();
    std::cout << "Outputs: " << t.getOutputs() << std::endl;
    std::cout << "Entropy: " << calculateEntropy(output) << std::endl;
    std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRng, true);
    std::cout << "Prediction: " << exchanges << std::endl;
    Hand e = mVideoPoker.exchange(exchanges);
    std::cout << "Ending Hand: " << e << std::endl;
    int score = mVideoPoker.score(mVideoPoker.getHandType(e));
    std::cout << "Score: " << score << std::endl;

    // Log progress to file for later analysis
    mLogFile << mNumBatches << ",";
    mLogFile << mIterations << ",";
    mLogFile << averageTotalScore << ",";
    mLogFile << averageRecentScore << ",";
    mLogFile << averageRecentEntropy << ",";
    logAndPrintNorms(t);
    mLogFile << std::endl;
    std::cout << std::endl;
}


void Agent::train(const std::atomic<bool>& stopSignal) {
    auto trainingStartTime = std::chrono::steady_clock::now();

    std::vector<Trainer> trainers(mConfig.numWorkers, Trainer(mNet.get()));
    std::vector<std::unique_ptr<BaselineCalculator>> baselineCalcs;
    baselineCalcs.reserve(mConfig.numWorkers);
    std::generate_n(std::back_inserter(baselineCalcs), mConfig.numWorkers, mBaselineFactory);

    auto completionStep = [&]() {
        mNumBatches += 1;
        for (size_t i = 1; i < trainers.size(); i++) {
            trainers[0].aggregate(trainers[i]);
        }
        trainers[0].batch(mConfig.getBatchSize());
        // mNet->update(mConfig.actorLearningRate, trainers[0].getTotalWeightGradients(), trainers[0].getTotalBiasGradients());
        mOptimizer->step(mNet.get(), trainers[0], mConfig.actorLearningRate);
        baselineCalcs[0]->update(baselineCalcs, mConfig.getBatchSize());
        if (mNumBatches % LOG_STEP == 0) {
            logProgress(trainers[0], baselineCalcs[0].get());
        }
    };


    std::barrier barrier(mConfig.numWorkers, completionStep);


    auto trainingLoop = [&](int workerId) {
        VideoPoker vp {mRngs[workerId]};
        Trainer& t = trainers[workerId];

        while (true) { // Break when stopSignal is set.
            t.reset(); // Clear accumulated gradients

            for (int i = 0; i < mConfig.numInBatch; i++) {
                Hand h = vp.deal();
                std::vector<float> input = translateHand(h);
                float baseline = baselineCalcs[workerId]->predict(input);
                t.feedForward(input);
                const std::vector<float>& output = t.getOutputs();
                std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, mRngs[workerId], true);
                Hand e = vp.exchange(exchanges);

                int score = vp.score(vp.getHandType(e));
                baselineCalcs[workerId]->train(score);
                mTotalScore += score;
                mRecentTotal += score;
                mIterations += 1;

                float advantage = (score - baseline);
                std::vector<float> policyError = mDiscardStrategy->calculateError(output, exchanges, advantage);
                float entropy = calculateEntropy(output);
                mRecentEntropy += entropy;
                if (mConfig.entropyCoeff != 0.0f) {
                    std::vector<float> entropyError = mDiscardStrategy->calculateEntropyError(output, entropy, mConfig.entropyCoeff);
                    for (size_t i = 0; i < policyError.size(); i++) {
                        policyError[i] += entropyError[i];
                    }
                }
                t.backpropagate(policyError);
            }

            barrier.arrive_and_wait(); // Runs completionStep once all threads arrive.
            if (stopSignal) {
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < mConfig.numWorkers; i++) {
        threads.emplace_back(std::thread(trainingLoop, i));
    }
    for (std::thread& t: threads) {
        // Main thread will set stopSignal to end training.
        t.join();
    }

    std::chrono::duration<double> trainingSeconds = std::chrono::steady_clock::now() - trainingStartTime;
    mTotalTrainingTime += trainingSeconds;
    std::cout << "Training time (this/total): " << trainingSeconds << " / " << mTotalTrainingTime << std::endl;
}

int Agent::getNumTrainingIterations() const {
    return mIterations;
}

void Agent::randomEval(int iterations, std::mt19937& rng) const {
    VideoPoker vp {rng};
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

    std::vector<double> gradientNormsSquared = trainer.getLayerGradientNormsSquared();
    std::cout << "Gradient Norms:" << std::endl;
    double totalGradientNormSquared = 0.0;
    for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
        std::cout << "Layer " << i << ": " << std::sqrt(gradientNormsSquared[i]) << std::endl;
        totalGradientNormSquared += gradientNormsSquared[i];
    }
    double globalGradientNorm = std::sqrt(totalGradientNormSquared);
    std::cout << "Overall Gradient Norm: " << globalGradientNorm << std::endl;

    mLogFile << globalWeightNorm << ",";
    mLogFile << globalGradientNorm << ",";
    for (size_t i = 0; i < weightNormsSquared.size(); i++) {
        mLogFile << std::sqrt(weightNormsSquared[i]) << ",";
    }
    for (size_t i = 0; i < gradientNormsSquared.size(); i++) {
        mLogFile << std::sqrt(gradientNormsSquared[i]) << ",";
    }
}