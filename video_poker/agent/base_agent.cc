#include "base_agent.h"

#include "neural.h"
#include "poker.h"
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

void BaseAgent::randomEval(int iterations, std::mt19937& rng) const {
    VideoPoker vp {rng};
    std::cout << "---Starting Eval, " <<  iterations << " iterations.---" << std::endl;
    int total_score = 0;
    for (int i = 0; i < iterations; i++) {
        Hand h = vp.deal();
        std::vector<float> input = translateHand(h);
        const std::vector<float>& output = predict(input);
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


void BaseAgent::targetedEval(std::mt19937& rng) const {
    std::vector<std::pair<std::string, Hand>> hands {
        {"Junk", {{{{CLUB, 2}, {SPADE, 7}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"Pair", {{{{CLUB, 2}, {SPADE, 2}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 8}}}}},
        {"High Pair", {{{{CLUB, 3}, {SPADE, 12}, {HEART, 10}, {CLUB, 4}, {DIAMOND, 12}}}}},
        {"Two Pair", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 10}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Trips", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 8}}}}},
        {"Quads", {{{{CLUB, 12}, {SPADE, 12}, {HEART, 12}, {CLUB, 10}, {DIAMOND, 12}}}}}
    };
    for (const auto& h : hands) {
        std::vector<float> output = predict(translateHand(h.second));
        std::cout << h.first << ": " << h.second << std::endl;
        std::cout << "Outputs: " << output << std::endl;
        std::vector<bool> exchanges = mDiscardStrategy->selectAction(output, rng, false);
        std::cout << "Decision: " << exchanges << std::endl;
    }
}
