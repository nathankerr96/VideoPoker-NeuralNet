#pragma once

#include "hand.h"
#include "deck.h"

#include <vector>
#include <array>
#include <ostream>
#include <random>

class VideoPoker {
public:
    VideoPoker(std::mt19937& rng) : mDeck(rng) {}
    const Hand& deal();
    const Hand& exchange(const std::vector<bool>& ex);
    PokerHand getHandType(const Hand& hand);
    int score(PokerHand handType);

private:
    Deck mDeck;
    Hand mHand;
    bool mInProgress = false;
};
