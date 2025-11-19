#pragma once

#include "card.h"

#include <random>

class Deck {
public:
    Deck(std::mt19937& rng);
    void shuffle();
    Card draw();
    bool operator==(const Deck& other) const;
    bool operator!=(const Deck& other) const;

private:
    std::mt19937& mRandomGenerator;
    std::vector<Card> mDeck;
    int mIndex = 0;
};
