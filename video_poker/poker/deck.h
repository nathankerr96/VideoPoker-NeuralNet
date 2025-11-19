#pragma once

#include "card.h"
#include "hand.h"

#include <random>

class Deck {
public:
    Deck(std::mt19937& rng);
    void shuffle();
    void stack(PokerHand hand, int rank, int rank2);
    Card draw();
    bool operator==(const Deck& other) const;
    bool operator!=(const Deck& other) const;

private:
    void swapNext(int rank);
    std::mt19937& mRandomGenerator;
    std::vector<Card> mDeck;
    int mIndex = 0;
};
