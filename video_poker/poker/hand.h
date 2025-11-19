#pragma once

#include "card.h"

#include <array>

enum PokerHand {
    HIGH_CARD,
    PAIR,
    HIGH_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
    ROYAL_FLUSH
};

class Hand {
public:
    Hand() {}
    Hand(std::array<Card, 5> h) : mHand(h) {};
    Card& operator[](int index);
    const Card& operator[](int index) const;
private:
    std::array<Card, 5> mHand;
};

std::ostream& operator<<(std::ostream& os, const Hand& hand);
