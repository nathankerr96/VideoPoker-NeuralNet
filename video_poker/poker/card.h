#pragma once

#include <ostream>

enum Suit {
    CLUB,
    DIAMOND,
    HEART,
    SPADE
};

struct Card {
    Suit suit;
    int rank;
};

bool operator==(const Card& lhs, const Card& rhs);
bool operator!=(const Card& lhs, const Card& rhs);
std::ostream& operator<<(std::ostream& os, const Card& card);
