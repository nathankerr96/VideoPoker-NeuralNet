#ifndef POKER_H
#define POKER_H

#include <vector>
#include <array>
#include <ostream>
#include <random>

enum Suit {
    CLUB,
    DIAMOND,
    HEART,
    SPADE
};

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

struct Card {
    Suit suit;
    int rank;
};

std::ostream& operator<<(std::ostream& os, const Card& card);
bool operator==(const Card& lhs, const Card& rhs);
bool operator!=(const Card& lhs, const Card& rhs);

class Deck {
public:
    Deck();
    void shuffle();
    Card draw();
    bool operator==(const Deck& other) const;
    bool operator!=(const Deck& other) const;

private:
    std::vector<Card> mDeck;
    int mIndex = 0;
    std::mt19937 mRandomGenerator {2242};

};

class Hand {
public:
    Hand();
    Card& operator[](int index);
    const Card& operator[](int index) const;
private:
    std::array<Card, 5> mHand;
};

std::ostream& operator<<(std::ostream& os, const Hand& hand);

class VideoPoker {
public:
    const Hand& deal();
    const Hand& exchange(bool e1, bool e2, bool e3, bool e4, bool e5);
    PokerHand getHandType(const Hand& hand);

private:
    Deck mDeck;
    Hand mHand;
    bool mInProgress = false;
    // payout table
    // in progress
    // 
};

#endif // POKER_H
