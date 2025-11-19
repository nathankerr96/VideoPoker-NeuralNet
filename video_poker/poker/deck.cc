#include "deck.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

Deck::Deck(std::mt19937& rng) : mRandomGenerator(rng) {
    mDeck.reserve(52);
    for (int s = 0; s < 4; s++) {
        for (int r = 2 ; r <= 14; r++) {
            mDeck.emplace_back(Card {static_cast<Suit>(s), r});
        }
    }
}

void Deck::shuffle() {
    std::shuffle(mDeck.begin(), mDeck.end(), mRandomGenerator);
    mIndex = 0;
}

void Deck::swapNext(int rank) {
    int next = mIndex;
    while (mDeck[next].rank != rank) {
        next++;
    }
    std::swap(mDeck[mIndex], mDeck[next]);
    mIndex++;
}

void Deck::stack(PokerHand h, int rank, int rank2) {
    if (h == PAIR && rank > 10) {
        throw std::runtime_error("Cannot use high rank with PAIR hand.");
    } else if (h == HIGH_PAIR && rank < 11) {
        throw std::runtime_error("Cannot use low rank with HIGH_PAIR hand.");
    }

    shuffle();
    switch(h) {
        case PokerHand::ROYAL_FLUSH:
        case PokerHand::STRAIGHT_FLUSH:
        case PokerHand::STRAIGHT:
        case PokerHand::FLUSH:
            // TODO: Implement
            return;
        // MULTI PAIR HANDS
        case PokerHand::FULL_HOUSE:
            swapNext(rank);
        case PokerHand::TWO_PAIR:
            swapNext(rank);
            swapNext(rank);
            swapNext(rank2);
            swapNext(rank2);
            break;
        // SINGLE PAIR HANDS
        case PokerHand::FOUR_OF_A_KIND:
            swapNext(rank);
        case PokerHand::THREE_OF_A_KIND:
            swapNext(rank);
        case PokerHand::HIGH_PAIR:
        case PokerHand::PAIR:
            swapNext(rank);
            swapNext(rank);
            break;
        case PokerHand::HIGH_CARD:
            // TODO: Implement
            break;
    }
    //shuffle
    std::shuffle(mDeck.begin(), mDeck.begin()+5, mRandomGenerator);
    std::shuffle(mDeck.begin()+5, mDeck.end(), mRandomGenerator);
    mIndex = 0;
}

Card Deck::draw() {
    // TODO: implement draw
    return mDeck.at(mIndex++);
}

bool Deck::operator==(const Deck& other) const {
    return mDeck == other.mDeck && mIndex == other.mIndex;
}

bool Deck::operator!=(const Deck& other) const {
    return !(*this == other);
}
