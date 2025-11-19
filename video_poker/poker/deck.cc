#include "deck.h"

#include <algorithm>

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
