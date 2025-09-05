#include "poker.h"

#include <algorithm>
#include <array>
#include <stdexcept>

//Card::Card(Suit suit, int rank) : suit(suit), rank(rank) {}

bool operator==(const Card& lhs, const Card& rhs) {
    return lhs.suit == rhs.suit && lhs.rank == rhs.rank;
}

bool operator!=(const Card& lhs, const Card& rhs) {
    return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const Card& card) {
    if (card.rank >= 2 && card.rank <= 9) {
        os << card.rank;
    } else {
        switch (card.rank) {
            case 10:
                os << "T";
                break;
            case 11:
                os << "J";
                break;
            case 12:
                os << "Q";
                break;
            case 13:
                os << "K";
                break;
            case 14:
                os << "A";
                break;
        }
    }
    switch (card.suit) {
        case CLUB:
            os << "♣"; // \u2663
            break;
        case DIAMOND:
            os << "♦"; // \u2666
            break;
        case HEART:
            os << "♥"; // \u2665
            break;
        case SPADE:
            os << "♠"; // \u2660
            break;
    }
    return os;
}

Deck::Deck() {
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

Card& Hand::operator[](int index) {
    return mHand[index];
}

const Card& Hand::operator[](int index) const {
    return mHand[index];
}

std::ostream& operator<<(std::ostream& os, const Hand& hand) {
    for (int i = 0; i < 5; i++) {
        os << hand[i] << " ";
    }
    return os;
}


const Hand& VideoPoker::deal() {
    if (mInProgress) throw std::runtime_error("Deal called while hand already in progress");
    mInProgress = true;
    mDeck.shuffle();
    for (int i = 0; i < 5; i++) {
        mHand[i] = mDeck.draw();
    }
    return mHand;
}

const Hand& VideoPoker::exchange(const std::vector<bool>& ex) {
    if (!mInProgress) throw std::runtime_error("Exchange called while and not in progress.");
    mInProgress = false;
    if (ex[0]) mHand[0] = mDeck.draw();
    if (ex[1]) mHand[1] = mDeck.draw();
    if (ex[2]) mHand[2] = mDeck.draw();
    if (ex[3]) mHand[3] = mDeck.draw();
    if (ex[4]) mHand[4] = mDeck.draw();
    return mHand;
}

PokerHand VideoPoker::getHandType(const Hand& hand) {

    bool hasFlush = true;
    Suit firstSuit = hand[0].suit;
    std::array<int, 13> counts{};
    for (int i=0; i < 5; i++) {
        if (hand[i].suit != firstSuit) hasFlush = false;
        counts[hand[i].rank-2]++;
    }

    bool hasPair = false;
    bool hasTwoPair = false;
    bool hasThree = false;
    bool hasStraight = false;
    bool hasFour = false;
    bool hasHighPair = false;
    int cardsToStraight = 0;
    int cardsSeen = 0;
    for (int i=0; i<13; i++) {
        int n = counts[i];
        if (n == 1) {
            cardsToStraight += 1;
        } else {
            cardsToStraight = 0;
        }
        if (cardsToStraight == 5) hasStraight = true;
        if (n == 4) hasFour = true;
        if (n == 3) hasThree = true;
        if (n == 2) {
            if (hasPair) hasTwoPair = true;
            hasPair = true;
            if (i >= 9) hasHighPair = true;
        }
        cardsSeen += n;
        if (cardsToStraight == 4 && i == 3 && counts[12] == 1) {
            // Special case, ace low straight.
            hasStraight = true;
            break;
        }
    }

    if (hasFlush && hasStraight) {
        if (counts[8] == 1 && counts[12] == 1) return PokerHand::ROYAL_FLUSH;
        else return PokerHand::STRAIGHT_FLUSH;
    }
    if (hasFlush) return PokerHand::FLUSH;
    if (hasStraight) return PokerHand::STRAIGHT;
    if (hasFour) return PokerHand::FOUR_OF_A_KIND;
    if (hasThree) {
        if (hasPair) return PokerHand::FULL_HOUSE;
        else return PokerHand::THREE_OF_A_KIND;
    }
    if (hasTwoPair) return PokerHand::TWO_PAIR;
    if (hasHighPair) return PokerHand::HIGH_PAIR;
    if (hasPair) return PokerHand::PAIR;
    return PokerHand::HIGH_CARD;
}

int VideoPoker::score(PokerHand handType) {
    switch(handType) {
        case PokerHand::ROYAL_FLUSH: return 800;
        case PokerHand::STRAIGHT_FLUSH: return 40;
        case PokerHand::FOUR_OF_A_KIND: return 20;
        case PokerHand::FULL_HOUSE: return 9;
        case PokerHand::FLUSH: return 6;
        case PokerHand::STRAIGHT: return 5;
        case PokerHand::THREE_OF_A_KIND: return 3;
        case PokerHand::TWO_PAIR: return 2;
        case PokerHand::HIGH_PAIR: return 1;
        case PokerHand::PAIR: return 0;
        case PokerHand::HIGH_CARD: return 0;
        default: throw std::runtime_error("Invalid Hand");
    }
}