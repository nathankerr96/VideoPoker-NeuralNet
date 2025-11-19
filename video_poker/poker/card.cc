#include "card.h"

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
