#include "hand.h"

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
